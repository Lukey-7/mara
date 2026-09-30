"""Job storage: live state + event log in Redis (or memory when Redis is down), and a JSON
file per finished run for history."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

import redis.asyncio as redis
from pydantic import BaseModel, Field

from mara.agents.state import ResearchState
from mara.core.schema import utc_now

log = logging.getLogger(__name__)


class JobEvent(BaseModel):
    seq: int
    ts: datetime = Field(default_factory=utc_now)
    agent: str
    message: str


class JobStore(Protocol):
    async def save(self, state: ResearchState) -> None: ...

    async def get(self, job_id: str) -> ResearchState | None: ...

    async def append_event(self, job_id: str, agent: str, message: str) -> JobEvent: ...

    async def events(self, job_id: str, after_seq: int = 0) -> list[JobEvent]: ...

    async def recent(self, limit: int = 20) -> list[ResearchState]: ...


class MemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ResearchState] = {}
        self._events: dict[str, list[JobEvent]] = {}

    async def save(self, state: ResearchState) -> None:
        self._jobs[state.job_id] = state.model_copy(deep=True)

    async def get(self, job_id: str) -> ResearchState | None:
        s = self._jobs.get(job_id)
        return s.model_copy(deep=True) if s else None

    async def append_event(self, job_id: str, agent: str, message: str) -> JobEvent:
        events = self._events.setdefault(job_id, [])
        ev = JobEvent(seq=len(events) + 1, agent=agent, message=message)
        events.append(ev)
        return ev

    async def events(self, job_id: str, after_seq: int = 0) -> list[JobEvent]:
        return [e for e in self._events.get(job_id, []) if e.seq > after_seq]

    async def recent(self, limit: int = 20) -> list[ResearchState]:
        jobs = sorted(self._jobs.values(), key=lambda s: s.created_at, reverse=True)
        return [j.model_copy(deep=True) for j in jobs[:limit]]


class RedisJobStore:
    """`mara:job:<id>` = state JSON, `mara:job:<id>:events` = list of event JSON,
    `mara:jobs` = sorted set by created_at. All with a TTL: history lives in files."""

    def __init__(self, client: redis.Redis, ttl_s: int = 7 * 24 * 3600) -> None:
        self._r, self._ttl = client, ttl_s

    async def save(self, state: ResearchState) -> None:
        key = f"mara:job:{state.job_id}"
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.set(key, state.model_dump_json(), ex=self._ttl)
            pipe.zadd("mara:jobs", {state.job_id: state.created_at.timestamp()})
            await pipe.execute()

    async def get(self, job_id: str) -> ResearchState | None:
        raw = await self._r.get(f"mara:job:{job_id}")
        return ResearchState.model_validate_json(raw) if raw else None

    async def append_event(self, job_id: str, agent: str, message: str) -> JobEvent:
        key = f"mara:job:{job_id}:events"
        seq = await self._r.llen(key) + 1
        ev = JobEvent(seq=seq, agent=agent, message=message)
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.rpush(key, ev.model_dump_json())
            pipe.expire(key, self._ttl)
            await pipe.execute()
        return ev

    async def events(self, job_id: str, after_seq: int = 0) -> list[JobEvent]:
        raws = await self._r.lrange(f"mara:job:{job_id}:events", after_seq, -1)
        return [JobEvent.model_validate_json(r) for r in raws]

    async def recent(self, limit: int = 20) -> list[ResearchState]:
        ids = await self._r.zrevrange("mara:jobs", 0, limit - 1)
        states = [await self.get(i.decode() if isinstance(i, bytes) else i) for i in ids]
        return [s for s in states if s]


class TraceArchive:
    """One JSON file per finished run: the durable history the interview demo reads from."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def write(self, state: ResearchState) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{state.job_id}.json"
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return path

    def read(self, job_id: str) -> ResearchState | None:
        path = self._dir / f"{job_id}.json"
        if not path.is_file():
            return None
        return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted((p.stem for p in self._dir.glob("*.json")), reverse=True)


def dump_state(state: ResearchState) -> dict:
    return json.loads(state.model_dump_json())
