"""Research jobs: start one (202 + job_id), poll it, or stream its agent steps over SSE."""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mara.agents.jobs import JobStore, TraceArchive
from mara.agents.runner import ResearchService
from mara.agents.state import JobStatus, ResearchOptions, ResearchState

router = APIRouter(tags=["research"])


def get_research_service(request: Request) -> ResearchService:
    svc = request.app.state.research
    if svc is None:
        raise HTTPException(503, f"research not available: {request.app.state.research_error}")
    return svc


def get_job_store(request: Request) -> JobStore:
    store = request.app.state.jobs
    if store is None:
        raise HTTPException(503, "job store not available")
    return store


Service = Annotated[ResearchService, Depends(get_research_service)]
Jobs = Annotated[JobStore, Depends(get_job_store)]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    options: ResearchOptions = Field(default_factory=ResearchOptions)


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus


class JobSummary(BaseModel):
    job_id: str
    question: str
    status: JobStatus
    created_at: datetime
    finished_at: datetime | None
    n_sources: int


@router.post("/research", response_model=JobCreated, status_code=202)
async def start_research(req: ResearchRequest, svc: Service) -> JobCreated:
    state = await svc.start(req.question, req.options)
    return JobCreated(job_id=state.job_id, status=state.status)


@router.get("/research", response_model=list[JobSummary])
async def list_research(jobs: Jobs, limit: int = 20) -> list[JobSummary]:
    return [
        JobSummary(
            job_id=s.job_id,
            question=s.question,
            status=s.status,
            created_at=s.created_at,
            finished_at=s.finished_at,
            n_sources=len(s.sources),
        )  # fmt: skip
        for s in await jobs.recent(limit)
    ]


@router.get("/research/{job_id}", response_model=ResearchState)
async def get_research(job_id: str, jobs: Jobs, request: Request) -> ResearchState:
    state = await jobs.get(job_id)
    if state is None:  # expired from the live store: try the archive
        archive: TraceArchive | None = request.app.state.trace_archive
        state = await asyncio.to_thread(archive.read, job_id) if archive else None
    if state is None:
        raise HTTPException(404, "job not found")
    return state


@router.get("/research/{job_id}/events")
async def stream_events(job_id: str, jobs: Jobs, request: Request) -> StreamingResponse:
    """Server-sent events: one `step` event per agent message, then a final `done` event.
    Implemented by polling the job store, so it works with Redis or the in-memory store
    and survives the client reconnecting (send `Last-Event-ID` to resume)."""
    if await jobs.get(job_id) is None:
        raise HTTPException(404, "job not found")
    last = int(request.headers.get("last-event-id", 0) or 0)

    async def gen() -> AsyncIterator[str]:
        after = last
        idle = 0.0
        while True:
            for ev in await jobs.events(job_id, after):
                after = ev.seq
                idle = 0.0
                yield f"id: {ev.seq}\nevent: step\ndata: {ev.model_dump_json()}\n\n"
            state = await jobs.get(job_id)
            if state is None or state.status in ("done", "failed"):
                payload = json.dumps({"status": state.status if state else "unknown"})
                yield f"event: done\ndata: {payload}\n\n"
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.3)
            idle += 0.3
            if idle >= 15:  # keep proxies from closing a quiet connection
                idle = 0.0
                yield ": keep-alive\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
