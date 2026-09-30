import fakeredis
import pytest

from mara.agents.jobs import MemoryJobStore, RedisJobStore, TraceArchive
from mara.agents.state import ResearchState


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return MemoryJobStore()
    return RedisJobStore(fakeredis.FakeAsyncRedis(), ttl_s=60)


async def test_save_get_events_recent(store):
    a = ResearchState(job_id="a", question="first")
    b = ResearchState(job_id="b", question="second")
    await store.save(a)
    await store.save(b)

    got = await store.get("a")
    assert got is not None and got.question == "first" and got.status == "pending"
    assert await store.get("missing") is None

    e1 = await store.append_event("a", "planner", "3 sub-questions")
    e2 = await store.append_event("a", "researcher", "12 chunks")
    assert (e1.seq, e2.seq) == (1, 2)
    assert [e.message for e in await store.events("a")] == ["3 sub-questions", "12 chunks"]
    assert [e.seq for e in await store.events("a", after_seq=1)] == [2]
    assert await store.events("nope") == []

    a.status = "done"
    await store.save(a)
    recent = await store.recent(10)
    assert [s.job_id for s in recent] == ["b", "a"] or [s.job_id for s in recent] == ["a", "b"]
    assert next(s for s in recent if s.job_id == "a").status == "done"


def test_trace_archive_round_trip(tmp_path):
    archive = TraceArchive(tmp_path / "traces")
    state = ResearchState(job_id="abc", question="q", status="done", answer="A [1].")
    path = archive.write(state)
    assert path.name == "abc.json"
    assert archive.read("abc") == state
    assert archive.read("nope") is None
    assert archive.list_ids() == ["abc"]
