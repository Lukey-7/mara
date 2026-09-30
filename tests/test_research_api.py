"""POST /research end to end: real ingestion + Haystack retrieval, scripted FakeLLM."""

import json
import time

import pytest

from tests.agent_fakes import critique_json, notes_from_prompt, plan_json, writer_json
from tests.fakes import FakeLLM, topic_embedder
from tests.test_search_api import KB

ANSWER = "Raft elects one leader per term [1]. No evidence was found for swallows."


def research_llm() -> FakeLLM:
    return FakeLLM(
        embedder=topic_embedder({"raft": 0, "paxos": 1, "cooking": 2}),
        replies_by_schema={
            "Plan": [plan_json(("How does raft elect a leader?", ["kb"]),
                               ("What is the airspeed of a swallow?", ["kb"]))],
            "SummarizerOutput": [notes_from_prompt()],
            "Critique": [critique_json(gaps=["q2"])],
            "WriterOutput": [writer_json(ANSWER)],
        },
    )  # fmt: skip


@pytest.fixture
def client(make_client, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    for name, text in KB.items():
        (kb / name).write_text(text, encoding="utf-8")
    with make_client(llm=research_llm(), knowledge_base_dir=str(kb)) as c:
        assert c.post("/ingest/kb").status_code == 200
        yield c


def wait_done(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/research/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_research_job_runs_in_background_and_exposes_trace(client, tmp_path):
    r = client.post("/research", json={"question": "How does Raft elect a leader, and swallows?"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "pending"

    body = wait_done(client, job_id)
    assert body["status"] == "done", body["warnings"]
    assert body["answer"] == ANSWER
    assert [s["agent"] for s in body["trace"]] == [
        "planner",
        "researcher",
        "summarizer",
        "critic",
        "writer",
    ]
    assert body["sources"][0]["title"] == "Raft" and body["sources"][0]["n"] == 1
    assert body["evidence"]["q1"][0]["chunk"]["title"] == "Raft"
    assert body["citation_coverage"] == 1.0
    # hybrid retrieval always returns candidates, so both sub-questions got verified notes and
    # code overrides the model's claimed gap for q2
    assert body["critique"]["covered"] == ["q1", "q2"] and body["critique"]["gaps"] == []
    assert (tmp_path / "traces" / f"{job_id}.json").is_file()  # archived for history

    listing = client.get("/research").json()
    assert listing[0]["job_id"] == job_id and listing[0]["status"] == "done"


def test_events_stream_ends_with_done(client):
    job_id = client.post("/research", json={"question": "How does Raft elect a leader?"}).json()[
        "job_id"
    ]
    steps, done = [], None
    with client.stream("GET", f"/research/{job_id}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        event = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if event == "step":
                    steps.append(payload)
                elif event == "done":
                    done = payload
                    break
    assert done == {"status": "done"}
    agents = [s["agent"] for s in steps]
    assert agents[0] == "orchestrator" and "planner" in agents and agents[-1] == "orchestrator"
    assert [s["seq"] for s in steps] == list(range(1, len(steps) + 1))


def test_unknown_job_is_404_and_validation(client):
    assert client.get("/research/nope").status_code == 404
    assert client.get("/research/nope/events").status_code == 404
    assert client.post("/research", json={"question": "hi"}).status_code == 422


def test_research_needs_a_configured_llm(make_client):
    with make_client(gemini_api_key=None) as c:
        r = c.post("/research", json={"question": "How does Raft elect a leader?"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
    with make_client(gemini_api_key=None) as c:
        assert c.get("/health").json()["checks"]["research"]["ready"] is False
