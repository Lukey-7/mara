"""POST /search through the real Haystack pipelines (fake embeddings, no reranker model)."""

import pytest

KB = {
    "raft.md": "---\ntitle: Raft\ntags: [consensus, raft]\npublished_date: 2024-01-10\n---\n"
    "## Election\nRaft elects one leader per term using randomised timeouts.\n"
    "## Replication\nThe raft leader replicates log entries to followers.\n",
    "paxos.md": "---\ntitle: Paxos\ntags: [consensus, paxos]\npublished_date: 2020-06-01\n---\n"
    "## Roles\nPaxos has proposers, acceptors and learners.\n",
    "pasta.md": "---\ntitle: Pasta\ntags: [cooking]\n---\n## Boil\n"
    "Cooking pasta needs salted water.\n",
}


@pytest.fixture
def client(make_client, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    for name, text in KB.items():
        (kb / name).write_text(text, encoding="utf-8")
    with make_client(knowledge_base_dir=str(kb)) as c:
        assert c.post("/ingest/kb").status_code == 200
        yield c


def search(client, query, **body):
    r = client.post("/search", json={"query": query, **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_hybrid_search_returns_ranked_chunks_with_metadata(client):
    body = search(client, "how does raft elect a leader?")

    assert body["mode"] == "hybrid" and body["rerank"] is False  # no reranker in tests
    assert body["score_kind"] == "rrf"
    assert body["stage_counts"]["bm25"] >= 1 and body["stage_counts"]["dense"] >= 1
    assert body["stage_counts"]["fused"] >= body["stage_counts"]["final"] >= 1
    top = body["chunks"][0]
    assert top["rank"] == 1 and "raft" in top["chunk"]["text"].lower()
    assert top["chunk"]["title"] == "Raft" and top["chunk"]["section"] in (
        "Election",
        "Replication",
    )
    assert body["latency_ms"] > 0


@pytest.mark.parametrize("mode,score_kind", [("bm25", "bm25"), ("dense", "cosine_similarity")])
def test_single_retriever_modes(client, mode, score_kind):
    body = search(client, "paxos acceptors", config={"mode": mode, "top_k": 3})
    assert body["mode"] == mode and body["score_kind"] == score_kind
    assert "paxos" in body["chunks"][0]["chunk"]["text"].lower()
    assert len(body["chunks"]) <= 3


def test_filters_apply_to_both_retrievers(client):
    only_cooking = search(client, "raft leader", filters={"tags": ["cooking"]})
    assert {c["chunk"]["title"] for c in only_cooking["chunks"]} == {"Pasta"}

    recent = search(client, "consensus", filters={"date_from": "2023-01-01"})
    assert {c["chunk"]["title"] for c in recent["chunks"]} == {"Raft"}

    none = search(client, "raft", filters={"source_types": ["pdf"]})
    assert none["chunks"] == [] and none["stage_counts"]["final"] == 0


def test_search_validation(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert (
        client.post("/search", json={"query": "x", "config": {"mode": "magic"}}).status_code == 422
    )
