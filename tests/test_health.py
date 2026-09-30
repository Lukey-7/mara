from uuid import uuid4

import chromadb

from mara.core.chunk_store import ChunkStore


def test_health_ok_when_everything_is_up(make_client):
    with make_client() as c:
        body = c.get("/health").json()
    assert body["status"] == "ok", body
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["llm"] == {"provider": "gemini", "configured": True}
    assert body["checks"]["store"] == {"ready": True}
    assert body["checks"]["retriever"]["ready"] is True
    assert body["checks"]["embeddings"]["model"] == "fake-embed"


def test_health_degraded_but_200_without_api_key(make_client):
    with make_client(gemini_api_key=None) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["llm"]["configured"] is False
    assert "GEMINI_API_KEY" in body["checks"]["llm"]["error"]
    # embeddings do not need the key, so ingestion and search still work
    assert body["checks"]["store"] == {"ready": True}
    assert body["checks"]["retriever"]["ready"] is True


def test_health_degraded_when_store_rejects_embedding_config(make_client, tmp_path):
    """A collection built with another embedding model must not be silently reused."""
    name = f"h_{uuid4().hex[:8]}"
    path = str(tmp_path / "chroma")
    ChunkStore(chromadb.PersistentClient(path), name, "openai/text-embedding-3-small", 1536)

    with make_client(chroma_collection=name, chroma_persist_path=path) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert "EmbeddingMismatchError" in body["checks"]["store"]["error"]
    assert body["checks"]["retriever"]["ready"] is False
