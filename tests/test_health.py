from uuid import uuid4

import chromadb
import fakeredis
import pytest
from fastapi.testclient import TestClient

import mara.api.main as main
from mara.core.config import Settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.redis.Redis, "from_url", lambda url: fakeredis.FakeAsyncRedis())

    async def chroma_ok(settings):
        return "ok"

    monkeypatch.setattr(main, "_check_chroma", chroma_ok)

    def make(**overrides) -> TestClient:
        overrides.setdefault("chroma_collection", f"h_{uuid4().hex[:8]}")
        settings = Settings(_env_file=None, **overrides)
        return TestClient(main.create_app(settings, chroma_client=chromadb.EphemeralClient()))

    return make


def test_health_ok_when_everything_is_up(client):
    with client(gemini_api_key="k") as c:
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["llm"] == {"provider": "gemini", "configured": True}
    assert body["checks"]["store"] == {"ready": True}


def test_health_degraded_but_200_without_api_key(client):
    with client(gemini_api_key=None) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["llm"]["configured"] is False
    assert "GEMINI_API_KEY" in body["checks"]["llm"]["error"]
    assert body["checks"]["store"] == {"ready": False}


def test_health_degraded_when_store_rejects_embedding_config(monkeypatch, client):
    """A collection built with another embedding model must not be silently reused."""
    name = f"h_{uuid4().hex[:8]}"
    from mara.core.chunk_store import ChunkStore

    ChunkStore(chromadb.EphemeralClient(), name, "openai/text-embedding-3-small", 1536)
    with client(gemini_api_key="k", chroma_collection=name) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert "EmbeddingMismatchError" in body["checks"]["store"]["error"]
