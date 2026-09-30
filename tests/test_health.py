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
        settings = Settings(_env_file=None, **overrides)
        return TestClient(main.create_app(settings))

    return make


def test_health_ok_when_everything_is_up(client):
    with client(gemini_api_key="k") as c:
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["llm"] == {"provider": "gemini", "configured": True}


def test_health_degraded_but_200_without_api_key(client):
    with client(gemini_api_key=None) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["llm"]["configured"] is False
    assert "GEMINI_API_KEY" in body["checks"]["llm"]["error"]
