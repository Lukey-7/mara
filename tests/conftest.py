"""Shared fixtures: an app wired with fakes (fakeredis, FakeLLM, embedded Chroma, no models)."""

from collections.abc import Callable
from uuid import uuid4

import fakeredis
import pytest
from fastapi.testclient import TestClient

import mara.api.main as main
from mara.core.config import Settings
from mara.llm.factory import wrap_provider
from tests.fakes import FakeLLM, topic_embedder

TOPICS = {"raft": 0, "paxos": 1, "cap": 2, "cooking": 3}


@pytest.fixture
def make_client(monkeypatch, tmp_path) -> Callable[..., TestClient]:
    """make_client(**settings_overrides, llm=FakeLLM(...)) -> TestClient (use as a context
    manager so the lifespan runs). Chroma is embedded under tmp_path; the reranker is off."""
    monkeypatch.setattr(main.redis.Redis, "from_url", lambda url, **kw: fakeredis.FakeAsyncRedis())

    def make(llm: FakeLLM | None = None, **overrides) -> TestClient:
        llm = llm or FakeLLM(embedder=topic_embedder(TOPICS))
        monkeypatch.setattr(main, "build_llm", lambda settings, r: wrap_provider(llm, settings, r))
        overrides.setdefault("gemini_api_key", "k")
        overrides.setdefault("embedding_provider", "gemini")  # never load the local model here
        overrides.setdefault("reranker", "none")  # never load the cross-encoder here
        overrides.setdefault("chunking_strategy", "fixed")
        overrides.setdefault("cache_enabled", False)
        overrides.setdefault("chroma_persist_path", str(tmp_path / "chroma"))
        overrides.setdefault("chroma_collection", f"t_{uuid4().hex[:8]}")
        overrides.setdefault("knowledge_base_dir", str(tmp_path / "kb"))
        overrides.setdefault("trace_dir", str(tmp_path / "traces"))
        return TestClient(main.create_app(Settings(_env_file=None, **overrides)))

    return make
