"""End-to-end through FastAPI with fakes: fakeredis, EphemeralClient, FakeLLM, stubbed fetch."""

from pathlib import Path
from uuid import uuid4

import chromadb
import fakeredis
import pytest
from fastapi.testclient import TestClient

import mara.api.ingest_routes as routes
import mara.api.main as main
import mara.llm.factory as factory
from mara.core.config import Settings
from tests.fakes import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main.redis.Redis, "from_url", lambda url: fakeredis.FakeAsyncRedis())
    monkeypatch.setattr(factory, "build_base_provider", lambda settings: FakeLLM())

    async def chroma_ok(settings):
        return "ok"

    monkeypatch.setattr(main, "_check_chroma", chroma_ok)

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "raft.md").write_text(
        "---\ntitle: Raft notes\ntags: [raft]\n---\n## Election\nLeaders are elected per term.\n"
        "## Replication\nLogs are replicated to followers.\n",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        gemini_api_key="k",
        chunking_strategy="fixed",
        knowledge_base_dir=str(kb),
        cache_enabled=False,
        chroma_collection=f"api_{uuid4().hex[:8]}",  # EphemeralClient is process-shared
    )
    app = main.create_app(settings, chroma_client=chromadb.EphemeralClient())
    with TestClient(app) as c:
        yield c


def test_health_reports_store_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["checks"]["store"] == {"ready": True}


def test_pdf_upload_then_list_then_delete(client):
    pdf = (FIXTURES / "two_pages.pdf").read_bytes()
    r = client.post(
        "/ingest/pdf",
        files={"file": ("two_pages.pdf", pdf, "application/pdf")},
        data={"tags": "raft, test", "title": "Two pages"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ingested" and body["n_source_units"] == 2

    docs = client.get("/documents", params={"source_type": "pdf"}).json()
    assert len(docs) == 1
    assert docs[0]["title"] == "Two pages" and docs[0]["tags"] == ["raft", "test"]
    assert docs[0]["pages"] == 2 and docs[0]["published_date"] == "2024-02-15"

    again = client.post("/ingest/pdf", files={"file": ("x.pdf", pdf, "application/pdf")})
    assert again.json()["status"] == "skipped_duplicate"

    assert client.delete(f"/documents/{body['doc_id']}").status_code == 200
    assert client.get("/documents").json() == []


def test_non_pdf_upload_is_rejected(client):
    r = client.post("/ingest/pdf", files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert r.status_code == 400


def test_kb_rescan_is_idempotent(client):
    first = client.post("/ingest/kb").json()
    assert [r["status"] for r in first["results"]] == ["ingested"]
    assert first["results"][0]["n_source_units"] == 2  # two sections

    second = client.post("/ingest/kb").json()
    assert [r["status"] for r in second["results"]] == ["skipped_duplicate"]

    docs = client.get("/documents", params={"tag": "raft"}).json()
    assert docs[0]["title"] == "Raft notes" and docs[0]["source_type"] == "kb"


def test_url_ingest_collects_per_url_errors(client, monkeypatch):
    pages = {
        "https://ok.example/raft": "<html><head><title>Raft</title></head><body><article>"
        "<p>Raft is a consensus algorithm for managing a replicated log. It elects a leader.</p>"
        "<p>The leader replicates entries to followers and commits them safely.</p>"
        "</article></body></html>",
    }

    async def fake_fetch(url, timeout_s, max_bytes):
        if url in pages:
            return pages[url]
        raise routes.FetchError(f"HTTP 404 for {url}")

    monkeypatch.setattr(routes, "fetch_html", fake_fetch)

    r = client.post(
        "/ingest/url",
        json={"urls": ["https://ok.example/raft", "https://bad.example/x"], "tags": ["web"]},
    )
    body = r.json()
    assert r.status_code == 200, r.text
    assert [x["status"] for x in body["results"]] == ["ingested"]
    assert body["results"][0]["title"] == "Raft"
    assert "HTTP 404" in body["errors"]["https://bad.example/x"]

    docs = client.get("/documents", params={"q": "raft"}).json()
    assert docs[0]["url_or_path"] == "https://ok.example/raft"


def test_documents_filter_validation(client):
    assert client.get("/documents", params={"source_type": "bogus"}).status_code == 422
