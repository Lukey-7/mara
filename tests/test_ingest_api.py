"""End-to-end through FastAPI with fakes: fakeredis, embedded Chroma, FakeLLM, stubbed fetch."""

from pathlib import Path

import pytest

import mara.api.ingest_routes as routes

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(make_client, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "raft.md").write_text(
        "---\ntitle: Raft notes\ntags: [raft]\n---\n## Election\nLeaders are elected per term.\n"
        "## Replication\nLogs are replicated to followers.\n",
        encoding="utf-8",
    )
    with make_client(knowledge_base_dir=str(kb)) as c:
        yield c


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

    assert client.get("/health").json()["checks"]["retriever"]["bm25_documents"] == body["n_chunks"]
    assert client.delete(f"/documents/{body['doc_id']}").status_code == 200
    assert client.get("/documents").json() == []
    assert client.get("/health").json()["checks"]["retriever"]["bm25_documents"] == 0


def test_pdf_published_date_override(client):
    pdf = (FIXTURES / "two_pages.pdf").read_bytes()
    r = client.post(
        "/ingest/pdf",
        files={"file": ("two_pages.pdf", pdf, "application/pdf")},
        data={"published_date": "2014-05-20"},
    )
    assert r.status_code == 200, r.text
    assert client.get("/documents").json()[0]["published_date"] == "2014-05-20"


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
