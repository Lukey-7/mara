"""Ingestion pipeline + ChunkStore against an in-memory Chroma (EphemeralClient)."""

from datetime import date
from uuid import uuid4

import chromadb
import pytest

from mara.core.chunk_store import ChunkStore, EmbeddingMismatchError
from mara.core.filters import MetadataFilter
from mara.ingest.chunking import FixedChunker
from mara.ingest.loaders import SourceDocument
from mara.ingest.pipeline import IngestionPipeline
from tests.fakes import FakeLLM


@pytest.fixture
def client():
    return chromadb.EphemeralClient()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def store(client, llm):
    # EphemeralClient shares one in-process database, so each test gets its own collection.
    name = f"test_{uuid4().hex[:8]}"
    return ChunkStore(client, name, llm.embedding_model, llm.embedding_dimensions)


@pytest.fixture
def pipeline(store, llm):
    return IngestionPipeline(store, FixedChunker(chunk_size_tokens=64, overlap_tokens=0), llm)


def pdf_pages(n: int = 2, prefix: str = "Page") -> list[SourceDocument]:
    return [
        SourceDocument(
            text=f"{prefix} {i} talks about raft leader election and log replication. " * 3,
            source_type="pdf",
            title="Raft paper",
            url_or_path="raft.pdf",
            page=i,
            published_date=date(2014, 6, 1),
            tags=["consensus"],
        )
        for i in range(1, n + 1)
    ]


async def test_ingest_writes_chunks_with_metadata(pipeline, store):
    res = await pipeline.ingest(pdf_pages())

    assert res.status == "ingested" and res.n_source_units == 2 and res.n_chunks >= 2
    chunks = await store.all_chunks()
    assert len(chunks) == res.n_chunks
    c = chunks[0]
    assert c.doc_id == res.doc_id and c.source_type == "pdf" and c.title == "Raft paper"
    assert c.tags == ["consensus"] and c.published_date == date(2014, 6, 1)
    assert {x.page for x in chunks} == {1, 2}


async def test_ingest_is_idempotent(pipeline, store, llm):
    first = await pipeline.ingest(pdf_pages())
    second = await pipeline.ingest(pdf_pages())

    assert second.status == "skipped_duplicate" and second.doc_id == first.doc_id
    assert await store.count() == first.n_chunks
    assert len(llm.embed_calls) == 1  # no re-embedding


async def test_force_replaces_existing_chunks(pipeline, store):
    first = await pipeline.ingest(pdf_pages())
    replaced = await pipeline.ingest(pdf_pages(), force=True)

    assert replaced.status == "replaced" and replaced.doc_id == first.doc_id
    assert await store.count() == first.n_chunks


async def test_different_content_is_a_different_document(pipeline, store):
    a = await pipeline.ingest(pdf_pages(prefix="Alpha"))
    b = await pipeline.ingest(pdf_pages(prefix="Beta"))

    assert a.doc_id != b.doc_id
    docs = await store.list_documents()
    assert {d.doc_id for d in docs} == {a.doc_id, b.doc_id}
    assert all(d.n_chunks >= 2 and d.pages == 2 for d in docs)


async def test_list_documents_applies_filters(pipeline, store):
    await pipeline.ingest(pdf_pages())
    note = SourceDocument(
        text="Paxos notes. Proposers, acceptors, learners. " * 4,
        source_type="kb",
        title="Paxos",
        url_or_path="kb/paxos.md",
        section="Roles",
        tags=["consensus", "paxos"],
        published_date=date(2024, 1, 15),
    )
    await pipeline.ingest([note])

    kb = await store.list_documents(MetadataFilter(source_types=["kb"]))
    assert [d.title for d in kb] == ["Paxos"]
    both = await store.list_documents(MetadataFilter(tags=["consensus"]))
    assert {d.title for d in both} == {"Raft paper", "Paxos"}
    recent = await store.list_documents(MetadataFilter(date_from=date(2020, 1, 1)))
    assert [d.title for d in recent] == ["Paxos"]
    paxos_only = await store.list_documents(MetadataFilter(tags=["paxos"]))
    assert [d.title for d in paxos_only] == ["Paxos"]


async def test_delete_document(pipeline, store):
    res = await pipeline.ingest(pdf_pages())
    assert await store.delete_document(res.doc_id) == res.n_chunks
    assert await store.count() == 0
    assert await store.delete_document(res.doc_id) == 0


async def test_query_returns_similarity_best_first(pipeline, store, llm):
    await pipeline.ingest(pdf_pages())
    chunks = await store.all_chunks()
    target = chunks[-1]
    [vec] = await llm.embed([target.text])

    hits = await store.query(vec, top_k=3)
    assert hits[0][0].chunk_id == target.chunk_id
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1))

    filtered = await store.query(vec, top_k=3, filters=MetadataFilter(source_types=["web"]))
    assert filtered == []


def test_store_refuses_a_collection_built_with_another_embedding_model(client):
    name = f"shared_{uuid4().hex[:8]}"
    ChunkStore(client, name, "gemini/embedding-001", 768)
    with pytest.raises(EmbeddingMismatchError):
        ChunkStore(client, name, "openai/text-embedding-3-small", 768)
    with pytest.raises(EmbeddingMismatchError):
        ChunkStore(client, name, "gemini/embedding-001", 1536)
    ChunkStore(client, name, "gemini/embedding-001", 768)  # same config is fine
