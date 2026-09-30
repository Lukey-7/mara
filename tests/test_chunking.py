"""Semantic vs fixed chunking, exercised with a topic-axis fake embedder (no model)."""

import pytest

from mara.ingest.chunking import FixedChunker, SemanticChunker, make_chunk_id
from mara.ingest.embeddings import ProviderEmbedding
from mara.ingest.loaders import SourceDocument
from tests.fakes import FakeLLM, topic_embedder

RAFT = (
    "Raft elects a single leader per term. "
    "The raft leader accepts client requests and replicates them. "
    "If the raft leader crashes, followers time out and start a new election. "
)
COOKING = (
    "Cooking pasta starts with salted boiling water. "
    "Fresh cooking herbs are added at the end. "
    "A good cooking pan makes the sauce even. "
)


def src(text: str, **kw) -> SourceDocument:
    base = dict(source_type="kb", title="T", url_or_path="kb/t.md", section="S", tags=["x"])
    return SourceDocument(text=text, **{**base, **kw})


@pytest.fixture
def semantic() -> SemanticChunker:
    llm = FakeLLM(embedder=topic_embedder({"raft": 0, "cooking": 1}))
    return SemanticChunker(
        ProviderEmbedding(llm, batch_size=10), breakpoint_percentile=90, buffer_size=0
    )


async def test_semantic_chunker_splits_at_the_topic_shift(semantic):
    chunks = await semantic.chunk("doc1", [src(RAFT + COOKING)])

    assert len(chunks) == 2
    assert "Raft" in chunks[0].text and "Cooking" not in chunks[0].text
    assert "Cooking pasta" in chunks[1].text and "raft" not in chunks[1].text.lower()


async def test_semantic_chunker_keeps_single_topic_together(semantic):
    chunks = await semantic.chunk("doc1", [src(RAFT)])
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Raft elects")


async def test_chunks_inherit_source_metadata_and_get_stable_ids(semantic):
    chunks = await semantic.chunk("doc1", [src(RAFT + COOKING, page=3)])

    c = chunks[0]
    assert (c.doc_id, c.title, c.section, c.page, c.tags) == ("doc1", "T", "S", 3, ["x"])
    assert c.chunk_id == make_chunk_id("doc1", 0, c.text)
    assert chunks[1].chunk_id == make_chunk_id("doc1", 1, chunks[1].text)
    again = await semantic.chunk("doc1", [src(RAFT + COOKING, page=3)])
    assert [x.chunk_id for x in again] == [x.chunk_id for x in chunks]


async def test_semantic_chunker_caps_giant_chunks():
    llm = FakeLLM(embedder=topic_embedder({"raft": 0}))
    chunker = SemanticChunker(
        ProviderEmbedding(llm, batch_size=10), breakpoint_percentile=100, max_chunk_chars=300
    )
    chunks = await chunker.chunk("d", [src(RAFT * 6)])  # ~1000 chars, one topic, no split

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


async def test_fixed_chunker_produces_bounded_windows_regardless_of_topic():
    chunker = FixedChunker(chunk_size_tokens=40, overlap_tokens=10)
    chunks = await chunker.chunk("d", [src(RAFT + COOKING, page=1), src(COOKING, page=2)])

    assert len(chunks) >= 3
    assert {c.page for c in chunks} == {1, 2}
    page1 = [c.text for c in chunks if c.page == 1]
    assert len(page1) >= 2  # ~55 words do not fit one 40-token window
    assert all(len(t.split()) <= 40 for t in page1)
    assert chunks[0].chunk_id == make_chunk_id("d", 0, chunks[0].text)
