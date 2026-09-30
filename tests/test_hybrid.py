"""HaystackHybridRetriever against embedded Chroma + in-memory BM25, fake embeddings."""

from datetime import date
from uuid import uuid4

import chromadb
import pytest
from haystack import Document, component
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from mara.core.chunk_store import ChunkStore
from mara.core.filters import MetadataFilter
from mara.ingest.chunking import FixedChunker
from mara.ingest.loaders import SourceDocument
from mara.ingest.pipeline import IngestionPipeline
from mara.retrieval.bm25_index import BM25Index
from mara.retrieval.hybrid import HaystackHybridRetriever, RetrievalConfig
from tests.fakes import FakeLLM, topic_embedder

TOPICS = {"raft": 0, "paxos": 1, "cooking": 2}


@component
class ReversingRanker:
    """Stands in for the cross-encoder: reverses the fused order and scores 0.9, 0.8, ..."""

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict:
        docs = list(reversed(documents))[: top_k or self.top_k]
        for i, d in enumerate(docs):
            d.score = 0.9 - i * 0.1
        return {"documents": docs}


def src(text, title, tags, pub=None, source_type="kb"):
    return SourceDocument(
        text=text, source_type=source_type, title=title, url_or_path=f"{title}.md",
        section="S", tags=tags, published_date=pub,
    )  # fmt: skip


@pytest.fixture
async def stack(tmp_path):
    path = str(tmp_path / "chroma")
    name = f"h_{uuid4().hex[:8]}"
    llm = FakeLLM(embedder=topic_embedder(TOPICS))
    store = ChunkStore(
        chromadb.PersistentClient(path), name, llm.embedding_model, llm.embedding_dimensions
    )
    bm25 = BM25Index()
    pipeline = IngestionPipeline(store, FixedChunker(64, 0), llm, indexes=[bm25])
    docs = [
        [src("Raft elects a single leader per term. The raft leader sends heartbeats.", "raft",
             ["consensus", "raft"], date(2024, 1, 1))],
        [src("Paxos has proposers and acceptors. Paxos needs a majority.", "paxos",
             ["consensus", "paxos"], date(2019, 1, 1))],
        [src("Cooking pasta needs salted water. Cooking well takes time.", "pasta", ["cooking"],
             source_type="web")],
    ]  # fmt: skip
    for d in docs:
        await pipeline.ingest(d)
    chroma_store = ChromaDocumentStore(
        collection_name=name, persist_path=path, distance_function="cosine"
    )
    return dict(llm=llm, store=store, bm25=bm25, pipeline=pipeline, chroma=chroma_store)


def retriever(stack, ranker_factory=None, **kw) -> HaystackHybridRetriever:
    return HaystackHybridRetriever(
        stack["llm"], stack["bm25"], stack["chroma"], ranker_factory, candidates=10, top_k=5,
        remote_chroma=False, **kw,
    )  # fmt: skip


async def test_bm25_mode(stack):
    res = await retriever(stack).retrieve(
        "raft leader heartbeats", config=RetrievalConfig(mode="bm25")
    )
    assert res.mode == "bm25" and res.score_kind == "bm25" and res.rerank is False
    assert res.chunks[0].chunk.title == "raft" and res.chunks[0].rank == 1
    assert res.stage_counts == {"bm25": res.stage_counts["bm25"], "final": len(res.chunks)}
    assert "dense" not in res.stage_counts


async def test_dense_mode_uses_query_embedding(stack):
    res = await retriever(stack).retrieve("paxos", config=RetrievalConfig(mode="dense", top_k=1))
    assert res.score_kind == "cosine_similarity"
    assert res.chunks[0].chunk.title == "paxos"
    assert res.chunks[0].score == pytest.approx(1.0, abs=1e-5)  # same topic axis
    assert stack["llm"].embed_calls[-1] == ["paxos"]  # query embedded once, as a query


async def test_hybrid_fuses_both_lists(stack):
    res = await retriever(stack).retrieve("raft leader")
    assert res.mode == "hybrid" and res.score_kind == "rrf"
    assert res.chunks[0].chunk.title == "raft"
    c = res.stage_counts
    assert c["fused"] <= c["bm25"] + c["dense"] and c["final"] == len(res.chunks)
    assert res.chunks[0].score > res.chunks[-1].score


async def test_filters_reach_both_retrievers(stack):
    r = retriever(stack)
    only_web = await r.retrieve("raft", MetadataFilter(source_types=["web"]))
    assert {c.chunk.title for c in only_web.chunks} == {"pasta"}
    by_tag = await r.retrieve("majority leader", MetadataFilter(tags=["paxos", "cooking"]))
    assert {c.chunk.title for c in by_tag.chunks} == {"paxos", "pasta"}
    recent = await r.retrieve("consensus", MetadataFilter(date_from=date(2020, 1, 1)))
    assert {c.chunk.title for c in recent.chunks} == {"raft"}
    by_doc = await r.retrieve("anything", MetadataFilter(doc_ids=["nope"]))
    assert by_doc.chunks == []


async def test_rerank_is_skipped_without_a_reranker_and_applied_with_one(stack):
    plain = await retriever(stack).retrieve("raft leader", config=RetrievalConfig(rerank=True))
    assert plain.rerank is False and plain.score_kind == "rrf"

    reranked = await retriever(stack, ranker_factory=ReversingRanker).retrieve(
        "raft leader", config=RetrievalConfig(rerank=True, top_k=3)
    )
    assert reranked.rerank is True and reranked.score_kind == "cross_encoder_probability"
    assert [c.score for c in reranked.chunks] == pytest.approx(
        [0.9, 0.8, 0.7][: len(reranked.chunks)]
    )
    assert [c.chunk.chunk_id for c in reranked.chunks] != [
        c.chunk.chunk_id for c in plain.chunks[:3]
    ]


async def test_delete_keeps_bm25_and_chroma_in_sync(stack):
    r = retriever(stack)
    doc_id = (await r.retrieve("paxos", config=RetrievalConfig(mode="bm25"))).chunks[0].chunk.doc_id
    await stack["pipeline"].delete_document(doc_id)

    after_bm25 = await r.retrieve("paxos", config=RetrievalConfig(mode="bm25"))
    after_dense = await r.retrieve("paxos", config=RetrievalConfig(mode="dense"))
    assert all(c.chunk.doc_id != doc_id for c in after_bm25.chunks + after_dense.chunks)
    assert await stack["bm25"].count() == await stack["store"].count()


async def test_pipelines_are_cached_per_configuration(stack):
    r = retriever(stack)
    assert r.pipeline("hybrid", False) is r.pipeline("hybrid", False)
    assert r.pipeline("bm25", False) is not r.pipeline("dense", False)
    assert set(r.pipeline("hybrid", False).graph.nodes) == {
        "bm25",
        "embedder",
        "dense",
        "joiner",
        "ranker",
    }
    assert set(r.pipeline("bm25", False).graph.nodes) == {"bm25", "ranker"}
