"""The query-time retrieval pipeline, built with Haystack components:

    query ─┬─ InMemoryBM25Retriever ───────────────────────┐
           └─ ProviderQueryEmbedder → ChromaEmbeddingRetriever ┴→ DocumentJoiner(RRF) → ranker

Each (mode, rerank) combination is its own small Pipeline graph, built lazily and cached, so
the eval can time "BM25 only" or "dense only" honestly instead of running the whole graph and
discarding stages. Metadata filters are passed to both retrievers in Haystack's filter
grammar (`MetadataFilter.to_haystack`).
"""

import time
from collections.abc import Callable
from typing import Literal, Protocol

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from pydantic import BaseModel, Field

from mara.core.filters import MetadataFilter, to_haystack
from mara.core.schema import Chunk
from mara.llm.base import LLMProvider
from mara.retrieval.bm25_index import BM25Index, document_to_chunk
from mara.retrieval.components import (
    ChromaDenseRetriever,
    PassthroughRanker,
    ProviderQueryEmbedder,
    Reranker,
)

RetrievalMode = Literal["bm25", "dense", "hybrid"]


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = "hybrid"
    rerank: bool = True
    candidates: int | None = None  # per retriever before fusion; None = settings default
    top_k: int | None = None  # final; None = settings default


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    chunk: Chunk


class RetrievalResult(BaseModel):
    query: str
    mode: RetrievalMode
    rerank: bool
    score_kind: str  # what `score` means for this configuration
    chunks: list[RetrievedChunk]
    stage_counts: dict[str, int] = Field(default_factory=dict)  # bm25 / dense / fused / final
    latency_ms: float


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult: ...


class HaystackHybridRetriever:
    def __init__(
        self,
        llm: LLMProvider,
        bm25: BM25Index,
        chroma_store: ChromaDocumentStore,
        ranker_factory: Callable[[int], Reranker] | None,  # None = no cross-encoder available
        candidates: int = 20,
        top_k: int = 8,
        remote_chroma: bool = True,  # False = embedded on-disk Chroma (no native async)
    ) -> None:
        self._llm, self._bm25, self._chroma = llm, bm25, chroma_store
        self._ranker_factory = ranker_factory
        self._remote = remote_chroma
        self._candidates, self._top_k = candidates, top_k
        self._pipelines: dict[tuple[RetrievalMode, bool], Pipeline] = {}

    # --- graph construction ------------------------------------------------------------

    def pipeline(self, mode: RetrievalMode, rerank: bool) -> Pipeline:
        key = (mode, rerank)
        if key not in self._pipelines:
            self._pipelines[key] = self._build(mode, rerank)
        return self._pipelines[key]

    def _build(self, mode: RetrievalMode, rerank: bool) -> Pipeline:
        p = Pipeline()
        if mode in ("bm25", "hybrid"):
            p.add_component("bm25", InMemoryBM25Retriever(self._bm25.store, top_k=self._candidates))
        if mode in ("dense", "hybrid"):
            p.add_component("embedder", ProviderQueryEmbedder(self._llm))
            inner = ChromaEmbeddingRetriever(self._chroma, top_k=self._candidates)
            p.add_component("dense", ChromaDenseRetriever(inner, remote=self._remote))
            p.connect("embedder.embedding", "dense.query_embedding")
        if mode == "hybrid":
            p.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
            p.connect("bm25.documents", "joiner.documents")
            p.connect("dense.documents", "joiner.documents")
        last = {"bm25": "bm25", "dense": "dense", "hybrid": "joiner"}[mode]

        if rerank and self._ranker_factory is not None:
            p.add_component("ranker", self._ranker_factory(self._top_k))
        else:
            p.add_component("ranker", PassthroughRanker(top_k=self._top_k))
        p.connect(f"{last}.documents", "ranker.documents")
        return p

    def warm_up(self) -> None:
        """Load the cross-encoder now rather than on the first request."""
        self.pipeline("hybrid", rerank=True).warm_up()

    @property
    def has_reranker(self) -> bool:
        return self._ranker_factory is not None

    # --- querying ---------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = config or RetrievalConfig()
        rerank = cfg.rerank and self.has_reranker
        top_k = cfg.top_k or self._top_k
        hs_filters = to_haystack(filters)
        pipe = self.pipeline(cfg.mode, rerank)

        data: dict[str, dict] = {"ranker": {"query": query, "top_k": top_k}}
        if cfg.mode in ("bm25", "hybrid"):
            data["bm25"] = {"query": query, "filters": hs_filters, "top_k": cfg.candidates}
        if cfg.mode in ("dense", "hybrid"):
            data["embedder"] = {"text": query}
            data["dense"] = {"filters": hs_filters, "top_k": cfg.candidates}

        start = time.perf_counter()
        out = await pipe.run_async(data, include_outputs_from={"bm25", "dense", "joiner"})
        latency_ms = (time.perf_counter() - start) * 1000

        docs: list[Document] = out["ranker"]["documents"][:top_k]
        counts = {name: len(out[name]["documents"]) for name in ("bm25", "dense") if name in out}
        if "joiner" in out:
            counts["fused"] = len(out["joiner"]["documents"])
        counts["final"] = len(docs)

        score_kind = _score_kind(cfg.mode, rerank)
        return RetrievalResult(
            query=query,
            mode=cfg.mode,
            rerank=rerank,
            score_kind=score_kind,
            chunks=[
                RetrievedChunk(rank=i + 1, score=_score(d, score_kind), chunk=document_to_chunk(d))
                for i, d in enumerate(docs)
            ],
            stage_counts=counts,
            latency_ms=latency_ms,
        )


def _score_kind(mode: RetrievalMode, rerank: bool) -> str:
    if rerank:
        return "cross_encoder_probability"
    return {"bm25": "bm25", "dense": "cosine_similarity", "hybrid": "rrf"}[mode]


def _score(doc: Document, kind: str) -> float:
    s = float(doc.score or 0.0)
    return 1.0 - s if kind == "cosine_similarity" else s  # Chroma reports cosine *distance*
