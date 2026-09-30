"""Builds the retrieval stack from settings.

Order matters: the Haystack ChromaDocumentStore must be opened AFTER ChunkStore created the
collection, otherwise it would create it with its own defaults (L2 distance, no embedding
metadata) and ChunkStore would then refuse it.
"""

from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from mara.core.config import Settings
from mara.llm.base import LLMProvider
from mara.retrieval.bm25_index import BM25Index
from mara.retrieval.hybrid import HaystackHybridRetriever


def build_chroma_document_store(settings: Settings) -> ChromaDocumentStore:
    if settings.chroma_persist_path:
        return ChromaDocumentStore(
            collection_name=settings.chroma_collection,
            persist_path=settings.chroma_persist_path,
            distance_function="cosine",
        )
    return ChromaDocumentStore(
        collection_name=settings.chroma_collection,
        host=settings.chroma_host,
        port=settings.chroma_port,
        distance_function="cosine",
    )


def ranker_factory(settings: Settings):  # noqa: ANN201 - returns a Haystack component factory
    if settings.reranker == "none":
        return None
    from haystack_integrations.components.rankers.sentence_transformers import (
        SentenceTransformersSimilarityRanker,
    )

    def make(top_k: int):  # noqa: ANN202
        # scale_score=True: sigmoid of the cross-encoder logit, so scores are comparable
        # across queries (a plain logit is not).
        return SentenceTransformersSimilarityRanker(
            model=settings.reranker_model, top_k=top_k, scale_score=True
        )

    return make


def build_retriever(
    settings: Settings, llm: LLMProvider, bm25: BM25Index, chroma_store: ChromaDocumentStore
) -> HaystackHybridRetriever:
    return HaystackHybridRetriever(
        llm,
        bm25,
        chroma_store,
        ranker_factory(settings),
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
        remote_chroma=not settings.chroma_persist_path,
    )
