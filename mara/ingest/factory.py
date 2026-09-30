"""Builds the ingestion pipeline from settings (same Factory pattern as mara.llm.factory)."""

from collections.abc import Sequence

from chromadb.api import ClientAPI

from mara.core.chunk_store import ChunkStore
from mara.core.config import Settings
from mara.ingest.chunking import Chunker, FixedChunker, SemanticChunker
from mara.ingest.embeddings import ProviderEmbedding
from mara.ingest.pipeline import IngestionPipeline, SecondaryIndex
from mara.llm.base import LLMProvider


def embedding_label(llm: LLMProvider) -> str:
    return f"{llm.embedding_provider}/{llm.embedding_model}"


def build_store(settings: Settings, client: ClientAPI, llm: LLMProvider) -> ChunkStore:
    return ChunkStore(
        client,
        settings.chroma_collection,
        embedding_model=embedding_label(llm),
        embedding_dimensions=llm.embedding_dimensions,
    )


def build_chunker(settings: Settings, llm: LLMProvider) -> Chunker:
    if settings.chunking_strategy == "fixed":
        return FixedChunker(settings.fixed_chunk_size_tokens, settings.fixed_chunk_overlap_tokens)
    return SemanticChunker(
        ProviderEmbedding(llm, batch_size=settings.embedding_batch_size),
        breakpoint_percentile=settings.semantic_breakpoint_percentile,
        buffer_size=settings.semantic_buffer_size,
        max_chunk_chars=settings.max_chunk_chars,
    )


def build_pipeline(
    settings: Settings,
    store: ChunkStore,
    llm: LLMProvider,
    indexes: Sequence[SecondaryIndex] = (),
) -> IngestionPipeline:
    return IngestionPipeline(store, build_chunker(settings, llm), llm, indexes=indexes)
