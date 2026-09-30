"""Local embeddings with sentence-transformers: no API key, no rate limit, runs on a CPU.

Default model BAAI/bge-small-en-v1.5 (33M params, 384 dims, ~130 MB on disk). bge models
want a short instruction prefixed to *queries* (not passages) for retrieval.
"""

import asyncio
import logging

from mara.llm.base import EmbedKind

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddingProvider:
    """Embedding-only provider. `generate` is intentionally absent: compose it with an LLM
    provider through `CompositeProvider`."""

    embedding_provider = "local"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep it local

        self.embedding_model = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        self.embedding_dimensions: int | None = self._model.get_sentence_embedding_dimension()
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        log.info("loaded local embedding model %s (%s dims)", model_name, self.embedding_dimensions)

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        if kind == "query" and self._query_prefix:
            texts = [self._query_prefix + t for t in texts]
        # encode() is CPU-bound and synchronous: keep it off the event loop.
        vectors = await asyncio.to_thread(
            self._model.encode,
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # unit vectors: cosine == dot product
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
