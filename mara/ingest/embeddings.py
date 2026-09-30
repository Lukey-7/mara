"""Adapter that lets LlamaIndex's node parsers embed through our LLMProvider.

LlamaIndex expects a `BaseEmbedding`; we hand it this wrapper so the semantic splitter's
sentence embeddings go through the same cache + rate limiter + retry stack as everything
else. Only the async path is implemented, because ingestion runs on the event loop
(`SemanticSplitterNodeParser.aget_nodes_from_documents`).
"""

import asyncio
from typing import Any

from llama_index.core.embeddings import BaseEmbedding

from mara.llm.base import LLMProvider


class ProviderEmbedding(BaseEmbedding):
    _provider: LLMProvider

    def __init__(self, provider: LLMProvider, batch_size: int, **kwargs: Any) -> None:
        super().__init__(
            model_name=f"{provider.name}/{provider.embedding_model}",
            embed_batch_size=batch_size,
            **kwargs,
        )
        self._provider = provider

    @classmethod
    def class_name(cls) -> str:
        return "ProviderEmbedding"

    # -- async: the real implementation --
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed(texts, kind="document")

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._provider.embed([text], kind="document"))[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._provider.embed([query], kind="query"))[0]

    # -- sync: only safe when no event loop is running (scripts, never the API) --
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return _run_sync(self._provider.embed(texts, kind="document"))

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return _run_sync(self._provider.embed([query], kind="query"))[0]


def _run_sync(coro):  # noqa: ANN001, ANN202 - tiny helper
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "ProviderEmbedding sync methods called inside an event loop; use the async node "
        "parser API (aget_nodes_from_documents) instead"
    )
