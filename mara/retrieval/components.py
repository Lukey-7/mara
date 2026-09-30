"""Custom Haystack components that bridge the pipeline to our own abstractions."""

import asyncio

from haystack import Document, component
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from mara.llm.base import LLMProvider


@component
class ProviderQueryEmbedder:
    """Embeds the query through our LLMProvider stack (cache + rate limit + retry), with
    `kind="query"` so asymmetric models use their query task type."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict:
        return {"embedding": asyncio.run(self._llm.embed([text], kind="query"))[0]}

    @component.output_types(embedding=list[float])
    async def run_async(self, text: str) -> dict:
        return {"embedding": (await self._llm.embed([text], kind="query"))[0]}


@component
class ChromaDenseRetriever:
    """Wraps chroma-haystack's ChromaEmbeddingRetriever. Its `run_async` only supports remote
    (host/port) connections; for an embedded on-disk Chroma we run the sync method in a
    thread instead of failing."""

    def __init__(self, inner: ChromaEmbeddingRetriever, remote: bool) -> None:
        self._inner, self._remote = inner, remote

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> dict:
        return self._inner.run(query_embedding=query_embedding, filters=filters, top_k=top_k)

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> dict:
        if self._remote:
            return await self._inner.run_async(
                query_embedding=query_embedding, filters=filters, top_k=top_k
            )
        return await asyncio.to_thread(self.run, query_embedding, filters, top_k)


@component
class PassthroughRanker:
    """No-op reranker (RERANKER=none, and tests): keeps the fused order, truncates to top_k."""

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict:
        return {"documents": documents[: top_k or self._top_k]}

    @component.output_types(documents=list[Document])
    async def run_async(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> dict:
        return self.run(query, documents, top_k)
