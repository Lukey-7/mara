"""CompositeProvider: one LLMProvider whose text generation and embeddings come from
different backends (e.g. Gemini for generation, a local sentence-transformers model for
embeddings). Also lets ingestion and search work with no API key at all."""

from typing import Protocol

from pydantic import BaseModel

from mara.llm.base import EmbedKind, LLMError, LLMResponse


class TextGenerator(Protocol):
    name: str
    model: str

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse: ...


class Embedder(Protocol):
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int | None

    async def embed(
        self, texts: list[str], *, kind: EmbedKind = "document"
    ) -> list[list[float]]: ...


class CompositeProvider:
    def __init__(
        self, generator: TextGenerator | None, embedder: Embedder, missing_llm_reason: str
    ) -> None:
        self._generator = generator
        self._embedder = embedder
        self._missing = missing_llm_reason
        self.name = generator.name if generator else "none"
        self.model = generator.model if generator else "none"
        self.embedding_provider = embedder.embedding_provider
        self.embedding_model = embedder.embedding_model
        self.embedding_dimensions = embedder.embedding_dimensions

    @property
    def has_generator(self) -> bool:
        return self._generator is not None

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        if self._generator is None:
            raise LLMError(f"LLM not configured: {self._missing}")
        return await self._generator.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens,
            json_schema=json_schema,
        )  # fmt: skip

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        return await self._embedder.embed(texts, kind=kind)
