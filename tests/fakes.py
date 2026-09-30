"""Test doubles. FakeLLM satisfies the LLMProvider protocol with no network or API key."""

import hashlib
from collections.abc import Callable

from pydantic import BaseModel

from mara.llm.base import EmbedKind, LLMResponse


class FakeLLM:
    name = "fake"
    model = "fake-model"
    embedding_provider = "fake"
    embedding_model = "fake-embed"

    def __init__(
        self,
        replies: list[str | Exception] | None = None,
        dim: int = 4,
        embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.dim = dim
        self.embedding_dimensions = dim
        self._embedder = embedder
        self.generate_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.generate_calls.append(
            dict(prompt=prompt, system=system, temperature=temperature, json_schema=json_schema)
        )
        reply = self.replies.pop(0) if self.replies else f"echo: {prompt}"
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, model=self.model, input_tokens=10, output_tokens=5)

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        self.embed_calls.append(list(texts))
        embedder = self._embedder or self._hash_embedder
        return [embedder(t) for t in texts]

    def _hash_embedder(self, text: str) -> list[float]:
        # Deterministic per text (independent of batch position); k/256 values are exactly
        # representable in float32, so cached vectors round-trip bit-for-bit.
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 256 for b in digest[: self.dim]]


def topic_embedder(topics: dict[str, int], dim: int = 4) -> Callable[[str], list[float]]:
    """Embedder whose vectors point along one axis per topic keyword, so sentences about
    the same topic are identical and sentences about different topics are orthogonal.
    Lets tests exercise semantic chunking with no model."""

    def embed(text: str) -> list[float]:
        vec = [0.0] * dim
        for word, axis in topics.items():
            if word in text.lower():
                vec[axis] = 1.0
        if not any(vec):
            vec[dim - 1] = 1.0
        return vec

    return embed
