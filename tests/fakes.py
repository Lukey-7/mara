"""Test doubles. FakeLLM satisfies the LLMProvider protocol with no network or API key."""

from pydantic import BaseModel

from mara.llm.base import LLMResponse


class FakeLLM:
    name = "fake"
    model = "fake-model"
    embedding_model = "fake-embed"

    def __init__(self, replies: list[str | Exception] | None = None, dim: int = 4) -> None:
        self.replies = list(replies or [])
        self.dim = dim
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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        # Deterministic, distinct per text; values exactly representable in float32.
        return [[float(len(t)), float(i), 0.5, 0.25][: self.dim] for i, t in enumerate(texts)]
