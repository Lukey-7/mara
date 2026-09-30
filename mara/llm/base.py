"""The LLMProvider interface and the types every provider shares.

Agents depend only on `LLMProvider`, never on a vendor SDK. Swapping Gemini for OpenAI
(or a fake in tests) is a config change, not a code change: the Strategy pattern.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False  # set by CachedLLM on a cache hit; the trace reports it


class LLMError(Exception):
    """A provider call failed and retrying will not help (bad request, auth, etc.)."""


class RetryableLLMError(LLMError):
    """A provider call failed transiently (429, 5xx, timeout); safe to retry with backoff."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str  # "gemini" | "openai" | "fake"; part of every cache key
    model: str
    embedding_model: str

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Return one completion. With `json_schema`, the provider asks the model for JSON
        matching that Pydantic model (validation happens in the caller)."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...
