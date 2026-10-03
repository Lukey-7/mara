"""OpenAI implementation of LLMProvider, using the `openai` SDK (async client)."""

import time

import openai
from pydantic import BaseModel

from mara.llm.base import EmbedKind, LLMError, LLMResponse, RetryableLLMError
from mara.llm.schema_utils import inline_refs

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIProvider:
    name = "openai"
    embedding_provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        embedding_model: str,
        embedding_dimensions: int | None = None,
        timeout_s: float = 60.0,
        client: openai.AsyncOpenAI | None = None,  # injectable for tests
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        # max_retries=0: ResilientLLM owns retries, so they are not stacked twice.
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {"model": self.model, "messages": messages}
        # Reasoning models (configured with a reasoning effort) reject any temperature other
        # than the default, so it is only sent to non-reasoning models, and only if set.
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.__name__,
                    "schema": inline_refs(json_schema.model_json_schema()),
                    # strict mode demands every field be required + no extra keys; we validate
                    # with Pydantic ourselves, so non-strict keeps schemas simple.
                    "strict": False,
                },
            }

        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except RETRYABLE_ERRORS as e:
            raise RetryableLLMError(f"openai: {e}") from e
        except openai.OpenAIError as e:
            raise LLMError(f"openai: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        # OpenAI embeddings are symmetric: `kind` is intentionally unused.
        kwargs: dict = {"model": self.embedding_model, "input": texts}
        if self.embedding_dimensions is not None:
            kwargs["dimensions"] = self.embedding_dimensions
        try:
            resp = await self._client.embeddings.create(**kwargs)
        except RETRYABLE_ERRORS as e:
            raise RetryableLLMError(f"openai: {e}") from e
        except openai.OpenAIError as e:
            raise LLMError(f"openai: {e}") from e
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
