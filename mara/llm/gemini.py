"""Gemini implementation of LLMProvider, using the `google-genai` SDK (async client)."""

import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from mara.llm.base import LLMError, LLMResponse, RetryableLLMError

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        embedding_model: str,
        timeout_s: float = 60.0,
        client: genai.Client | None = None,  # injectable for tests
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        # HttpOptions.timeout is in milliseconds. SDK-level retries stay off: ResilientLLM owns
        # retry policy so it is applied once, identically, for every provider.
        self._client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
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
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if json_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = json_schema.model_json_schema()

        start = time.perf_counter()
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except errors.APIError as e:
            raise _translate(e) from e
        latency_ms = (time.perf_counter() - start) * 1000

        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            model=self.model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            latency_ms=latency_ms,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.aio.models.embed_content(
                model=self.embedding_model, contents=texts
            )
        except errors.APIError as e:
            raise _translate(e) from e
        return [list(e.values or []) for e in resp.embeddings or []]


def _translate(e: errors.APIError) -> LLMError:
    """Map SDK errors onto our two error kinds so retry logic stays provider-agnostic."""
    if e.code in RETRYABLE_STATUS:
        return RetryableLLMError(f"gemini {e.code}: {e.message}")
    return LLMError(f"gemini {e.code}: {e.message}")
