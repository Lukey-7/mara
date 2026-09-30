"""Factory: builds the configured LLMProvider and stacks the decorators around it.

CachedLLM( ResilientLLM( GeminiProvider | OpenAIProvider ) )
   │            │                 └─ talks to the vendor SDK
   │            └─ rate limit + retry with exponential backoff
   └─ Redis cache (a hit skips everything below)
"""

import redis.asyncio as redis

from mara.core.cache import RedisCache
from mara.core.config import Settings
from mara.llm.base import LLMProvider
from mara.llm.cached import CachedLLM
from mara.llm.resilience import AsyncRateLimiter, ResilientLLM


def build_base_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        from mara.llm.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            embedding_model=settings.gemini_embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            timeout_s=settings.llm_timeout_s,
        )
    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        from mara.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")


def wrap_provider(
    base: LLMProvider, settings: Settings, redis_client: redis.Redis | None
) -> LLMProvider:
    provider: LLMProvider = ResilientLLM(
        base,
        AsyncRateLimiter(per_minute=settings.llm_requests_per_minute),
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_backoff_base_s,
        max_delay=settings.llm_backoff_max_s,
        embedding_limiter=AsyncRateLimiter(per_minute=settings.embedding_requests_per_minute),
    )
    if settings.cache_enabled and redis_client is not None:
        provider = CachedLLM(
            provider,
            RedisCache(redis_client),
            llm_ttl_s=settings.llm_cache_ttl_s,
            embedding_ttl_s=settings.embedding_cache_ttl_s,
            key_version=settings.cache_key_version,
        )
    return provider


def build_llm(settings: Settings, redis_client: redis.Redis | None) -> LLMProvider:
    return wrap_provider(build_base_provider(settings), settings, redis_client)
