"""Factory: builds the configured providers and stacks the decorators around them.

    CachedLLM( ResilientLLM( CompositeProvider(generator, embedder) ) )
       │            │                 ├─ generator: GeminiProvider | OpenAIProvider | None
       │            │                 └─ embedder:  LocalEmbeddingProvider | Gemini | OpenAI
       │            └─ rate limit + retry with exponential backoff
       └─ Redis cache (a hit skips everything below)

A missing API key does not stop the app: the composite still embeds (local model by
default), so ingestion and search work; only generate() raises until a key is configured.
"""

import redis.asyncio as redis

from mara.core.cache import RedisCache
from mara.core.config import Settings
from mara.llm.base import LLMProvider
from mara.llm.cached import CachedLLM
from mara.llm.composite import CompositeProvider, Embedder, TextGenerator
from mara.llm.resilience import AsyncRateLimiter, ResilientLLM


def llm_configuration_error(settings: Settings) -> str | None:
    """Why generate() would fail with these settings, or None if it is configured."""
    if settings.llm_provider == "gemini" and settings.gemini_api_key is None:
        return "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set"
    if settings.llm_provider == "openai" and settings.openai_api_key is None:
        return "LLM_PROVIDER=openai but OPENAI_API_KEY is not set"
    return None


def _build_vendor(vendor: str, settings: Settings):  # noqa: ANN202 - Gemini | OpenAI provider
    if vendor == "gemini":
        from mara.llm.gemini import GeminiProvider

        assert settings.gemini_api_key is not None
        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            embedding_model=settings.gemini_embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            timeout_s=settings.llm_timeout_s,
        )
    if vendor == "openai":
        from mara.llm.openai_provider import OpenAIProvider

        assert settings.openai_api_key is not None
        return OpenAIProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"unknown provider: {vendor}")


def build_base_provider(settings: Settings) -> CompositeProvider:
    error = llm_configuration_error(settings)
    generator: TextGenerator | None = (
        None if error else _build_vendor(settings.llm_provider, settings)
    )

    embedder: Embedder
    if settings.embedding_provider == "local":
        from mara.llm.local_embeddings import LocalEmbeddingProvider

        embedder = LocalEmbeddingProvider(settings.local_embedding_model)
    elif generator is not None and settings.embedding_provider == settings.llm_provider:
        embedder = generator  # same vendor: one client serves both
    else:
        key = getattr(settings, f"{settings.embedding_provider}_api_key")
        if key is None:
            raise ValueError(
                f"EMBEDDING_PROVIDER={settings.embedding_provider} needs "
                f"{settings.embedding_provider.upper()}_API_KEY"
            )
        embedder = _build_vendor(settings.embedding_provider, settings)

    return CompositeProvider(generator, embedder, missing_llm_reason=error or "")


def wrap_provider(
    base: LLMProvider, settings: Settings, redis_client: redis.Redis | None
) -> LLMProvider:
    embedding_limiter = (
        None  # local model: no quota to respect
        if base.embedding_provider == "local"
        else AsyncRateLimiter(per_minute=settings.embedding_requests_per_minute)
    )
    provider: LLMProvider = ResilientLLM(
        base,
        AsyncRateLimiter(per_minute=settings.llm_requests_per_minute),
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_backoff_base_s,
        max_delay=settings.llm_backoff_max_s,
        embedding_limiter=embedding_limiter,
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
    """Blocking when the local embedding model has to load: call it off the event loop."""
    return wrap_provider(build_base_provider(settings), settings, redis_client)
