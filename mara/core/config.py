"""Application settings, loaded from environment variables / .env via pydantic-settings.

Every tunable lives here so nothing else reads os.environ directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "mara"
    log_level: str = "INFO"

    # --- LLM provider selection (Strategy chosen by the factory) ---
    llm_provider: Literal["gemini", "openai"] = "gemini"

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Embedding vector size. Both gemini-embedding-001 (default 3072) and text-embedding-3-*
    # support truncation via Matryoshka representation; 768 is 4x cheaper to store and
    # search than 3072 with a small quality loss. Changing it requires re-ingesting.
    embedding_dimensions: int | None = 768
    embedding_batch_size: int = 100  # Gemini's per-request cap is 100 texts

    # --- Resilience: every LLM call goes through rate limiter + retry ---
    llm_requests_per_minute: int = 10  # Gemini free tier is ~10 RPM for flash models
    embedding_requests_per_minute: int = 100  # embedding limits are separate and higher
    llm_max_retries: int = 4
    llm_backoff_base_s: float = 1.0
    llm_backoff_max_s: float = 30.0
    llm_timeout_s: float = 60.0

    # --- Cache ---
    cache_enabled: bool = True
    llm_cache_ttl_s: int = 7 * 24 * 3600
    embedding_cache_ttl_s: int = 30 * 24 * 3600
    cache_key_version: str = "v1"  # bump to invalidate every cached entry at once

    # --- Ingestion ---
    knowledge_base_dir: str = "knowledge_base"
    chunking_strategy: Literal["semantic", "fixed"] = "semantic"
    # Split where the cosine distance between neighbouring sentence windows is above this
    # percentile of all distances in the document (LlamaIndex's SemanticSplitterNodeParser).
    semantic_breakpoint_percentile: int = 90
    semantic_buffer_size: int = 1  # sentences of context on each side when embedding
    max_chunk_chars: int = 2000  # semantic chunks larger than this are re-split (fixed-size)
    fixed_chunk_size_tokens: int = 256
    fixed_chunk_overlap_tokens: int = 32
    web_fetch_timeout_s: float = 15.0
    web_max_bytes: int = 5_000_000

    # --- Infrastructure ---
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "mara_chunks"


@lru_cache
def get_settings() -> Settings:
    return Settings()
