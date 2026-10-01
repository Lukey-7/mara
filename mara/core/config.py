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

    # --- Embeddings: local by default so ingestion + search need no key and no rate limit ---
    embedding_provider: Literal["local", "gemini", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"  # 33M params, 384 dims, CPU-friendly
    # API providers only. Both gemini-embedding-001 (native 3072) and text-embedding-3-*
    # support Matryoshka truncation; 768 is 4x cheaper to store/search with a small quality
    # loss. Changing it (or the model) requires re-ingesting: the store refuses mismatches.
    embedding_dimensions: int | None = 768
    embedding_batch_size: int = 100  # Gemini's per-request cap is 100 texts

    # --- API ---
    api_rate_limit_per_minute: int = 60  # per client, POST /research and /search; 0 disables

    # --- Agents (Phase 4) ---
    agent_temperature: float = 0.2
    agent_timeout_s: float = 120.0  # per agent step; a timed-out step degrades, not aborts
    research_top_k: int = 6  # chunks per sub-question handed to the summarizer
    max_critic_loops: int = 1  # hard cap on extra research rounds
    summarizer_max_quote_chars: int = 300
    web_search_provider: Literal["none", "duckduckgo", "tavily"] = "none"
    tavily_api_key: SecretStr | None = None
    web_search_results: int = 3  # pages fetched + ingested per sub-question
    web_search_cache_ttl_s: int = 24 * 3600
    trace_dir: str | None = "data/traces"  # JSON per finished run; None disables
    job_ttl_s: int = 7 * 24 * 3600

    # --- Retrieval (Phase 3) ---
    reranker: Literal["cross-encoder", "none"] = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22M params, ~90 MB
    retrieval_candidates: int = 20  # per retriever, before rank fusion
    retrieval_top_k: int = 8  # after fusion (+ reranking)

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
    # 2 = RESP2, understood by every Redis version (native Windows builds are 3.x and reject
    # the RESP3 HELLO handshake); nothing here needs RESP3 features.
    redis_protocol: int = 2
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "mara_chunks"
    # Set to a directory to use an embedded on-disk Chroma instead of the server (no Docker).
    chroma_persist_path: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
