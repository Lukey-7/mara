"""Wires the agents, orchestrator and research service from settings."""

import redis.asyncio as redis

from mara.agents.base import StructuredLLM
from mara.agents.critic import Critic
from mara.agents.jobs import JobStore, MemoryJobStore, RedisJobStore, TraceArchive
from mara.agents.orchestrator import Orchestrator
from mara.agents.planner import KnownTags, Planner
from mara.agents.researcher import Researcher
from mara.agents.runner import ResearchService
from mara.agents.summarizer import Summarizer
from mara.agents.web_search import (
    CachedWebSearch,
    DuckDuckGoWebSearch,
    NoopWebSearch,
    TavilyWebSearch,
    WebSearchProvider,
)
from mara.agents.writer import Writer
from mara.core.cache import RedisCache
from mara.core.config import Settings
from mara.ingest.pipeline import IngestionPipeline
from mara.llm.base import LLMProvider
from mara.retrieval.hybrid import Retriever


def build_web_search(settings: Settings, cache: RedisCache | None = None) -> WebSearchProvider:
    provider: WebSearchProvider
    if settings.web_search_provider == "duckduckgo":
        provider = DuckDuckGoWebSearch()
    elif settings.web_search_provider == "tavily":
        if settings.tavily_api_key is None:
            raise ValueError("WEB_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set")
        provider = TavilyWebSearch(settings.tavily_api_key.get_secret_value())
    else:
        return NoopWebSearch()
    if cache is not None and settings.cache_enabled:
        provider = CachedWebSearch(provider, cache, settings.web_search_cache_ttl_s)
    return provider


def build_orchestrator(
    settings: Settings,
    llm: LLMProvider,
    retriever: Retriever,
    ingest: IngestionPipeline | None,
    known_tags: KnownTags = None,
    cache: RedisCache | None = None,
) -> Orchestrator:
    structured = StructuredLLM(llm, temperature=settings.agent_temperature)
    return Orchestrator(
        planner=Planner(structured, known_tags),
        researcher=Researcher(
            retriever,
            build_web_search(settings, cache),
            ingest,
            web_results=settings.web_search_results,
            fetch_timeout_s=settings.web_fetch_timeout_s,
            fetch_max_bytes=settings.web_max_bytes,
        ),
        summarizer=Summarizer(structured, max_quote_chars=settings.summarizer_max_quote_chars),
        critic=Critic(structured),
        writer=Writer(structured),
        step_timeout_s=settings.agent_timeout_s,
    )


def build_job_store(
    settings: Settings, redis_client: redis.Redis | None, redis_ok: bool
) -> JobStore:
    if redis_client is not None and redis_ok:
        return RedisJobStore(redis_client, ttl_s=settings.job_ttl_s)
    return MemoryJobStore()


def build_research_service(
    settings: Settings, orchestrator: Orchestrator, store: JobStore
) -> ResearchService:
    archive = TraceArchive(settings.trace_dir) if settings.trace_dir else None
    return ResearchService(orchestrator, store, archive)
