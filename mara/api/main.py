"""FastAPI application: lifespan wiring + /health. Routers plug in here."""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request

from mara import __version__
from mara.agents.factory import build_job_store, build_orchestrator, build_research_service
from mara.agents.jobs import TraceArchive
from mara.api import ingest_routes, research_routes, search_routes
from mara.core.chunk_store import ChunkStore, make_chroma_client
from mara.core.config import Settings, get_settings
from mara.ingest.factory import build_pipeline, build_store
from mara.llm.factory import build_llm, llm_configuration_error
from mara.retrieval.bm25_index import BM25Index
from mara.retrieval.factory import build_chroma_document_store, build_retriever

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level)
        st = app.state
        st.settings = settings
        st.redis = redis.Redis.from_url(settings.redis_url)
        st.llm = st.store = st.pipeline = st.bm25 = st.retriever = None
        st.research = st.jobs = st.trace_archive = None
        st.llm_error = llm_configuration_error(settings)  # generate() unavailable, embed() fine
        st.store_error = st.retriever_error = st.research_error = None
        try:
            # Loads the local embedding model: blocking, so off the event loop.
            st.llm = await asyncio.to_thread(build_llm, settings, st.redis)
        except Exception as e:  # noqa: BLE001 - e.g. model download failed: start degraded
            st.llm_error = f"{type(e).__name__}: {e}"
            log.warning("provider not available: %s", e)

        if st.llm is not None:
            try:
                client = await asyncio.to_thread(make_chroma_client, settings)
                st.store = build_store(settings, client, st.llm)
                st.bm25 = BM25Index()
                await st.bm25.reload(await st.store.all_chunks())  # Chroma is the source of truth
                st.pipeline = build_pipeline(settings, st.store, st.llm, indexes=[st.bm25])
            except Exception as e:  # noqa: BLE001 - chroma down / mismatch: start degraded
                st.store_error = f"{type(e).__name__}: {e}"
                log.warning("Chroma not available: %s", e)

        if st.store is not None:
            try:
                chroma_store = await asyncio.to_thread(build_chroma_document_store, settings)
                st.retriever = build_retriever(settings, st.llm, st.bm25, chroma_store)
                await asyncio.to_thread(st.retriever.warm_up)  # loads the cross-encoder
            except Exception as e:  # noqa: BLE001
                st.retriever_error = f"{type(e).__name__}: {e}"
                st.retriever = None
                log.warning("retriever not available: %s", e)

        if st.llm_error:
            st.research_error = f"LLM not configured: {st.llm_error}"
        elif st.retriever is None:
            st.research_error = "retriever not available"
        else:
            try:
                redis_ok = await _check_redis(st.redis) == "ok"
                st.jobs = build_job_store(settings, st.redis, redis_ok)  # memory if Redis is down
                orchestrator = build_orchestrator(
                    settings, st.llm, st.retriever, st.pipeline,
                    known_tags=lambda: _known_tags(st.store),
                )  # fmt: skip
                st.research = build_research_service(settings, orchestrator, st.jobs)
                st.trace_archive = TraceArchive(settings.trace_dir) if settings.trace_dir else None
            except Exception as e:  # noqa: BLE001
                st.research_error = f"{type(e).__name__}: {e}"
                log.warning("research service not available: %s", e)
        yield
        if st.research is not None:  # let in-flight jobs finish their last step
            await asyncio.wait_for(st.research.wait_all(), timeout=10)
        await st.redis.aclose()

    app = FastAPI(title="MARA", version=__version__, lifespan=lifespan)
    app.include_router(ingest_routes.router)
    app.include_router(search_routes.router)
    app.include_router(research_routes.router)

    @app.get("/health")
    async def health(request: Request) -> dict:
        """Liveness + dependency checks. Always 200 while the process is up; `status` is
        "degraded" if a dependency is down, so a load balancer does not kill the API
        just because Redis restarted."""
        st = request.app.state
        redis_ok, chroma_ok = await asyncio.gather(_check_redis(st.redis), _check_chroma(settings))
        checks = {
            "redis": redis_ok,
            "chroma": chroma_ok,
            "llm": {
                "provider": settings.llm_provider,
                "configured": st.llm is not None and not st.llm_error,
                **({"error": st.llm_error} if st.llm_error else {}),
            },
            "embeddings": {
                "provider": settings.embedding_provider,
                "model": st.llm.embedding_model if st.llm else None,
                "dimensions": st.llm.embedding_dimensions if st.llm else None,
            },
            "store": {
                "ready": st.store is not None,
                **({"error": st.store_error} if st.store_error else {}),
            },
            "research": {
                "ready": st.research is not None,
                "job_store": type(st.jobs).__name__ if st.jobs else None,
                "web_search": settings.web_search_provider,
                **({"error": st.research_error} if st.research_error else {}),
            },
            "retriever": {
                "ready": st.retriever is not None,
                "reranker": settings.reranker,
                "bm25_documents": await st.bm25.count() if st.bm25 else 0,
                **({"error": st.retriever_error} if st.retriever_error else {}),
            },
        }
        healthy = (
            redis_ok == "ok"
            and chroma_ok == "ok"
            and checks["llm"]["configured"]
            and st.store is not None
            and st.retriever is not None
        )
        return {"status": "ok" if healthy else "degraded", "version": __version__, "checks": checks}

    return app


async def _known_tags(store: ChunkStore) -> list[str]:
    """Tags currently in the corpus, offered to the Planner as filter candidates."""
    return sorted({t for d in await store.list_documents() for t in d.tags})


async def _check_redis(client: redis.Redis) -> str:
    try:
        await asyncio.wait_for(client.ping(), timeout=1.0)
        return "ok"
    except (redis.RedisError, TimeoutError, OSError) as e:
        return f"error: {type(e).__name__}"


async def _check_chroma(settings: Settings) -> str:
    if settings.chroma_persist_path:
        return "ok"  # embedded: no server to ping
    url = f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(url)
        return "ok" if resp.status_code == 200 else f"error: HTTP {resp.status_code}"
    except httpx.HTTPError as e:
        return f"error: {type(e).__name__}"


app = create_app()
