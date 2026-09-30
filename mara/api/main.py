"""FastAPI application: lifespan wiring + /health. Routers for later phases plug in here."""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from chromadb.api import ClientAPI
from fastapi import FastAPI, Request

from mara import __version__
from mara.api import ingest_routes
from mara.core.chunk_store import make_chroma_client
from mara.core.config import Settings, get_settings
from mara.ingest.factory import build_pipeline, build_store
from mara.llm.factory import build_llm

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, chroma_client: ClientAPI | None = None) -> FastAPI:
    """`chroma_client` is injectable so tests can pass an EphemeralClient."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level)
        app.state.settings = settings
        app.state.redis = redis.Redis.from_url(settings.redis_url)
        app.state.llm = app.state.llm_error = None
        app.state.store = app.state.store_error = app.state.pipeline = None
        try:
            app.state.llm = build_llm(settings, app.state.redis)
        except ValueError as e:  # missing API key: start anyway, /health reports it
            app.state.llm_error = str(e)
            log.warning("LLM not configured: %s", e)
        if app.state.llm is not None:
            try:
                client = chroma_client or await asyncio.to_thread(make_chroma_client, settings)
                app.state.store = build_store(settings, client, app.state.llm)
                app.state.pipeline = build_pipeline(settings, app.state.store, app.state.llm)
            except Exception as e:  # noqa: BLE001 - chroma down / mismatch: start degraded
                app.state.store_error = f"{type(e).__name__}: {e}"
                log.warning("Chroma not available: %s", e)
        yield
        await app.state.redis.aclose()

    app = FastAPI(title="MARA", version=__version__, lifespan=lifespan)
    app.include_router(ingest_routes.router)

    @app.get("/health")
    async def health(request: Request) -> dict:
        """Liveness + dependency checks. Always 200 while the process is up; `status` is
        "degraded" if a dependency is down, so a load balancer does not kill the API
        just because Redis restarted."""
        state = request.app.state
        redis_ok, chroma_ok = await asyncio.gather(
            _check_redis(state.redis), _check_chroma(settings)
        )
        checks = {
            "redis": redis_ok,
            "chroma": chroma_ok,
            "llm": {
                "provider": settings.llm_provider,
                "configured": state.llm is not None,
                **({"error": state.llm_error} if state.llm_error else {}),
            },
            "store": {
                "ready": state.store is not None,
                **({"error": state.store_error} if state.store_error else {}),
            },
        }
        healthy = (
            redis_ok == "ok" and chroma_ok == "ok" and state.llm is not None
            and state.store is not None
        )  # fmt: skip
        return {"status": "ok" if healthy else "degraded", "version": __version__, "checks": checks}

    return app


async def _check_redis(client: redis.Redis) -> str:
    try:
        await asyncio.wait_for(client.ping(), timeout=1.0)
        return "ok"
    except (redis.RedisError, TimeoutError, OSError) as e:
        return f"error: {type(e).__name__}"


async def _check_chroma(settings: Settings) -> str:
    url = f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(url)
        return "ok" if resp.status_code == 200 else f"error: HTTP {resp.status_code}"
    except httpx.HTTPError as e:
        return f"error: {type(e).__name__}"


app = create_app()
