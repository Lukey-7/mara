"""FastAPI dependencies that pull wired objects off app.state (set in the lifespan)."""

from fastapi import HTTPException, Request

from mara.core.chunk_store import ChunkStore
from mara.core.config import Settings
from mara.ingest.pipeline import IngestionPipeline
from mara.llm.base import LLMProvider


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_llm(request: Request) -> LLMProvider:
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(503, f"LLM not configured: {request.app.state.llm_error}")
    return llm


def get_store(request: Request) -> ChunkStore:
    store = request.app.state.store
    if store is None:
        raise HTTPException(503, f"Chroma not available: {request.app.state.store_error}")
    return store


def get_pipeline(request: Request) -> IngestionPipeline:
    get_llm(request), get_store(request)  # surface the specific 503 reason
    return request.app.state.pipeline
