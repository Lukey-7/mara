"""FastAPI dependencies that pull wired objects off app.state (set in the lifespan)."""

from fastapi import HTTPException, Request

from mara.core.chunk_store import ChunkStore
from mara.core.config import Settings
from mara.ingest.pipeline import IngestionPipeline
from mara.llm.base import LLMProvider
from mara.retrieval.hybrid import Retriever


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_llm(request: Request) -> LLMProvider:
    """The provider stack; embeds always, generates only if an API key is configured."""
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(503, f"provider not available: {request.app.state.llm_error}")
    return llm


def get_generating_llm(request: Request) -> LLMProvider:
    llm = get_llm(request)
    if request.app.state.llm_error:
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


def get_retriever(request: Request) -> Retriever:
    get_store(request)
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(503, f"retriever not available: {request.app.state.retriever_error}")
    return retriever
