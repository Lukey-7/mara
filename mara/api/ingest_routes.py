"""Ingestion + document listing endpoints."""

import asyncio
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from mara.api.deps import get_pipeline, get_settings_dep, get_store
from mara.core.chunk_store import ChunkStore
from mara.core.config import Settings
from mara.core.filters import MetadataFilter
from mara.core.schema import DocumentInfo, SourceType
from mara.ingest.loaders import load_html, load_kb_dir, load_pdf
from mara.ingest.pipeline import IngestionPipeline, IngestResult
from mara.ingest.web import FetchError, fetch_html

router = APIRouter(tags=["ingest"])

Pipeline = Annotated[IngestionPipeline, Depends(get_pipeline)]
Store = Annotated[ChunkStore, Depends(get_store)]
Config = Annotated[Settings, Depends(get_settings_dep)]


def _parse_tags(raw: str | None) -> list[str]:
    return [t.strip() for t in (raw or "").split(",") if t.strip()]


@router.post("/ingest/pdf", response_model=IngestResult)
async def ingest_pdf(
    pipeline: Pipeline,
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
    tags: Annotated[str | None, Form(description="comma-separated")] = None,
    published_date: Annotated[
        date | None, Form(description="overrides the PDF's creation date")
    ] = None,
    force: Annotated[bool, Form()] = False,
) -> IngestResult:
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "not a PDF file")
    filename = file.filename or "upload.pdf"
    # PDF parsing is CPU-bound: keep it off the event loop.
    docs = await asyncio.to_thread(load_pdf, data, filename, title, _parse_tags(tags))
    if not docs:
        raise HTTPException(422, "PDF has no extractable text (scanned? try OCR first)")
    if published_date is not None:
        for d in docs:
            d.published_date = published_date
    return await pipeline.ingest(docs, force=force)


class UrlIngestRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=20)
    tags: list[str] = Field(default_factory=list)
    force: bool = False


class UrlIngestResponse(BaseModel):
    results: list[IngestResult]
    errors: dict[str, str]  # url -> reason; one bad URL must not fail the batch


@router.post("/ingest/url", response_model=UrlIngestResponse)
async def ingest_urls(req: UrlIngestRequest, pipeline: Pipeline, cfg: Config) -> UrlIngestResponse:
    results, errors = [], {}

    async def one(url: str) -> None:
        try:
            html = await fetch_html(url, cfg.web_fetch_timeout_s, cfg.web_max_bytes)
            doc = await asyncio.to_thread(load_html, html, url, None, req.tags)
            if doc is None:
                errors[url] = "no main text could be extracted"
                return
            results.append(await pipeline.ingest([doc], force=req.force))
        except FetchError as e:
            errors[url] = str(e)

    await asyncio.gather(*(one(u) for u in req.urls))
    return UrlIngestResponse(results=results, errors=errors)


class KbIngestResponse(BaseModel):
    directory: str
    results: list[IngestResult]


@router.post("/ingest/kb", response_model=KbIngestResponse)
async def ingest_kb(pipeline: Pipeline, cfg: Config, force: bool = False) -> KbIngestResponse:
    """Re-scan the knowledge_base folder. Unchanged notes are skipped (content hash)."""
    root = Path(cfg.knowledge_base_dir)
    if not await asyncio.to_thread(root.is_dir):
        raise HTTPException(404, f"knowledge base directory not found: {root}")
    results = []
    for sections in await asyncio.to_thread(load_kb_dir, root):
        if sections:
            results.append(await pipeline.ingest(sections, force=force))
    return KbIngestResponse(directory=str(root), results=results)


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    store: Store,
    source_type: Annotated[list[SourceType] | None, Query()] = None,
    tag: Annotated[list[str] | None, Query(description="any of")] = None,
    date_from: date | None = None,
    date_to: date | None = None,
    q: Annotated[str | None, Query(description="case-insensitive title substring")] = None,
) -> list[DocumentInfo]:
    f = MetadataFilter(source_types=source_type, tags=tag, date_from=date_from, date_to=date_to)
    docs = await store.list_documents(f)
    if q:
        docs = [d for d in docs if q.lower() in d.title.lower()]
    return docs


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, store: Store) -> dict:
    deleted = await store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(404, "document not found")
    return {"doc_id": doc_id, "deleted_chunks": deleted}
