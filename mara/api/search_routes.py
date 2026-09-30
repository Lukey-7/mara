"""POST /search: run the retrieval pipeline on its own (debugging, evals, the UI)."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from mara.api.deps import get_retriever
from mara.core.filters import MetadataFilter
from mara.retrieval.hybrid import RetrievalConfig, RetrievalResult, Retriever

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    filters: MetadataFilter | None = None
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)


@router.post("/search", response_model=RetrievalResult)
async def search(
    req: SearchRequest, retriever: Annotated[Retriever, Depends(get_retriever)]
) -> RetrievalResult:
    return await retriever.retrieve(req.query, req.filters, req.config)
