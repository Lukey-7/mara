"""The single chunk schema shared by LlamaIndex (ingestion) and Haystack (retrieval).

Both frameworks convert to/from this model, so a chunk written by one is readable by
the other. Defined in Phase 1 so every later phase builds on the same contract.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "web", "kb"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source_type: SourceType
    title: str
    url_or_path: str
    page: int | None = None
    section: str | None = None
    published_date: date | None = None
    tags: list[str] = Field(default_factory=list)
    ingested_at: datetime
