"""The single chunk schema shared by LlamaIndex (ingestion) and Haystack (retrieval).

Both frameworks convert to/from this model, so a chunk written by one is readable by
the other. The Chroma record layout (ids / documents / metadatas) is defined here too, so
there is exactly one place that knows how a Chunk is stored.
"""

from datetime import UTC, date, datetime
from typing import Any, Literal

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

    # --- Chroma record layout -------------------------------------------------------
    # Chroma metadata values must be str / int / float / bool / list-of-those; no None and no
    # nested dicts. Dates are stored as int YYYYMMDD so range filters ($gte/$lte) work. Tags
    # are stored as a list (display) plus one boolean flag per tag (`tag:raft = True`) that
    # both Chroma and Haystack's in-memory store can filter with a plain equality.

    def to_chroma_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "title": self.title,
            "url_or_path": self.url_or_path,
            "ingested_at": self.ingested_at.isoformat(),
            **{tag_flag(t): True for t in self.tags},
        }
        if self.tags:  # Chroma rejects empty lists as metadata values
            meta["tags"] = list(self.tags)
        if self.page is not None:
            meta["page"] = self.page
        if self.section is not None:
            meta["section"] = self.section
        if self.published_date is not None:
            meta["published_date"] = date_to_int(self.published_date)
        return meta

    @classmethod
    def from_chroma(cls, chunk_id: str, text: str, meta: dict[str, Any]) -> "Chunk":
        pub = meta.get("published_date")
        return cls(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            text=text,
            source_type=meta["source_type"],
            title=meta["title"],
            url_or_path=meta["url_or_path"],
            page=meta.get("page"),
            section=meta.get("section"),
            published_date=int_to_date(pub) if pub is not None else None,
            tags=list(meta.get("tags") or []),
            ingested_at=datetime.fromisoformat(meta["ingested_at"]),
        )


class DocumentInfo(BaseModel):
    """One ingested document, as summarised from its chunks (what GET /documents returns)."""

    doc_id: str
    title: str
    source_type: SourceType
    url_or_path: str
    tags: list[str] = Field(default_factory=list)
    published_date: date | None = None
    ingested_at: datetime
    n_chunks: int
    pages: int | None = None


def tag_flag(tag: str) -> str:
    return f"tag:{tag.strip().lower()}"


def date_to_int(d: date) -> int:
    return d.year * 10_000 + d.month * 100 + d.day


def int_to_date(n: int) -> date:
    return date(n // 10_000, (n // 100) % 100, n % 100)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)
