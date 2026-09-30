"""Metadata filter model + the builder that turns it into a Chroma `where` clause.

`MetadataFilter` is what the API and (from Phase 4) the Planner produce; `build_where` is
the one function that knows Chroma's filter grammar. Keep it small enough to whiteboard.
"""

from datetime import date
from typing import Any

from pydantic import BaseModel

from mara.core.schema import SourceType, date_to_int


class MetadataFilter(BaseModel):
    source_types: list[SourceType] | None = None
    tags: list[str] | None = None  # match ANY of these tags
    doc_ids: list[str] | None = None
    date_from: date | None = None  # inclusive
    date_to: date | None = None  # inclusive

    def is_empty(self) -> bool:
        return build_where(self) is None


def build_where(f: MetadataFilter | None) -> dict[str, Any] | None:
    """Chroma grammar: one condition is a bare `{field: {op: value}}`; several must be
    wrapped in `{"$and": [...]}`. Tags are any-of, so they become an inner `$or`."""
    if f is None:
        return None
    conds: list[dict[str, Any]] = []
    if f.source_types:
        conds.append({"source_type": {"$in": list(f.source_types)}})
    if f.doc_ids:
        conds.append({"doc_id": {"$in": list(f.doc_ids)}})
    if f.tags:
        tag_conds = [{"tags": {"$contains": t}} for t in f.tags]
        conds.append(tag_conds[0] if len(tag_conds) == 1 else {"$or": tag_conds})
    if f.date_from:
        conds.append({"published_date": {"$gte": date_to_int(f.date_from)}})
    if f.date_to:
        conds.append({"published_date": {"$lte": date_to_int(f.date_to)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}
