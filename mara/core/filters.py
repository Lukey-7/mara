"""Metadata filter model + the two builders that turn it into a store-specific filter.

`MetadataFilter` is what the API and (from Phase 4) the Planner produce.
- `to_haystack()` builds Haystack's filter dict, which BOTH retrievers in the query pipeline
  accept (chroma-haystack translates it to Chroma's `where`; InMemoryBM25Retriever evaluates
  it in Python).
- `build_where()` builds Chroma's native `where` for admin paths (GET /documents).
Both are short enough to whiteboard.

Tags are stored twice on purpose: `tags` (list, for display) and one boolean flag per tag
(`tag:raft = True`, for filtering), because Haystack's in-memory filters have no
list-membership operator while `==` on a flag works everywhere.
"""

from datetime import date
from typing import Any

from pydantic import BaseModel

from mara.core.schema import SourceType, date_to_int, tag_flag


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
        tag_conds = [{tag_flag(t): {"$eq": True}} for t in f.tags]
        conds.append(tag_conds[0] if len(tag_conds) == 1 else {"$or": tag_conds})
    if f.date_from:
        conds.append({"published_date": {"$gte": date_to_int(f.date_from)}})
    if f.date_to:
        conds.append({"published_date": {"$lte": date_to_int(f.date_to)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def to_haystack(f: MetadataFilter | None) -> dict[str, Any] | None:
    """Haystack grammar: leaves are `{"field": "meta.x", "operator": op, "value": v}`,
    combined with `{"operator": "AND"|"OR", "conditions": [...]}`."""
    if f is None:
        return None
    conds: list[dict[str, Any]] = []
    if f.source_types:
        conds.append({"field": "meta.source_type", "operator": "in", "value": list(f.source_types)})
    if f.doc_ids:
        conds.append({"field": "meta.doc_id", "operator": "in", "value": list(f.doc_ids)})
    if f.tags:
        tag_conds = [
            {"field": f"meta.{tag_flag(t)}", "operator": "==", "value": True} for t in f.tags
        ]
        conds.append(
            tag_conds[0] if len(tag_conds) == 1 else {"operator": "OR", "conditions": tag_conds}
        )
    if f.date_from:
        conds.append(
            {"field": "meta.published_date", "operator": ">=", "value": date_to_int(f.date_from)}
        )
    if f.date_to:
        conds.append(
            {"field": "meta.published_date", "operator": "<=", "value": date_to_int(f.date_to)}
        )
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"operator": "AND", "conditions": conds}
