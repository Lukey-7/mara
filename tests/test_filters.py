from datetime import date

from mara.core.filters import MetadataFilter, build_where, to_haystack


def test_empty_filter_is_none():
    assert build_where(None) is None
    assert build_where(MetadataFilter()) is None
    assert to_haystack(MetadataFilter()) is None
    assert MetadataFilter().is_empty()


def test_single_condition_is_not_wrapped():
    assert build_where(MetadataFilter(source_types=["pdf"])) == {"source_type": {"$in": ["pdf"]}}
    assert build_where(MetadataFilter(tags=["Raft"])) == {"tag:raft": {"$eq": True}}
    assert to_haystack(MetadataFilter(tags=["Raft"])) == {
        "field": "meta.tag:raft", "operator": "==", "value": True
    }  # fmt: skip


def test_multiple_tags_are_any_of():
    assert build_where(MetadataFilter(tags=["raft", "paxos"])) == {
        "$or": [{"tag:raft": {"$eq": True}}, {"tag:paxos": {"$eq": True}}]
    }
    assert to_haystack(MetadataFilter(tags=["raft", "paxos"])) == {
        "operator": "OR",
        "conditions": [
            {"field": "meta.tag:raft", "operator": "==", "value": True},
            {"field": "meta.tag:paxos", "operator": "==", "value": True},
        ],
    }


def test_full_filter_is_anded_with_int_dates():
    f = MetadataFilter(
        source_types=["kb", "web"],
        doc_ids=["d1"],
        tags=["raft"],
        date_from=date(2024, 1, 1),
        date_to=date(2025, 6, 30),
    )
    assert build_where(f) == {
        "$and": [
            {"source_type": {"$in": ["kb", "web"]}},
            {"doc_id": {"$in": ["d1"]}},
            {"tag:raft": {"$eq": True}},
            {"published_date": {"$gte": 20240101}},
            {"published_date": {"$lte": 20250630}},
        ]
    }
    assert to_haystack(f) == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.source_type", "operator": "in", "value": ["kb", "web"]},
            {"field": "meta.doc_id", "operator": "in", "value": ["d1"]},
            {"field": "meta.tag:raft", "operator": "==", "value": True},
            {"field": "meta.published_date", "operator": ">=", "value": 20240101},
            {"field": "meta.published_date", "operator": "<=", "value": 20250630},
        ],
    }
