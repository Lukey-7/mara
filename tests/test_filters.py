from datetime import date

from mara.core.filters import MetadataFilter, build_where


def test_empty_filter_is_none():
    assert build_where(None) is None
    assert build_where(MetadataFilter()) is None
    assert MetadataFilter().is_empty()


def test_single_condition_is_not_wrapped_in_and():
    assert build_where(MetadataFilter(source_types=["pdf"])) == {"source_type": {"$in": ["pdf"]}}
    assert build_where(MetadataFilter(tags=["raft"])) == {"tags": {"$contains": "raft"}}


def test_multiple_tags_are_any_of():
    assert build_where(MetadataFilter(tags=["raft", "paxos"])) == {
        "$or": [{"tags": {"$contains": "raft"}}, {"tags": {"$contains": "paxos"}}]
    }


def test_full_filter_is_anded_with_int_dates():
    where = build_where(
        MetadataFilter(
            source_types=["kb", "web"],
            doc_ids=["d1"],
            tags=["raft"],
            date_from=date(2024, 1, 1),
            date_to=date(2025, 6, 30),
        )
    )
    assert where == {
        "$and": [
            {"source_type": {"$in": ["kb", "web"]}},
            {"doc_id": {"$in": ["d1"]}},
            {"tags": {"$contains": "raft"}},
            {"published_date": {"$gte": 20240101}},
            {"published_date": {"$lte": 20250630}},
        ]
    }
