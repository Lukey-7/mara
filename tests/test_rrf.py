from haystack import Document
from haystack.utils.misc import _reciprocal_rank_fusion as haystack_rrf

from mara.retrieval.rrf import reciprocal_rank_fusion


def ids(fused):
    return [item for item, _ in fused]


def test_item_ranked_by_both_systems_beats_single_top_hit():
    bm25 = ["a", "b", "c"]
    dense = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([bm25, dense])
    assert ids(fused)[:2] == ["b", "a"]  # b: 1/62+1/61, a: 1/61+1/63
    assert ids(fused) == ["b", "a", "c", "d"] or ids(fused) == ["b", "a", "d", "c"]


def test_scores_follow_the_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=60))
    assert fused["a"] == 1 / 61
    assert fused["b"] == 1 / 62 + 1 / 61


def test_weights_and_ties():
    # equal evidence, deterministic tie-break by id
    assert ids(reciprocal_rank_fusion([["y"], ["x"]])) == ["x", "y"]
    # weight the dense list 3x: its top hit now wins
    assert ids(reciprocal_rank_fusion([["y"], ["x"]], weights=[1.0, 3.0]))[0] == "x"


def test_empty_and_single_list():
    assert reciprocal_rank_fusion([]) == []
    assert ids(reciprocal_rank_fusion([["a", "b"]])) == ["a", "b"]


def test_matches_haystack_document_joiner_ordering():
    lists = [["a", "b", "c", "d"], ["c", "a", "e"], ["e", "b"]]
    docs = [[Document(id=i, content=i) for i in ranking] for ranking in lists]
    ours = ids(reciprocal_rank_fusion(lists))
    theirs = [d.id for d in sorted(haystack_rrf(docs), key=lambda d: (-d.score, d.id))]
    assert ours == theirs
