"""Reciprocal Rank Fusion, written out so it can be reproduced on a whiteboard.

Each ranked list votes for its items with weight 1 / (k + rank); items sum their votes.
k (60 in the original paper) flattens the curve so that rank 1 vs rank 2 is not a landslide
and an item ranked well by two systems beats an item ranked first by only one.
Haystack's DocumentJoiner(join_mode="reciprocal_rank_fusion") computes the same ordering
(tests/test_rrf.py checks that).
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """rankings: one list of ids per retriever, best first. Returns (id, score), best first;
    ties broken by id so the output is deterministic."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
