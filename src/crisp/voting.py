from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def weighted_vote(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Similarity-weighted voting.

    Score(class) = sum(similarity of retrieved neighbors from that class)
    """
    if not neighbors:
        raise ValueError("neighbors must not be empty.")

    scores = defaultdict(float)
    for item in neighbors:
        scores[str(item["label"])] += float(item["similarity"])

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    predicted_label = next(iter(sorted_scores.keys()))

    return {
        "predicted_label": predicted_label,
        "scores": sorted_scores,
    }


def majority_vote(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Majority voting based on label frequency in top-k retrieval results.
    """
    if not neighbors:
        raise ValueError("neighbors must not be empty.")

    scores = defaultdict(int)
    for item in neighbors:
        scores[str(item["label"])] += 1

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    predicted_label = next(iter(sorted_scores.keys()))

    return {
        "predicted_label": predicted_label,
        "scores": sorted_scores,
    }
