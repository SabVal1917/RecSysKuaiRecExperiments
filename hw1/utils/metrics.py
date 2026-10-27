"""Top-K ranking metrics."""

import numpy as np


def recall_at_k(
    recommended: np.ndarray,
    relevant: set[int],
    k: int = 10,
) -> float:
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)

    return hits / len(relevant)


def ndcg_at_k(
    recommended: np.ndarray,
    relevant: set[int],
    k: int = 10,
) -> float:
    if not relevant:
        return 0.0

    top_k = recommended[:k]

    relevance = np.isin(
        top_k, list(relevant)
    ).astype(np.float64)

    discounts = 1.0 / np.log2(
        np.arange(2, len(top_k) + 2)
    )

    dcg = np.sum(relevance * discounts)

    n_relevant = min(len(relevant), k)

    idcg = np.sum(discounts[:n_relevant])

    return float(dcg / idcg)
