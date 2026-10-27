"""Unified recommender evaluation."""

from collections import defaultdict

import numpy as np
import polars as pl

from utils.metrics import recall_at_k, ndcg_at_k


def evaluate(
    model,
    train: pl.DataFrame,
    holdout: pl.DataFrame,
    n_items: int,
    k: int = 10,
) -> dict:

    seen = defaultdict(set)

    for user, item in train.select("user_idx", "item_idx").iter_rows():
        seen[user].add(item)

    relevant = defaultdict(set)

    positives = holdout.filter(pl.col("label") == 1)

    for user, item in positives.select("user_idx", "item_idx").iter_rows():
        if item not in seen[user]:
            relevant[user].add(item)

    recalls = []
    ndcgs = []

    for user, ground_truth in relevant.items():
        if not ground_truth:
            continue

        # кандидаты весь каталог без известных видео
        mask = np.ones(n_items, dtype=bool)

        if seen[user]:
            mask[list(seen[user])] = False

        candidates = np.flatnonzero(mask)

        if len(candidates) == 0:
            continue

        # предсказания модели
        scores = np.asarray(
            model.score(user, candidates)
        )

        if scores.shape != (len(candidates),):
            raise ValueError("Invalid scores shape")

        scores = np.where(
            np.isfinite(scores), scores, -np.inf
        )

        top_indices = np.argsort(
            -scores, kind="stable"
        )[:k]

        top_k = candidates[top_indices]

        recalls.append(
            recall_at_k(top_k, ground_truth, k)
        )
        ndcgs.append(
            ndcg_at_k(top_k, ground_truth, k)
        )

    if not recalls:
        raise ValueError("No evaluable users")

    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "n_users": len(recalls),
    }
