"""Item-based collaborative filtering."""

import numpy as np
from scipy.sparse import csr_matrix


class ItemCF:
    def __init__(self, sim: str = "cosine", k: int = 50):
        if sim not in ("cosine", "jaccard"):
            raise ValueError(f"Unknown similarity: {sim}")
        if k < 1:
            raise ValueError("k must be positive")

        self.sim = sim
        self.k = k
        self.R = None
        self.similarity = None

    def fit(self, train: csr_matrix) -> None:
        self.R = train.tocsr().astype(np.float32)

        cooccurrence = (self.R.T @ self.R).toarray()

        if self.sim == "cosine":
            norms = np.sqrt(
                self.R.power(2).sum(axis=0)
            ).A1

            denominator = np.outer(norms, norms)

        elif self.sim == "jaccard":
            counts = np.asarray(
                self.R.sum(axis=0)
            ).ravel()

            denominator = (
                counts[:, None]
                + counts[None, :]
                - cooccurrence
            )

        similarity = np.divide(
            cooccurrence,
            denominator,
            out=np.zeros_like(cooccurrence),
            where=denominator > 0,
        )

        np.fill_diagonal(similarity, 0)

        n_items = similarity.shape[0]
        k = min(self.k, n_items - 1)

        if k > 0:
            indices = np.argpartition(
                similarity, -k, axis=1
            )[:, -k:]

            mask = np.zeros_like(similarity, dtype=bool)
            rows = np.arange(n_items)[:, None]
            mask[rows, indices] = True

            similarity[~mask] = 0
        else:
            similarity[:] = 0

        self.similarity = similarity

    def score(
        self,
        user_id: int,
        item_ids: np.ndarray,
    ) -> np.ndarray:

        if self.similarity is None:
            raise RuntimeError("Model is not fitted")

        user_vector = self.R.getrow(user_id)

        scores = user_vector @ self.similarity

        return np.asarray(scores).ravel()[item_ids]
