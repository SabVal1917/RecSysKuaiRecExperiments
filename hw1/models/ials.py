
"""Implicit Alternating Least Squares."""

import numpy as np
from scipy.sparse import csr_matrix


class IALS:
    def __init__(
        self,
        n_factors: int = 32,
        alpha: float = 40.0,
        reg: float = 0.1,
        n_iters: int = 10,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.alpha = alpha
        self.reg = reg
        self.n_iters = n_iters
        self.seed = seed

        self.user_factors = None
        self.item_factors = None

    def _update_factors(
        self,
        interactions: csr_matrix,
        fixed_factors: np.ndarray,
        weighted_targets: csr_matrix | None = None,
    ) -> np.ndarray:

        n_entities = interactions.shape[0]

        updated = np.zeros(
            (n_entities, self.n_factors),
            dtype=np.float64,
        )

        gram = fixed_factors.T @ fixed_factors
        reg_eye = self.reg * np.eye(self.n_factors)

        for u in range(n_entities):
            start = interactions.indptr[u]
            end = interactions.indptr[u + 1]

            indices = interactions.indices[start:end]
            values = interactions.data[start:end]

            neighbors = fixed_factors[indices]

            confidence = 1.0 + self.alpha * values

            A = (
                gram
                + neighbors.T @ (
                    (confidence - 1.0)[:, None] * neighbors
                )
                + reg_eye
            )

            if weighted_targets is None:
                b = confidence @ neighbors

            else:
                t_start = weighted_targets.indptr[u]
                t_end = weighted_targets.indptr[u + 1]

                target_indices = weighted_targets.indices[
                    t_start:t_end
                ]
                target_values = weighted_targets.data[
                    t_start:t_end
                ]

                b = (
                    target_values
                    @ fixed_factors[target_indices]
                )

            updated[u] = np.linalg.solve(A, b)

        return updated

    @staticmethod
    def _prepare_sparse(matrix: csr_matrix) -> csr_matrix:
        result = matrix.tocsr().astype(np.float64).copy()
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()

        if not np.all(np.isfinite(result.data)):
            raise ValueError("Non-finite matrix values")

        return result

    def fit(
        self,
        train: csr_matrix,
        targets: csr_matrix | None = None,
    ) -> None:
        if self.n_factors < 1 or self.n_iters < 1:
            raise ValueError("Invalid factors or iterations")

        if self.reg <= 0 or self.alpha < 0:
            raise ValueError("Invalid reg or alpha")

        R = self._prepare_sparse(train)

        if np.any(R.data != 1):
            raise ValueError("Expected binary positive matrix")

        weighted_targets = None

        if targets is not None:
            T = self._prepare_sparse(targets)

            if T.shape != R.shape:
                raise ValueError("Targets shape mismatch")

            if (T != R).nnz != 0:
                weighted_targets = (
                    T + self.alpha * R.multiply(T)
                ).tocsr()

                weighted_targets.sum_duplicates()
                weighted_targets.eliminate_zeros()
                weighted_targets.sort_indices()

        n_users, n_items = R.shape

        rng = np.random.default_rng(self.seed)

        self.item_factors = rng.normal(
            0, 0.1, (n_items, self.n_factors)
        )
        self.user_factors = None

        R_T = R.T.tocsr()

        T_T = (
            weighted_targets.T.tocsr()
            if weighted_targets is not None
            else None
        )

        for _ in range(self.n_iters):
            self.user_factors = self._update_factors(
                R,
                self.item_factors,
                weighted_targets,
            )

            self.item_factors = self._update_factors(
                R_T,
                self.user_factors,
                T_T,
            )

    def score(
        self,
        user_id: int,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model is not fitted")

        user_vector = self.user_factors[user_id]
        item_vectors = self.item_factors[item_ids]

        return item_vectors @ user_vector
