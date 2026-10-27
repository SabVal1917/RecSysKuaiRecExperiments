"""EASE"""

import numpy as np
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, reg: float = 100.0):
        self.reg = reg
        self.R = None
        self.weights = None

    def fit(self, train: csr_matrix) -> None:
        self.R = train.tocsr().astype(np.float32)

        gram = (self.R.T @ self.R).toarray()
        gram = gram.astype(np.float64)

        n_items = gram.shape[0]
        gram += self.reg * np.eye(n_items)

        P = np.linalg.inv(gram)

        B = -P / np.diag(P)[None, :]

        np.fill_diagonal(B, 0)
        self.weights = B.astype(np.float32)

    def score(
        self,
        user_id: int,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")

        user_vector = self.R.getrow(user_id)

        scores = user_vector @ self.weights

        return np.asarray(scores).ravel()[item_ids]

