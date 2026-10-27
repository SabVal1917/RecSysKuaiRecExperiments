
"""Most Popular baseline."""

import numpy as np
from scipy.sparse import csr_matrix


class Popularity:
    def __init__(self):
        self.popularity = None

    def fit(self, train: csr_matrix) -> None:
        self.popularity = np.asarray(
            train.sum(axis=0)
        ).ravel()

    def score(
        self,
        user_id: int,
        item_ids: np.ndarray,
    ) -> np.ndarray:

        if self.popularity is None:
            raise RuntimeError("Model is not fitted")

        return self.popularity[item_ids]
