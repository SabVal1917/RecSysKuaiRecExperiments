
"""ALS with custom piecewise-linear target."""

import numpy as np
from scipy.sparse import csr_matrix

from models.ials import IALS


class CustomALS(IALS):
    def __init__(
        self,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.gamma = gamma

    def fit(
        self,
        train: csr_matrix,
        weak_signal: csr_matrix,
    ) -> None:

        R = train.tocsr().astype(np.float64).copy()
        W = weak_signal.tocsr().astype(np.float64).copy()

        R.sum_duplicates()
        W.sum_duplicates()
        R.eliminate_zeros()
        W.eliminate_zeros()

        if R.shape != W.shape:
            raise ValueError("Matrix shapes differ")

        if not np.all(np.isfinite(W.data)):
            raise ValueError("Non-finite weak signal")

        if np.any((W.data < 0) | (W.data > 1)):
            raise ValueError("Weak signal must be in [0, 1]")

        if R.multiply(W).nnz != 0:
            raise ValueError("Positive and weak pairs overlap")

        # Custom target:
        # positive -> 1
        # weak -> -gamma * (1 - watch_ratio / 0.8)
        # missing -> 0
        targets = R - self.gamma * W

        super().fit(R, targets=targets)
