"""Second-order FM using PyTorch autograd and static video categories.

Only observed binary feedback is used; unknown pairs are never labelled negative.
"""
import ast

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F


def build_item_categories(item_map: pl.DataFrame, categories: pl.DataFrame):
    """Join original video IDs to fixed item_idx; vocabulary uses catalog only.

    Missing metadata gives an empty multi-hot vector; duplicate tags count once.
    """
    mapping = item_map.select("video_id", "item_idx").sort("item_idx")
    if mapping["video_id"].n_unique() != mapping.height:
        raise ValueError("Duplicate video_id in item_map")
    if not np.array_equal(mapping["item_idx"].to_numpy(), np.arange(mapping.height)):
        raise ValueError("item_idx must be contiguous from zero")
    if categories["video_id"].n_unique() != categories.height:
        raise ValueError("Duplicate video_id in categories")
    rows = mapping.join(categories.select("video_id", "feat"), on="video_id", how="left").sort("item_idx")
    parsed = []
    for value in rows["feat"].to_list():
        tags = ast.literal_eval(value) if isinstance(value, str) else value
        if tags is None:
            tags = []
        if not isinstance(tags, (list, tuple)) or any(type(tag) is not int or tag < 0 for tag in tags):
            raise ValueError(f"Invalid category list: {value!r}")
        parsed.append(sorted(set(tags)))
    vocabulary = sorted({tag for tags in parsed for tag in tags})
    return parsed, {tag: index for index, tag in enumerate(vocabulary)}


class FM(nn.Module):
    """Logistic FM with one-hot user/item and multi-hot video categories.

    Loss: mean observed BCE + reg/2 * mean squared non-padding parameters.
    Observed label=0 and label=1 both have weight 1; no negative sampling.
    """
    def __init__(self, n_factors=16, reg=1e-4, learning_rate=1e-2,
                 epochs=10, batch_size=8192, seed=42):
        super().__init__()
        if n_factors < 1 or epochs < 1 or batch_size < 1 or reg < 0 or learning_rate <= 0:
            raise ValueError("Invalid FM hyperparameters")
        self.n_factors = n_factors
        self.reg = reg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._fitted = False

    def _initialize(self, n_users, item_map, categories):
        parsed, self.category_map_ = build_item_categories(item_map, categories)
        self.n_users_ = n_users
        self.n_items_ = item_map.height
        self.n_features_ = n_users + self.n_items_ + len(self.category_map_)
        self.padding_idx_ = self.n_features_
        # Preserve the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.bias = nn.Parameter(torch.zeros(()))
            self.linear = nn.Embedding(self.n_features_ + 1, 1, padding_idx=self.padding_idx_)
            self.factors = nn.Embedding(self.n_features_ + 1, self.n_factors, padding_idx=self.padding_idx_)
            nn.init.zeros_(self.linear.weight)
            nn.init.normal_(self.factors.weight, std=0.01)
            with torch.no_grad():
                self.factors.weight[self.padding_idx_].zero_()
        width = 1 + max((len(tags) for tags in parsed), default=0)
        features = torch.full((self.n_items_, width), self.padding_idx_, dtype=torch.long)
        for item_idx, tags in enumerate(parsed):
            features[item_idx, 0] = n_users + item_idx
            if tags:
                features[item_idx, 1:1 + len(tags)] = torch.tensor([
                    n_users + self.n_items_ + self.category_map_[tag] for tag in tags
                ])
        if "item_features_" in self._buffers:
            self.item_features_ = features
        else:
            self.register_buffer("item_features_", features)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Exact FM formula, with active feature values x_j=1."""
        indices = torch.cat((users[:, None], self.item_features_[items]), dim=1)
        vectors = self.factors(indices)
        linear = self.linear(indices).sum(dim=(1, 2))
        pairwise = 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
        return self.bias + linear + pairwise

    def fit(self, train: pl.DataFrame, item_map: pl.DataFrame,
            categories: pl.DataFrame, n_users: int | None = None):
        """Fit observed rows only; caller controls the train/time boundary.

        No holdout argument is accepted. For final fitting the caller supplies
        train+validation with the original fixed ID maps and catalog.
        """
        self._fitted = False
        if train.height == 0:
            raise ValueError("Empty training data")
        if train.select(pl.struct("user_idx", "item_idx").n_unique()).item() != train.height:
            raise ValueError("Expected one observed label per user/item pair")
        data = train.select("user_idx", "item_idx", "label").sort(["user_idx", "item_idx"])
        if any(data.null_count().row(0)) or not set(data["label"].unique().to_list()) <= {0, 1}:
            raise ValueError("Expected observed binary labels and non-null IDs")
        inferred_users = int(data["user_idx"].max()) + 1
        n_users = inferred_users if n_users is None else n_users
        if n_users < inferred_users or data["user_idx"].min() < 0:
            raise ValueError("user_idx outside user catalog")
        if data["item_idx"].min() < 0 or data["item_idx"].max() >= item_map.height:
            raise ValueError("item_idx outside train catalog")
        self._initialize(n_users, item_map, categories)
        users = torch.tensor(data["user_idx"].to_numpy().copy(), dtype=torch.long)
        items = torch.tensor(data["item_idx"].to_numpy().copy(), dtype=torch.long)
        labels = torch.tensor(data["label"].to_numpy().copy(), dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        self.history_ = []
        self.label_counts_ = {label: int((labels == label).sum()) for label in (0, 1)}
        self.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(users), generator=generator)
            total_bce = 0.0
            for batch in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = self(users[batch], items[batch])
                bce = F.binary_cross_entropy_with_logits(logits, labels[batch])
                squared = self.linear.weight[:-1].square().sum() + self.factors.weight[:-1].square().sum()
                count = self.n_features_ * (1 + self.n_factors)
                loss = bce + 0.5 * self.reg * squared / count
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite FM loss")
                loss.backward()
                optimizer.step()
                total_bce += bce.item() * len(batch)
            self.history_.append({"epoch": epoch + 1, "train_bce": total_bce / len(users)})
        self.eval()
        self._cache_catalog()
        self._fitted = True
        return self

    @torch.no_grad()
    def _cache_catalog(self):
        """Factor the exact FM expression for efficient full-catalog scoring."""
        vectors = self.factors(self.item_features_)
        self._item_sums = vectors.sum(dim=1).numpy().copy()
        self._item_base = (
            self.linear(self.item_features_).sum(dim=(1, 2))
            + 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
        ).numpy().copy()
        self._user_vectors = self.factors.weight[:self.n_users_].numpy().copy()
        self._user_linear = self.linear.weight[:self.n_users_, 0].numpy().copy()
        self._bias = self.bias.item()

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Return raw logits for requested candidates from the fixed catalog."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        ids = np.asarray(item_ids)
        if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("item_ids must be a one-dimensional integer array")
        ids = ids.astype(np.int64, copy=False)
        if not 0 <= user_id < self.n_users_ or np.any((ids < 0) | (ids >= self.n_items_)):
            raise ValueError("ID outside train catalog")
        return self._bias + self._user_linear[user_id] + self._item_base[ids] + self._item_sums[ids] @ self._user_vectors[user_id]
