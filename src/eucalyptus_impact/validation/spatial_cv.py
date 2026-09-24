"""Spatial block cross-validation splitters.

Random k-fold on spatially autocorrelated data leaks information between neighbouring train and
test cells and overstates skill. Every predictive model here is validated by holding out whole
spatial blocks.
"""

from __future__ import annotations

import numpy as np


class SpatialBlockKFold:
    """K-fold over spatial blocks so test cells are not neighbours of train cells.

    Blocks are shuffled and dealt to folds greedily by size so folds hold similar row counts.
    """

    def __init__(self, n_splits: int = 5, seed: int = 0):
        self.n_splits = n_splits
        self.seed = seed

    def fold_of(self, groups: np.ndarray) -> np.ndarray:
        groups = np.asarray(groups)
        uniq, counts = np.unique(groups, return_counts=True)
        if len(uniq) < self.n_splits:
            raise ValueError(f"{len(uniq)} blocks is fewer than n_splits={self.n_splits}")
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(uniq))
        order = order[np.argsort(-counts[order], kind="stable")]
        load = np.zeros(self.n_splits)
        block_fold = np.empty(len(uniq), dtype=int)
        for i in order:
            f = int(np.argmin(load))
            block_fold[i] = f
            load[f] += counts[i]
        return block_fold[np.searchsorted(uniq, groups)]

    def split(self, X=None, y=None, groups=None):
        fold = self.fold_of(groups)
        idx = np.arange(len(fold))
        for f in range(self.n_splits):
            yield idx[fold != f], idx[fold == f]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
