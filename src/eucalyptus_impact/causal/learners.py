"""Nuisance learners for DML.

Gradient-boosted trees approximate smooth, near-linear relationships (cover fractions that sum to
one, climate gradients) with piecewise-constant steps. The approximation error correlates with the
treatment residual and biases the DML estimate towards zero. On the synthetic landscape this cost
about 17% of the soil-moisture effect even with error-free cover. Fitting a ridge regression first
and boosting its residuals captures the linear part exactly and leaves the trees the nonlinearity.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class LinearPlusBoost(BaseEstimator, RegressorMixin):
    """Ridge regression plus gradient boosting on its residuals."""

    def __init__(
        self,
        seed: int = 0,
        max_iter: int = 250,
        learning_rate: float = 0.08,
        min_samples_leaf: int = 40,
    ):
        self.seed = seed
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.linear_ = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z, y)
        self.boost_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=31,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        ).fit(X, y - self.linear_.predict(Z))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.linear_.predict(self.scaler_.transform(X)) + self.boost_.predict(X)
