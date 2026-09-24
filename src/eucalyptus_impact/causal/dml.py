"""Double/debiased ML for the partially linear model with spatially clustered SEs.

Model: Y = theta * D + g(X) + e,  D = m(X) + v.
theta is estimated by residual-on-residual regression with cross-fitted nuisance functions
(Chernozhukov et al. 2018). Folds are spatial blocks, and the variance is clustered on the same
blocks, because fire and hydrology outcomes are spatially correlated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..validation.spatial_cv import SpatialBlockKFold


@dataclass
class EffectEstimate:
    name: str
    estimate: float
    se: float
    method: str
    n: int
    truth: float | None = None
    reliability: float | None = None

    @property
    def ci95(self) -> tuple[float, float]:
        return self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se

    @property
    def covers_truth(self) -> bool | None:
        if self.truth is None or not np.isfinite(self.se):
            return None
        lo, hi = self.ci95
        return bool(lo <= self.truth <= hi)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci95"] = list(self.ci95)
        d["covers_truth"] = self.covers_truth
        return d


def _default_learner(seed: int):
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=seed
    )


def cluster_se(psi: np.ndarray, groups: np.ndarray, jacobian: float) -> float:
    """Cluster-robust SE for a moment estimator with score psi and scalar Jacobian."""
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=psi)
    g = len(sums)
    n = len(psi)
    var = (g / max(g - 1, 1)) * np.sum(sums**2) / n**2 / jacobian**2
    return float(np.sqrt(var))


def dml_plr(
    y,
    d,
    X,
    groups,
    n_folds: int = 5,
    seed: int = 0,
    learner_factory=None,
    name: str = "theta",
    d_error_var: float | None = None,
) -> EffectEstimate:
    """Partially linear DML estimate of the effect of continuous treatment d on y.

    d_error_var: variance of classical measurement error in d (e.g. mapped cover fraction minus
    reference-plot fraction, from the accuracy assessment). When given, the estimate is
    regression-calibrated: the error variance is subtracted from the residual treatment variance.
    This matters because partialling out X removes true signal from d but not noise, so the
    attenuation after residualising is much stronger than the raw reliability ratio suggests.
    """
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    make = learner_factory or _default_learner
    res_y = np.empty_like(y)
    res_d = np.empty_like(d)
    for k, (tr, te) in enumerate(SpatialBlockKFold(n_folds, seed).split(groups=groups)):
        res_y[te] = y[te] - make(seed + k).fit(X[tr], y[tr]).predict(X[te])
        res_d[te] = d[te] - make(seed + 100 + k).fit(X[tr], d[tr]).predict(X[te])
    jac = float(np.mean(res_d**2))
    method = "DML-PLR"
    if d_error_var:
        jac = max(jac - d_error_var, 0.05 * jac)
        method = "DML-PLR (ME-corrected)"
    theta = float(np.mean(res_d * res_y) / jac)
    psi = res_d * res_y - theta * (res_d**2 - (d_error_var or 0.0))
    reliability = float(1 - (d_error_var or 0.0) / np.mean(res_d**2))
    return EffectEstimate(
        name, theta, cluster_se(psi, groups, jac), method, len(y), reliability=reliability
    )


def ols(y, X, groups=None, name: str = "coef", col: int = 0) -> EffectEstimate:
    """OLS with intercept; returns the coefficient on column `col`, cluster-robust if groups."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    Z = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    bread = np.linalg.pinv(Z.T @ Z)
    if groups is None:
        meat = (Z * resid[:, None] ** 2).T @ Z
    else:
        _, inv = np.unique(groups, return_inverse=True)
        s = np.zeros((inv.max() + 1, Z.shape[1]))
        np.add.at(s, inv, Z * resid[:, None])
        meat = s.T @ s
    cov = bread @ meat @ bread
    j = col + 1
    return EffectEstimate(name, float(beta[j]), float(np.sqrt(cov[j, j])), "OLS", len(y))
