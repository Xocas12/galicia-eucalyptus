"""Covariate matching with a propensity caliper, bias correction and balance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .dml import EffectEstimate


def standardized_mean_diff(X: np.ndarray, treated: np.ndarray, w_control=None) -> np.ndarray:
    xt, xc = X[treated], X[~treated]
    wc = np.ones(len(xc)) if w_control is None else w_control
    mt = xt.mean(0)
    mc = np.average(xc, axis=0, weights=wc)
    pooled = np.sqrt((xt.var(0) + xc.var(0)) / 2) + 1e-12
    return (mt - mc) / pooled


def match_att(
    y, treated, X, caliper: float = 0.2, feature_names=None, name: str = "att"
) -> tuple[EffectEstimate, pd.DataFrame]:
    """ATT by 1-NN matching (with replacement) on standardised covariates.

    Treated units whose propensity score has no control within `caliper` standard deviations of
    the logit score are dropped (lack of overlap). The estimate is regression bias-corrected
    (Abadie & Imbens 2011). Returns the estimate and a balance table of standardised mean
    differences before and after matching (|SMD| < 0.1 is the usual bar). `balance.attrs` holds
    the matched pairs and the share of treated units dropped for lack of overlap.
    """
    y = np.asarray(y, float)
    treated = np.asarray(treated, bool)
    Xa = np.asarray(X, float)
    Xs = StandardScaler().fit_transform(Xa)
    ps = LogisticRegression(max_iter=2000).fit(Xs, treated).predict_proba(Xs)[:, 1]
    lps = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / np.clip(1 - ps, 1e-6, 1))
    t_idx, c_idx = np.flatnonzero(treated), np.flatnonzero(~treated)
    nn = NearestNeighbors(n_neighbors=1).fit(Xs[c_idx])
    _, j = nn.kneighbors(Xs[t_idx])
    c_match = c_idx[j[:, 0]]
    ok = np.abs(lps[t_idx] - lps[c_match]) <= caliper * lps.std()
    t_m, c_m = t_idx[ok], c_match[ok]
    mu0 = LinearRegression().fit(Xa[c_idx], y[c_idx])
    bias = mu0.predict(Xa[t_m]) - mu0.predict(Xa[c_m])
    diffs = y[t_m] - y[c_m] - bias
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    est = EffectEstimate(name, float(diffs.mean()), se, "Matching (bias-corrected)", int(ok.sum()))
    w = np.bincount(np.searchsorted(c_idx, c_m), minlength=len(c_idx)).astype(float)
    names = feature_names or [f"x{i}" for i in range(Xa.shape[1])]
    before = standardized_mean_diff(Xa, treated)
    after_X = np.vstack([Xa[t_m], Xa[c_idx]])
    after_t = np.r_[np.ones(len(t_m), bool), np.zeros(len(c_idx), bool)]
    after = standardized_mean_diff(after_X, after_t, w_control=w + 1e-12)
    balance = pd.DataFrame({"feature": names, "smd_before": before, "smd_after": after})
    balance.attrs["pairs"] = (t_m, c_m)
    balance.attrs["dropped_share"] = float(1 - ok.mean())
    return est, balance
