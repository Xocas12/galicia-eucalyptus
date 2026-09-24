"""SIMEX correction for map error in cover fractions (Cook & Stefanski 1994).

Regression calibration (dml_plr's `d_error_var`) corrects error in the treatment only. Mapped cover
has error in *every* fraction, and the other fractions enter as controls, so part of the bias
survives. SIMEX handles this generically. It adds extra error of variance lambda * sigma^2 to all
cover fractions (with the same clip-and-renormalise structure as map error), re-estimates at each
lambda, and extrapolates the trend back to lambda = -1, the error-free case.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .dml import EffectEstimate


def perturb_cover(df: pd.DataFrame, cover_cols: list[str], sd: float, rng) -> pd.DataFrame:
    out = df.copy()
    F = out[cover_cols].to_numpy() + sd * rng.standard_normal((len(out), len(cover_cols)))
    F = np.clip(F, 0, None)
    F /= np.maximum(F.sum(axis=1, keepdims=True), 1e-12)
    out[cover_cols] = F
    return out


def simex(
    estimator: Callable[[pd.DataFrame], EffectEstimate],
    df: pd.DataFrame,
    cover_cols: list[str],
    error_var: float,
    lambdas=(0.5, 1.0, 1.5, 2.0),
    n_rep: int = 3,
    seed: int = 0,
) -> tuple[EffectEstimate, pd.DataFrame]:
    """Quadratic SIMEX extrapolation of `estimator` to zero map error.

    Returns the corrected estimate (SE taken from the lambda = 0 fit, which understates the
    extrapolation uncertainty) and the lambda path for plotting.
    """
    rng = np.random.default_rng(seed)
    base = estimator(df)
    path = [{"lambda": 0.0, "estimate": base.estimate}]
    for lam in lambdas:
        vals = [
            estimator(perturb_cover(df, cover_cols, np.sqrt(lam * error_var), rng)).estimate
            for _ in range(n_rep)
        ]
        path.append({"lambda": lam, "estimate": float(np.mean(vals))})
    path = pd.DataFrame(path)
    coef = np.polyfit(path["lambda"], path["estimate"], 2)
    corrected = float(np.polyval(coef, -1.0))
    est = EffectEstimate(base.name, corrected, base.se, "DML-PLR (SIMEX)", base.n, truth=base.truth)
    return est, path
