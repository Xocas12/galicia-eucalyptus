"""Sensitivity of DML estimates to unobserved confounding.

Omitted-variable-bias bounds for the partially linear model (Cinelli & Hazlett 2020; Chernozhukov,
Cinelli, Newey, Sharma & Syrgkanis 2022). A latent confounder A that explains a share R2_D of the
residual variance of the treatment and R2_Y of the residual variance of the outcome can shift the
estimate by at most

    |bias| <= sqrt(R2_Y * R2_D / (1 - R2_D)) * sd(eps) / sd(v),

where v is the treatment residual and eps the outcome residual after removing the effect. The
*robustness value* is the equal strength R2_Y = R2_D at which that bound wipes out the estimate
(or its 95% CI reaches zero). In Galicia the obvious candidates for A are arson intent, land
ownership disputes on community forests, and forest-management intensity. None is mapped, which
is why this analysis is reported next to every fire effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dml import EffectEstimate


def _scales(est: EffectEstimate) -> tuple[float, float]:
    r = est.diag
    var_v = float(np.mean(r["res_d"] ** 2)) - r["d_error_var"]
    eps = r["res_y"] - est.estimate * r["res_d"]
    return float(np.sqrt(np.mean(eps**2))), float(np.sqrt(var_v))


def bias_bound(est: EffectEstimate, r2_y: float, r2_d: float) -> float:
    """Maximum absolute bias from a confounder with partial R2s (r2_y, r2_d)."""
    sd_eps, sd_v = _scales(est)
    return float(np.sqrt(r2_y * r2_d / (1 - r2_d)) * sd_eps / sd_v)


def robustness_value(
    est: EffectEstimate, target: float = 0.0, alpha_z: float | None = None
) -> float:
    """Equal-strength partial R2 of a confounder needed to move the estimate to `target`.

    With alpha_z (e.g. 1.96), the value at which the CI bound nearest to `target` reaches it.
    """
    sd_eps, sd_v = _scales(est)
    gap = abs(est.estimate - target)
    if alpha_z:
        gap = max(gap - alpha_z * est.se, 0.0)
    f2 = (gap * sd_v / sd_eps) ** 2
    # Solve r^2 / (1 - r) = f2 for r in [0, 1).
    return float(0.5 * (np.sqrt(f2**2 + 4 * f2) - f2))


def sensitivity_table(est: EffectEstimate, r2_grid=(0.01, 0.02, 0.05, 0.1)) -> pd.DataFrame:
    """Bias bounds and the resulting worst-case intervals over a grid of equal-strength R2s."""
    rows = []
    lo, hi = est.ci95
    for r2 in r2_grid:
        b = bias_bound(est, r2, r2)
        rows.append(
            {"r2_confounder": r2, "max_bias": b, "worst_ci_low": lo - b, "worst_ci_high": hi + b}
        )
    return pd.DataFrame(rows)
