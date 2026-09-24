"""Water: catchment panel fixed effects, Budyko (Fu) fit, pixel soil-moisture effect."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..causal.dml import EffectEstimate, dml_plr, ols
from ..causal.matching import match_att
from ..data.synthetic import fu_et


def _demean(df: pd.DataFrame, cols, unit, time) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    for _ in range(50):  # alternating projections: exact for balanced panels after one pass
        prev = out.copy()
        out = out - out.groupby(df[unit]).transform("mean")
        out = out - out.groupby(df[time]).transform("mean")
        if np.max(np.abs(out.to_numpy() - prev.to_numpy())) < 1e-10:
            break
    return out


def twfe(df: pd.DataFrame, y: str, x: str, unit: str, time: str, controls=()) -> EffectEstimate:
    """Two-way fixed-effects coefficient on x, SEs clustered by unit."""
    cols = [y, x, *controls]
    w = _demean(df, cols, unit, time)
    est = ols(
        w[y].to_numpy(), w[[x, *controls]].to_numpy(), df[unit].to_numpy(), name=f"{x} -> {y}"
    )
    est.method = "TWFE"
    # Degrees-of-freedom correction for the absorbed fixed effects.
    n, k = len(df), df[unit].nunique() + df[time].nunique() + len(controls)
    est.se *= np.sqrt((n - 1) / max(n - k - 1, 1))
    return est


def fit_budyko(df: pd.DataFrame) -> dict:
    """Fit Fu's curve with w = w0 + a_euc*f_euc + a_pine*f_pine + a_native*f_native.

    Nonlinear least squares on catchment-year ET = P - Q. Returns parameters with asymptotic SEs.
    """
    P = df["precip"].to_numpy()
    PET = df["pet"].to_numpy()
    ET = P - df["runoff"].to_numpy()
    F = df[["f_eucalyptus", "f_pine", "f_native_broadleaf"]].to_numpy()

    def resid(theta):
        w = theta[0] + F @ theta[1:]
        return (fu_et(P, PET, np.maximum(w, 1.01)) - ET) / P

    fit = least_squares(resid, x0=[2.0, 0.5, 0.5, 0.5])
    J = fit.jac
    dof = max(len(P) - len(fit.x), 1)
    s2 = np.sum(fit.fun**2) / dof
    cov = s2 * np.linalg.pinv(J.T @ J)
    names = ["w0", "w_euc", "w_pine", "w_native"]
    return {
        n: (float(v), float(np.sqrt(cov[i, i])))
        for i, (n, v) in enumerate(zip(names, fit.x, strict=True))
    }


def budyko_runoff_effect(df: pd.DataFrame, params: dict, delta: float = 0.01) -> float:
    """Average change in runoff (mm/yr) per unit eucalyptus share implied by the Budyko fit."""
    P, PET = df["precip"].to_numpy(), df["pet"].to_numpy()
    F = df[["f_eucalyptus", "f_pine", "f_native_broadleaf"]].to_numpy()
    th = np.array([params[k][0] for k in ("w0", "w_euc", "w_pine", "w_native")])
    w = th[0] + F @ th[1:]
    return float(np.mean(-(fu_et(P, PET, w + th[1] * delta) - fu_et(P, PET, w)) / delta))


SM_COVARIATES = [
    "elev",
    "slope",
    "continentality",
    "precip_mean",
    "pet_mean",
    "summer_temp",
    "log_pop",
    "dist_coast_km",
    "f_pine",
    "f_native_broadleaf",
    "f_shrub",
]


def water_effects(
    catch: pd.DataFrame, xsec: pd.DataFrame, n_folds=5, seed=0, d_error_var: float | None = None
) -> dict:
    controls = ["precip", "pet", "f_pine", "f_native_broadleaf"]
    truth = float(catch["true_te_water"].mean())
    naive = ols(
        catch["runoff"], catch["f_eucalyptus"], catch["catchment"], name="eucalyptus -> runoff"
    )
    naive.truth = truth
    fe = twfe(catch, "runoff", "f_eucalyptus", "catchment", "year", controls)
    fe.truth = truth
    native_fe = twfe(
        catch,
        "runoff",
        "f_native_broadleaf",
        "catchment",
        "year",
        ["precip", "pet", "f_pine", "f_eucalyptus"],
    )
    low = twfe(catch, "low_flow", "f_eucalyptus", "catchment", "year", controls)
    budyko = fit_budyko(catch)
    b_eff = budyko_runoff_effect(catch, budyko)
    budyko_est = EffectEstimate(
        "eucalyptus -> runoff", b_eff, float("nan"), "Budyko-Fu", len(catch), truth
    )

    sm_cov = SM_COVARIATES
    sm_dml = dml_plr(
        xsec["soil_moisture"],
        xsec["f_eucalyptus"],
        xsec[sm_cov].to_numpy(),
        xsec["block"],
        n_folds,
        seed,
        name="eucalyptus -> summer soil moisture",
    )
    sm_me = None
    if d_error_var:
        sm_me = dml_plr(
            xsec["soil_moisture"],
            xsec["f_eucalyptus"],
            xsec[sm_cov].to_numpy(),
            xsec["block"],
            n_folds,
            seed,
            name="eucalyptus -> summer soil moisture",
            d_error_var=d_error_var,
        )
    sm_naive = ols(
        xsec["soil_moisture"],
        xsec["f_eucalyptus"],
        xsec["block"],
        name="eucalyptus -> summer soil moisture",
    )
    # Matching: plantation-dominated cells (>40% eucalyptus) vs cells with <10%.
    sub = xsec[(xsec["f_eucalyptus"] > 0.4) | (xsec["f_eucalyptus"] < 0.1)]
    treated = (sub["f_eucalyptus"] > 0.4).to_numpy()
    match_cov = [c for c in sm_cov if c not in ("f_pine", "f_native_broadleaf", "f_shrub")]
    att, balance = match_att(
        sub["soil_moisture"],
        treated,
        sub[match_cov].to_numpy(),
        feature_names=match_cov,
        name="eucalyptus stand vs other -> soil moisture",
    )
    return {
        "runoff_naive": naive,
        "runoff_twfe": fe,
        "native_runoff_twfe": native_fe,
        "runoff_budyko": budyko_est,
        "low_flow_twfe": low,
        "budyko_params": budyko,
        "soil_moisture_dml": sm_dml,
        "soil_moisture_naive": sm_naive,
        "soil_moisture_dml_me": sm_me,
        "soil_moisture_matching": att,
        "matching_balance": balance,
    }
