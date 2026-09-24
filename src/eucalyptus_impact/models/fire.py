"""Wildfire susceptibility (prediction) and eucalyptus effect on fire (causal).

The two are kept apart on purpose. The susceptibility model answers "where will it burn?" and is
judged by spatial-CV AUC and calibration. Its feature importances are *associations*. The effect of
eucalyptus comes from DML with confounders partialled out, and is checked against a naive
regression so the confounding bias is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..causal.dml import dml_plr, ols
from ..validation.spatial_cv import SpatialBlockKFold

SUSCEPTIBILITY_FEATURES = [
    "f_eucalyptus",
    "f_pine",
    "f_native_broadleaf",
    "f_shrub",
    "f_agriculture",
    "fwi",
    "elev",
    "slope",
    "continentality",
    "precip_mean",
    "summer_temp",
    "log_pop",
    "dist_road_km",
    "recent_fire",
]
# Confounders for the eucalyptus effect: everything except the treatment itself. The other cover
# fractions are included, so the effect is "eucalyptus instead of agriculture/other, holding pine,
# native and shrub fixed".
CONFOUNDERS = [f for f in SUSCEPTIBILITY_FEATURES if f not in ("f_eucalyptus", "f_agriculture")]


def _subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    return df.sample(max_rows, random_state=seed) if len(df) > max_rows else df


def fit_susceptibility(panel: pd.DataFrame, n_folds=5, seed=0, max_rows=250_000) -> dict:
    df = _subsample(panel, max_rows, seed)
    X = df[SUSCEPTIBILITY_FEATURES].to_numpy()
    y = df["burned"].to_numpy()
    groups = df["block"].to_numpy()
    prob = np.empty(len(y))

    def make():
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            random_state=seed,
        )

    for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=groups):
        prob[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    model = make().fit(X, y)
    bins = pd.qcut(prob, 10, labels=False, duplicates="drop")
    calib = (
        pd.DataFrame({"bin": bins, "predicted": prob, "observed": y})
        .groupby("bin")[["predicted", "observed"]]
        .mean()
        .reset_index()
    )
    return {
        "model": model,
        "features": SUSCEPTIBILITY_FEATURES,
        "spatial_cv_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "base_rate": float(y.mean()),
        "calibration": calib,
    }


def estimate_fire_effects(
    panel: pd.DataFrame, n_folds=5, seed=0, max_rows=250_000, d_error_var: float | None = None
) -> dict:
    """Naive vs DML effect of eucalyptus fraction on fire occurrence and severity.

    Occurrence effects are also estimated for pine, native broadleaf and shrub, so that policy
    scenarios can price whatever mix of cover a policy actually changes.
    """
    df = _subsample(panel, max_rows, seed)
    d = df["f_eucalyptus"].to_numpy()
    X = df[CONFOUNDERS].to_numpy()
    g = df["block"].to_numpy()
    y = df["burned"].to_numpy().astype(float)
    truth_occ = float(df["true_te_fire"].mean()) if "true_te_fire" in df else None
    occ_dml = dml_plr(y, d, X, g, n_folds, seed, name="eucalyptus -> P(burn)")
    occ_naive = ols(y, d, g, name="eucalyptus -> P(burn)")
    occ_dml.truth = occ_naive.truth = truth_occ
    occ_me = None
    if d_error_var:
        occ_me = dml_plr(
            y, d, X, g, n_folds, seed, name="eucalyptus -> P(burn)", d_error_var=d_error_var
        )
        occ_me.truth = truth_occ
    # Effect of every other cover class (vs the agriculture/other reference) so scenarios can price
    # the full cover change, not only the eucalyptus share.
    cover_effects = {"f_eucalyptus": occ_me or occ_dml}
    for cls in ("f_pine", "f_native_broadleaf", "f_shrub"):
        Xc = df[[c for c in CONFOUNDERS if c != cls] + ["f_eucalyptus"]].to_numpy()
        cover_effects[cls] = dml_plr(
            y,
            df[cls].to_numpy(),
            Xc,
            g,
            n_folds,
            seed,
            name=f"{cls[2:]} -> P(burn)",
            d_error_var=d_error_var,
        )

    b = panel[panel["burned"] == 1]
    truth_sev = float(b["true_te_severity"].mean()) if "true_te_severity" in b else None
    sev_dml = dml_plr(
        b["dnbr"].to_numpy(),
        b["f_eucalyptus"].to_numpy(),
        b[CONFOUNDERS].to_numpy(),
        b["block"].to_numpy(),
        n_folds,
        seed,
        name="eucalyptus -> dNBR",
    )
    sev_naive = ols(
        b["dnbr"].to_numpy(),
        b["f_eucalyptus"].to_numpy(),
        b["block"].to_numpy(),
        name="eucalyptus -> dNBR",
    )
    sev_dml.truth = sev_naive.truth = truth_sev
    sev_me = None
    if d_error_var:
        sev_me = dml_plr(
            b["dnbr"].to_numpy(),
            b["f_eucalyptus"].to_numpy(),
            b[CONFOUNDERS].to_numpy(),
            b["block"].to_numpy(),
            n_folds,
            seed,
            name="eucalyptus -> dNBR",
            d_error_var=d_error_var,
        )
        sev_me.truth = truth_sev

    # Reverse-causality check: future eucalyptus gain (t -> t+5) cannot cause fire at t. A non-zero
    # "effect" means fire predicts later planting, which is the fire -> plantation feedback, and
    # shows why treatment must be lagged cover and never contemporaneous or future cover.
    placebo = None
    if "f_eucalyptus_lead5" in df:
        ok = df["f_eucalyptus_lead5"].notna().to_numpy()
        Xp = np.column_stack([X[ok], d[ok]])
        placebo = dml_plr(
            y[ok],
            df["f_eucalyptus_lead5"].to_numpy()[ok] - d[ok],
            Xp,
            g[ok],
            n_folds,
            seed,
            name="reverse check: future eucalyptus gain ~ P(burn)",
        )
    keep = [c for c in ("x", "y", "continentality", "fwi", "true_te_fire") if c in df]
    return {
        "occurrence_frame": df[keep].reset_index(drop=True),
        "severity_frame": b.reset_index(drop=True),
        "occurrence_dml": occ_dml,
        "occurrence_naive": occ_naive,
        "occurrence_dml_me": occ_me,
        "cover_effects": cover_effects,
        "severity_dml_me": sev_me,
        "severity_dml": sev_dml,
        "severity_naive": sev_naive,
        "placebo": placebo,
    }
