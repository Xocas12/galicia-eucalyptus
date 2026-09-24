"""Native-forest conversion: transition matrices, loss attribution and driver models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ..causal.dml import dml_plr, ols
from ..data.synthetic import CLASSES, N_CLASSES
from ..validation.spatial_cv import SpatialBlockKFold


def transition_matrix(cover_a: np.ndarray, cover_b: np.ndarray, weights=None) -> pd.DataFrame:
    """Area transition matrix (rows: class at A, cols: class at B) from paired pixel labels."""
    w = np.ones(len(cover_a)) if weights is None else np.asarray(weights, float)
    M = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(M, (cover_a, cover_b), w)
    return pd.DataFrame(M, index=CLASSES, columns=CLASSES)


def attribute_loss_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Split observed tree-cover loss into fire, plantation rotation and conversion.

    Only observable evidence is used: the burned-area product, and the change in mapped cover
    between consecutive years. The rules are:
    - fire: loss in a cell-year that burned;
    - conversion: in unburned cell-years, the part of loss matched by a native/pine decline with a
      simultaneous eucalyptus gain;
    - rotation: the remaining loss (clear-fell with no land-use change).
    Returns attributed and (where available) true totals per driver.
    """
    loss = panel["tree_loss"].to_numpy()
    burned = panel["burned"].to_numpy().astype(bool)
    forest_decline = np.clip(-(panel["d_native"] + panel["d_pine"]).to_numpy(), 0, None)
    euc_gain = np.clip(panel["d_euc"].to_numpy(), 0, None)
    # Map noise creates small spurious cover changes; only count changes above a detection floor.
    floor = 0.02
    conv = np.where(
        ~burned & (forest_decline > floor) & (euc_gain > floor),
        np.minimum(np.minimum(forest_decline, euc_gain), loss),
        0.0,
    )
    fire = np.where(burned, loss, 0.0)
    rotation = np.clip(loss - fire - conv, 0, None)
    out = pd.DataFrame(
        {
            "driver": ["fire", "rotation", "conversion"],
            "attributed": [fire.sum(), rotation.sum(), conv.sum()],
        }
    )
    true_cols = [f"true_loss_{d}" for d in out["driver"]]
    if all(c in panel for c in true_cols):
        out["true"] = [panel[c].sum() for c in true_cols]
    out["attributed_share"] = out["attributed"] / out["attributed"].sum()
    if "true" in out:
        out["true_share"] = out["true"] / out["true"].sum()
    return out


DRIVER_FEATURES = [
    "dist_mill_km",
    "elev",
    "slope",
    "neigh_euc",
    "recent_fire",
    "f_native_broadleaf",
    "f_shrub",
    "f_agriculture",
    "f_pine",
    "continentality",
    "log_pop",
    "dist_road_km",
    "year",
]


def conversion_drivers(panel: pd.DataFrame, n_folds=5, seed=0, max_rows=150_000) -> dict:
    """Predictive driver model of the conversion rate plus the causal fire -> plantation effect."""
    df = panel[panel["f_eucalyptus"] < 0.95]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    X = df[DRIVER_FEATURES].to_numpy()
    y = df["conv_share"].to_numpy()
    groups = df["block"].to_numpy()
    folds = SpatialBlockKFold(n_folds, seed)
    r2 = []
    for tr, te in folds.split(groups=groups):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2.append(1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    model = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(X, y)
    sub = np.random.default_rng(seed).choice(len(X), size=min(20_000, len(X)), replace=False)
    imp = permutation_importance(model, X[sub], y[sub], n_repeats=5, random_state=seed)
    importance = (
        pd.DataFrame({"feature": DRIVER_FEATURES, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    confounders = [f for f in DRIVER_FEATURES if f != "recent_fire"]
    truth = float(df["true_te_conversion"].mean()) if "true_te_conversion" in df else None
    dml = dml_plr(
        y,
        df["recent_fire"].to_numpy(),
        df[confounders].to_numpy(),
        groups,
        n_folds,
        seed,
        name="recent fire -> conversion rate",
    )
    dml.truth = truth
    naive = ols(y, df["recent_fire"].to_numpy(), groups, name="recent fire -> conversion rate")
    naive.truth = truth
    return {
        "model": model,
        "spatial_cv_r2": float(np.mean(r2)),
        "importance": importance,
        "dml": dml,
        "naive": naive,
    }
