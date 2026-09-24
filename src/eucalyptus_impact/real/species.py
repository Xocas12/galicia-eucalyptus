"""Species / cover mapping for Galicia from Sentinel-2 composites and OSM-derived labels.

Classes follow data.synthetic: eucalyptus, pine, native broadleaf, shrub, agriculture, other.

Labels:
- eucalyptus, pine and native broadleaf from OSM forest polygons (tags, see layers.py),
  restricted to pixels WorldCover 2021 calls tree cover;
- shrub from OSM scrub/heath polygons, agriculture from OSM meadow/farmland/orchard/vineyard
  polygons, each restricted to compatible WorldCover classes;
- other from WorldCover built-up, bare, water and wetland.

The 2024 model trains on 2024 composites. The 2017 model trains on 2017 composites using only
labelled pixels with no Hansen loss 2017-2024 and no EFFIS burn 2018-2023 (stable pixels), so
labels drawn today stay valid in 2017.

Validation is spatial-block cross-validation on the labelled pixels. OSM labels are neither a
probability sample nor an official inventory, so the accuracy figures and the error-adjusted
areas describe agreement with OSM, not certified map accuracy. Because labels are sampled per
true class, areas are corrected by confusion-matrix inversion, not the Olofsson estimator
(which needs a sample stratified by mapped class).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from ..features.spectral import harmonic_features
from ..geo.grid import block_ids
from ..validation.spatial_cv import SpatialBlockKFold
from .common import FINE_PER_CELL, GRID_1KM, GRID_40M, INTERIM, cached_npz, log
from .layers import AGRI, EUC, NATIVE, OTHER, PINE, SHRUB, all_layers
from .s2 import build_period

MONTHS = np.array([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9])
CLASS_NAMES = ["eucalyptus", "pine", "native_broadleaf", "shrub", "agriculture", "other"]
PIXEL_HA = 0.16


def training_labels(L: dict, period: str) -> np.ndarray:
    """40 m label raster (255 = none) for one period."""
    lab = L["osm"]["label40"].copy()
    wc = L["worldcover"]["wc40"]
    forest = np.isin(lab, [EUC, PINE, NATIVE])
    lab[forest & (wc != 10)] = 255
    lab[(lab == SHRUB) & ~np.isin(wc, [20, 30])] = 255
    lab[(lab == AGRI) & ~np.isin(wc, [30, 40])] = 255
    other = np.isin(wc, [50, 60, 80, 90]) & (lab == 255)
    lab[other] = OTHER
    lab[L["aoi"]["mask40"] == 0] = 255
    if period == "2017":
        ly = L["hansen"]["lossyear40"]
        unstable = (ly >= 17) & (ly <= 24)
        for y in range(2018, 2024):
            unstable |= L["effis"][f"burned40_{y}"].astype(bool)
        lab[unstable] = 255
    return lab


def _pixels_features(cube, rows, cols) -> np.ndarray:
    series = np.asarray(cube[:, :, rows, cols], dtype="float32")  # (12, 3, n)
    series = np.transpose(series, (2, 0, 1))  # (n, 12, 3)
    F, _ = harmonic_features(series, MONTHS)
    # Raw monthly values as well (NaN where cloudy; gradient boosting handles missing values).
    # They add ~3 points of spatial-CV accuracy over the harmonic summary alone, mostly on the
    # eucalyptus / native broadleaf split (checked on a 48k-pixel 2024 sample: 0.758 -> 0.788).
    return np.column_stack([F, series.reshape(len(series), -1)])


def sample_training(lab: np.ndarray, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    rows, cols, ys = [], [], []
    for k in range(6):
        r, c = np.nonzero(lab == k)
        if len(r) == 0:
            continue
        take = rng.choice(len(r), size=min(per_class, len(r)), replace=False)
        rows.append(r[take])
        cols.append(c[take])
        ys.append(np.full(len(take), k))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(ys)


def make_classifier(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def train_period(period: str, per_class: int = 30_000, seed: int = 0, n_folds: int = 5):
    L = all_layers()
    cube = build_period(period)
    lab = training_labels(L, period)
    rows, cols, y = sample_training(lab, per_class, seed)
    X = _pixels_features(cube, rows, cols)
    blocks = block_ids(GRID_40M, 20)[rows, cols]
    cv_pred = np.empty_like(y)
    for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=blocks):
        cv_pred[te] = make_classifier(seed).fit(X[tr], y[tr]).predict(X[te])
    model = make_classifier(seed).fit(X, y)
    labels = np.arange(6)
    metrics = {
        "n_train": len(y),
        "per_class_train": np.bincount(y, minlength=6).tolist(),
        "spatial_cv_accuracy": float(accuracy_score(y, cv_pred)),
        "kappa": float(cohen_kappa_score(y, cv_pred)),
        "f1": f1_score(y, cv_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion": confusion_matrix(y, cv_pred, labels=labels).tolist(),
    }
    log.info(
        "species %s: spatial-CV accuracy %.3f, F1 %s",
        period,
        metrics["spatial_cv_accuracy"],
        np.round(metrics["f1"], 2),
    )
    return model, metrics, (y, cv_pred)


def predict_period(period: str, model, chunk_rows: int = 160):
    """Class map and max probability (40 m) and soft class fractions (1 km) for the whole AOI."""
    L = all_layers()
    cube = build_period(period)
    mask = L["aoi"]["mask40"].astype(bool)
    ny, nx = GRID_40M.shape
    cls = np.full((ny, nx), 255, "uint8")
    pmax = np.zeros((ny, nx), "uint8")
    psum = np.zeros((6, ny, nx), "float32")
    for r0 in range(0, ny, chunk_rows):
        r1 = min(ny, r0 + chunk_rows)
        rr, cc = np.nonzero(mask[r0:r1])
        if len(rr) == 0:
            continue
        rows = rr + r0
        X = _pixels_features(cube, rows, cc)
        P = model.predict_proba(X)
        k = P.argmax(1)
        cls[rows, cc] = k
        pmax[rows, cc] = np.round(P.max(1) * 100).astype("uint8")
        for j in range(6):
            psum[j, rows, cc] = P[:, j]
    ny1, nx1 = GRID_1KM.shape
    npx = mask.reshape(ny1, FINE_PER_CELL, nx1, FINE_PER_CELL).sum(axis=(1, 3))
    frac = psum.reshape(6, ny1, FINE_PER_CELL, nx1, FINE_PER_CELL).sum(axis=(2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(npx > 0, frac / npx, np.nan).astype("float32")
    return cls, pmax, frac


@cached_npz("species")
def species_maps(per_class: int = 30_000, seed: int = 0):
    """Train, validate and apply the classifier for both periods (cached)."""
    out = {}
    for period in ("2017", "2024"):
        model, metrics, (y, cvp) = train_period(period, per_class, seed)
        cls, pmax, frac = predict_period(period, model)
        out[f"class40_{period}"] = cls
        out[f"pmax40_{period}"] = pmax
        out[f"frac_{period}"] = frac
        out[f"cv_true_{period}"] = y
        out[f"cv_pred_{period}"] = cvp
        pd.Series(metrics).to_json(INTERIM / f"species_metrics_{period}.json")
    return out


def confusion_inversion(map_props: np.ndarray, y: np.ndarray, pred: np.ndarray, k: int = 6):
    """True class shares from mapped shares and P(pred | true), on the simplex.

    Valid when the reference labels are sampled per *true* class (as the capped OSM training
    sample is): P(pred | true) does not depend on how many labels each class got, whereas the
    Olofsson estimator needs a sample stratified by *mapped* class. Solves
    map_props = C^T pi by non-negative least squares with a sum-to-one row.
    """
    from scipy.optimize import nnls

    C = np.zeros((k, k))
    np.add.at(C, (y, pred), 1)
    C = C / np.maximum(C.sum(1, keepdims=True), 1)
    A = np.vstack([C.T, 100 * np.ones(k)])
    b = np.concatenate([map_props, [100.0]])
    pi, _ = nnls(A, b)
    return pi / pi.sum(), C


def area_table(maps: dict, period: str, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Pixel-count area, error-adjusted area (confusion inversion) with bootstrap 95% CI."""
    cls = maps[f"class40_{period}"]
    valid = cls < 255
    m = np.bincount(cls[valid].astype(int), minlength=6) / valid.sum()
    total_ha = valid.sum() * PIXEL_HA
    y, p = maps[f"cv_true_{period}"].astype(int), maps[f"cv_pred_{period}"].astype(int)
    pi, C = confusion_inversion(m, y, p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in range(6) if (y == c).any()]
        )
        boots.append(confusion_inversion(m, y[idx], p[idx])[0])
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    # User's accuracy under the estimated true shares (Bayes): P(true = j | pred = j).
    joint = pi[:, None] * C
    users = np.diag(joint) / np.maximum(joint.sum(0), 1e-12)
    return pd.DataFrame(
        {
            "class": np.arange(6),
            "name": CLASS_NAMES,
            "map_area_ha": m * total_ha,
            "est_area_ha": pi * total_ha,
            "ci95_ha": (hi - lo) / 2 * total_ha,
            "soft_area_ha": [np.nansum(maps[f"frac_{period}"][k]) * 100 for k in range(6)],
            "users_accuracy": users,
            "producers_accuracy": np.diag(C),
        }
    )


def transition_table(maps: dict, conf: int = 70) -> pd.DataFrame:
    """2017 -> 2024 transitions at 40 m, all pixels and confident pixels only (both maps >= conf%)."""
    a, b = maps["class40_2017"], maps["class40_2024"]
    ok = (a < 255) & (b < 255)
    confident = ok & (maps["pmax40_2017"] >= conf) & (maps["pmax40_2024"] >= conf)
    rows = []
    for i in range(6):
        for j in range(6):
            sel = (a == i) & (b == j)
            rows.append(
                {
                    "from": CLASS_NAMES[i],
                    "to": CLASS_NAMES[j],
                    "area_ha": float((sel & ok).sum() * PIXEL_HA),
                    "area_confident_ha": float((sel & confident).sum() * PIXEL_HA),
                }
            )
    return pd.DataFrame(rows)
