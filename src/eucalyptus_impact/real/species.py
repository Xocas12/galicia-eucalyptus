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

from ..features.spectral import _design, harmonic_features, harmonic_fit
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
    """Harmonic phenology summary plus gap-filled monthly values for the given pixels.

    Cloud gaps are filled from each pixel's own fitted annual curve, and the clear-observation
    share is dropped. Otherwise the pattern of missing months, which follows each Sentinel-2
    tile's acquisition footprint, lets the classifier recognise the tile. With 91% of the
    eucalyptus labels inside one 100 km square, it did exactly that, and the 2017 map showed a
    tile-shaped block.
    """
    series = np.asarray(cube[:, :, rows, cols], dtype="float32")  # (12, 3, n)
    series = np.transpose(series, (2, 0, 1))  # (n, 12, 3)
    F, names = harmonic_features(series, MONTHS)
    F = F[:, [k for k, nm in enumerate(names) if nm != "clear_share"]]
    X = _design(MONTHS, 2)
    filled = np.empty_like(series)
    for b in range(series.shape[2]):
        coef = harmonic_fit(series[:, :, b], MONTHS)
        curve = coef @ X.T
        filled[:, :, b] = np.where(np.isfinite(series[:, :, b]), series[:, :, b], curve)
    return np.column_stack([F, filled.reshape(len(filled), -1)])


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


class DropEmptyColumns:
    """Classifier wrapper that drops feature columns with no finite value in training.

    The 2017 window has no imagery for October 2016 (the L2A archive starts in November), so
    those monthly columns are all missing and gradient boosting cannot bin them.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.keep_ = np.isfinite(X).any(axis=0)
        self.model.fit(X[:, self.keep_], y)
        return self

    def predict(self, X):
        return self.model.predict(X[:, self.keep_])

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, self.keep_])


def make_classifier(seed=0):
    return DropEmptyColumns(
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    )


NORTH_SQUARE = 548  # 100 km square (easting 500-600 km, northing 4800-4900 km) holding the OSM
# eucalyptus labels: A Coruna, Ferrol, Ortegal.
WINTER, SUMMER = [2, 3, 4], [8, 9, 10]  # Dec-Feb and Jun-Aug in the Oct-Sep cube


def _square(rows, cols) -> np.ndarray:
    x, y = GRID_40M.centers()
    return (x[rows, cols] // 1e5).astype(int) * 100 + (y[rows, cols] // 1e5).astype(int)


def winter_metrics(cube, rows, cols):
    """Winter NDVI, winter NDMI and summer NDVI (monthly medians) for the given pixels."""
    import warnings

    raw = np.asarray(cube[:, :, rows, cols], dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            np.nanmedian(raw[WINTER, 0], 0),
            np.nanmedian(raw[WINTER, 1], 0),
            np.nanmedian(raw[SUMMER, 0], 0),
        )


def harvest_share(L: dict) -> np.ndarray:
    """Share of 2001-2016 Hansen stand-replacing loss within ~280 m (7x7 pixels at 40 m).

    Pre-2017 only, so it cannot carry the 2018-2023 fire outcomes into the exposure map.
    """
    from scipy import ndimage

    ly = L["hansen"]["lossyear40"].astype(int)
    return ndimage.uniform_filter(((ly >= 1) & (ly <= 16)).astype("float32"), 7)


def clean_mask(cube, L, rows, cols, y) -> np.ndarray:
    """Keep only labels consistent with each class's winter behaviour.

    Galician eucalyptus is evergreen broadleaf. OSM "eucalyptus" pixels that lose greenness in
    winter are mislabelled or cleared, and "native broadleaf" pixels that stay green and moist
    through winter in harvested stands are very likely eucalyptus. Checked on 2024: southern
    OSM eucalyptus labels had native-like winter profiles.
    """
    w0, w1, s0 = winter_metrics(cube, rows, cols)
    drop = s0 - w0
    h = harvest_share(L)[rows, cols]
    keep = np.ones(len(y), bool)
    keep[(y == 0) & ~((w0 >= 0.7) & (drop < 0.12))] = False
    keep[(y == 2) & ~((drop >= 0.12) | (w1 < 0.2))] = False
    keep[(y == 1) & (w1 >= 0.4) & (drop < 0.1) & (h >= 0.2)] = False
    return keep


def pseudo_eucalyptus_2024(L, per_square: int = 8000, candidates: int = 80_000, seed: int = 0):
    """Eucalyptus pseudo-labels across Galicia from 2024 imagery and harvest history.

    Rule: tree cover (WorldCover), evergreen (winter NDVI >= 0.75, summer-winter NDVI drop <
    0.1), moist in winter (winter NDMI >= 0.4, which separates it from pine), and in a harvested
    neighbourhood (>= 20% of pixels within ~280 m lost 2001-2016, the short-rotation plantation
    regime native forest lacks). Sampled evenly per 100 km square so no region dominates.
    """
    cube = build_period("2024")
    lab = L["osm"]["label40"]
    cond = (L["aoi"]["mask40"] == 1) & (L["worldcover"]["wc40"] == 10) & (lab == 255)
    rr, cc = np.nonzero(cond)
    sq = _square(rr, cc)
    h = harvest_share(L)
    rng = np.random.default_rng(seed)
    out_r, out_c = [], []
    for s_ in np.unique(sq):
        idx = np.flatnonzero(sq == s_)
        idx = rng.choice(idx, min(candidates, len(idx)), replace=False)
        r, c = rr[idx], cc[idx]
        w0, w1, s0 = winter_metrics(cube, r, c)
        ok = (w0 >= 0.75) & (w1 >= 0.4) & (s0 - w0 < 0.1) & (h[r, c] >= 0.2)
        out_r.append(r[ok][:per_square])
        out_c.append(c[ok][:per_square])
    return np.concatenate(out_r), np.concatenate(out_c)


def build_training(period: str, per_class: int, seed: int, exclude_square: int | None = None):
    """OSM labels cleaned by winter behaviour, plus Galicia-wide eucalyptus pseudo-labels.

    For 2017 the 2024 pseudo-labels are reused where the pixel is stable between the two maps
    (no Hansen loss 2017-2024, no EFFIS burn 2018-2023), as with the OSM labels. Returns rows,
    cols, labels and source (0 = OSM, 1 = pseudo-label).
    """
    L = all_layers()
    cube = build_period(period)
    lab = training_labels(L, period)
    r, c, y = sample_training(lab, per_class, seed)
    keep = clean_mask(cube, L, r, c, y)
    r, c, y = r[keep], c[keep], y[keep]
    pr, pc = pseudo_eucalyptus_2024(L, seed=seed)
    if period == "2017":
        ly = L["hansen"]["lossyear40"][pr, pc]
        stable = ~((ly >= 17) & (ly <= 24))
        for yr in range(2018, 2024):
            stable &= ~L["effis"][f"burned40_{yr}"][pr, pc].astype(bool)
        pr, pc = pr[stable], pc[stable]
    rows, cols = np.concatenate([r, pr]), np.concatenate([c, pc])
    y = np.concatenate([y, np.zeros(len(pr), int)])
    src = np.concatenate([np.zeros(len(r), int), np.ones(len(pr), int)])
    if exclude_square is not None:
        ok = _square(rows, cols) != exclude_square
        rows, cols, y, src = rows[ok], cols[ok], y[ok], src[ok]
    return rows, cols, y, src


def north_transfer(period: str, seed: int = 0, n_test: int = 60_000) -> dict:
    """Train without the northern square, test on its OSM labels (never seen, never cleaned).

    This is the transfer test the eucalyptus map has to pass: the pseudo-labels come from a rule
    applied elsewhere, and the OSM eucalyptus labels are almost all in this square.
    """
    L = all_layers()
    cube = build_period(period)
    lab = training_labels(L, period)
    r, c, y, _ = build_training(period, 25_000, seed, exclude_square=NORTH_SQUARE)
    model = make_classifier(seed).fit(_pixels_features(cube, r, c), y)
    tr_, tc_ = np.nonzero(lab < 255)
    in_n = _square(tr_, tc_) == NORTH_SQUARE
    tr_, tc_ = tr_[in_n], tc_[in_n]
    idx = np.random.default_rng(seed).choice(len(tr_), min(n_test, len(tr_)), replace=False)
    tr_, tc_ = tr_[idx], tc_[idx]
    ty = lab[tr_, tc_].astype(int)
    p = model.predict(_pixels_features(cube, tr_, tc_))
    from sklearn.metrics import precision_score, recall_score

    return {
        "euc_f1": float(f1_score(ty == 0, p == 0)),
        "euc_precision": float(precision_score(ty == 0, p == 0, zero_division=0)),
        "euc_recall": float(recall_score(ty == 0, p == 0, zero_division=0)),
        "accuracy": float(accuracy_score(ty, p)),
        "f1": f1_score(ty, p, labels=np.arange(6), average=None, zero_division=0).tolist(),
        "n_test": len(ty),
    }


def train_period(period: str, per_class: int = 30_000, seed: int = 0, n_folds: int = 5):
    cube = build_period(period)
    rows, cols, y, src = build_training(period, per_class, seed)
    X = _pixels_features(cube, rows, cols)
    blocks = block_ids(GRID_40M, 20)[rows, cols]
    cv_pred = np.empty_like(y)
    for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=blocks):
        cv_pred[te] = make_classifier(seed).fit(X[tr], y[tr]).predict(X[te])
    model = make_classifier(seed).fit(X, y)
    labels = np.arange(6)
    osm = src == 0
    metrics = {
        "n_train": len(y),
        "n_pseudo_eucalyptus": int((src == 1).sum()),
        "per_class_train": np.bincount(y, minlength=6).tolist(),
        "spatial_cv_accuracy": float(accuracy_score(y, cv_pred)),
        "kappa": float(cohen_kappa_score(y, cv_pred)),
        "f1": f1_score(y, cv_pred, labels=labels, average=None, zero_division=0).tolist(),
        "f1_osm_only": f1_score(
            y[osm], cv_pred[osm], labels=labels, average=None, zero_division=0
        ).tolist(),
        "confusion": confusion_matrix(y, cv_pred, labels=labels).tolist(),
        "north_transfer": north_transfer(period, seed),
    }
    nt = metrics["north_transfer"]
    log.info(
        "species %s: spatial-CV accuracy %.3f, F1 %s; north transfer eucalyptus F1 %.2f "
        "(P %.2f, R %.2f)",
        period,
        metrics["spatial_cv_accuracy"],
        np.round(metrics["f1"], 2),
        nt["euc_f1"],
        nt["euc_precision"],
        nt["euc_recall"],
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
