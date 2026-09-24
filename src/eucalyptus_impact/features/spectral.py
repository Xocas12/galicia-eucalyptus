"""Spectral indices and harmonic (phenology) features for species mapping.

Galicia is cloudy for much of the year, so single-date composites are unreliable. A per-pixel
harmonic regression fits the annual cycle through whatever clear observations exist; its mean,
amplitude and phase are what separate evergreen eucalyptus from deciduous oak and chestnut.
"""

from __future__ import annotations

import numpy as np


def normalized_difference(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def ndvi(nir, red):
    return normalized_difference(nir, red)


def ndmi(nir, swir1):
    return normalized_difference(nir, swir1)


def nbr(nir, swir2):
    return normalized_difference(nir, swir2)


def dnbr(nbr_pre, nbr_post):
    """Differenced NBR scaled by 1000 (the usual burn-severity convention)."""
    return 1000.0 * (np.asarray(nbr_pre) - np.asarray(nbr_post))


def _design(months: np.ndarray, n_harmonics: int) -> np.ndarray:
    t = 2 * np.pi * np.asarray(months, float) / 12.0
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    return np.stack(cols, axis=1)


def harmonic_fit(series: np.ndarray, months: np.ndarray, n_harmonics: int = 2, ridge=1e-3):
    """Least-squares harmonic coefficients per pixel, skipping NaN (cloud) observations.

    series: (n_pixels, n_obs). Returns coefficients (n_pixels, 1 + 2*n_harmonics).
    """
    X = _design(months, n_harmonics)
    ok = np.isfinite(series)
    Y = np.where(ok, series, 0.0)
    W = ok.astype(float)
    XtX = np.einsum("no,oi,oj->nij", W, X, X) + ridge * np.eye(X.shape[1])
    XtY = np.einsum("no,oi->ni", W * Y, X)
    return np.linalg.solve(XtX, XtY[..., None])[..., 0]


def row_nanpercentile(a: np.ndarray, q: float) -> np.ndarray:
    """Per-row percentile ignoring NaN, vectorised; matches np.nanpercentile (linear method).

    np.nanpercentile falls back to a Python loop over rows when NaNs are present, which is far
    too slow for tens of millions of pixels.
    """
    a = np.sort(np.asarray(a, dtype=float), axis=1)  # NaN sorts last
    k = np.isfinite(a).sum(axis=1)
    pos = (np.maximum(k, 1) - 1) * q / 100.0
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(k - 1, 0))
    rows = np.arange(len(a))
    frac = pos - lo
    out = a[rows, lo] * (1 - frac) + a[rows, hi] * frac
    out[k == 0] = np.nan
    return out


def harmonic_features(series: np.ndarray, months: np.ndarray, names=("ndvi", "ndmi", "nbr")):
    """Feature matrix from a (n_pixels, n_obs, n_indices) cube of index time series.

    Per index: harmonic mean, first-harmonic amplitude and phase (as cos/sin to avoid the wrap),
    second-harmonic amplitude, and robust 10th/90th percentiles and range. The share of clear
    observations is appended once.
    """
    feats, cols = [], []
    for j, nm in enumerate(names):
        s = series[:, :, j]
        c = harmonic_fit(s, months)
        amp1 = np.hypot(c[:, 1], c[:, 2])
        phase = np.arctan2(c[:, 2], c[:, 1])
        amp2 = np.hypot(c[:, 3], c[:, 4])
        p10 = row_nanpercentile(s, 10)
        p90 = row_nanpercentile(s, 90)
        feats += [c[:, 0], amp1, np.cos(phase), np.sin(phase), amp2, p10, p90, p90 - p10]
        cols += [
            f"{nm}_{k}" for k in ("mean", "amp1", "phcos", "phsin", "amp2", "p10", "p90", "rng")
        ]
    feats.append(np.isfinite(series[:, :, 0]).mean(axis=1))
    cols.append("clear_share")
    F = np.column_stack(feats)
    return np.nan_to_num(F, nan=0.0), cols
