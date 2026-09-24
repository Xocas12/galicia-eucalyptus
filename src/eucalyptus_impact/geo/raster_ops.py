"""Raster neighbourhood operations: focal means, distance transforms, zonal statistics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def focal_mean(arr: np.ndarray, radius: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean of `arr` in a (2r+1)^2 window, counting only cells inside `mask`."""
    size = 2 * radius + 1
    if mask is None:
        mask = np.ones(arr.shape, dtype=bool)
    vals = np.where(mask, arr, 0.0)
    num = ndimage.uniform_filter(vals.astype(float), size=size, mode="constant")
    den = ndimage.uniform_filter(mask.astype(float), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, 0.0)


def distance_to(mask: np.ndarray, resolution_m: float) -> np.ndarray:
    """Euclidean distance (m) from every cell to the nearest True cell of `mask`."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask) * resolution_m


def zonal_mean(values: np.ndarray, zones: np.ndarray, n_zones: int) -> np.ndarray:
    """Mean of `values` per zone id in [0, n_zones); negative zone ids are ignored."""
    ok = zones >= 0
    s = np.bincount(zones[ok], weights=values[ok], minlength=n_zones)
    c = np.bincount(zones[ok], minlength=n_zones)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


def zonal_sum(values: np.ndarray, zones: np.ndarray, n_zones: int) -> np.ndarray:
    ok = zones >= 0
    return np.bincount(zones[ok], weights=values[ok], minlength=n_zones)


def gaussian_field(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Standardised spatially correlated random field (smoothed white noise)."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    return (f - f.mean()) / (f.std() + 1e-12)
