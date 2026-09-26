"""Shared grid, area-of-interest and I/O helpers for the real-data pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from ..geo.grid import Grid

log = logging.getLogger("eucalyptus_impact.real")

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
for _d in (RAW, INTERIM):
    _d.mkdir(parents=True, exist_ok=True)

# ETRS89 / UTM 29N, with a 5 km margin around Galicia.
BBOX = (465_000.0, 4_620_000.0, 695_000.0, 4_855_000.0)
GRID_1KM = Grid.from_bbox(BBOX, 1000)
GRID_40M = Grid.from_bbox(BBOX, 40)
LONLAT_BBOX = (-9.45, 41.72, -6.60, 43.85)
FINE_PER_CELL = 25  # 40 m pixels per 1 km cell edge

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
# Public buckets only: never sign requests with whatever AWS keys the environment carries.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")


def transform_of(grid: Grid):
    from rasterio.transform import from_origin

    return from_origin(grid.xmin, grid.ymax, grid.resolution_m, grid.resolution_m)


def curl(url: str, timeout: int = 120) -> str:
    return subprocess.run(
        ["curl", "-sS", "--retry", "4", "-m", str(timeout), url],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def download(url: str, dest: Path, timeout: int = 1800) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(
        ["curl", "-sS", "--retry", "4", "-m", str(timeout), "-o", str(tmp), url], check=True
    )
    tmp.rename(dest)
    return dest


def s3_prefixes(bucket_url: str, prefix: str) -> list[str]:
    """Common prefixes ("folders") under `prefix` in a public S3 bucket."""
    out, token = [], None
    while True:
        u = f"{bucket_url}/?list-type=2&delimiter=/&prefix={prefix}"
        if token:
            from urllib.parse import quote

            u += f"&continuation-token={quote(token)}"
        t = curl(u)
        out += re.findall(r"<Prefix>(.*?)</Prefix>", t)[1:]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", t)
        if not m:
            return out
        token = m.group(1)


def fetch_json(url: str) -> dict:
    return json.loads(curl(url))


def cached_npz(name: str):
    """Decorator: cache a function returning dict[str, ndarray] as data/interim/<name>.npz."""

    def deco(fn):
        def wrapper(*args, refresh: bool = False, **kwargs):
            path = INTERIM / f"{name}.npz"
            if path.exists() and not refresh:
                with np.load(path, allow_pickle=False) as z:
                    return {k: z[k] for k in z.files}
            log.info("building %s", name)
            out = fn(*args, **kwargs)
            np.savez_compressed(path, **out)
            return out

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


def galicia_polygon():
    """Galicia (the four provinces) from Natural Earth admin-1, in EPSG:25829."""
    import geopandas as gpd

    z = download(
        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip",
        RAW / "ne_admin1.zip",
    )
    shp_dir = RAW / "ne_admin1"
    if not shp_dir.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(shp_dir)
    gdf = gpd.read_file(next(shp_dir.glob("*.shp")))
    gal = gdf[(gdf["adm0_a3"] == "ESP") & (gdf["region"] == "Galicia")]
    if len(gal) != 4:
        raise RuntimeError(f"expected 4 Galician provinces, got {sorted(gal['name'])}")
    return gal.to_crs("EPSG:25829").dissolve()[["geometry"]]


@cached_npz("aoi")
def aoi_masks():
    """Share of each 1 km cell inside Galicia, and the 40 m land mask."""
    from rasterio.features import rasterize

    poly = galicia_polygon()
    m40 = rasterize(
        poly.geometry,
        out_shape=GRID_40M.shape,
        transform=transform_of(GRID_40M),
        fill=0,
        default_value=1,
        dtype="uint8",
    )
    ny, nx = GRID_1KM.shape
    frac = m40.reshape(ny, FINE_PER_CELL, nx, FINE_PER_CELL).mean(axis=(1, 3))
    return {"mask40": m40, "frac1km": frac.astype("float32")}


def block_to_1km(a40: np.ndarray, how: str = "mean") -> np.ndarray:
    """Aggregate a 40 m array to the 1 km grid (NaN-aware)."""
    ny, nx = GRID_1KM.shape
    b = a40.reshape(ny, FINE_PER_CELL, nx, FINE_PER_CELL)
    fn = {"mean": np.nanmean, "sum": np.nansum}[how]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(b, axis=(1, 3))


def reproject_to(
    src_arr,
    src_transform,
    src_crs,
    grid: Grid,
    resampling="average",
    dtype="float32",
    src_nodata=None,
):
    from rasterio.warp import Resampling, reproject

    dst = np.full(grid.shape, np.nan, dtype=dtype)
    reproject(
        src_arr,
        dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform_of(grid),
        dst_crs=grid.crs,
        resampling=getattr(Resampling, resampling),
        src_nodata=src_nodata,
        dst_nodata=np.nan,
        num_threads=4,
    )
    return dst
