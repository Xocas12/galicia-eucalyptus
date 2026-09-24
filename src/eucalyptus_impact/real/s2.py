"""Monthly Sentinel-2 L2A index composites on the 40 m Galicia grid.

For each 12-month window and each month, up to three least-cloudy scenes per MGRS tile are read
from the AWS COG archive at 40 m (via internal overviews), masked with the scene classification
layer, converted to NDVI, NDMI and NBR with each scene's own scale and offset (the 2022
processing-baseline offset is already removed from the pixels by the archive), median-combined per tile, reprojected to
the Galicia grid and mosaicked. The result is a float16 memmap of shape (12, 3, ny, nx).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .common import GRID_40M, INTERIM, fetch_json, log, s3_prefixes

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
os.environ.setdefault("GDAL_CACHEMAX", "256")
TILES = ["29TMG", "29TMH", "29TMJ", "29TNG", "29TNH", "29TNJ", "29TPG", "29TPH", "29TPJ"]
PERIODS = {
    "2017": [(2016, m) for m in (10, 11, 12)] + [(2017, m) for m in range(1, 10)],
    "2024": [(2023, m) for m in (10, 11, 12)] + [(2024, m) for m in range(1, 10)],
}
# SCL classes treated as invalid: no data, saturated, cloud shadow, cloud medium/high, cirrus,
# snow.
BAD_SCL = np.array([0, 1, 3, 8, 9, 10, 11])
INDEX_NAMES = ("ndvi", "ndmi", "nbr")


@dataclass
class Scene:
    prefix: str
    cloud: float
    scale: dict
    offset: dict


def band_scale_offset(meta: dict) -> tuple[dict, dict]:
    """Reflectance scale and offset per band from an Earth Search STAC item.

    Earth Search harmonises processing baseline >= 04.00 pixels itself and flags it with
    `earthsearch:boa_offset_applied`. The -0.1 offset left in the band metadata must then not be
    applied again (raw values match pre-2022 scenes, checked on tile 29TNH).
    """
    applied = bool(meta["properties"].get("earthsearch:boa_offset_applied", False))
    sc, of = {}, {}
    for key in ("red", "nir", "swir16", "swir22"):
        rb = meta["assets"][key].get("raster:bands", [{}])[0]
        sc[key] = rb.get("scale", 1e-4)
        of[key] = 0.0 if applied else rb.get("offset", 0.0)
    return sc, of


def list_scenes(tile: str, year: int, month: int, max_scenes: int = 3, max_cloud: float = 80):
    zone, band, sq = tile[:2], tile[2], tile[3:]
    prefixes = s3_prefixes(BUCKET, f"sentinel-s2-l2a-cogs/{zone}/{band}/{sq}/{year}/{month}/")
    scenes = []
    for p in prefixes:
        name = p.rstrip("/").rsplit("/", 1)[-1]
        try:
            meta = fetch_json(f"{BUCKET}/{p}{name}.json")
        except Exception:
            continue
        cc = meta["properties"].get("eo:cloud_cover", 100.0)
        if cc > max_cloud:
            continue
        sc, of = band_scale_offset(meta)
        scenes.append(Scene(p, cc, sc, of))
    scenes.sort(key=lambda s: s.cloud)
    return scenes[:max_scenes]


ASSET_FILE = {
    "red": ("B04.tif", 4),
    "nir": ("B08.tif", 4),
    "swir16": ("B11.tif", 2),
    "swir22": ("B12.tif", 2),
    "scl": ("SCL.tif", 2),
}


def _read(url: str, factor: int):
    import rasterio

    for attempt in range(4):
        try:
            with rasterio.open(f"/vsicurl/{url}") as src:
                shape = (src.height // factor, src.width // factor)
                arr = src.read(1, out_shape=shape)
                tr = src.transform * src.transform.scale(
                    src.width / shape[1], src.height / shape[0]
                )
                return arr, tr, src.crs
        except Exception as e:  # transient network errors
            if attempt == 3:
                raise
            log.info("retry %s (%s)", url.rsplit("/", 2)[-2:], e)
            time.sleep(2 * (attempt + 1))


def scene_indices(scene: Scene):
    """(3, h, w) float32 NDVI/NDMI/NBR at 40 m in the tile's native grid, NaN where invalid."""
    bands = {}
    tr = crs = None
    for key, (fname, factor) in ASSET_FILE.items():
        arr, t, c = _read(f"{BUCKET}/{scene.prefix}{fname}", factor)
        if key == "red":
            tr, crs = t, c
        bands[key] = arr
    ok = ~np.isin(bands["scl"], BAD_SCL)
    refl = {}
    for k in ("red", "nir", "swir16", "swir22"):
        r = bands[k].astype("float32") * scene.scale[k] + scene.offset[k]
        refl[k] = np.where((bands[k] > 0) & ok, r, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = lambda a, b: (a - b) / (a + b)  # noqa: E731
        out = np.stack(
            [
                nd(refl["nir"], refl["red"]),
                nd(refl["nir"], refl["swir16"]),
                nd(refl["nir"], refl["swir22"]),
            ]
        )
    out[(out < -1) | (out > 1)] = np.nan
    return out.astype("float16"), tr, crs


def tile_month(tile: str, year: int, month: int):
    scenes = list_scenes(tile, year, month)
    if not scenes:
        return None
    stack, tr, crs = [], None, None
    for sc in scenes:
        try:
            idx, tr, crs = scene_indices(sc)
            stack.append(idx)
        except Exception as e:
            log.info("scene failed %s: %s", sc.prefix, e)
    if not stack:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.stack(stack).astype("float32"), axis=0)
    n_used = len(stack)
    del stack
    return med, tr, crs, n_used


def build_period(period: str, threads: int = 4) -> np.memmap:
    """Build (or reuse) the monthly composite cube for one period."""
    from rasterio.warp import Resampling, reproject

    from .common import transform_of

    path = INTERIM / f"s2_{period}.f16"
    done = INTERIM / f"s2_{period}.done"
    ny, nx = GRID_40M.shape
    shape = (12, 3, ny, nx)
    if done.exists():
        return np.memmap(path, dtype="float16", mode="r", shape=shape)
    cube = np.memmap(path, dtype="float16", mode="w+", shape=shape)
    for mi, (year, month) in enumerate(PERIODS[period]):
        t0 = time.time()
        acc = np.zeros((3, ny, nx), "float32")
        cnt = np.zeros((3, ny, nx), "uint8")
        with ThreadPoolExecutor(threads) as ex:
            results = list(ex.map(lambda t, y=year, m=month: tile_month(t, y, m), TILES))
        n_sc = 0
        for res in results:
            if res is None:
                continue
            med, tr, crs, k = res
            n_sc += k
            for b in range(3):
                dst = np.full((ny, nx), np.nan, "float32")
                reproject(
                    med[b],
                    dst,
                    src_transform=tr,
                    src_crs=crs,
                    dst_transform=transform_of(GRID_40M),
                    dst_crs=GRID_40M.crs,
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
                ok = np.isfinite(dst)
                acc[b][ok] += dst[ok]
                cnt[b][ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            cube[mi] = (acc / cnt).astype("float16")
        cube.flush()
        valid = np.isfinite(cube[mi, 0]).mean()
        log.info(
            "S2 %s %d-%02d: %d scenes, %.0f%% valid, %.0fs",
            period,
            year,
            month,
            n_sc,
            100 * valid,
            time.time() - t0,
        )
    done.write_text("ok")
    return np.memmap(path, dtype="float16", mode="r", shape=shape)
