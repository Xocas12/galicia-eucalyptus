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
    "2018": [(2017, m) for m in (10, 11, 12)] + [(2018, m) for m in range(1, 10)],
    "2024": [(2023, m) for m in (10, 11, 12)] + [(2024, m) for m in range(1, 10)],
}
# SCL classes treated as invalid: no data, saturated, cloud shadow, cloud medium/high, cirrus,
# snow.
BAD_SCL = np.array([0, 1, 3, 8, 9, 10, 11])
INDEX_NAMES = ("ndvi", "ndmi", "nbr")
REFLECTANCE_KEYS = ("red", "nir", "swir16", "swir22", "rededge1", "rededge3", "nir08")


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
    for key in REFLECTANCE_KEYS:
        if key not in meta["assets"]:
            continue
        rb = meta["assets"][key].get("raster:bands", [{}])[0]
        sc[key] = rb.get("scale", 1e-4)
        of[key] = 0.0 if applied else rb.get("offset", 0.0)
    return sc, of


def list_scenes(
    tile: str, year: int, month: int, max_scenes: int | None = 3, max_cloud: float = 80
):
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
    return scenes if max_scenes is None else scenes[:max_scenes]


def scene_date(scene: Scene) -> str:
    return scene.prefix.rstrip("/").rsplit("/", 1)[-1].split("_")[2]


def select_month_dates(year: int, month: int, n_dates: int = 5, min_clear_tiles: float = 0.3):
    """Acquisition dates for one month and, per date, the scenes of every tile acquired then.

    Dates are ranked by expected clear area, the sum over the tiles they cover of
    (1 - cloud cover), so a clear date over most of Galicia beats a clear sliver of one tile.
    Every pixel is then composited from the same pool of dates, with no per-tile choice
    (per-tile choices left tile seams and swath edges in the first maps).
    """
    by_date: dict[str, list[Scene]] = {}
    for t in TILES:
        for sc in list_scenes(t, year, month, max_scenes=None, max_cloud=100):
            by_date.setdefault(scene_date(sc), []).append(sc)
    clear = {d: sum(1 - sc.cloud / 100 for sc in v) for d, v in by_date.items()}
    ranked = [d for d in sorted(clear, key=clear.get, reverse=True) if clear[d] >= min_clear_tiles]
    return {d: by_date[d] for d in sorted(ranked[:n_dates])}


def tile_offsets(grids: list[np.ndarray], min_overlap: int = 2000, ridge: float = 1e-3):
    """Additive offsets per tile and band that best reconcile overlapping tiles of one date.

    Tiles overlap by ~10 km and show the same acquisition there, so the median difference over
    an overlap measures the pair's radiometric offset (the archive's per-tile atmospheric
    correction differs slightly). Offsets o solve min sum_ab w_ab (o_a - o_b - d_ab)^2 + ridge *
    sum o^2, which anchors their mean near zero. Returns an array (n_tiles, n_bands).
    """
    if len(grids) < 2:
        return np.zeros((len(grids), 3))
    n, nb = len(grids), grids[0].shape[0]
    out = np.zeros((n, nb))
    step = 4  # subsample the grid for speed; overlaps hold ~10^5 pixels at 40 m
    subs = [g[:, ::step, ::step].astype("float32") for g in grids]
    for b in range(nb):
        A, y, w = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                both = np.isfinite(subs[i][b]) & np.isfinite(subs[j][b])
                k = int(both.sum())
                if k * step * step < min_overlap:
                    continue
                d = float(np.median(subs[i][b][both] - subs[j][b][both]))
                row = np.zeros(n)
                row[i], row[j] = 1.0, -1.0
                A.append(row)
                y.append(d)
                w.append(np.sqrt(k))
        if not A:
            continue
        A, y, w = np.array(A), np.array(y), np.array(w)
        lhs = (A * w[:, None]).T @ A + ridge * np.eye(n)
        rhs = (A * w[:, None]).T @ y
        out[:, b] = np.linalg.solve(lhs, rhs)
    return out


ASSET_FILE = {
    "red": ("B04.tif", 4),
    "nir": ("B08.tif", 4),
    "swir16": ("B11.tif", 2),
    "swir22": ("B12.tif", 2),
    "scl": ("SCL.tif", 2),
}
# Red-edge product: NDRE (B8A vs B05), a red-edge slope (B07 vs B05) and the SWIR ratio
# (B11 vs B12). Red-edge reflectance tracks leaf chlorophyll and canopy structure, which
# differ between eucalyptus and pine; the three indices extend the NDVI/NDMI/NBR cube.
RE_ASSET_FILE = {
    "rededge1": ("B05.tif", 2),
    "rededge3": ("B07.tif", 2),
    "nir08": ("B8A.tif", 2),
    "swir16": ("B11.tif", 2),
    "swir22": ("B12.tif", 2),
    "scl": ("SCL.tif", 2),
}
PRODUCTS = {"idx": ASSET_FILE, "re": RE_ASSET_FILE}
PRODUCT_INDICES = {"idx": INDEX_NAMES, "re": ("ndre", "nd_re", "nd_swir")}


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


def scene_indices(scene: Scene, product: str = "idx"):
    """(3, h, w) index stack at 40 m in the tile's native grid, NaN where invalid.

    product "idx": NDVI, NDMI, NBR. product "re": NDRE, red-edge slope, SWIR ratio.
    """
    bands = {}
    tr = crs = None
    for key, (fname, factor) in PRODUCTS[product].items():
        arr, t, c = _read(f"{BUCKET}/{scene.prefix}{fname}", factor)
        if tr is None:
            tr, crs = t, c
        bands[key] = arr
    ok = ~np.isin(bands["scl"], BAD_SCL)
    refl = {}
    for k in bands:
        if k == "scl":
            continue
        r = bands[k].astype("float32") * scene.scale.get(k, 1e-4) + scene.offset.get(k, 0.0)
        refl[k] = np.where((bands[k] > 0) & ok, r, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = lambda a, b: (a - b) / (a + b)  # noqa: E731
        if product == "idx":
            layers = [
                nd(refl["nir"], refl["red"]),
                nd(refl["nir"], refl["swir16"]),
                nd(refl["nir"], refl["swir22"]),
            ]
        else:
            layers = [
                nd(refl["nir08"], refl["rededge1"]),
                nd(refl["rededge3"], refl["rededge1"]),
                nd(refl["swir16"], refl["swir22"]),
            ]
        out = np.stack(layers)
    out[(out < -1) | (out > 1)] = np.nan
    return out.astype("float16"), tr, crs


def build_period(period: str, threads: int = 4, product: str = "idx") -> np.memmap:
    """Build (or reuse) the monthly composite cube for one period and product."""
    from rasterio.warp import Resampling, reproject

    from .common import transform_of

    suffix = "" if product == "idx" else f"_{product}"
    path = INTERIM / f"s2_{period}{suffix}.f16"
    done = INTERIM / f"s2_{period}{suffix}.done"
    ny, nx = GRID_40M.shape
    shape = (12, 3, ny, nx)
    if done.exists():
        return np.memmap(path, dtype="float16", mode="r", shape=shape)
    cube = np.memmap(path, dtype="float16", mode="w+", shape=shape)

    def to_grid(res):
        idx, tr, crs = res
        out = np.full((3, ny, nx), np.nan, "float32")
        for b in range(3):
            reproject(
                idx[b].astype("float32"),
                out[b],
                src_transform=tr,
                src_crs=crs,
                dst_transform=transform_of(GRID_40M),
                dst_crs=GRID_40M.crs,
                resampling=Resampling.nearest,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
        return out

    def safe_indices(sc):
        try:
            return scene_indices(sc, product)
        except Exception as e:  # missing band files happen occasionally in the archive
            log.info("scene failed %s: %s", sc.prefix, e)
            return None

    for mi, (year, month) in enumerate(PERIODS[period]):
        t0 = time.time()
        dates = select_month_dates(year, month)
        mosaics = []
        n_sc = 0
        for scenes in dates.values():
            # One mosaic per acquisition date. Overlapping tiles show the same acquisition, but
            # the archive's per-tile atmospheric correction differs slightly, so tiles are
            # harmonised with offsets estimated on their overlaps before averaging.
            with ThreadPoolExecutor(threads) as ex:
                grids = [
                    to_grid(r).astype("float16")
                    for r in ex.map(safe_indices, scenes)
                    if r is not None
                ]
            n_sc += len(grids)
            offs = tile_offsets(grids)
            acc = np.zeros((3, ny, nx), "float32")
            cnt = np.zeros((3, ny, nx), "uint8")
            for g, o in zip(grids, offs, strict=True):
                g = g.astype("float32") - o[:, None, None].astype("float32")
                ok = np.isfinite(g)
                acc[ok] += g[ok]
                cnt[ok] += 1
            del grids
            with np.errstate(invalid="ignore", divide="ignore"):
                mosaics.append((acc / cnt).astype("float16"))
            del acc, cnt
        if mosaics:
            import warnings

            for r0 in range(0, ny, 500):
                r1 = min(ny, r0 + 500)
                block = np.stack([m[:, r0:r1] for m in mosaics]).astype("float32")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cube[mi, :, r0:r1] = np.nanmedian(block, axis=0).astype("float16")
        else:
            cube[mi] = np.nan
        cube.flush()
        valid = np.isfinite(cube[mi, 0]).mean()
        log.info(
            "S2 %s %d-%02d: %d scenes on %d dates (%s), %.0f%% valid, %.0fs",
            period,
            year,
            month,
            n_sc,
            len(dates),
            ",".join(dates),
            100 * valid,
            time.time() - t0,
        )
    done.write_text("ok")
    return np.memmap(path, dtype="float16", mode="r", shape=shape)
