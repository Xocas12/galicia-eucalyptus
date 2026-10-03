"""Landsat back-cast of the species map to 1990, 2000 and 2010.

Source: the Landsat Collection 1 Level-1 archive on Google Cloud Storage
(gcp-public-data-landsat), the only Landsat archive reachable from this environment. Galicia
is covered by WRS-2 paths 204-205, rows 30-31.

Per epoch (three years centred on the target year) and path/row, the clearest winter
(November-March) and summer (June-September) Tier-1 scenes are converted to top-of-atmosphere
reflectance, cloud-masked with the BQA band, turned into NDVI, NDMI and NBR, reprojected to the
40 m grid and median-composited per season. Features: winter and summer medians of the three
indices and their summer-winter differences. Eucalyptus is evergreen and stays moist in winter,
which this seasonal contrast captures.

A classifier is trained in each epoch on the same pixels: confident and undisturbed in the
Sentinel-2 maps (same class in both, no Hansen loss 2001-2024, no EFFIS fire). The 2000 epoch is checked against the IFN3
inventory plots, which were surveyed around 1998.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .common import GRID_40M, INTERIM, RAW, cached_npz, log, transform_of

GCS = "https://storage.googleapis.com/gcp-public-data-landsat"
PATH_ROWS = [(204, 30), (204, 31), (205, 30), (205, 31)]
EPOCHS = {"1990": (1989, 1991), "2000": (1999, 2001), "2010": (2008, 2010), "2017": (2016, 2018)}
WINTER, SUMMER = (11, 12, 1, 2, 3), (6, 7, 8, 9)
# red, nir, swir1, swir2 band numbers per sensor
BANDS = {
    "LANDSAT_4": (3, 4, 5, 7),
    "LANDSAT_5": (3, 4, 5, 7),
    "LANDSAT_7": (3, 4, 5, 7),
    "LANDSAT_8": (4, 5, 6, 7),
}
FEATURES = [
    "ndvi_w",
    "ndmi_w",
    "nbr_w",
    "ndvi_s",
    "ndmi_s",
    "nbr_s",
    "d_ndvi",
    "d_ndmi",
    "d_nbr",
]


def scene_index() -> pd.DataFrame:
    """Galicia Tier-1 scenes from the archive index (filtered copy made by `euc real fetch`)."""
    d = pd.read_csv(RAW / "landsat" / "index_galicia.csv")
    d = d[(d["COLLECTION_CATEGORY"] == "T1") & d["SPACECRAFT_ID"].isin(list(BANDS))]
    d["date"] = pd.to_datetime(d["DATE_ACQUIRED"])
    d["season_year"] = d["date"].dt.year + (d["date"].dt.month >= 11).astype(int)
    return d


def choose_scenes(
    idx: pd.DataFrame, epoch: str, per_season: int = 4, max_cloud: float = 40
) -> pd.DataFrame:
    """Clearest scenes per path/row and season within the epoch window.

    Winters are labelled by the year they end in (Nov 1999-Mar 2000 is winter 2000). Landsat 7
    after May 2003 (scan-line corrector off) is used only when nothing else is available, since
    a fifth of each of its scenes is missing.
    """
    y0, y1 = EPOCHS[epoch]
    out = []
    for (p, r), g in idx.groupby(["WRS_PATH", "WRS_ROW"]):
        if (p, r) not in PATH_ROWS:
            continue
        for season, months in (("winter", WINTER), ("summer", SUMMER)):
            s = g[
                g["date"].dt.month.isin(months)
                & (g["season_year"] >= y0)
                & (g["season_year"] <= y1)
                & (g["CLOUD_COVER"] <= max_cloud)
            ].copy()
            slc_off = (s["SPACECRAFT_ID"] == "LANDSAT_7") & (s["date"] >= "2003-06-01")
            s["rank"] = s["CLOUD_COVER"] + 100 * slc_off
            out.append(s.nsmallest(per_season, "rank").assign(season=season))
    return pd.concat(out, ignore_index=True)


def _mtl(url: str) -> dict:
    from .common import curl

    txt = curl(url + "_MTL.txt")
    out = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = (x.strip() for x in line.split("=", 1))
            out[k] = v.strip('"')
    return out


def _qa_bad(qa: np.ndarray, sensor: str) -> np.ndarray:
    """Collection 1 BQA: fill, cloud, high-confidence cloud, cloud shadow, snow, cirrus (OLI)."""
    fill = (qa & 1) > 0
    cloud = (qa & (1 << 4)) > 0
    cloud_conf = ((qa >> 5) & 3) >= 2
    shadow = ((qa >> 7) & 3) >= 2
    snow = ((qa >> 9) & 3) >= 2
    bad = fill | cloud | cloud_conf | shadow | snow
    if sensor == "LANDSAT_8":
        bad |= ((qa >> 11) & 3) >= 2
    return bad


def scene_indices(row: pd.Series):
    """NDVI, NDMI, NBR (3, h, w) at 30 m in the scene's UTM grid, NaN where masked."""
    import rasterio

    base = row["BASE_URL"].replace("gs://gcp-public-data-landsat", GCS) + "/" + row["PRODUCT_ID"]
    meta = _mtl(base)
    sun = np.sin(np.radians(float(meta["SUN_ELEVATION"])))
    refl = []
    tr = crs = None
    for b in BANDS[row["SPACECRAFT_ID"]]:
        for attempt in range(4):
            try:
                with rasterio.open(f"/vsicurl/{base}_B{b}.TIF") as src:
                    dn = src.read(1)
                    tr, crs = src.transform, src.crs
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        m = float(meta[f"REFLECTANCE_MULT_BAND_{b}"])
        a = float(meta[f"REFLECTANCE_ADD_BAND_{b}"])
        refl.append(np.where(dn > 0, (dn * m + a) / sun, np.nan).astype("float32"))
    with rasterio.open(f"/vsicurl/{base}_BQA.TIF") as src:
        bad = _qa_bad(src.read(1).astype("uint16"), row["SPACECRAFT_ID"])
    red, nir, sw1, sw2 = refl
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.stack(
            [(nir - red) / (nir + red), (nir - sw1) / (nir + sw1), (nir - sw2) / (nir + sw2)]
        )
    idx[:, bad] = np.nan
    idx[(idx < -1) | (idx > 1)] = np.nan
    return idx, tr, crs


def _to_grid(idx, tr, crs) -> np.ndarray:
    from rasterio.warp import Resampling, reproject

    ny, nx = GRID_40M.shape
    out = np.full((3, ny, nx), np.nan, "float32")
    for b in range(3):
        reproject(
            idx[b],
            out[b],
            src_transform=tr,
            src_crs=crs,
            dst_transform=transform_of(GRID_40M),
            dst_crs=GRID_40M.crs,
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return out.astype("float16")


def season_composite(scenes: pd.DataFrame, threads: int = 2) -> np.ndarray:
    """Per-pixel median over the scenes' index grids, computed in row blocks from a memmap."""
    ny, nx = GRID_40M.shape
    tmp = INTERIM / "landsat_tmp.f16"
    stack = np.memmap(tmp, dtype="float16", mode="w+", shape=(len(scenes), 3, ny, nx))

    def work(i_row):
        i, row = i_row
        try:
            stack[i] = _to_grid(*scene_indices(row))
            return True
        except Exception as e:  # a missing band file occasionally
            log.info("landsat scene failed %s: %s", row["PRODUCT_ID"], e)
            stack[i] = np.nan
            return False

    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(work, scenes.reset_index(drop=True).iterrows()))
    stack.flush()
    out = np.full((3, ny, nx), np.nan, "float16")
    for r0 in range(0, ny, 400):
        r1 = min(ny, r0 + 400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[:, r0:r1] = np.nanmedian(stack[:, :, r0:r1].astype("float32"), axis=0)
    stack = None  # release the memmap before deleting its file
    tmp.unlink()
    log.info("landsat composite: %d/%d scenes", sum(ok), len(scenes))
    return out


def epoch_features(epoch: str) -> np.memmap:
    """(9, ny, nx) float16 feature stack for one epoch, cached on disk."""
    ny, nx = GRID_40M.shape
    path = INTERIM / f"landsat_{epoch}.f16"
    done = INTERIM / f"landsat_{epoch}.done"
    if done.exists():
        return np.memmap(path, dtype="float16", mode="r", shape=(9, ny, nx))
    sc = choose_scenes(scene_index(), epoch)
    log.info(
        "landsat %s: %d scenes (%s)",
        epoch,
        len(sc),
        sc.groupby("SPACECRAFT_ID").size().to_dict(),
    )
    w = season_composite(sc[sc["season"] == "winter"])
    s = season_composite(sc[sc["season"] == "summer"])
    f = np.memmap(path, dtype="float16", mode="w+", shape=(9, ny, nx))
    f[0:3] = w
    f[3:6] = s
    f[6:9] = s.astype("float32") - w.astype("float32")
    f.flush()
    sc.to_csv(INTERIM / f"landsat_{epoch}_scenes.csv", index=False)
    done.write_text("ok")
    return np.memmap(path, dtype="float16", mode="r", shape=(9, ny, nx))


# ---------------------------------------------------------------- classification


def stable_training_mask(conf: int = 70) -> np.ndarray:
    """Pixels whose class is trustworthy and unchanged: same class in both Sentinel-2 maps,
    confident in 2024, no Hansen loss 2001-2024 and no EFFIS fire 2018-2023."""
    sp = np.load(INTERIM / "species.npz")
    ly = np.load(INTERIM / "hansen.npz")["lossyear40"]
    ef = np.load(INTERIM / "effis.npz")
    burnt = np.zeros(ly.shape, bool)
    for y in range(2018, 2024):
        burnt |= ef[f"burned40_{y}"]
    c17, c24 = sp["class40_2017"], sp["class40_2024"]
    return (c17 == c24) & (c24 < 6) & (sp["pmax40_2024"] >= conf) & (ly == 0) & ~burnt


def _X(F, rows, cols) -> np.ndarray:
    return np.asarray(F[:, rows, cols], dtype="float32").T


@cached_npz("landsat_maps_v2")
def landsat_maps(per_class: int = 20_000, seed: int = 0, n_folds: int = 5):
    """Class maps (40 m) for each epoch, each from a classifier trained in its own epoch.

    Training pixels are the same in every epoch: confident, unchanged Sentinel-2 pixels with no
    Hansen loss 2001-2024 and no fire, which were very probably the same class in 2000 and
    2010 (for 1990 the assumption is weaker: Hansen does not reach back that far). Training
    per epoch avoids cross-sensor normalisation; an earlier version trained on 2017 and
    quantile-matched older epochs onto it, which also matched away real cover change (the
    eucalyptus area came out flat, 523-572 kha, 1990-2017).
    """
    from sklearn.metrics import accuracy_score, f1_score

    from ..geo.grid import block_ids
    from ..validation.spatial_cv import SpatialBlockKFold
    from .species import make_classifier, sample_training

    sp = np.load(INTERIM / "species.npz")
    aoi = np.load(INTERIM / "aoi.npz")["mask40"].astype(bool)
    stable = stable_training_mask() & aoi
    lab = np.where(stable, sp["class40_2024"], 255).astype("uint8")
    rows0, cols0, y0 = sample_training(lab, per_class, seed)
    blocks0 = block_ids(GRID_40M, 20)[rows0, cols0]
    ny, nx = GRID_40M.shape
    out = {}
    for epoch in EPOCHS:
        F = epoch_features(epoch)
        X = _X(F, rows0, cols0)
        ok = np.isfinite(X).sum(1) >= 6
        X, y, blocks = X[ok], y0[ok], blocks0[ok]
        cvp = np.empty_like(y)
        for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=blocks):
            cvp[te] = make_classifier(seed).fit(X[tr], y[tr]).predict(X[te])
        out[f"cv_accuracy_{epoch}"] = np.array(accuracy_score(y, cvp))
        out[f"cv_f1_{epoch}"] = np.array(
            f1_score(y, cvp, labels=range(6), average=None, zero_division=0)
        )
        model = make_classifier(seed).fit(X, y)
        cls = np.full((ny, nx), 255, "uint8")
        for r0 in range(0, ny, 200):
            rr, cc = np.nonzero(aoi[r0 : r0 + 200])
            rr = rr + r0
            Xe = _X(F, rr, cc)
            good = np.isfinite(Xe).sum(1) >= 6
            if good.any():
                cls[rr[good], cc[good]] = model.predict(Xe[good])
        out[f"class40_{epoch}"] = cls
        v = cls[aoi]
        log.info(
            "landsat %s: CV accuracy %.3f, eucalyptus F1 %.2f, eucalyptus %.0f kha",
            epoch,
            out[f"cv_accuracy_{epoch}"],
            out[f"cv_f1_{epoch}"][0],
            (v == 0).sum() * 0.16 / 1e3,
        )
    return out


def fetch_index() -> None:
    """Keep the Galicia rows of the archive index (the full index is ~770 MB compressed)."""
    import subprocess

    out = RAW / "landsat" / "index_galicia.csv"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = " || ".join(f"($10=={p} && $11=={r})" for p, r in PATH_ROWS)
    cmd = (
        f"curl -sS --retry 4 {GCS}/index.csv.gz | gunzip -c | "
        f"awk -F, 'NR==1 || {rows}' > {out}.part && mv {out}.part {out}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)


def backcast_validation(min_f1: float = 0.7, min_loss_ratio: float = 2.0) -> dict:
    """Is the back-cast good enough to date cover change?

    Two checks: the eucalyptus F1 of each epoch's classifier in spatial cross-validation, and
    whether pixels that turn eucalyptus between 2000 and 2010 had a Hansen loss in 2001-2010
    (planting follows a clear-cut) clearly more often than pixels whose class did not change.
    Without the second, map-to-map "change" is classification noise.
    """
    m = backcast_maps()
    aoi = np.load(INTERIM / "aoi.npz")["mask40"] > 0
    ly = np.load(INTERIM / "hansen.npz")["lossyear40"]
    a, b = m["class40_2000"], m["class40_2010"]
    loss = (ly >= 1) & (ly <= 10)
    gain = aoi & (a < 6) & (a != 0) & (b == 0)
    same = aoi & (a < 6) & (a != 0) & (b == a)
    r_gain, r_same = float(loss[gain].mean()), float(loss[same].mean())
    f1 = {e: float(m[f"cv_f1_{e}"][0]) for e in EPOCHS}
    area = {e: float((m[f"class40_{e}"][aoi] == 0).sum() * 0.16) for e in EPOCHS}
    ratio = r_gain / max(r_same, 1e-9)
    return {
        "euc_f1": f1,
        "cv_accuracy": {e: float(m[f"cv_accuracy_{e}"]) for e in EPOCHS},
        "euc_area_ha": area,
        "gain_with_loss": r_gain,
        "same_with_loss": r_same,
        "loss_ratio": ratio,
        "passed": bool(min(f1.values()) >= min_f1 and ratio >= min_loss_ratio),
        "source": "C2" if (INTERIM / "landsat_maps_c2.npz").exists() else "C1",
    }


# ---------------------------------------------------------------- Collection 2 (surface refl.)
#
# The Level-1 back-cast above failed validation. This version uses Landsat Collection 2 Level-2
# surface reflectance from Microsoft Planetary Computer and builds, per epoch, the same monthly
# NDVI/NDMI/NBR cube as the Sentinel-2 maps (October-September order, all clear scenes of the
# three-year window pooled by calendar month), so the species classifier and its harmonic
# features carry over unchanged.

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
C2_PLATFORMS = ("landsat-4", "landsat-5", "landsat-7", "landsat-8")
C2_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # cube order, as the Sentinel-2 periods
C2_SCALE, C2_OFFSET = 2.75e-5, -0.2
# QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
C2_BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)


def c2_items(epoch: str, max_cloud: float = 60, per_month: int = 16) -> list:
    """Clear Collection 2 Level-2 scenes over Galicia for the epoch window, by calendar month.

    Keeps the four Galicia path/rows, and per calendar month at most `per_month` scenes, the
    least cloudy first (Landsat 7 after the May 2003 scan-line failure ranked last).
    """
    import planetary_computer as pc
    import pystac_client

    y0, y1 = EPOCHS[epoch]
    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = cat.search(
        collections=["landsat-c2-l2"],
        bbox=[-9.35, 41.8, -6.7, 43.8],
        datetime=f"{y0 - 1}-10-01/{y1}-09-30",
        query={"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": list(C2_PLATFORMS)}},
    )
    items = [
        it
        for it in search.items()
        if (int(it.properties["landsat:wrs_path"]), int(it.properties["landsat:wrs_row"]))
        in PATH_ROWS
    ]
    by_month: dict[int, list] = {}
    for it in items:
        d = pd.Timestamp(it.properties["datetime"]).tz_localize(None)
        slc_off = it.properties["platform"] == "landsat-7" and d >= pd.Timestamp("2003-06-01")
        it.properties["_rank"] = it.properties["eo:cloud_cover"] + 100 * slc_off
        by_month.setdefault(d.month, []).append(it)
    return {
        m: sorted(v, key=lambda i: i.properties["_rank"])[:per_month] for m, v in by_month.items()
    }


def c2_scene_indices(item, factor: int = 2):
    """NDVI, NDMI, NBR (3, h, w) at 60 m (overview `factor` 2) from surface reflectance."""
    import planetary_computer as pc
    import rasterio

    item = pc.sign(item)
    refl, tr, crs = {}, None, None
    for key in ("red", "nir08", "swir16", "swir22", "qa_pixel"):
        href = item.assets[key].href
        for attempt in range(4):
            try:
                with rasterio.open(href) as src:
                    shape = (src.height // factor, src.width // factor)
                    a = src.read(1, out_shape=shape)
                    if tr is None:
                        tr = src.transform * src.transform.scale(
                            src.width / shape[1], src.height / shape[0]
                        )
                        crs = src.crs
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
                item = pc.sign(item)
                href = item.assets[key].href
        refl[key] = a
    bad = (refl.pop("qa_pixel").astype("uint16") & C2_BAD_BITS) > 0
    r = {
        k: np.where((v > 0) & ~bad, v.astype("float32") * C2_SCALE + C2_OFFSET, np.nan)
        for k, v in refl.items()
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = lambda a, b: (a - b) / (a + b)  # noqa: E731
        out = np.stack(
            [nd(r["nir08"], r["red"]), nd(r["nir08"], r["swir16"]), nd(r["nir08"], r["swir22"])]
        )
    out[(out < -1) | (out > 1)] = np.nan
    return out, tr, crs


def _composite(items: list, threads: int = 4) -> np.ndarray:
    """Per-pixel median of the scenes' index grids (row blocks from a disk memmap)."""
    ny, nx = GRID_40M.shape
    tmp = INTERIM / "landsat_c2_tmp.f16"
    stack = np.memmap(tmp, dtype="float16", mode="w+", shape=(len(items), 3, ny, nx))

    def work(i_it):
        i, it = i_it
        try:
            stack[i] = _to_grid(*c2_scene_indices(it))
            return True
        except Exception as e:  # a failed read leaves the scene out
            log.info("landsat C2 scene failed %s: %s", it.id, e)
            stack[i] = np.nan
            return False

    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(work, enumerate(items)))
    stack.flush()
    out = np.full((3, ny, nx), np.nan, "float16")
    for r0 in range(0, ny, 300):
        r1 = min(ny, r0 + 300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[:, r0:r1] = np.nanmedian(stack[:, :, r0:r1].astype("float32"), axis=0)
    stack = None  # release the memmap before deleting its file
    tmp.unlink()
    return out, sum(ok)


def c2_cube(epoch: str) -> np.memmap:
    """(12, 3, ny, nx) float16 monthly NDVI/NDMI/NBR cube for one epoch, cached on disk."""
    ny, nx = GRID_40M.shape
    path = INTERIM / f"landsat_c2_{epoch}.f16"
    done = INTERIM / f"landsat_c2_{epoch}.done"
    shape = (12, 3, ny, nx)
    if done.exists():
        return np.memmap(path, dtype="float16", mode="r", shape=shape)
    by_month = c2_items(epoch)
    cube = np.memmap(path, dtype="float16", mode="w+", shape=shape)
    for k, month in enumerate(C2_MONTHS):
        items = by_month.get(month, [])
        t0 = time.time()
        if items:
            cube[k], n_ok = _composite(items)
        else:
            cube[k], n_ok = np.nan, 0
        cube.flush()
        log.info(
            "landsat C2 %s month %02d: %d/%d scenes, %.0f%% valid, %.0fs",
            epoch,
            month,
            n_ok,
            len(items),
            100 * np.isfinite(cube[k, 0]).mean(),
            time.time() - t0,
        )
    done.write_text("ok")
    return np.memmap(path, dtype="float16", mode="r", shape=shape)


@cached_npz("landsat_maps_c2")
def landsat_maps_c2(per_class: int = 20_000, seed: int = 0, n_folds: int = 5):
    """Per-epoch class maps from the Collection 2 monthly cubes (same scheme as landsat_maps).

    Features are the Sentinel-2 ones (harmonic phenology plus gap-filled monthly values);
    training pixels are the confident, undisturbed Sentinel-2 pixels (stable_training_mask).
    """
    from sklearn.metrics import accuracy_score, f1_score

    from ..geo.grid import block_ids
    from ..validation.spatial_cv import SpatialBlockKFold
    from .species import _pixels_features, make_classifier, sample_training

    sp = np.load(INTERIM / "species.npz")
    aoi = np.load(INTERIM / "aoi.npz")["mask40"].astype(bool)
    lab = np.where(stable_training_mask() & aoi, sp["class40_2024"], 255).astype("uint8")
    rows0, cols0, y0 = sample_training(lab, per_class, seed)
    blocks0 = block_ids(GRID_40M, 20)[rows0, cols0]
    ny, nx = GRID_40M.shape
    out = {}
    for epoch in EPOCHS:
        cube = c2_cube(epoch)
        X = _pixels_features(cube, rows0, cols0)
        ok = np.isfinite(X).sum(1) >= X.shape[1] // 2
        X, y, blocks = X[ok], y0[ok], blocks0[ok]
        cvp = np.empty_like(y)
        for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=blocks):
            cvp[te] = make_classifier(seed).fit(X[tr], y[tr]).predict(X[te])
        out[f"cv_accuracy_{epoch}"] = np.array(accuracy_score(y, cvp))
        out[f"cv_f1_{epoch}"] = np.array(
            f1_score(y, cvp, labels=range(6), average=None, zero_division=0)
        )
        model = make_classifier(seed).fit(X, y)
        cls = np.full((ny, nx), 255, "uint8")
        for r0 in range(0, ny, 160):
            rr, cc = np.nonzero(aoi[r0 : r0 + 160])
            if len(rr) == 0:
                continue
            rr = rr + r0
            Xe = _pixels_features(cube, rr, cc)
            good = np.isfinite(Xe).sum(1) >= Xe.shape[1] // 2
            if good.any():
                cls[rr[good], cc[good]] = model.predict(Xe[good])
        out[f"class40_{epoch}"] = cls
        v = cls[aoi]
        log.info(
            "landsat C2 %s: CV accuracy %.3f, eucalyptus F1 %.2f, eucalyptus %.0f kha",
            epoch,
            out[f"cv_accuracy_{epoch}"],
            out[f"cv_f1_{epoch}"][0],
            (v == 0).sum() * 0.16 / 1e3,
        )
    return out


def backcast_maps() -> dict:
    """The best available back-cast: Collection 2 if built, else the Level-1 one."""
    if (INTERIM / "landsat_maps_c2.npz").exists():
        return landsat_maps_c2()
    return landsat_maps()
