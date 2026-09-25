"""Independent reference points for the species map, from the GBIF occurrence archive.

The official references (IFN4 plots, Mapa Forestal de España) are not reachable from this
environment. GBIF's monthly snapshot is (public S3, Parquet), and its Galician tree records
(iNaturalist, Observation.org, herbaria, forest-inventory datasets republished on GBIF) share
no source with the OpenStreetMap labels the classifier was trained on.

They are presence-only and opportunistic: observers favour roadsides, parks and edges, and a
point may be a single tree inside a pixel of something else. So the check reported is
per-species agreement (share of points of each species the map puts in each class), overall and
by region, not a design-based accuracy. The official inventory remains the proper check.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .common import GRID_40M, INTERIM, LONLAT_BBOX, log

GBIF_BUCKET = "gbif-open-data-eu-central-1"
GBIF_SNAPSHOT = "2026-09-01"
COLUMNS = [
    "gbifid",
    "datasetkey",
    "genus",
    "species",
    "decimallatitude",
    "decimallongitude",
    "coordinateuncertaintyinmeters",
    "year",
    "basisofrecord",
    "establishmentmeans",
]

# Map classes: EUC, PINE, NATIVE, SHRUB, AGRI, OTHER = range(6) (see layers.osm_forest_labels).
GENUS_CLASS = {
    "Eucalyptus": 0,
    "Pinus": 1,
    "Quercus": 2,
    "Castanea": 2,
    "Betula": 2,
    "Alnus": 2,
    "Fraxinus": 2,
    "Ilex": 2,
    "Sorbus": 2,
    "Arbutus": 2,
    "Fagus": 2,
    "Ulex": 3,
    "Cytisus": 3,
    "Erica": 3,
    "Calluna": 3,
    "Genista": 3,
    "Pterospartum": 3,
}
# Native genera that are usually single trees in hedgerows or riparian strips rather than
# stands; kept for the table but excluded from the stand-level summary.
NON_STAND = {"Fraxinus", "Ilex", "Sorbus", "Arbutus", "Alnus"}


def _s3():
    import pyarrow.fs as pfs

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    kw = {"proxy_options": proxy} if proxy else {}
    return pfs.S3FileSystem(anonymous=True, region="eu-central-1", **kw)


def _scan_file(fs, path: str) -> pd.DataFrame | None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    lon0, lat0, lon1, lat1 = LONLAT_BBOX
    for attempt in range(4):
        try:
            pf = pq.ParquetFile(path, filesystem=fs)
            # Cheap pass first: country codes are tiny, most files hold no Spanish records.
            cc = pf.read(columns=["countrycode"]).column(0)
            if not pc.any(pc.equal(cc, "ES")).as_py():
                return None
            t = pf.read(columns=[*COLUMNS, "countrycode"])
            conds = [
                pc.equal(t["countrycode"], "ES"),
                pc.is_in(t["genus"], value_set=pa.array(list(GENUS_CLASS))),
                pc.greater_equal(t["decimallatitude"], lat0),
                pc.less_equal(t["decimallatitude"], lat1),
                pc.greater_equal(t["decimallongitude"], lon0),
                pc.less_equal(t["decimallongitude"], lon1),
            ]
            m = conds[0]
            for c in conds[1:]:
                m = pc.and_kleene(m, c)
            m = pc.fill_null(m, False)
            if not pc.any(m).as_py():
                return None
            return t.filter(m).drop(["countrycode"]).to_pandas()
        except Exception as e:  # transient S3 errors
            if attempt == 3:
                log.info("GBIF file failed %s: %s", path, e)
                return None


def gbif_records(threads: int = 16) -> pd.DataFrame:
    """All GBIF records of the reference genera inside the Galicia bounding box (cached)."""
    import pyarrow.fs as pfs

    out = INTERIM / "gbif_trees.parquet"
    if out.exists():
        return pd.read_parquet(out)
    fs = _s3()
    files = fs.get_file_info(
        pfs.FileSelector(f"{GBIF_BUCKET}/occurrence/{GBIF_SNAPSHOT}/occurrence.parquet/")
    )
    paths = [f.path for f in files if f.size > 0]
    log.info("GBIF: scanning %d files", len(paths))
    parts = []
    with ThreadPoolExecutor(threads) as ex:
        for i, df in enumerate(ex.map(lambda p: _scan_file(fs, p), paths)):
            if df is not None:
                parts.append(df)
            if i % 500 == 0:
                log.info("GBIF: %d/%d files, %d records", i, len(paths), sum(map(len, parts)))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    df.to_parquet(out)
    return df


def filter_records(
    df: pd.DataFrame,
    max_uncertainty: float = 60.0,
    years: tuple[int, int] = (2019, 2025),
) -> pd.DataFrame:
    """Records usable as map reference: precise, recent, observed (not specimens of cultivated
    plants), one per species and 40 m pixel."""
    d = df.copy()
    unc = pd.to_numeric(d["coordinateuncertaintyinmeters"], errors="coerce")
    d = d[unc.notna() & (unc <= max_uncertainty)]
    y = pd.to_numeric(d["year"], errors="coerce")
    d = d[(y >= years[0]) & (y <= years[1])]
    d = d[d["basisofrecord"].isin(["HUMAN_OBSERVATION", "OBSERVATION", "MACHINE_OBSERVATION"])]
    # Coordinates rounded to ~1 m by several apps; exact duplicate positions of different
    # species are usually a centroid (a park, a village) rather than a tree.
    key = d["decimallatitude"].round(5).astype(str) + d["decimallongitude"].round(5).astype(str)
    d = d[key.map(key.value_counts()) <= 3]
    d["ref_class"] = d["genus"].map(GENUS_CLASS).astype(int)
    return d


def to_pixels(d: pd.DataFrame) -> pd.DataFrame:
    """Row/col on the 40 m grid (ETRS89 / UTM 29N), one record per genus and pixel."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", GRID_40M.crs, always_xy=True)
    x, y = tr.transform(d["decimallongitude"].to_numpy(), d["decimallatitude"].to_numpy())
    col = np.floor((x - GRID_40M.xmin) / GRID_40M.resolution_m).astype(int)
    row = np.floor((GRID_40M.ymax - y) / GRID_40M.resolution_m).astype(int)
    ny, nx = GRID_40M.shape
    ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
    d = d.assign(row=row, col=col)[ok]
    return d.drop_duplicates(["genus", "row", "col"])


def agreement_table(pts: pd.DataFrame, pred: np.ndarray, group: str | None = None) -> pd.DataFrame:
    """Share of reference points of each class that the map assigns to each class.

    The diagonal is a presence-only recall: of the pixels where observers saw class c, the share
    the map calls c. The EUC column off the diagonal is how often the map calls eucalyptus
    where observers saw something else.
    """
    p = pred[pts["row"].to_numpy(), pts["col"].to_numpy()]
    d = pts.assign(pred=p)
    d = d[d["pred"] < 255]
    keys = ["ref_class"] + ([group] if group else [])
    tab = pd.crosstab([d[k] for k in keys], d["pred"], normalize="index")
    tab = tab.reindex(columns=range(6), fill_value=0.0)
    tab["n"] = d.groupby(keys).size()
    return tab


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (c - h, c + h)


def eucalyptus_scores(pts: pd.DataFrame, pred: np.ndarray, prior_euc: float) -> dict:
    """Recall, and precision/F1 re-weighted to a stated eucalyptus share among forest pixels.

    `pred` is the class map, or already the predicted class of each point (1-D).

    Presence-only points do not give prevalence, so precision is computed at an assumed share of
    eucalyptus among the forest reference classes (from the map itself, which the brief says).
    """
    p = pred[pts["row"].to_numpy(), pts["col"].to_numpy()] if pred.ndim == 2 else pred
    ok = p < 255
    y = pts["ref_class"].to_numpy()[ok]
    p = p[ok]
    forest = np.isin(y, [0, 1, 2])
    euc = y == 0
    n_e, n_o = int((euc & forest).sum()), int((~euc & forest).sum())
    tp = int(((p == 0) & euc).sum())
    fp_rate = float(((p == 0) & ~euc & forest).sum() / max(n_o, 1))
    recall = tp / max(n_e, 1)
    prec = prior_euc * recall / max(prior_euc * recall + (1 - prior_euc) * fp_rate, 1e-9)
    f1 = 2 * prec * recall / max(prec + recall, 1e-9)
    lo, hi = wilson(tp, n_e)
    return {
        "n_euc": n_e,
        "n_other_forest": n_o,
        "recall": recall,
        "recall_ci": [lo, hi],
        "false_euc_rate": fp_rate,
        "precision_at_prior": prec,
        "f1_at_prior": f1,
        "prior_euc": prior_euc,
    }


def _square_of(rows, cols) -> np.ndarray:
    """100 km square code (as species._square) from 40 m row/col."""
    x = GRID_40M.xmin + (np.asarray(cols) + 0.5) * GRID_40M.resolution_m
    y = GRID_40M.ymax - (np.asarray(rows) + 0.5) * GRID_40M.resolution_m
    return (x // 1e5).astype(int) * 100 + (y // 1e5).astype(int)


def _tolerant(pts: pd.DataFrame, pred: np.ndarray, radius: int = 1) -> np.ndarray:
    """Class matched if any pixel within `radius` of the point has the reference class
    (absorbs GPS error and points on stand edges); otherwise the pixel's own class."""
    r, c, y = pts["row"].to_numpy(), pts["col"].to_numpy(), pts["ref_class"].to_numpy()
    ny, nx = pred.shape
    own = pred[r, c].copy()
    hit = np.zeros(len(r), bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = np.clip(r + dr, 0, ny - 1), np.clip(c + dc, 0, nx - 1)
            hit |= pred[rr, cc] == y
    return np.where(hit, y, own)


def reference_check(min_points: int = 30) -> dict:
    """Agreement of the species maps with GBIF points, overall, outside the north and outside
    the OSM training polygons. 2024 map vs 2019-2025 records; 2017 map vs 2014-2018 records."""
    from .species import NORTH_SQUARE

    raw = gbif_records()
    sp = np.load(INTERIM / "species.npz")
    osm = np.load(INTERIM / "osm_labels_v2.npz")["label40"]
    aoi = np.load(INTERIM / "aoi.npz")["mask40"]
    out: dict = {"n_raw": len(raw), "snapshot": GBIF_SNAPSHOT}
    for period, years, key in (
        ("2024", (2019, 2025), "class40_2024"),
        ("2017", (2014, 2018), "class40_2017"),
        ("2017_independent", (2014, 2018), "class40_2017_independent"),
    ):
        pts = to_pixels(filter_records(raw, years=years))
        pts = pts[aoi[pts["row"], pts["col"]] > 0]
        pred = sp[key]
        pts = pts.assign(
            region=np.where(_square_of(pts["row"], pts["col"]) == NORTH_SQUARE, "norte", "resto"),
            in_training=osm[pts["row"], pts["col"]] < 255,
        )
        f = pred[aoi > 0]
        forest = np.isin(f, [0, 1, 2])
        prior = float((f[forest] == 0).mean())
        res: dict = {"n_points": len(pts), "prior_euc": prior}
        res["by_genus"] = (
            pts.assign(pred=pred[pts["row"], pts["col"]])
            .groupby("genus")
            .agg(n=("pred", "size"), share_mapped_euc=("pred", lambda s: float((s == 0).mean())))
            .reset_index()
            .to_dict(orient="records")
        )
        subsets = {
            "todo": pts,
            "fora_do_norte": pts[pts["region"] == "resto"],
            "norte": pts[pts["region"] == "norte"],
            "fora_das_etiquetas": pts[~pts["in_training"]],
            "fora_do_norte_e_etiquetas": pts[(pts["region"] == "resto") & ~pts["in_training"]],
        }
        for name, d in subsets.items():
            if (d["ref_class"] == 0).sum() < min_points:
                res[name] = {"n_euc": int((d["ref_class"] == 0).sum()), "too_few": True}
                continue
            s = eucalyptus_scores(d, pred, prior)
            s["recall_3x3"] = eucalyptus_scores(d, _tolerant(d, pred), prior)["recall"]
            s["agreement"] = agreement_table(d, pred).reset_index().to_dict(orient="records")
            res[name] = s
        out[period] = res
    return out


# The Third Spanish National Forest Inventory (IFN3) plots, published on GBIF by the Ministry
# (institution code MAGRAMA, collection code IFN3, occurrence ids "MAGRAMA:IFN3:<n>", licence
# CC BY-NC 4.0): a systematic 1 km grid of plots with the species present in each, no counts
# and no date. IFN3 fieldwork in Galicia was around 1997-1998, so the plots predate the
# Sentinel-2 maps by two decades; the Landsat 2000 epoch is the date-matched comparison.
INVENTORY_KEY = "fab4c599-802a-4bfc-8a59-fc7515001bfa"
TREE_GENERA = {"Eucalyptus", "Pinus", "Quercus", "Castanea", "Betula", "Alnus", "Fagus"}


def inventory_plots(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per plot with the genera recorded in it, on the 40 m grid."""
    from pyproj import Transformer

    g = raw[raw["datasetkey"] == INVENTORY_KEY]
    tr = Transformer.from_crs("EPSG:4326", GRID_40M.crs, always_xy=True)
    x, y = tr.transform(g["decimallongitude"].to_numpy(), g["decimallatitude"].to_numpy())
    g = g.assign(px=np.round(x, -3), py=np.round(y, -3), x=x, y=y)
    plots = g.groupby(["px", "py"]).agg(
        x=("x", "mean"), y=("y", "mean"), genera=("genus", lambda s: frozenset(s))
    )
    plots = plots.reset_index()
    col = np.floor((plots["x"] - GRID_40M.xmin) / GRID_40M.resolution_m).astype(int)
    row = np.floor((GRID_40M.ymax - plots["y"]) / GRID_40M.resolution_m).astype(int)
    ny, nx = GRID_40M.shape
    ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
    plots = plots.assign(row=row, col=col)[ok].reset_index(drop=True)
    has = lambda gen: plots["genera"].map(lambda s: gen in s)  # noqa: E731
    trees = plots["genera"].map(lambda s: bool(s & TREE_GENERA))
    plots["plot_type"] = np.select(
        [has("Eucalyptus"), has("Pinus"), trees],
        ["eucalipto", "piñeiro sen eucalipto", "frondosas sen eucalipto nin piñeiro"],
        "só mato",
    )
    return plots


def _plot_scores(p: pd.DataFrame, pred: np.ndarray) -> dict:
    m = pred[p["row"], p["col"]]
    ok = m < 6
    p, m = p[ok], m[ok]
    euc = (p["plot_type"] == "eucalipto").to_numpy()
    tp = int(((m == 0) & euc).sum())
    n_map = int((m == 0).sum())
    n_e = int(euc.sum())
    prec, rec = tp / max(n_map, 1), tp / max(n_e, 1)
    return {
        "n_plots": len(p),
        "n_euc_plots": n_e,
        "recall": rec,
        "recall_ci": list(wilson(tp, n_e)),
        "precision": prec,
        "precision_ci": list(wilson(tp, n_map)),
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "false_euc_rate": float(((m == 0) & ~euc).sum() / max((~euc).sum(), 1)),
    }


def inventory_check() -> dict:
    """Map agreement with the inventory plots: overall, outside the north, by plot type, and
    whether map-eucalyptus plots without eucalyptus show later disturbance (Hansen loss from
    2001, the first year it covers, or EFFIS fire), which would point to planting after the inventory rather than map error."""
    from .species import NORTH_SQUARE

    raw = gbif_records()
    plots = inventory_plots(raw)
    sp = np.load(INTERIM / "species.npz")
    osm = np.load(INTERIM / "osm_labels_v2.npz")["label40"]
    ly = np.load(INTERIM / "hansen.npz")["lossyear40"].astype(int)
    ef = np.load(INTERIM / "effis.npz")
    burnt = np.zeros(ly.shape, bool)
    for y in range(2018, 2024):
        burnt |= ef[f"burned40_{y}"]
    r, c = plots["row"].to_numpy(), plots["col"].to_numpy()
    # 3 x 3 window: the plot is 25 m in radius and its position is known to ~10 m.
    dist = np.zeros(len(plots), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = np.clip(r + dr, 0, ly.shape[0] - 1)
            cc = np.clip(c + dc, 0, ly.shape[1] - 1)
            dist |= (ly[rr, cc] >= 1) | burnt[rr, cc]
    plots = plots.assign(
        region=np.where(_square_of(r, c) == NORTH_SQUARE, "norte", "resto"),
        in_training=osm[r, c] < 255,
        disturbed_since_2001=dist,
    )
    out: dict = {
        "n_plots": len(plots),
        "n_records": int((raw["datasetkey"] == INVENTORY_KEY).sum()),
    }
    for key, name in (
        ("class40_2024", "2024"),
        ("class40_2017", "2017"),
        ("class40_2017_independent", "2017_independent"),
    ):
        pred = sp[key]
        res = {
            "todo": _plot_scores(plots, pred),
            "fora_do_norte": _plot_scores(plots[plots["region"] == "resto"], pred),
            "norte": _plot_scores(plots[plots["region"] == "norte"], pred),
            "fora_das_etiquetas": _plot_scores(plots[~plots["in_training"]], pred),
        }
        m = pred[r, c]
        tab = pd.crosstab(plots["plot_type"], np.where(m < 6, m, 6), normalize="index")
        tab = tab.reindex(columns=range(7), fill_value=0.0)
        tab["n"] = plots.groupby("plot_type").size()
        res["by_type"] = tab.reset_index().to_dict(orient="records")
        fp = (m == 0) & (plots["plot_type"] != "eucalipto").to_numpy()
        tn = (m < 6) & (m != 0) & (plots["plot_type"] != "eucalipto").to_numpy()
        res["disturbed_share_false_euc"] = float(plots["disturbed_since_2001"][fp].mean())
        res["disturbed_share_other"] = float(plots["disturbed_since_2001"][tn].mean())
        # Share of forest plots (any tree genus) that list eucalyptus, against the map's share
        # of eucalyptus at the same plots: both describe the same sample.
        forest = plots["plot_type"] != "só mato"
        res["plot_euc_share"] = float((plots["plot_type"][forest] == "eucalipto").mean())
        mf = m[forest.to_numpy()]
        res["map_euc_share_at_forest_plots"] = float((mf[mf < 6] == 0).mean())
        out[name] = res
    return out


PLOT_LABEL = {
    "eucalipto": 0,
    "piñeiro sen eucalipto": 1,
    "frondosas sen eucalipto nin piñeiro": 2,
    "só mato": 3,
}


def inventory_training_experiment(seed: int = 0, weights=(5.0, 20.0)) -> dict:
    """Does adding inventory plots to the training labels improve the map?

    Plots are split by 10 km blocks into halves. Plots in the training half with no
    disturbance since 2001 (so their label probably still holds) add 3 x 3 pixels each to the
    usual training set (cleaned OSM labels + pseudo-labels), with extra weight. Every model is
    scored on the plots of the other half, the current map included.
    """
    from ..geo.grid import block_ids
    from .s2 import build_period
    from .species import _pixels_features, build_training, make_classifier

    plots = inventory_plots(gbif_records())
    aoi = np.load(INTERIM / "aoi.npz")["mask40"]
    plots = plots[aoi[plots["row"], plots["col"]] > 0].reset_index(drop=True)
    ly = np.load(INTERIM / "hansen.npz")["lossyear40"].astype(int)
    ef = np.load(INTERIM / "effis.npz")
    burnt = np.zeros(ly.shape, bool)
    for y in range(2018, 2024):
        burnt |= ef[f"burned40_{y}"]
    r, c = plots["row"].to_numpy(), plots["col"].to_numpy()
    dist = np.zeros(len(r), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = np.clip(r + dr, 0, ly.shape[0] - 1), np.clip(c + dc, 0, ly.shape[1] - 1)
            dist |= (ly[rr, cc] >= 1) | burnt[rr, cc]
    plots["disturbed"] = dist
    plots["y"] = plots["plot_type"].map(PLOT_LABEL)
    blk = block_ids(GRID_40M, 10)[r, c]
    rng = np.random.default_rng(seed)
    ub = np.unique(blk)
    train_blocks = set(rng.choice(ub, len(ub) // 2, replace=False).tolist())
    is_train = np.array([b in train_blocks for b in blk])
    te = plots[~is_train]
    tr = plots[is_train & ~plots["disturbed"].to_numpy()]

    def score(pred, d):
        e = d["y"].to_numpy() == 0
        m = pred == 0
        tp = int((m & e).sum())
        p, rc = tp / max(int(m.sum()), 1), tp / max(int(e.sum()), 1)
        return {"n": len(d), "recall": rc, "precision": p, "f1": 2 * p * rc / max(p + rc, 1e-9)}

    cube = build_period("2024")
    cur = np.load(INTERIM / "species.npz")["class40_2024"]
    out = {"n_train_plots": len(tr), "n_test_plots": len(te)}
    out["current_map"] = score(cur[te["row"], te["col"]], te)
    rows, cols, y, _ = build_training("2024", 30_000, seed)
    X = _pixels_features(cube, rows, cols)
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    ri = np.concatenate([tr["row"].to_numpy() + dr for dr, _ in offs])
    ci = np.concatenate([tr["col"].to_numpy() + dc for _, dc in offs])
    yi = np.tile(tr["y"].to_numpy(), len(offs))
    Xi = _pixels_features(cube, ri, ci)
    Xt = _pixels_features(cube, te["row"].to_numpy(), te["col"].to_numpy())
    variants = {"refit": (X, y, np.ones(len(y)))}
    for w in weights:
        variants[f"inventory_w{w:g}"] = (
            np.vstack([X, Xi]),
            np.concatenate([y, yi]),
            np.concatenate([np.ones(len(y)), np.full(len(yi), w)]),
        )
    ag = y >= 4
    variants["inventory_only"] = (
        np.vstack([X[ag], Xi]),
        np.concatenate([y[ag], yi]),
        np.concatenate([np.ones(int(ag.sum())), np.full(len(yi), weights[0])]),
    )
    for name, (XX, YY, W) in variants.items():
        m = make_classifier(seed)
        m.keep_ = np.isfinite(XX).any(axis=0)
        m.model.fit(XX[:, m.keep_], YY, sample_weight=W)
        out[name] = score(m.predict(Xt), te)
        log.info("inventory experiment %s: %s", name, out[name])
    return out


def landsat_inventory_check() -> dict:
    """The inventory plots against each Landsat epoch map and the Sentinel-2 maps.

    The plots date from around 1998, so the 2000 Landsat map should agree with them best; a
    steady fall in agreement after 2000 points to change since the survey rather than map error.
    """
    from .landsat import EPOCHS, landsat_maps

    plots = inventory_plots(gbif_records())
    lm = landsat_maps()
    sp = np.load(INTERIM / "species.npz")
    maps = {f"Landsat {e}": lm[f"class40_{e}"] for e in EPOCHS}
    maps["Sentinel-2 2017"] = sp["class40_2017"]
    maps["Sentinel-2 2024"] = sp["class40_2024"]
    out = {}
    for name, m in maps.items():
        s = _plot_scores(plots, m)
        forest = plots["plot_type"] != "só mato"
        v = m[plots["row"], plots["col"]][forest.to_numpy()]
        s["map_euc_share_at_forest_plots"] = float((v[v < 6] == 0).mean())
        out[name] = s
    out["plot_euc_share"] = float(
        (plots["plot_type"][plots["plot_type"] != "só mato"] == "eucalipto").mean()
    )
    return out
