"""Real static, forest-loss, fire, weather and human-pressure layers on the 1 km / 40 m grids."""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from .common import (
    GRID_1KM,
    GRID_40M,
    INTERIM,
    LONLAT_BBOX,
    RAW,
    block_to_1km,
    cached_npz,
    curl,
    download,
    log,
    reproject_to,
    transform_of,
)

W, S, E, N = LONLAT_BBOX


def _read_lonlat_window(url: str, decimate: int = 1, bounds=LONLAT_BBOX):
    """Read a lon/lat window of a remote raster; returns (array, transform, crs)."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(url) as src:
        win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        shape = (max(1, int(win.height) // decimate), max(1, int(win.width) // decimate))
        arr = src.read(1, window=win, out_shape=shape)
        tr = src.window_transform(win) * rasterio.Affine.scale(
            win.width / shape[1], win.height / shape[0]
        )
        return arr, tr, src.crs, src.nodata


# ---------------------------------------------------------------- terrain


@cached_npz("dem")
def dem_layers():
    """Copernicus DEM (90 m): elevation and slope on the 1 km grid, elevation on the 40 m grid."""
    import rasterio
    from rasterio.merge import merge

    urls = []
    for lat in (41, 42, 43):
        for lon in (10, 9, 8, 7):
            name = f"Copernicus_DSM_COG_30_N{lat:02d}_00_W{lon:03d}_00_DEM"
            urls.append(f"/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif")
    srcs = []
    for u in urls:
        try:
            srcs.append(rasterio.open(u))
        except rasterio.errors.RasterioIOError:
            log.info("no DEM tile %s (sea)", u.rsplit("/", 1)[-1])
    mosaic, tr = merge(srcs, bounds=LONLAT_BBOX)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    from ..geo.grid import Grid
    from .common import BBOX

    g90 = Grid.from_bbox(BBOX, 100)
    elev100 = reproject_to(mosaic[0].astype("float32"), tr, crs, g90, "bilinear")
    gy, gx = np.gradient(np.nan_to_num(elev100), 100.0)
    slope100 = np.degrees(np.arctan(np.hypot(gx, gy)))
    ny, nx = GRID_1KM.shape
    elev1 = np.nanmean(elev100.reshape(ny, 10, nx, 10), axis=(1, 3))
    slope1 = np.nanmean(slope100.reshape(ny, 10, nx, 10), axis=(1, 3))
    elev40 = reproject_to(mosaic[0].astype("float32"), tr, crs, GRID_40M, "bilinear")
    return {"elev": elev1, "slope": slope1, "elev40": elev40.astype("float32")}


# ---------------------------------------------------------------- land cover

WORLDCOVER_CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}


@cached_npz("worldcover")
def worldcover_layers():
    """ESA WorldCover 2021 (10 m, read at ~40 m): 40 m class map and 1 km class fractions."""
    tiles = ["N42W009", "N39W009", "N42W012", "N39W012"]
    parts = []
    for t in tiles:
        url = (
            "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
        )
        try:
            parts.append(_read_lonlat_window(url, decimate=4))
        except Exception as e:  # tile outside the window or missing
            log.info("worldcover %s skipped: %s", t, e)
    wc40 = np.zeros(GRID_40M.shape, dtype="uint8")
    for arr, tr, crs, _ in parts:
        if arr.size <= 1:
            continue
        r = reproject_to(arr, tr, crs, GRID_40M, "nearest", dtype="float32", src_nodata=0)
        ok = np.isfinite(r) & (r > 0)
        wc40[ok] = r[ok].astype("uint8")
    out = {"wc40": wc40}
    for code, name in WORLDCOVER_CLASSES.items():
        if name in ("snow", "mangrove", "moss"):
            continue
        out[f"wc_{name}"] = block_to_1km((wc40 == code).astype("float32")).astype("float32")
    return out


# ---------------------------------------------------------------- Hansen GFC

HANSEN = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/"


@cached_npz("hansen")
def hansen_layers():
    """Hansen GFC v1.12: tree cover 2000, annual loss 2001-2024, gain, on the 1 km grid.

    Also a 40 m loss-year map (nearest) for pixel-level loss attribution.
    """

    def read(layer):
        url = f"/vsicurl/{HANSEN}Hansen_GFC-2024-v1.12_{layer}_50N_010W.tif"
        return _read_lonlat_window(url)

    tc, tr, crs, _ = read("treecover2000")
    ly, _, _, _ = read("lossyear")
    gain, _, _, _ = read("gain")
    out = {
        "treecover2000": reproject_to(tc.astype("float32"), tr, crs, GRID_1KM) / 100.0,
        "gain": reproject_to(gain.astype("float32"), tr, crs, GRID_1KM),
    }
    forest = tc >= 30
    for k in range(1, 25):
        ind = ((ly == k) & forest).astype("float32")
        out[f"loss_{2000 + k}"] = reproject_to(ind, tr, crs, GRID_1KM)
    out["lossyear40"] = np.nan_to_num(
        reproject_to(np.where(forest, ly, 0).astype("float32"), tr, crs, GRID_40M, "mode")
    ).astype("uint8")
    return out


# ---------------------------------------------------------------- EFFIS severity

EFFIS = "https://effis-gwis-cms.s3.eu-west-1.amazonaws.com/effis/applications/data-and-services/"
EFFIS_YEARS = range(2018, 2024)


def _effis_year(year: int):
    path = download(f"{EFFIS}severity_{year}.tiff", RAW / f"effis_severity_{year}.tiff")
    # Read at ~25 m (2x decimation, nearest): still finer than the 40 m and 1 km grids, and a
    # quarter of the memory of the native ~12 m Europe-wide strips.
    arr, tr, crs, _ = _read_lonlat_window(str(path), decimate=2)
    burned = (arr > 0).astype("float32")
    sev = np.where(arr > 0, arr, np.nan).astype("float32")
    return {
        f"burned_{year}": reproject_to(burned, tr, crs, GRID_1KM),
        # NaN must be declared as nodata, or any partly burned cell averages to NaN.
        f"severity_{year}": reproject_to(sev, tr, crs, GRID_1KM, "average", src_nodata=np.nan),
        f"burned40_{year}": np.nan_to_num(reproject_to(burned, tr, crs, GRID_40M)) > 0.5,
        f"values_{year}": np.unique(arr),
    }


@cached_npz("effis")
def effis_layers():
    """EFFIS burn severity (Sentinel-2 based, ~12 m) 2018-2023: burned share and mean class."""
    out = {}
    for year in EFFIS_YEARS:  # sequential: each year decompresses Europe-wide row strips
        out.update(_effis_year(year))
    return out


# ---------------------------------------------------------------- Overture (OSM-derived)

OVERTURE = "overturemaps-us-west-2/release/2026-09-23.0"


def _overture_table(theme: str, typ: str, columns: list[str]):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    fs = pafs.S3FileSystem(
        anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY")
    )
    d = ds.dataset(f"{OVERTURE}/theme={theme}/type={typ}", filesystem=fs, format="parquet")
    f = (
        (pc.field("bbox", "xmin") > W)
        & (pc.field("bbox", "xmax") < E)
        & (pc.field("bbox", "ymin") > S)
        & (pc.field("bbox", "ymax") < N)
    )
    return d.to_table(filter=f, columns=columns)


@cached_npz("buildings")
def building_layers():
    """Building count per 1 km cell (Overture buildings; ignition-pressure proxy)."""
    from pyproj import Transformer

    t = _overture_table("buildings", "building", ["bbox"])
    bb = t.column("bbox").combine_chunks()
    x = (np.asarray(bb.field("xmin")) + np.asarray(bb.field("xmax"))) / 2
    y = (np.asarray(bb.field("ymin")) + np.asarray(bb.field("ymax"))) / 2
    ex, ny_ = Transformer.from_crs("EPSG:4326", GRID_1KM.crs, always_xy=True).transform(x, y)
    ny, nx = GRID_1KM.shape
    col = ((ex - GRID_1KM.xmin) // 1000).astype(int)
    row = ((GRID_1KM.ymax - ny_) // 1000).astype(int)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    counts = np.zeros((ny, nx), "float32")
    np.add.at(counts, (row[ok], col[ok]), 1)
    from ..geo.raster_ops import distance_to

    dense = counts >= 50
    return {"buildings": counts, "dist_settlement_km": distance_to(dense, 1000) / 1000.0}


# Label rules from OSM tags. In Galicia evergreen broadleaved forest is almost entirely eucalyptus
# (holm and cork oak are rare); deciduous broadleaved is native oak, chestnut and birch;
# needleleaved is pine. Scrub/heath polygons label shrub; meadows, farmland, orchards and
# vineyards label agriculture. Class codes match data.synthetic.
EUC, PINE, NATIVE, SHRUB, AGRI, OTHER = range(6)
AGRI_CLASSES = {"meadow", "farmland", "orchard", "vineyard", "farmyard", "allotments"}


def _label_from_tags(tags: dict) -> int | None:
    g = (tags.get("genus") or "").lower()
    sp = (tags.get("species") or "").lower()
    lt = (tags.get("leaf_type") or "").lower()
    lc = (tags.get("leaf_cycle") or "").lower()
    if "eucalyptus" in g or "eucalyptus" in sp:
        return EUC
    if g.startswith("pinus") or sp.startswith("pinus") or lt == "needleleaved":
        return PINE
    if any(k in g + sp for k in ("quercus", "castanea", "betula", "querqus", "fagus", "alnus")):
        return NATIVE
    if lt == "broadleaved" and lc == "evergreen":
        return EUC
    if lt == "broadleaved" and lc == "deciduous":
        return NATIVE
    return None


@cached_npz("osm_labels_v2")
def osm_forest_labels():
    """Cover labels on the 40 m grid from OSM polygons (Overture land and land-use layers).

    255 = no label. Polygons are shrunk by one pixel so edges do not leak into training.
    Forest labels are drawn last, so they win where polygons overlap.
    """
    import geopandas as gpd
    import shapely
    from rasterio.features import rasterize

    rows = []
    land = _overture_table("base", "land", ["geometry", "subtype", "class", "source_tags"])
    geoms = shapely.from_wkb(np.asarray(land.column("geometry").to_pylist(), dtype=object))
    for geom, sub, cls, tags in zip(
        geoms,
        land.column("subtype").to_pylist(),
        land.column("class").to_pylist(),
        land.column("source_tags").to_pylist(),
        strict=True,
    ):
        if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if sub in ("forest", "tree"):
            lab = _label_from_tags(dict(tags or []))
        elif sub == "shrub" and cls in ("scrub", "heath"):
            lab = SHRUB
        else:
            lab = None
        if lab is not None:
            rows.append((geom, lab))
    lu = _overture_table("base", "land_use", ["geometry", "subtype", "class"])
    geoms = shapely.from_wkb(np.asarray(lu.column("geometry").to_pylist(), dtype=object))
    for geom, cls in zip(geoms, lu.column("class").to_pylist(), strict=True):
        if (
            cls in AGRI_CLASSES
            and geom is not None
            and geom.geom_type in ("Polygon", "MultiPolygon")
        ):
            rows.append((geom, AGRI))
    gdf = gpd.GeoDataFrame(
        {"label": [r[1] for r in rows]}, geometry=[r[0] for r in rows], crs="EPSG:4326"
    ).to_crs(GRID_40M.crs)
    gdf["geometry"] = gdf.buffer(-40)
    gdf = gdf[~gdf.is_empty]
    gdf["order"] = gdf["label"].map({AGRI: 0, SHRUB: 1}).fillna(2)
    gdf = gdf.sort_values("order")
    lab40 = rasterize(
        ((g, v) for g, v in zip(gdf.geometry, gdf["label"], strict=True)),
        out_shape=GRID_40M.shape,
        transform=transform_of(GRID_40M),
        fill=255,
        dtype="uint8",
    )
    counts = np.array([(gdf["label"] == k).sum() for k in range(5)])
    return {"label40": lab40, "polygons_per_class": counts}


# ---------------------------------------------------------------- weather stations

GHCN_STATIONS = {
    "SPE00119711": (43.3669, -8.4192),  # A Coruna
    "SPE00119729": (42.8878, -8.4106),  # Santiago de Compostela
    "SPE00120260": (43.1153, -7.4558),  # Lugo
    "SPE00120368": (42.3278, -7.8603),  # Ourense
    "SPE00120395": (42.4400, -8.6164),  # Pontevedra
    "SPE00120413": (42.2392, -8.6239),  # Vigo
}


def ghcn_station_years() -> pd.DataFrame:
    """Fire-season weather per station and year: mean Jul-Sep Tmax and Jun-Sep precipitation."""
    rows = []
    for sid in GHCN_STATIONS:
        txt = curl(f"https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{sid}.csv", 300)
        df = pd.read_csv(io.StringIO(txt), dtype={"ID": str}, low_memory=False)
        df.columns = [c.upper() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")
        df = df[df["ELEMENT"].isin(["TMAX", "PRCP"]) & (df["DATE"].dt.year >= 2000)]
        df["v"] = df["DATA_VALUE"] / 10.0
        df["year"] = df["DATE"].dt.year
        df["month"] = df["DATE"].dt.month
        tmax = df[(df["ELEMENT"] == "TMAX") & df["month"].isin([7, 8, 9])].groupby("year")["v"]
        prcp = df[(df["ELEMENT"] == "PRCP") & df["month"].isin([6, 7, 8, 9])].groupby("year")["v"]
        t = pd.DataFrame(
            {
                "tmax_summer": tmax.mean(),
                "n_t": tmax.size(),
                "prcp_summer": prcp.sum(),
                "n_p": prcp.size(),
            }
        )
        t = t[(t["n_t"] > 60) & (t["n_p"] > 90)].drop(columns=["n_t", "n_p"])
        t["station"] = sid
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


@cached_npz("weather")
def weather_layers(years=range(2001, 2025)):
    """Cell-year fire-weather index from station anomalies, inverse-distance weighted.

    index = z(summer Tmax anomaly) - z(summer precipitation anomaly), each anomaly relative to
    the station's own mean, so it measures the year, not the site. Site climate is handled by
    terrain and location covariates.
    """
    from pyproj import Transformer

    sy = ghcn_station_years()
    sy.to_csv(INTERIM / "ghcn_station_years.csv", index=False)
    for c in ("tmax_summer", "prcp_summer"):
        sy[c + "_anom"] = sy[c] - sy.groupby("station")[c].transform("mean")
    tr = Transformer.from_crs("EPSG:4326", GRID_1KM.crs, always_xy=True)
    st_xy = {s: tr.transform(lon, lat) for s, (lat, lon) in GHCN_STATIONS.items()}
    gx, gy = GRID_1KM.centers()
    out = {}
    zt = sy["tmax_summer_anom"].std()
    zp = sy["prcp_summer_anom"].std()
    for yr in years:
        sub = sy[sy["year"] == yr]
        if sub.empty:
            out[f"fwi_{yr}"] = np.zeros(GRID_1KM.shape, "float32")
            continue
        num = np.zeros(GRID_1KM.shape)
        den = np.zeros(GRID_1KM.shape)
        for _, r in sub.iterrows():
            sx, syy = st_xy[r["station"]]
            w = 1.0 / (np.hypot(gx - sx, gy - syy) / 1000 + 10) ** 2
            val = r["tmax_summer_anom"] / zt - r["prcp_summer_anom"] / zp
            num += w * val
            den += w
        out[f"fwi_{yr}"] = (num / den).astype("float32")
    return out


def all_layers():
    """Build (or load from cache) every non-imagery layer."""
    from .common import aoi_masks

    return {
        "aoi": aoi_masks(),
        "dem": dem_layers(),
        "worldcover": worldcover_layers(),
        "hansen": hansen_layers(),
        "effis": effis_layers(),
        "buildings": building_layers(),
        "osm": osm_forest_labels(),
        "weather": weather_layers(),
    }
