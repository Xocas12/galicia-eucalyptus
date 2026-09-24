"""Real-data ingestion adapters (Sentinel-2 STAC, FIRMS, EFFIS, MFE, gauging stations).

These map real sources onto the same analysis grid and panel layout the synthetic pipeline uses
(`features/panel.py`), so the models run unchanged on real data. They need the optional `geo`
dependencies (`pip install -e '.[geo]'`) and network access to the providers.

NOT YET EXERCISED. The providers were unreachable from the environment this was written in, so
these functions are untested against live endpoints. Treat them as the M2 starting point, and
check field names, endpoints and licences before relying on them.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..geo.grid import CRS, Grid
from .synthetic import AGRI, EUC, NATIVE, OTHER, PINE, SHRUB

GALICIA_BBOX_LONLAT = (-9.35, 41.80, -6.70, 43.80)
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


def _require(module: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"{module} is needed for real-data ingestion: pip install -e '.[geo]'"
        ) from e


# --------------------------------------------------------------------------------------------
# Sentinel-2
# --------------------------------------------------------------------------------------------


def search_sentinel2(
    bbox_lonlat=GALICIA_BBOX_LONLAT,
    start="2023-01-01",
    end="2023-12-31",
    max_cloud: float = 60,
    stac_url: str = EARTH_SEARCH,
):
    """STAC search for Sentinel-2 L2A scenes over the bbox. Returns a list of pystac Items."""
    pystac_client = _require("pystac_client")
    cat = pystac_client.Client.open(stac_url)
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_lonlat,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def sentinel2_monthly_indices(items, grid: Grid, resolution: float = 20.0):
    """Monthly median NDVI/NDMI/NBR cube (time, y, x, index) on the grid's bbox and CRS.

    Cloud and shadow pixels are masked with the scene classification layer (SCL classes 3, 8,
    9, 10). The cube feeds `features.spectral.harmonic_features` after reshaping to pixels.
    """
    stackstac = _require("stackstac")
    epsg = int(CRS.split(":")[1])
    da = stackstac.stack(
        items,
        assets=["red", "nir", "swir16", "swir22", "scl"],
        epsg=epsg,
        resolution=resolution,
        bounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        chunksize=2048,
    )
    scl = da.sel(band="scl")
    clear = ~scl.isin([0, 1, 3, 8, 9, 10])
    refl = da.sel(band=["red", "nir", "swir16", "swir22"]).where(clear)
    red, nir, sw1, sw2 = (refl.sel(band=b) for b in ["red", "nir", "swir16", "swir22"])
    idx = [(nir - red) / (nir + red), (nir - sw1) / (nir + sw1), (nir - sw2) / (nir + sw2)]
    xr = _require("xarray")
    cube = xr.concat(idx, dim="index").assign_coords(index=["ndvi", "ndmi", "nbr"])
    return cube.resample(time="1MS").median().transpose("time", "y", "x", "index")


# --------------------------------------------------------------------------------------------
# Vector layers -> grid
# --------------------------------------------------------------------------------------------


def rasterize_fraction(gdf, grid: Grid, oversample: int = 10) -> np.ndarray:
    """Fraction of each grid cell covered by the polygons in gdf (supersampled rasterisation)."""
    features = _require("rasterio.features")
    transform_mod = _require("rasterio.transform")
    gdf = gdf.to_crs(grid.crs)
    ny, nx = grid.shape
    fine = grid.resolution_m / oversample
    tr = transform_mod.from_origin(grid.xmin, grid.ymax, fine, fine)
    burned = features.rasterize(
        ((g, 1) for g in gdf.geometry if g is not None and not g.is_empty),
        out_shape=(ny * oversample, nx * oversample),
        transform=tr,
        fill=0,
        dtype="uint8",
    )
    return burned.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))


# MFE species codes are numeric (IFN "especie" codes). The ones below are the main species in
# Galicia: Eucalyptus globulus 61, E. nitens 62 (check against the MFE legend in use), Pinus pinaster
# 26, P. radiata 28, P. sylvestris 21, Quercus robur 41, Q. pyrenaica 43, Castanea sativa 72,
# Betula spp. 73. Unlisted forest species go to native broadleaf.
MFE_SPECIES_TO_CLASS = {
    61: EUC,
    62: EUC,
    64: EUC,
    26: PINE,
    28: PINE,
    21: PINE,
    41: NATIVE,
    43: NATIVE,
    72: NATIVE,
    73: NATIVE,
}


def load_mfe_labels(path: str | Path, species_col: str = "ESPECIE1", tree_col: str = "TIPO_ESTR"):
    """Read an MFE shapefile/GeoPackage and attach a class label per polygon."""
    gpd = _require("geopandas")
    gdf = gpd.read_file(path)
    sp = pd.to_numeric(gdf.get(species_col), errors="coerce")
    label = sp.map(MFE_SPECIES_TO_CLASS)
    label = label.fillna(NATIVE).where(sp.notna(), SHRUB)
    if tree_col in gdf:
        nonforest = gdf[tree_col].astype(str).str.contains("Agr|Cultiv", case=False, na=False)
        label[nonforest] = AGRI
        urban = gdf[tree_col].astype(str).str.contains("Artif|Urban", case=False, na=False)
        label[urban] = OTHER
    gdf["label"] = label.astype(int)
    return gdf


def load_effis_burnt_areas(path: str | Path, bbox_lonlat=GALICIA_BBOX_LONLAT):
    """EFFIS burnt-area perimeters clipped to Galicia with a `year` column."""
    gpd = _require("geopandas")
    gdf = gpd.read_file(path, bbox=bbox_lonlat)
    date_col = next(c for c in ("FIREDATE", "firedate", "initialdate") if c in gdf)
    gdf["year"] = pd.to_datetime(gdf[date_col], errors="coerce").dt.year
    return gdf.dropna(subset=["year"])


def burned_panel(effis_gdf, grid: Grid, years) -> np.ndarray:
    """(T, ny, nx) burned fraction per cell and year from EFFIS perimeters."""
    ny, nx = grid.shape
    out = np.zeros((len(years), ny, nx))
    for t, yr in enumerate(years):
        sub = effis_gdf[effis_gdf["year"] == yr]
        if len(sub):
            out[t] = rasterize_fraction(sub, grid)
    return out


# --------------------------------------------------------------------------------------------
# FIRMS active fire
# --------------------------------------------------------------------------------------------


def fetch_firms(
    start: str,
    days: int = 10,
    source: str = "VIIRS_SNPP_SP",
    bbox_lonlat=GALICIA_BBOX_LONLAT,
    map_key: str | None = None,
) -> pd.DataFrame:
    """Active-fire detections from the FIRMS area API (up to 10 days per call).

    Needs a free MAP_KEY (env FIRMS_MAP_KEY). Loop over date windows for longer periods.
    """
    requests = _require("requests")
    key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("set FIRMS_MAP_KEY (free key from firms.modaps.eosdis.nasa.gov)")
    area = ",".join(str(v) for v in bbox_lonlat)
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{start}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def points_to_grid_counts(df: pd.DataFrame, grid: Grid, lon="longitude", lat="latitude"):
    """Count points per grid cell (reprojects lon/lat to the grid CRS)."""
    pyproj = _require("pyproj")
    tr = pyproj.Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tr.transform(df[lon].to_numpy(), df[lat].to_numpy())
    ny, nx = grid.shape
    col = ((x - grid.xmin) // grid.resolution_m).astype(int)
    row = ((grid.ymax - y) // grid.resolution_m).astype(int)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    out = np.zeros((ny, nx))
    np.add.at(out, (row[ok], col[ok]), 1)
    return out


# --------------------------------------------------------------------------------------------
# Gauging stations
# --------------------------------------------------------------------------------------------


def load_gauge_daily(path: str | Path, date_col="fecha", flow_col="caudal", station_col="indroea"):
    """Daily flow table (e.g. the CEDEX Anuario de Aforos export) with parsed dates."""
    df = pd.read_csv(path, sep=None, engine="python")
    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    return df.rename(columns={flow_col: "q_m3s", station_col: "station"})[
        ["station", "date", "q_m3s"]
    ]


def gauge_annual_runoff(daily: pd.DataFrame, basin_area_km2: dict) -> pd.DataFrame:
    """Hydrological-year (Oct-Sep) runoff in mm and summer (Jul-Sep) 10th-percentile low flow.

    Converts m3/s to mm over the upstream basin area, which must come from a basin delineation
    on the DEM for each station.
    """
    d = daily.dropna(subset=["date"]).copy()
    d["hyear"] = d["date"].dt.year + (d["date"].dt.month >= 10)
    d["mm_day"] = d["q_m3s"] * 86400 / (d["station"].map(basin_area_km2) * 1e6) * 1000
    ann = d.groupby(["station", "hyear"]).agg(runoff=("mm_day", "sum"), n=("mm_day", "size"))
    summer = d[d["date"].dt.month.isin([7, 8, 9])]
    low = summer.groupby(["station", "hyear"])["mm_day"].quantile(0.1).rename("low_flow")
    out = ann.join(low).reset_index()
    return (
        out[out["n"] >= 330]
        .drop(columns="n")
        .rename(columns={"station": "catchment", "hyear": "year"})
    )
