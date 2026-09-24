"""Catalogue of real data sources for the Galicia study (docs/SCOPE.md section 3).

URLs point at the provider's landing or API page. Check licence terms and current endpoints
before bulk download; several providers require a free account or API key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSource:
    key: str
    name: str
    provider: str
    url: str
    resolution: str
    period: str
    role: str
    workstream: str
    access: str


CATALOG: list[DataSource] = [
    DataSource(
        "s2",
        "Sentinel-2 L2A",
        "ESA Copernicus (via Earth Search STAC)",
        "https://earth-search.aws.element84.com/v1",
        "10-20 m, 5-day",
        "2017-",
        "Species mapping, NBR/dNBR, NDMI",
        "forest, fire, water",
        "STAC, open",
    ),
    DataSource(
        "landsat",
        "Landsat Collection 2 L2",
        "USGS",
        "https://landsatlook.usgs.gov/stac-server",
        "30 m, 16-day",
        "1985-",
        "Back-cast plantation expansion",
        "forest",
        "STAC, open",
    ),
    DataSource(
        "s1",
        "Sentinel-1 GRD",
        "ESA Copernicus",
        "https://dataspace.copernicus.eu",
        "10 m",
        "2015-",
        "Cloud-free structure, soil-moisture proxy",
        "forest, water",
        "open, account",
    ),
    DataSource(
        "mfe",
        "Mapa Forestal de Espana (MFE25/MFE50)",
        "MITECO",
        "https://www.miteco.gob.es/es/biodiversidad/servicios/banco-datos-naturaleza/informacion-disponible/mfe.html",
        "1:25k-1:50k polygons",
        "multi-epoch",
        "Species training labels",
        "forest",
        "open download",
    ),
    DataSource(
        "ifn",
        "Inventario Forestal Nacional (IFN3/IFN4)",
        "MITECO",
        "https://www.miteco.gob.es/es/biodiversidad/servicios/banco-datos-naturaleza/informacion-disponible/ifn.html",
        "field plots",
        "IFN3 ~2000, IFN4 ~2010s",
        "Independent accuracy reference",
        "forest",
        "open download",
    ),
    DataSource(
        "clc",
        "CORINE Land Cover",
        "Copernicus Land Monitoring Service",
        "https://land.copernicus.eu/en/products/corine-land-cover",
        "100 m",
        "1990-2018",
        "Covariates, cross-check",
        "forest",
        "open",
    ),
    DataSource(
        "worldcover",
        "ESA WorldCover",
        "ESA",
        "https://esa-worldcover.org",
        "10 m",
        "2020, 2021",
        "Tree cover mask",
        "forest",
        "open",
    ),
    DataSource(
        "gfc",
        "Global Forest Change",
        "Hansen / UMD",
        "https://glad.earthengine.app/view/global-forest-change",
        "30 m",
        "2001-",
        "Tree-cover loss events to attribute",
        "forest",
        "open",
    ),
    DataSource(
        "effis",
        "EFFIS burnt areas",
        "JRC",
        "https://forest-fire.emergency.copernicus.eu",
        "perimeters",
        "2000-",
        "Fire outcome",
        "fire",
        "open",
    ),
    DataSource(
        "mcd64",
        "MODIS MCD64A1 burned area",
        "NASA LP DAAC",
        "https://lpdaac.usgs.gov/products/mcd64a1v061/",
        "500 m, monthly",
        "2000-",
        "Fire outcome (cross-check)",
        "fire",
        "open, Earthdata login",
    ),
    DataSource(
        "firms",
        "FIRMS active fire (MODIS/VIIRS)",
        "NASA",
        "https://firms.modaps.eosdis.nasa.gov/api/",
        "375 m-1 km, daily",
        "2000-",
        "Fire timing and ignition density",
        "fire",
        "API key (free)",
    ),
    DataSource(
        "fwi",
        "Fire Weather Index (CEMS)",
        "Copernicus CEMS / ECMWF",
        "https://cds.climate.copernicus.eu",
        "~0.25 deg, daily",
        "1940-",
        "Weather confounder",
        "fire",
        "CDS account",
    ),
    DataSource(
        "era5land",
        "ERA5-Land",
        "ECMWF",
        "https://cds.climate.copernicus.eu",
        "0.1 deg, hourly",
        "1950-",
        "Precipitation, PET, temperature",
        "fire, water",
        "CDS account",
    ),
    DataSource(
        "dem",
        "Copernicus DEM GLO-30",
        "ESA",
        "https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model",
        "30 m",
        "static",
        "Elevation, slope, TWI",
        "all",
        "open",
    ),
    DataSource(
        "gauges",
        "Anuario de Aforos (gauging stations)",
        "CEDEX / Augas de Galicia",
        "https://ceh.cedex.es/anuarioaforos/default.asp",
        "daily flow",
        "decades",
        "Runoff and low-flow outcomes",
        "water",
        "open download",
    ),
    DataSource(
        "mod16",
        "MODIS MOD16A2 evapotranspiration",
        "NASA LP DAAC",
        "https://lpdaac.usgs.gov/products/mod16a2v061/",
        "500 m, 8-day",
        "2001-",
        "ET mechanism check",
        "water",
        "open, Earthdata login",
    ),
    DataSource(
        "esacci_sm",
        "ESA CCI Soil Moisture",
        "ESA",
        "https://climate.esa.int/en/projects/soil-moisture/",
        "0.25 deg, daily",
        "1978-",
        "Soil-moisture outcome",
        "water",
        "open",
    ),
    DataSource(
        "pop",
        "Population grid",
        "INE",
        "https://www.ine.es",
        "1 km",
        "census years",
        "Ignition pressure confounder",
        "fire",
        "open",
    ),
    DataSource(
        "osm",
        "OpenStreetMap roads",
        "OSM contributors (Geofabrik extracts)",
        "https://download.geofabrik.de/europe/spain/galicia.html",
        "vector",
        "current",
        "Access / ignition confounder",
        "fire, forest",
        "ODbL",
    ),
    DataSource(
        "catastro",
        "Cadastral parcels",
        "Direccion General del Catastro",
        "https://www.catastro.hacienda.gob.es",
        "parcel polygons",
        "current",
        "Parcel size / ownership confounder",
        "forest",
        "INSPIRE WFS",
    ),
]


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in CATALOG])


def get(key: str) -> DataSource:
    for s in CATALOG:
        if s.key == key:
            return s
    raise KeyError(key)
