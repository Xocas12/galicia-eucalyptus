"""Real-data adapters against local fixtures (the live providers are not contacted)."""

import numpy as np
import pandas as pd
import pytest

from eucalyptus_impact.data import sources
from eucalyptus_impact.data.synthetic import AGRI, EUC, NATIVE, PINE, SHRUB
from eucalyptus_impact.geo.grid import Grid

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
shapely_geometry = pytest.importorskip("shapely.geometry")
box = shapely_geometry.box

GRID = Grid.from_bbox((500_000, 4_700_000, 510_000, 4_710_000), 1000)


def test_rasterize_fraction_half_cell():
    gdf = gpd.GeoDataFrame(geometry=[box(500_000, 4_709_000, 500_500, 4_710_000)], crs=GRID.crs)
    frac = sources.rasterize_fraction(gdf, GRID)
    assert frac.shape == GRID.shape
    assert np.isclose(frac[0, 0], 0.5)
    assert np.isclose(frac.sum(), 0.5)


def test_effis_to_burned_panel(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"FIREDATE": ["2017-10-15", "2018-08-01"]},
        geometry=[
            box(500_000, 4_700_000, 502_000, 4_702_000),
            box(505_000, 4_705_000, 506_000, 4_706_000),
        ],
        crs=GRID.crs,
    ).to_crs("EPSG:4326")
    path = tmp_path / "effis.gpkg"
    gdf.to_file(path)
    loaded = sources.load_effis_burnt_areas(path)
    assert sorted(loaded["year"].astype(int)) == [2017, 2018]
    panel = sources.burned_panel(loaded, GRID, [2016, 2017, 2018])
    assert panel.shape == (3, 10, 10)
    assert panel[0].sum() == 0
    assert np.isclose(panel[1].sum(), 4, atol=0.2)
    assert np.isclose(panel[2].sum(), 1, atol=0.1)


def test_mfe_labels(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "ESPECIE1": [61, 26, 41, None, 99, None],
            "TIPO_ESTR": ["Bosque", "Bosque", "Bosque", "Matorral", "Bosque", "Agricola"],
        },
        geometry=[
            box(500_000 + i * 100, 4_700_000, 500_050 + i * 100, 4_700_050) for i in range(6)
        ],
        crs=GRID.crs,
    )
    path = tmp_path / "mfe.gpkg"
    gdf.to_file(path)
    out = sources.load_mfe_labels(path)
    assert out["label"].tolist() == [EUC, PINE, NATIVE, SHRUB, NATIVE, AGRI]


def test_points_to_grid_counts():
    pyproj = pytest.importorskip("pyproj")
    inv = pyproj.Transformer.from_crs(GRID.crs, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform([500_500, 500_600, 509_500], [4_709_500, 4_709_400, 4_700_500])
    counts = sources.points_to_grid_counts(pd.DataFrame({"longitude": lon, "latitude": lat}), GRID)
    assert counts[0, 0] == 2 and counts[9, 9] == 1 and counts.sum() == 3


def test_gauge_annual_runoff(tmp_path):
    dates = pd.date_range("2019-10-01", "2021-09-30", freq="D")
    df = pd.DataFrame({"indroea": 1, "fecha": dates.strftime("%d/%m/%Y"), "caudal": 10.0})
    path = tmp_path / "aforos.csv"
    df.to_csv(path, index=False, sep=";")
    daily = sources.load_gauge_daily(path)
    ann = sources.gauge_annual_runoff(daily, {1: 100.0})
    # 10 m3/s over 100 km2 = 8.64 mm/day.
    assert ann["year"].tolist() == [2020, 2021]
    assert np.allclose(ann["runoff"], 8.64 * np.array([366, 365]))
    assert np.allclose(ann["low_flow"], 8.64)


def test_write_geotiff_roundtrip(tmp_path):
    import rasterio

    from eucalyptus_impact.geo.io import write_geotiff

    arr = np.arange(100, dtype=float).reshape(10, 10)
    arr[0, 0] = np.nan
    p = write_geotiff(arr, GRID, tmp_path / "a.tif")
    with rasterio.open(p) as src:
        back = src.read(1, masked=True)
        assert src.crs.to_epsg() == 25829
        assert src.transform.c == GRID.xmin and src.transform.f == GRID.ymax
    assert back.mask[0, 0] and back[9, 9] == 99
