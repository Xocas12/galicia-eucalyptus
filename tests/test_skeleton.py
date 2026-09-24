import importlib

MODULES = [
    "config",
    "data.catalog",
    "data.synthetic",
    "data.sources",
    "geo.grid",
    "geo.raster_ops",
    "features.spectral",
    "features.panel",
    "models.landcover",
    "models.conversion",
    "models.fire",
    "models.hydrology",
    "causal.dml",
    "causal.matching",
    "validation.spatial_cv",
    "scenarios",
    "reporting",
    "pipeline",
    "cli",
]


def test_all_modules_import():
    for m in MODULES:
        importlib.import_module(f"eucalyptus_impact.{m}")
