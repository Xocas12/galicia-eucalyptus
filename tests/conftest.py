import pytest

from eucalyptus_impact.config import config_from_dict

TINY = {
    "seed": 3,
    "grid": {"resolution_m": 5000, "block_size_km": 40},
    "n_catchments": 15,
    "causal": {"n_folds": 3, "max_rows": 40000},
    "landcover": {"n_train_pixels": 1500, "n_map_pixels": 5000, "reference_per_class": 60},
    "output_dir": "outputs/test",
}


@pytest.fixture(scope="session")
def tiny_cfg():
    return config_from_dict(TINY)


@pytest.fixture(scope="session")
def tiny_land(tiny_cfg):
    from eucalyptus_impact.data.synthetic import simulate_landscape

    return simulate_landscape(tiny_cfg)
