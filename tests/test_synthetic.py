import numpy as np

from eucalyptus_impact.data.synthetic import EUC, simulate_landscape, simulate_spectral_pixels
from eucalyptus_impact.features.panel import build_catchment_panel, build_cell_panel
from eucalyptus_impact.models.conversion import attribute_loss_events


def test_cover_is_a_partition(tiny_land):
    assert np.allclose(tiny_land.cover.sum(-1), 1, atol=1e-9)
    assert np.allclose(tiny_land.cover_obs.sum(-1), 1, atol=1e-9)
    assert np.allclose(tiny_land.joint.sum(axis=1), tiny_land.cover[-1], atol=1e-9)
    assert (tiny_land.cover >= -1e-12).all()


def test_eucalyptus_expands_and_fire_happens(tiny_land):
    assert tiny_land.cover[-1, :, EUC].mean() > tiny_land.cover[0, :, EUC].mean()
    assert 0.005 < tiny_land.burned.mean() < 0.1


def test_reproducible(tiny_cfg, tiny_land):
    again = simulate_landscape(tiny_cfg)
    assert np.array_equal(again.burned, tiny_land.burned)


def test_confounding_is_present(tiny_land):
    """Plantations sit in low fire-weather places: the design problem the causal models solve."""
    r = np.corrcoef(tiny_land.cover[0, :, EUC], tiny_land.static["fwi_base"])[0, 1]
    assert r < -0.15


def test_spectral_pixels_shape_and_clouds():
    rng = np.random.default_rng(0)
    s, m = simulate_spectral_pixels(np.array([0, 1, 2, 3, 4, 5] * 20), np.zeros(120), rng)
    assert s.shape == (120, 36, 3) and len(m) == 36
    assert 0.2 < np.isnan(s[:, :, 0]).mean() < 0.8


def test_panels(tiny_land):
    p = build_cell_panel(tiny_land)
    assert len(p) == tiny_land.n * (len(tiny_land.years) - 1)
    assert p["burned"].isin([0, 1]).all()
    c = build_catchment_panel(tiny_land)
    assert len(c) == tiny_land.n_catchments * len(tiny_land.years)
    assert (c["runoff_ratio"].between(0, 1)).all()
    att = attribute_loss_events(p)
    assert np.isclose(att["attributed"].sum(), p["tree_loss"].sum())
