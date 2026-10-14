"""The IFN3 aggregate eucalyptus shares: plot list and map, taken over the same plots.

`plot_euc_share` used to divide by every forested plot while the map share dropped the plots
whose pixel has no mapped class (6 and above), so the two numbers described different samples
(27.5% over 5,058 plots against 30.4% over about 4,604 in the 2024 brief). These tests build a
tiny grid and plot table in which the two samples differ, and check that both shares now come
from the same plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eucalyptus_impact.real import reference

# (plot type, mapped class) for ten plots. Class 6 = no mapped class.
PLOTS = [
    ("eucalipto", 0),
    ("eucalipto", 0),
    ("eucalipto", 1),
    ("eucalipto", 6),  # forested, unmapped: must leave both shares
    ("piñeiro sen eucalipto", 0),
    ("piñeiro sen eucalipto", 1),
    ("piñeiro sen eucalipto", 6),  # forested, unmapped
    ("frondosas sen eucalipto nin piñeiro", 2),
    ("frondosas sen eucalipto nin piñeiro", 2),
    ("só mato", 0),  # not forest: in neither share
]
# Forested plots with a mapped class: 7, of which 3 list eucalyptus and 3 are mapped eucalyptus.
N_SAME = 7
PLOT_SHARE = 3 / 7
MAP_SHARE = 3 / 7


@pytest.fixture
def fake_inventory(tmp_path, monkeypatch):
    shape = (12, 12)
    rows = np.arange(len(PLOTS)) + 1
    cols = np.arange(len(PLOTS)) + 1
    cls = np.full(shape, 6, np.uint8)
    cls[rows, cols] = [m for _, m in PLOTS]
    np.savez(
        tmp_path / "species.npz",
        class40_2024=cls,
        class40_2017=cls,
        class40_2017_independent=cls,
    )
    np.savez(tmp_path / "osm_labels_v2.npz", label40=np.full(shape, 255, np.uint8))
    np.savez(tmp_path / "hansen.npz", lossyear40=np.zeros(shape, np.uint8))
    np.savez(
        tmp_path / "effis.npz",
        **{f"burned40_{y}": np.zeros(shape, bool) for y in range(2018, 2024)},
    )
    plots = pd.DataFrame({"row": rows, "col": cols, "plot_type": [t for t, _ in PLOTS]})
    raw = pd.DataFrame({"datasetkey": [reference.INVENTORY_KEY] * len(PLOTS)})
    monkeypatch.setattr(reference, "INTERIM", tmp_path)
    monkeypatch.setattr(reference, "gbif_records", lambda *a, **k: raw)
    monkeypatch.setattr(reference, "inventory_plots", lambda _raw: plots.copy())
    return cls


def test_forest_euc_shares_use_the_same_plots():
    shares = reference.forest_euc_shares([t for t, _ in PLOTS], np.array([m for _, m in PLOTS]))
    assert shares["n_forest_plots_compared"] == N_SAME
    assert shares["plot_euc_share"] == pytest.approx(PLOT_SHARE)
    assert shares["map_euc_share_at_forest_plots"] == pytest.approx(MAP_SHARE)


def test_inventory_check_shares_share_a_denominator(fake_inventory):
    out = reference.inventory_check()
    for period in ("2024", "2017", "2017_independent"):
        res = out[period]
        # All forested plots would give 4 / 9; only the mapped ones are compared.
        assert res["plot_euc_share"] == pytest.approx(PLOT_SHARE)
        assert res["map_euc_share_at_forest_plots"] == pytest.approx(MAP_SHARE)
        assert res["n_forest_plots_compared"] == N_SAME


def test_landsat_inventory_check_shares_share_a_denominator(fake_inventory, monkeypatch):
    from eucalyptus_impact.real import landsat

    monkeypatch.setattr(
        landsat, "backcast_maps", lambda: {f"class40_{e}": fake_inventory for e in landsat.EPOCHS}
    )
    out = reference.landsat_inventory_check()
    for name, res in out.items():
        assert res["plot_euc_share"] == pytest.approx(PLOT_SHARE), name
        assert res["map_euc_share_at_forest_plots"] == pytest.approx(MAP_SHARE), name
        assert res["n_forest_plots_compared"] == N_SAME, name


# The brief's wording: the near-match may be reported, but not as validating the area.
INV_2024 = {
    "n_forest_plots_compared": 4604,
    "plot_euc_share": 0.291,
    "map_euc_share_at_forest_plots": 0.304,
    "fora_do_norte": {"f1": 0.45},
}
VALIDATING = ("cadra", "achégase", "plausible")


def test_brief_summary_does_not_call_the_ifn3_match_a_validation():
    from eucalyptus_impact.real.brief import _ref_resumo

    txt = _ref_resumo(None, {"n_plots": 6903, "2024": INV_2024})
    assert not any(w in txt for w in VALIDATING), txt
    assert "Non é unha validación da superficie" in txt
    assert "30,4 %" in txt and "29,1 %" in txt


def test_brief_aggregate_bullet_names_the_shared_plots_and_the_caveat():
    from eucalyptus_impact.real.brief import _aggregate_txt

    txt = _aggregate_txt(INV_2024)
    assert not any(w in txt for w in VALIDATING), txt
    assert "4 604 parcelas arboradas con clase no mapa" in txt
    assert "non valida a superficie" in txt
    # an older cached result without the count still renders
    old = {k: v for k, v in INV_2024.items() if k != "n_forest_plots_compared"}
    assert "parcelas arboradas" in _aggregate_txt(old)
