"""The fire exposure must not be built from the fire outcome.

The backdated 2017 map keeps the 2024 class only where a pixel had no EFFIS burn in 2018-2023
and no Hansen loss in 2017-2024, so it picks each pixel's classifier by whether it later
burned. The fire estimates use the independently classified 2017 map instead, and the brief
labels the backdated and 2024 versions as post-outcome sensitivity checks.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eucalyptus_impact.dynamic import COVER
from eucalyptus_impact.geo.grid import Grid
from eucalyptus_impact.real import analysis as A
from eucalyptus_impact.real import common as C
from eucalyptus_impact.real import species as S

NY, NX, F = 4, 6, 5  # 1 km cells, and 40 m pixels per cell edge
YEARS = range(2018, 2024)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_grid(monkeypatch):
    """Shrink the Galicia grid to NY x NX cells of F x F pixels for the 1 km aggregation."""
    g1 = Grid.from_bbox((0.0, 0.0, NX * 1000.0, NY * 1000.0), 1000)
    for mod in (A, S, C):
        monkeypatch.setattr(mod, "GRID_1KM", g1)
    for mod in (S, C):
        monkeypatch.setattr(mod, "FINE_PER_CELL", F)


def _layers(rng) -> dict:
    s1, s40 = (NY, NX), (NY * F, NX * F)

    def u(scale=1.0):
        return (rng.uniform(0, 1, s1) * scale).astype("float32")

    aoi, water = np.ones(s1, "float32"), np.zeros(s1, "float32")
    aoi[0, 0], water[0, 0] = 0, 1  # one sea cell
    L = {
        "aoi": {"mask40": np.ones(s40, "uint8"), "frac1km": aoi},
        "worldcover": {"wc_water": water},
        "dem": {"elev": u(500), "slope": u(20)},
        "buildings": {"buildings": u(50), "dist_settlement_km": u(5)},
        "hansen": {
            "treecover2000": u(100),
            "lossyear40": rng.integers(0, 25, s40).astype("uint8"),
            **{f"loss_{y}": u(0.01) for y in range(2001, 2017)},
        },
        "effis": {},
        "weather": {f"fwi_{y}": rng.normal(0, 1, s1).astype("float32") for y in YEARS},
    }
    for y in YEARS:
        L["effis"][f"burned_{y}"] = u(0.05)
        L["effis"][f"severity_{y}"] = rng.integers(1, 5, s1).astype("float32")
        L["effis"][f"burned40_{y}"] = rng.uniform(size=s40) < 0.15
    return L


def _predictions(rng):
    """2024 and normalised-2017 classifier output (class, confidence %), disagreeing often."""
    s40 = (NY * F, NX * F)
    cls24 = rng.integers(0, 6, s40).astype("uint8")
    cls17r = np.where(rng.uniform(size=s40) < 0.5, cls24, rng.integers(0, 6, s40)).astype("uint8")
    pmax24 = rng.integers(40, 100, s40).astype("uint8")
    pmax17r = rng.integers(40, 100, s40).astype("uint8")
    return cls24, pmax24, cls17r, pmax17r


def _other_post_2017_records(L: dict, rng) -> dict:
    """Same layers, but different EFFIS burns 2018-2023 and Hansen loss 2017-2024 at 40 m."""
    L2 = copy.deepcopy(L)
    for y in YEARS:
        L2["effis"][f"burned40_{y}"] = rng.uniform(size=L["aoi"]["mask40"].shape) < 0.15
    ly = L["hansen"]["lossyear40"]
    ly2 = np.where(ly >= 17, 0, ly)
    ly2[(ly < 17) & (rng.uniform(size=ly.shape) < 0.2)] = 19
    L2["hansen"]["lossyear40"] = ly2.astype("uint8")
    return L2


def _cells_and_panel(L: dict, preds):
    maps, _ = S.assemble_maps(L, *preds)
    cells = A.cell_frame(L, maps)
    return cells, A.cell_year_panel(cells)


def test_fire_exposure_does_not_depend_on_post_2017_burns_or_loss(tiny_grid):
    rng = np.random.default_rng(0)
    L, preds = _layers(rng), _predictions(rng)
    L2 = _other_post_2017_records(L, rng)
    cells_a, panel_a = _cells_and_panel(L, preds)
    cells_b, panel_b = _cells_and_panel(L2, preds)
    assert len(cells_a) == len(cells_b) == NY * NX - 1

    # The records do move the backdated 2017 fractions, so the check below can fail...
    assert not np.allclose(cells_a[COVER].to_numpy(), cells_b[COVER].to_numpy())
    # ...but the fire panel's exposure is the same whatever burned or was cleared after 2017.
    pd.testing.assert_frame_equal(panel_a[COVER], panel_b[COVER])

    # And it is the independently classified 2017 map, aggregated to 1 km.
    ind = S.class_fractions(preds[2], L["aoi"]["mask40"].astype(bool))
    expected = ind[:, cells_a["row"], cells_a["col"]].T
    for yr in YEARS:
        got = panel_a.loc[panel_a["year"] == yr, COVER].to_numpy()
        assert np.allclose(got, expected)

    # The map-version table leads with that map; the other two are the post-outcome checks.
    assert next(iter(A.FIRE_MAPS.items())) == (A.MAIN_FIRE_MAP, "_ind")
    backdated = A.cell_year_panel(cells_a, A.FIRE_MAPS["2017 backdated (post-outcome)"])
    assert np.allclose(backdated.loc[backdated["year"] == 2018, COVER], cells_a[COVER])


def test_brief_labels_backdated_and_2024_maps_as_post_outcome_checks():
    from eucalyptus_impact.real import brief as B

    # The map-version table of the committed brief: independent, backdated, 2024.
    ms = pd.DataFrame(
        {
            "map": list(A.FIRE_MAPS),
            "estimate": [-0.01585, -0.02649, -0.04641],
            "se": [0.008087, 0.009941, 0.01349],
            "ci_low": [-0.0317, -0.04598, -0.07285],
            "ci_high": [-1.971e-06, -0.007008, -0.01998],
        }
    )
    txt = B._sensitivity_txt({"map_sensitivity": ms, "exposure_map": A.MAIN_FIRE_MAP})
    # No "all three versions agree": two of them are built with post-outcome records.
    assert "mesma conclusión" not in txt and "As tres versións" not in txt
    assert "posteriores aos lumes" in txt
    assert "comprobacións de sensibilidade, non corroboración" in txt
    assert "2017 retrodatado: -0,26" in txt and "2024: -0,46" in txt
    # The main estimate is reported on its own, not as one of its own checks.
    assert "independente" not in txt
    assert B.GL_MAP[A.MAIN_FIRE_MAP].endswith("(principal)")


def _flat(text: str) -> str:
    return " ".join(text.replace(">", " ").split())


def test_no_claim_that_map_error_shrinks_the_fire_effects_toward_zero():
    assert "fire cannot feed back" not in A.__doc__
    assert "post-outcome sensitivity checks" in _flat(A.__doc__)
    claims = [
        "atenúa os efectos cara a cero",
        "están atenuados por este erro",
        "map error attenuates every estimate that uses the map",
    ]
    for rel in ("src/eucalyptus_impact/real/brief.py", "README.md", "docs/galicia_real/informe.md"):
        text = _flat((ROOT / rel).read_text(encoding="utf-8"))
        for claim in claims:
            assert claim not in text, (rel, claim)
