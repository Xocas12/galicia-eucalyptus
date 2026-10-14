"""Conversion of open land without fire, which the backdated 2017 map cannot record (issue #10)."""

from pathlib import Path

import numpy as np
import pandas as pd

from eucalyptus_impact.dynamic import COVER, Components
from eucalyptus_impact.real.layers import AGRI, EUC, NATIVE, PINE, SHRUB
from eucalyptus_impact.real.species import PIXEL_HA, backdate, stable_since

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    """Six pixels, all eucalyptus in 2024.

    0 farmland planted 2019, no loss, no burn      -> hidden by backdating
    1 shrub planted 2020, no loss, no burn         -> hidden by backdating
    2 oak felled 2019 and planted                  -> recorded (Hansen loss)
    3 shrub burnt 2018 and planted                 -> recorded (EFFIS burn)
    4 eucalyptus all along                         -> no change
    5 pine in the independent 2017 map, but 2024 is not confident -> outside the bound
    """
    ly = np.array([[0, 0, 19, 0, 0, 0]], "uint8")
    effis = {f"burned40_{y}": np.zeros((1, 6), "uint8") for y in range(2018, 2024)}
    effis["burned40_2018"][0, 3] = 1
    L = {"hansen": {"lossyear40": ly}, "effis": effis}
    cls17r = np.array([[AGRI, SHRUB, NATIVE, SHRUB, EUC, PINE]], "uint8")
    pmax17r = np.full((1, 6), 90, "uint8")
    cls24 = np.full((1, 6), EUC, "uint8")
    pmax24 = np.array([[90, 90, 90, 90, 90, 50]], "uint8")
    cls17, kept = backdate(L, cls17r, pmax17r, cls24)
    maps = {
        "class40_2017": cls17,
        "pmax40_2017": np.where(kept, pmax24, pmax17r).astype("uint8"),
        "class40_2024": cls24,
        "pmax40_2024": pmax24,
        "class40_2017_independent": cls17r,
    }
    return L, maps


def test_backdating_hides_undisturbed_open_land_conversion():
    """The premise: farmland or shrub planted without loss or burn shows no change."""
    _, maps = _toy()
    a, b = maps["class40_2017"], maps["class40_2024"]
    changed = (a != b)[0]
    assert changed.tolist() == [False, False, True, True, False, False]


def test_bound_counts_backdated_pixels_the_independent_map_calls_non_eucalyptus():
    from eucalyptus_impact.real.analysis import undisturbed_gain_bound

    L, maps = _toy()
    b = undisturbed_gain_bound(maps, stable_since(L, 2017))
    assert np.isclose(b["hidden_ha"], 2 * PIXEL_HA)
    assert np.isclose(b["hidden_by_source_ha"]["agriculture"], PIXEL_HA)
    assert np.isclose(b["hidden_by_source_ha"]["shrub"], PIXEL_HA)
    assert b["hidden_by_source_ha"]["pine"] == 0
    assert "eucalyptus" not in b["hidden_by_source_ha"]
    # Recorded gain: the felled oak and the burnt shrub, out of those two confident pixels.
    assert np.isclose(b["recorded_ha"], 2 * PIXEL_HA)
    assert np.isclose(b["base_ha"], 2 * PIXEL_HA)
    assert np.isclose(b["recorded_rate"], 1 / 7)
    assert np.isclose(b["envelope_rate"], 1 / 7)


def test_bound_raises_the_rate_when_recorded_gain_is_partial():
    from eucalyptus_impact.real.analysis import undisturbed_gain_bound

    L, maps = _toy()
    # The burnt shrub stays shrub in 2024: recorded gain 1 of 2, hidden pixels add 2 of 2.
    maps["class40_2024"] = maps["class40_2024"].copy()
    maps["class40_2024"][0, 3] = SHRUB
    b = undisturbed_gain_bound(maps, stable_since(L, 2017))
    assert np.isclose(b["recorded_rate"], 0.5 / 7)
    assert np.isclose(b["envelope_rate"], 0.75 / 7)
    assert b["envelope_rate"] > b["recorded_rate"]


def test_brief_reports_the_bound_and_the_open_land_caveat():
    from eucalyptus_impact.real.brief import _undisturbed_txt

    L, maps = _toy()
    from eucalyptus_impact.real.analysis import undisturbed_gain_bound

    bound = undisturbed_gain_bound(maps, stable_since(L, 2017))
    bound["hidden_ha"] = 12_345.0
    txt = _undisturbed_txt({"undisturbed_gain_bound": bound})
    assert "12 345 ha" in txt
    assert "Non vale para mato → eucalipto nin para agricultura → eucalipto" in txt
    assert "envolvente" in txt
    # Older cached results without the bound still render the caveat.
    assert "non está dispoñible" in _undisturbed_txt({})


def _cells(n=300, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.dirichlet([3, 2, 2, 2, 2, 0.5], n)
    df = pd.DataFrame(f0, columns=COVER)
    for c in COVER:
        df[c + "_end"] = df[c]
    for yr in range(2018, 2024):
        df[f"burned_{yr}"] = rng.uniform(0, 0.05, n) * (rng.random(n) < 0.1)
    df["gross_gain"] = rng.uniform(0, 0.02, n)
    df["n_base_px"] = 100
    return df


def test_projections_run_the_planting_rate_up_as_well_as_down(monkeypatch):
    from eucalyptus_impact.real import analysis

    cells = _cells()
    n = len(cells)

    def fake_components(cells, years, cov, p_base, theta, burn_frac, weather, seed=0):
        return Components(
            conv_rate=np.full(n, 0.01),
            fire_boost=0.0,
            source=np.array([0, 0.2, 0.2, 0.4, 0.2, 0.0]),
            burn_loss=np.array([0, 0.6, 0.5, 0, 0, 0]),
            p_base=p_base,
            theta=theta,
            burn_frac=burn_frac,
            weather=weather,
        )

    monkeypatch.setattr(analysis, "estimate_components", fake_components)
    susc = {"p_base": np.full(n, 0.02)}
    proj = analysis.projections(cells, {"cover_effects": {}}, susc, horizon=2030)
    cap = {
        k: proj[k].set_index("scenario").loc["Cap / moratorium", "d_eucalyptus_ha"]
        for k in ("contrasts_half_conversion", "contrasts", "contrasts_double_conversion")
    }
    # The cap removes BAU planting, so its contrast grows with the planting rate.
    assert (
        cap["contrasts_double_conversion"] < cap["contrasts"] < cap["contrasts_half_conversion"] < 0
    )


def test_upper_bound_claim_is_gone_and_limitations_name_the_blind_spot():
    brief = (ROOT / "src/eucalyptus_impact/real/brief.py").read_text(encoding="utf-8")
    informe = (ROOT / "docs/galicia_real/informe.md").read_text(encoding="utf-8")
    scope = (ROOT / "docs/SCOPE.md").read_text(encoding="utf-8")
    for text in (brief, informe):
        assert "tendencial é un límite superior" not in text
        assert "non fai do escenario tendencial un límite superior" in text
        assert "**Cambio en terreo aberto.**" in text
        assert "Conversión a eucalipto rexistrada en terreo queimado" in text
    assert "Conversion of open land is invisible to the backdated maps" in scope
    assert "conversion recorded on burnt land" in scope
