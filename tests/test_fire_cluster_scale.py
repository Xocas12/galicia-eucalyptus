"""Fire SEs: the block-size table, the cluster scale chosen from it, and what uses that scale."""

import numpy as np
import pandas as pd
import pytest

from eucalyptus_impact.causal.dml import EffectEstimate, at_cluster_scale, reclustered_se, t_crit
from eucalyptus_impact.dynamic import COVER
from eucalyptus_impact.real import analysis as A
from eucalyptus_impact.real import brief as B
from eucalyptus_impact.real import lookup as LK
from eucalyptus_impact.real.common import GRID_1KM


def _cells(step: int = 9, seed: int = 0) -> pd.DataFrame:
    """Toy cell table on the real 1 km grid: every `step`-th cell, with a spatially smooth
    burn-risk field so the scores are correlated across blocks."""
    rng = np.random.default_rng(seed)
    ny, nx = GRID_1KM.shape
    rr, cc = np.meshgrid(np.arange(0, ny, step), np.arange(0, nx, step), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    n = len(rr)
    x, y = GRID_1KM.centers()
    df = pd.DataFrame(
        {
            "row": rr,
            "col": cc,
            "block": A.block_ids(GRID_1KM, 20)[rr, cc],
            "x_km": x[rr, cc] / 1000,
            "y_km": y[rr, cc] / 1000,
        }
    )
    for c in A.STATIC:
        if c not in df:
            df[c] = rng.normal(size=n)
    frac = rng.dirichlet(np.ones(len(COVER)), size=n)
    for j, c in enumerate(COVER):
        df[c] = frac[:, j]
    field = np.sin(df["x_km"] / 60) + np.cos(df["y_km"] / 45)
    for yr in A.FIRE_YEARS:
        p = 1 / (1 + np.exp(-(-2.0 + 0.8 * field + rng.normal(0, 0.5) - 0.5 * df[COVER[0]])))
        burned = rng.random(n) < p
        df[f"burned_{yr}"] = np.where(burned, rng.uniform(0.01, 1, n), 0.0)
        df[f"severity_{yr}"] = np.where(burned, rng.integers(1, 5, n).astype(float), np.nan)
        df[f"fwi_{yr}"] = rng.normal(size=n) + field
    return df


@pytest.fixture(scope="module")
def fire():
    panel = A.cell_year_panel(_cells())
    return panel, A.fire_analysis(panel, n_folds=3)


def test_9_twenty_km_row_of_the_block_table_reproduces_the_main_se(fire):
    panel, out = fire
    tab = out["se_by_block"].set_index("block_km")
    occ = out["occurrence_dml"]
    # The fit is clustered on panel["block"], the 20 km blocks counted from the bbox origin.
    assert tab.loc[20, "se"] == pytest.approx(reclustered_se(occ, panel["block"]), rel=1e-12)


def test_9_fire_ses_are_clustered_at_the_scale_the_table_supports(fire):
    _, out = fire
    tab = out["se_by_block"].set_index("block_km")
    sc = out["cluster_scale"]
    k = sc["block_km"]
    occ = out["occurrence_dml"]
    assert occ.se == pytest.approx(tab.loc[k, "se"], rel=1e-12)
    larger = tab.index[tab.index > k]
    if sc["still_rising"]:
        assert k == tab.index.max()
    else:
        assert tab.loc[larger[0], "se"] <= tab.loc[k, "se"]
    assert occ.crit == pytest.approx(t_crit(tab.loc[k, "n_blocks"]))
    # Every cover effect (what the lookup and the projections read) uses the same clustering.
    for e in out["cover_effects"].values():
        assert e.crit == occ.crit
        assert np.array_equal(e.diag["groups"], occ.diag["groups"])


def test_9_choose_cluster_scale_stops_where_the_se_stops_rising():
    def tab(ses):
        return pd.DataFrame(
            {"block_km": [5, 10, 20, 40, 80], "n_blocks": [900, 300, 90, 25, 9], "se": ses}
        )

    rising = A.choose_cluster_scale(tab([0.0066, 0.0088, 0.0105, 0.0120, 0.0137]))
    assert rising["block_km"] == 80 and rising["still_rising"]
    assert rising["crit"] == pytest.approx(t_crit(9)) and rising["crit"] > 2.3
    levels = A.choose_cluster_scale(tab([0.004, 0.005, 0.009, 0.011, 0.0105]))
    assert levels["block_km"] == 40 and not levels["still_rising"]
    flat = A.choose_cluster_scale(tab([0.004, 0.005, 0.009, 0.0085, 0.0089]))
    assert flat["block_km"] == 20 and not flat["still_rising"]


def test_9_at_cluster_scale_moves_only_the_inference():
    rng = np.random.default_rng(1)
    n = 400
    rd, ry = rng.normal(size=n), rng.normal(size=n)
    g20 = np.repeat(np.arange(40), 10)
    est = EffectEstimate(
        "x",
        float(np.mean(rd * ry) / np.mean(rd**2)),
        0.0,
        "DML-PLR",
        n,
        diag={"res_y": ry, "res_d": rd, "groups": g20, "d_error_var": 0.0},
    )
    g80 = g20 // 10
    big = at_cluster_scale(est, g80)
    assert big.estimate == est.estimate
    assert big.se == pytest.approx(reclustered_se(est, g80))
    assert big.crit == pytest.approx(t_crit(4))
    lo, hi = big.ci95
    assert hi - lo == pytest.approx(2 * t_crit(4) * big.se)


def test_9_lookup_flag_uses_the_chosen_critical_value():
    eff = {"cover_effects": {"f_eucalyptus": [-0.02, 0.01]}, "crit": 2.31, "cluster_km": 80}
    a = LK.eucalyptus_contribution(eff, 0.5)
    assert not a["significant"]  # |-0.02| / 0.01 = 2.0 < 2.31
    assert np.allclose(a["ci"], [-0.01 - 2.31 * 0.005, -0.01 + 2.31 * 0.005])


def _ms(ci_highs):
    return pd.DataFrame(
        {
            "map": ["2017 backdated", "2017 independent", "2024 (post-fire)"],
            "estimate": [-0.026, -0.016, -0.03],
            "se": [0.013, 0.011, 0.012],
            "block_km": [80.0] * 3,
            "ci_low": [-0.05] * 3,
            "ci_high": ci_highs,
        }
    )


def test_9_brief_states_what_the_block_table_implies():
    se = pd.DataFrame(
        {
            "block_km": [5, 10, 20, 40, 80],
            "n_blocks": [900, 300, 90, 25, 9],
            "se": [0.0066, 0.0088, 0.0105, 0.0120, 0.0137],
        }
    )
    fire = {"se_by_block": se, "cluster_scale": A.choose_cluster_scale(se)}
    txt = B._block_txt(fire)
    assert "medra con cada tamaño de bloque" in txt and "80 km" in txt
    assert "non se debe dar como significativo" in txt
    # The map-sensitivity sentence reads intervals clustered at that scale and says so.
    fire["map_sensitivity"] = _ms([-0.001, 0.002, -0.004])
    s = B._sensitivity_txt(fire)
    assert "non todas son distinguibles de cero" in s and "bloques de 80 km" in s
    fire["map_sensitivity"] = _ms([-0.001, -0.002, -0.004])
    s = B._sensitivity_txt(fire)
    assert "mesma conclusión" in s and "estreitos de máis" in s
