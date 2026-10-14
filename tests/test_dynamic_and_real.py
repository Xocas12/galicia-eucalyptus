import numpy as np

from eucalyptus_impact.data.synthetic import EUC, NATIVE, PINE, SHRUB
from eucalyptus_impact.dynamic import Components, project, risk_weights, scenario_contrasts


def _components(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p_base = rng.uniform(0.005, 0.08, n)
    return Components(
        conv_rate=np.full(n, 0.01),
        fire_boost=0.05,
        source=np.array([0, 0.2, 0.2, 0.4, 0.2, 0.0]),
        burn_loss=np.array([0, 0.6, 0.5, 0, 0, 0]),
        p_base=p_base,
        theta=np.array([0.03, 0.02, -0.02, 0.04, 0, 0]),
        burn_frac=0.4,
        weather=np.array([0.5, 1.0, 1.5]),
    )


def _cover(n=400, seed=1):
    return np.random.default_rng(seed).dirichlet([3, 2, 2, 2, 2, 0.5], n)


def test_policies_move_cover_as_specified():
    comp, f0 = _components(), _cover()
    kw = dict(cell_area_ha=100, n_sims=5, seed=0)
    bau = project(f0, comp, 10, "bau", **kw)
    cap = project(f0, comp, 10, "cap", **kw)
    restore = project(f0, comp, 10, "restore", np.ones(len(f0)), **kw)
    euc0 = f0[:, EUC].sum() * 100
    assert bau["eucalyptus_ha"].iloc[-1] > euc0
    assert np.isclose(cap["eucalyptus_ha"].iloc[-1], euc0)
    assert restore["eucalyptus_ha"].iloc[-1] < 0.2 * euc0
    assert restore["native_ha"].iloc[-1] > cap["native_ha"].iloc[-1]
    c = scenario_contrasts({"BAU": bau, "Cap / moratorium": cap, "Restore": restore})
    assert (c["d_eucalyptus_ha"] < 0).all()


def test_fire_turns_native_and_pine_into_shrub():
    comp, f0 = _components(), _cover()
    comp.conv_rate[:] = 0
    comp.fire_boost = 0
    comp.p_base[:] = 0.5
    out = project(f0, comp, 5, "cap", cell_area_ha=100, n_sims=3, seed=1)
    assert out["shrub_ha"].iloc[-1] > f0[:, SHRUB].sum() * 100
    assert out["native_ha"].iloc[-1] < f0[:, NATIVE].sum() * 100


def test_targeting_high_risk_cells_beats_random():
    comp, f0 = _components(n=2000, seed=3), _cover(n=2000, seed=4)
    comp.conv_rate[:] = 0
    order = np.argsort(-comp.p_base * f0[:, EUC])
    top = np.zeros(2000)
    top[order[:300]] = 1
    rnd = np.zeros(2000)
    rnd[np.random.default_rng(5).permutation(2000)[:300]] = 1
    kw = dict(cell_area_ha=100, n_sims=20, seed=2)
    base = project(f0, comp, 12, "cap", **kw)
    t = project(f0, comp, 12, "restore", top, **kw)
    r = project(f0, comp, 12, "restore", rnd, **kw)
    gain_t = base["expected_burned_ha"].sum() - t["expected_burned_ha"].sum()
    gain_r = base["expected_burned_ha"].sum() - r["expected_burned_ha"].sum()
    assert gain_t > gain_r > 0


def _restoration_gain_per_ha(comp, f0, sel, risk_scaling):
    """Drop in expected burned area in year 2 (after one year of restoration), per ha moved."""
    kw = dict(cell_area_ha=100, n_sims=1, seed=0, risk_scaling=risk_scaling)
    base = project(f0, comp, 2, "cap", **kw)
    rest = project(f0, comp, 2, "restore", sel, **kw)
    moved_ha = rest["native_ha"].iloc[-1] - base["native_ha"].iloc[-1]
    gain = base["expected_burned_ha"].iloc[-1] - rest["expected_burned_ha"].iloc[-1]
    return gain / moved_ha


def _static_components(n, seed):
    """No planting, no fire damage, one weather year: cover moves only by restoration."""
    comp = _components(n=n, seed=seed)
    comp.conv_rate[:] = 0
    comp.fire_boost = 0
    comp.burn_loss[:] = 0
    comp.weather = np.array([1.0])
    return comp


def _selections(comp, f0, k=300, seed=5):
    n = len(f0)
    top = np.zeros(n)
    top[np.argsort(-comp.p_base * f0[:, EUC])[:k]] = 1
    rnd = np.zeros(n)
    rnd[np.random.default_rng(seed).permutation(n)[:k]] = 1
    return top, rnd


def test_targeting_premium_comes_from_the_risk_scaling():
    # Small effects so nothing clips at p = 0: with a constant effect, a hectare restored in a
    # high-risk cell is worth exactly a hectare restored anywhere else.
    comp, f0 = _static_components(2000, 3), _cover(n=2000, seed=4)
    comp.theta = np.array([0.002, 0.0, -0.001, 0.0, 0.0, 0.0])  # |dp| < min p_base
    top, rnd = _selections(comp, f0)
    t_const = _restoration_gain_per_ha(comp, f0, top, risk_scaling=False)
    r_const = _restoration_gain_per_ha(comp, f0, rnd, risk_scaling=False)
    assert t_const > 0
    assert np.isclose(t_const / r_const, 1.0, rtol=1e-9)
    t_risk = _restoration_gain_per_ha(comp, f0, top, risk_scaling=True)
    r_risk = _restoration_gain_per_ha(comp, f0, rnd, risk_scaling=True)
    assert t_risk / r_risk > 1.3
    assert np.isclose(risk_weights(comp.p_base).mean(), 1.0)


def test_targeting_premium_is_only_right_when_the_hazard_form_is():
    """The engine's targeted/random ratio against two true fire processes.

    Each truth gives the same average effect of eucalyptus -> native per unit share (what a
    pooled DML fit estimates), and the engine is given that average. Under a logistic hazard
    the risk-scaled engine reproduces the true premium and the constant one misses it; under
    an additive hazard (not logistic in cover) it is the other way round. A synthetic check
    built on a logistic simulator therefore passes the risk scaling by construction.
    """
    n = 3000
    comp, f0 = _static_components(n, 7), _cover(n=n, seed=8)
    comp.p_base = np.random.default_rng(9).lognormal(np.log(0.01), 1.0, n).clip(1e-4, 0.4)
    top, rnd = _selections(comp, f0, k=450, seed=10)
    pb = comp.p_base
    beta = -0.8  # logit-scale effect of moving one unit of share from eucalyptus to native
    ame = float(np.mean(pb * (1 - pb)) * beta)  # average effect per unit share

    def true_gain_per_ha(sel, form):
        moved = f0[:, EUC] * sel * 0.95
        if form == "logistic":
            z = np.log(pb / (1 - pb)) + beta * moved
            dp = 1 / (1 + np.exp(-z)) - pb
        else:
            dp = ame * moved
        return -dp.sum() / moved.sum()

    comp.theta = np.array([0.0, 0.0, ame, 0.0, 0.0, 0.0])
    eng = {
        rs: _restoration_gain_per_ha(comp, f0, top, rs)
        / _restoration_gain_per_ha(comp, f0, rnd, rs)
        for rs in (True, False)
    }
    for form, right, wrong in (("logistic", True, False), ("additive", False, True)):
        truth = true_gain_per_ha(top, form) / true_gain_per_ha(rnd, form)
        assert abs(eng[right] - truth) < 0.1 * truth, (form, eng, truth)
        assert abs(eng[wrong] - truth) > 0.3 * truth, (form, eng, truth)


def test_osm_label_rules():
    from eucalyptus_impact.real.layers import _label_from_tags

    assert _label_from_tags({"genus": "Eucalyptus"}) == EUC
    assert _label_from_tags({"leaf_type": "broadleaved", "leaf_cycle": "evergreen"}) == EUC
    assert _label_from_tags({"leaf_type": "needleleaved"}) == PINE
    assert _label_from_tags({"species": "Quercus robur"}) == NATIVE
    assert _label_from_tags({"leaf_type": "broadleaved", "leaf_cycle": "deciduous"}) == NATIVE
    assert _label_from_tags({"leaf_type": "mixed"}) is None


def test_s2_offset_rule():
    from eucalyptus_impact.real.s2 import band_scale_offset

    def item(applied, offset):
        rb = {"raster:bands": [{"scale": 1e-4, "offset": offset}]}
        return {
            "properties": {"earthsearch:boa_offset_applied": applied},
            "assets": {k: rb for k in ("red", "nir", "swir16", "swir22")},
        }

    sc, of = band_scale_offset(item(True, -0.1))
    assert of["red"] == 0.0 and sc["nir"] == 1e-4
    _, of = band_scale_offset(item(False, -0.1))
    assert of["red"] == -0.1


def test_confusion_inversion_recovers_true_shares():
    from eucalyptus_impact.real.species import confusion_inversion

    rng = np.random.default_rng(0)
    k = 6
    C = np.full((k, k), 0.02)
    np.fill_diagonal(C, 0.9)
    C /= C.sum(1, keepdims=True)
    pi = np.array([0.25, 0.1, 0.2, 0.15, 0.25, 0.05])
    m = C.T @ pi
    y = np.repeat(np.arange(k), 5000)  # equal labels per true class, as in training
    pred = np.array([rng.choice(k, p=C[t]) for t in y])
    est, _ = confusion_inversion(m, y, pred)
    assert np.allclose(est, pi, atol=0.02)


def test_tile_offsets_recover_radiometric_shifts():
    from eucalyptus_impact.real.s2 import tile_offsets

    rng = np.random.default_rng(0)
    truth = rng.normal(0.5, 0.1, (1, 400, 600))
    shifts = [0.0, 0.03, -0.02]
    grids = []
    for k, (c0, c1) in enumerate([(0, 250), (200, 450), (400, 600)]):
        g = np.full_like(truth, np.nan)
        g[:, :, c0:c1] = truth[:, :, c0:c1] + shifts[k] + rng.normal(0, 0.005, (1, 400, c1 - c0))
        grids.append(g)
    off = tile_offsets(grids, min_overlap=100)[:, 0]
    rel = off - off.mean()
    expected = np.array(shifts) - np.mean(shifts)
    assert np.allclose(rel, expected, atol=0.003)


def _fake_dml(name, res_d, res_y, groups):
    from eucalyptus_impact.causal.dml import EffectEstimate

    th = float(np.mean(res_d * res_y) / np.mean(res_d**2))
    diag = {"res_y": res_y, "res_d": res_d, "groups": groups, "d_error_var": 0.0}
    return EffectEstimate(name, th, 0.0, "DML-PLR", len(res_d), diag=diag)


def test_restoration_contrast_by_risk_tells_the_two_forms_apart():
    import pandas as pd

    from eucalyptus_impact.real.analysis import restoration_contrast_by_risk

    rng = np.random.default_rng(0)
    n, years = 4000, 6
    cells = pd.DataFrame({"row": np.arange(n) // 80, "col": np.arange(n) % 80})
    p_base = rng.lognormal(np.log(0.01), 1.0, n).clip(1e-4, 0.4)
    panel = pd.concat([cells.assign(year=y) for y in range(years)], ignore_index=True)
    pb = np.tile(p_base, years)
    w = risk_weights(pb)
    groups = np.tile(np.arange(n) // 40, years)
    for form, truth in (("risk-scaled", -0.02 * w), ("constant", np.full(len(pb), -0.02))):
        rd_e, rd_n = rng.standard_normal(len(pb)), rng.standard_normal(len(pb))
        fire = {
            "cover_effects": {
                "f_eucalyptus": _fake_dml("e", rd_e, 0.01 * rng.standard_normal(len(pb)), groups),
                "f_native_broadleaf": _fake_dml(
                    "n", rd_n, truth * rd_n + 0.01 * rng.standard_normal(len(pb)), groups
                ),
            }
        }
        g = restoration_contrast_by_risk(fire, panel, cells, p_base)
        assert list(g["group"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert (g["cell_years"] == n * years // 5).all()
        assert g["mean_p_base"].is_monotonic_increasing
        right = "engine_risk_scaled" if form == "risk-scaled" else "constant"
        wrong = "constant" if form == "risk-scaled" else "engine_risk_scaled"
        err_right = np.abs(g["contrast"] - g[right]).max()
        err_wrong = np.abs(g["contrast"] - g[wrong]).max()
        assert err_right < 3 * g["contrast_se"].max(), (form, g)
        assert err_wrong > 10 * err_right, (form, g)


def test_brief_states_the_risk_scaling_and_the_constant_effect_figures():
    import pandas as pd

    from eucalyptus_impact.real.brief import _premium_txt, _risk_scaling_section

    def contrasts(t, r):
        return pd.DataFrame(
            {
                "scenario": ["Targeted restoration", "Random restoration"],
                "d_eucalyptus_ha": [-1.0, -1.0],
                "d_native_ha": [1.0, 1.0],
                "d_burned_ha_per_year": [t, r],
                "d_burned_p05": [t - 100, r - 100],
                "d_burned_p95": [t + 100, r + 100],
            }
        )

    proj = {
        "contrasts_constant_effect": contrasts(-412.5, -360.25),
        "risk_weight_selected": {"Targeted restoration": 2.42, "Random restoration": 0.97},
    }
    txt = _premium_txt(proj)
    assert "p(1\u2212p)" in txt and "-412,5" in txt and "-360,2" in txt
    fire = {
        "restoration_by_risk": pd.DataFrame(
            {
                "group": ["Q1", "Q5"],
                "mean_p_base": [0.002, 0.04],
                "contrast": [-0.01, -0.05],
                "contrast_se": [0.01, 0.02],
                "engine_risk_scaled": [-0.004, -0.06],
                "constant": [-0.0226, -0.0226],
                "cell_years": [100, 100],
            }
        )
    }
    sec = _risk_scaling_section(fire, proj)
    assert "hipótese" in sec and "2,42" in sec and "0,97" in sec
    assert "efecto constante" in sec and "Q5 (risco máis alto)" in sec
    assert "| escenario |" in sec
    # Without the new outputs (an older run), the disclosure is still there.
    assert "hipótese" in _premium_txt({}) and "hipótese" in _risk_scaling_section({}, {})
