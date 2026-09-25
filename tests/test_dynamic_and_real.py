import numpy as np

from eucalyptus_impact.data.synthetic import EUC, NATIVE, PINE, SHRUB
from eucalyptus_impact.dynamic import Components, project, scenario_contrasts


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
