import numpy as np
import pandas as pd

from eucalyptus_impact.causal.dml import boosted_trees, dml_plr, group_effects, reclustered_se
from eucalyptus_impact.causal.sensitivity import bias_bound, robustness_value, sensitivity_table
from eucalyptus_impact.causal.simex import perturb_cover, simex


def _linear_confounded(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    d = X @ rng.uniform(0.2, 0.6, 6) + rng.normal(0, 0.5, n)
    y = 0.7 * d + X @ rng.uniform(-1, 1, 6) + rng.normal(0, 0.5, n)
    return y, d, X, rng.integers(0, 50, n)


def test_linear_plus_boost_beats_trees_on_linear_nuisance():
    y, d, X, g = _linear_confounded()
    lpb = dml_plr(y, d, X, g, n_folds=4)
    trees = dml_plr(y, d, X, g, n_folds=4, learner_factory=boosted_trees)
    assert abs(lpb.estimate - 0.7) < abs(trees.estimate - 0.7)
    assert lpb.ci95[0] < 0.7 < lpb.ci95[1]


def test_robustness_value_is_the_bias_bound_fixed_point():
    y, d, X, g = _linear_confounded(seed=1)
    est = dml_plr(y, d, X, g, n_folds=4)
    rv = robustness_value(est)
    assert 0 < rv < 1
    assert np.isclose(bias_bound(est, rv, rv), abs(est.estimate), rtol=1e-6)
    assert robustness_value(est, alpha_z=1.96) < rv
    tab = sensitivity_table(est)
    assert (np.diff(tab["max_bias"]) > 0).all()


def test_group_effects_recover_heterogeneity():
    rng = np.random.default_rng(2)
    n = 8000
    X = rng.normal(size=(n, 3))
    grp = (X[:, 0] > 0).astype(int)
    d = X[:, 1] + rng.normal(0, 1, n)
    theta = np.where(grp == 1, 2.0, 0.5)
    y = theta * d + X[:, 1] + rng.normal(0, 0.5, n)
    est = dml_plr(y, d, X, rng.integers(0, 60, n), n_folds=4)
    gates = group_effects(est, grp, truth=theta)
    for _, r in gates.iterrows():
        assert abs(r["estimate"] - r["truth"]) < 3 * r["se"] + 0.05
    assert reclustered_se(est, np.zeros(n) + np.arange(n) % 10) > 0


def test_perturb_cover_keeps_partition():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.dirichlet([1, 1, 1], 100), columns=["a", "b", "c"])
    out = perturb_cover(df, ["a", "b", "c"], 0.05, rng)
    assert np.allclose(out[["a", "b", "c"]].sum(axis=1), 1)
    assert (out.to_numpy() >= 0).all()


def test_simex_undoes_attenuation():
    rng = np.random.default_rng(4)
    n = 6000
    F = rng.dirichlet([2, 2, 2], n)
    z = rng.normal(size=n)
    y = -0.5 * F[:, 0] + 0.2 * F[:, 1] + 0.1 * z + rng.normal(0, 0.02, n)
    df = pd.DataFrame(
        {"a": F[:, 0], "b": F[:, 1], "c": F[:, 2], "z": z, "y": y, "g": rng.integers(0, 40, n)}
    )
    noisy = perturb_cover(df, ["a", "b", "c"], 0.06, rng)

    def est(d):
        return dml_plr(d["y"], d["a"], d[["b", "z"]].to_numpy(), d["g"], n_folds=3)

    raw = est(noisy).estimate
    corrected, path = simex(est, noisy, ["a", "b", "c"], 0.06**2, n_rep=2)
    assert len(path) == 5
    assert abs(corrected.estimate + 0.5) < abs(raw + 0.5)
