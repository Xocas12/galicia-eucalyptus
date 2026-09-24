import numpy as np
import pandas as pd

from eucalyptus_impact.causal.dml import dml_plr, ols
from eucalyptus_impact.causal.matching import match_att
from eucalyptus_impact.models.hydrology import twfe


def _confounded(n=6000, theta=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    d = np.tanh(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.5, n)
    y = theta * d - 2.0 * np.tanh(2 * X[:, 0]) + np.cos(X[:, 2]) + rng.normal(0, 0.5, n)
    groups = rng.integers(0, 60, n)
    return y, d, X, groups


def test_dml_recovers_theta_where_naive_fails():
    y, d, X, g = _confounded(seed=11)
    est = dml_plr(y, d, X, g, n_folds=4, seed=0)
    lo, hi = est.ci95
    assert lo < 0.5 < hi
    naive = ols(y, d, g)
    assert abs(naive.estimate - 0.5) > 3 * naive.se


def test_measurement_error_correction_removes_attenuation():
    y, d, X, g = _confounded(n=8000, seed=5)
    noise_var = 0.15
    d_obs = d + np.random.default_rng(9).normal(0, np.sqrt(noise_var), len(d))
    raw = dml_plr(y, d_obs, X, g, n_folds=4, seed=0)
    fixed = dml_plr(y, d_obs, X, g, n_folds=4, seed=0, d_error_var=noise_var)
    assert raw.estimate < 0.4  # attenuated
    # Regression calibration removes most (not all) of the attenuation.
    assert abs(fixed.estimate - 0.5) < 0.4 * abs(raw.estimate - 0.5)


def test_ols_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 2 + 3 * x + rng.normal(size=500)
    est = ols(y, x)
    assert abs(est.estimate - 3) < 0.2 and est.se > 0


def test_matching_recovers_att_and_improves_balance():
    rng = np.random.default_rng(2)
    n = 4000
    X = rng.normal(size=(n, 2))
    t = rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    y = 1.0 * t + 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.5, n)
    est, bal = match_att(y, t, X)
    assert abs(est.estimate - 1.0) < 0.15
    assert (bal["smd_after"].abs() < bal["smd_before"].abs()).all()


def test_twfe_absorbs_unit_and_time_confounding():
    rng = np.random.default_rng(3)
    units, years = 40, 15
    u = np.repeat(np.arange(units), years)
    t = np.tile(np.arange(years), units)
    a = rng.normal(size=units)[u] * 5
    b = rng.normal(size=years)[t] * 3
    x = 0.8 * a + 0.5 * b + rng.normal(size=len(u))
    y = -2.0 * x + 4 * a + 2 * b + rng.normal(size=len(u))
    df = pd.DataFrame({"y": y, "x": x, "u": u, "t": t})
    est = twfe(df, "y", "x", "u", "t")
    lo, hi = est.ci95
    assert lo < -2.0 < hi
