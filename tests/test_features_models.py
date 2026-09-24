import numpy as np
import pandas as pd

from eucalyptus_impact.data.synthetic import TruthParams, fu_et
from eucalyptus_impact.features.spectral import dnbr, harmonic_features, harmonic_fit, ndvi
from eucalyptus_impact.models.hydrology import fit_budyko
from eucalyptus_impact.models.landcover import olofsson_area


def test_indices():
    assert np.isclose(ndvi(0.5, 0.1), 0.4 / 0.6)
    assert dnbr(0.5, 0.1) == 400


def test_harmonic_fit_with_gaps():
    months = np.arange(24)
    true = np.array([0.6, 0.2, -0.1, 0.0, 0.0])
    t = 2 * np.pi * months / 12
    s = true[0] + true[1] * np.cos(t) + true[2] * np.sin(t)
    series = np.tile(s, (3, 1))
    series[:, ::3] = np.nan
    coef = harmonic_fit(series, months)
    assert np.allclose(coef, true, atol=1e-2)
    F, cols = harmonic_features(np.repeat(series[:, :, None], 3, axis=2), months)
    assert F.shape == (3, len(cols)) and np.isfinite(F).all()


def test_olofsson_perfect_map_equals_map_area():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, 3000)
    ref = rng.choice(3000, 300, replace=False)
    out = olofsson_area(m, m[ref], m[ref], 3, 1000.0)
    assert np.allclose(out["est_area_ha"], out["map_area_ha"])
    assert np.allclose(out["ci95_ha"], 0)


def test_olofsson_corrects_commission_error():
    rng = np.random.default_rng(1)
    truth = rng.choice(2, 50000, p=[0.8, 0.2])
    mapped = truth.copy()
    flip = (truth == 0) & (rng.random(len(truth)) < 0.1)  # class 0 mapped as 1
    mapped[flip] = 1
    ref = np.concatenate(
        [rng.choice(np.flatnonzero(mapped == c), 500, replace=False) for c in (0, 1)]
    )
    out = olofsson_area(mapped, mapped[ref], truth[ref], 2, 50000.0)
    true_area_1 = (truth == 1).sum()
    assert out.loc[1, "map_area_ha"] > true_area_1 * 1.2
    assert abs(out.loc[1, "est_area_ha"] - true_area_1) < 2 * out.loc[1, "ci95_ha"] + 1


def test_fu_curve_bounds():
    p = np.array([500.0, 1500, 2500])
    pet = np.array([900.0, 800, 700])
    et = fu_et(p, pet, 2.5)
    assert (et <= p).all() and (et <= pet).all() and (et > 0).all()


def test_budyko_fit_recovers_parameters():
    tp = TruthParams()
    rng = np.random.default_rng(4)
    n = 800
    f = rng.dirichlet([2, 2, 2, 2], n)[:, :3]
    P = rng.uniform(700, 2200, n)
    PET = rng.uniform(550, 900, n)
    w = tp.w0 + tp.w_euc * f[:, 0] + tp.w_pine * f[:, 1] + tp.w_native * f[:, 2]
    Q = P - fu_et(P, PET, w) + rng.normal(0, 5, n)
    df = pd.DataFrame(
        {
            "precip": P,
            "pet": PET,
            "runoff": Q,
            "f_eucalyptus": f[:, 0],
            "f_pine": f[:, 1],
            "f_native_broadleaf": f[:, 2],
        }
    )
    est = fit_budyko(df)
    assert abs(est["w_euc"][0] - tp.w_euc) < 3 * est["w_euc"][1] + 0.05
