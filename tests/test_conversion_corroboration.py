"""Native -> eucalyptus conversion: no disturbance "corroboration" on top of backdating (#12).

The backdated 2017 map keeps the 2024 class on every pixel without Hansen loss 2017-2024 or an
EFFIS burn 2018-2023, so every recorded 2017 -> 2024 transition already lies on a disturbed
pixel. A "confident and corroborated by loss or fire" figure therefore repeats the backdating
rule (it can only drop pixels whose sole disturbance is a calendar-2017 loss) and cannot reject a
wrong 2017 class on a pixel that really was disturbed.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from eucalyptus_impact.dynamic import COVER
from eucalyptus_impact.real import analysis
from eucalyptus_impact.real.brief import _conversion_resumo, _conversion_txt
from eucalyptus_impact.real.layers import EUC, NATIVE, PINE
from eucalyptus_impact.real.species import backdate, stable_since

FIRE_YEARS = range(2018, 2024)


def _all_histories():
    """One pixel per disturbance history: Hansen loss year 0..24 x every 2018-2023 burn pattern."""
    hist = list(itertools.product(range(25), itertools.product([0, 1], repeat=len(FIRE_YEARS))))
    ly = np.array([h[0] for h in hist], dtype="uint8")
    burns = np.array([h[1] for h in hist], dtype="uint8")
    L = {
        "hansen": {"lossyear40": ly},
        "effis": {f"burned40_{y}": burns[:, i] for i, y in enumerate(FIRE_YEARS)},
    }
    return L, ly, burns.any(axis=1)


def test_backdated_transitions_lie_on_disturbed_pixels_by_construction():
    L, ly, burned = _all_histories()
    n = len(ly)
    assert n == 1600
    cls17 = np.full(n, NATIVE, "uint8")
    cls24 = np.full(n, EUC, "uint8")
    out, _kept = backdate(L, cls17, np.full(n, 95, "uint8"), cls24)
    trans = (out == NATIVE) & (cls24 == EUC)
    assert trans.sum() == 1583
    # Every transition is on a pixel the backdating rule calls disturbed ...
    assert not (trans & stable_since(L, 2017)).any()
    # ... so the old "loss 2018-2024 or fire" filter could only drop 2017-loss-only pixels.
    old_filter = ((ly >= 18) & (ly <= 24)) | burned
    dropped = trans & ~old_filter
    assert dropped.sum() == 1
    assert (ly[dropped] == 17).all() and not burned[dropped].any()


class _StubModel:
    def fit(self, X, y):
        return self


def _stub_importance(m, X, y, n_repeats, random_state):
    return type("Imp", (), {"importances_mean": np.zeros(X.shape[1])})()


def test_conversion_analysis_reports_no_loss_or_fire_corroboration(monkeypatch):
    monkeypatch.setattr(analysis, "dml_plr", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "ols", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "HistGradientBoostingRegressor", lambda **k: _StubModel())
    monkeypatch.setattr(analysis, "permutation_importance", _stub_importance)

    L, ly, _ = _all_histories()
    n = len(ly)
    cls24 = np.full(n, EUC, "uint8")
    a17 = np.where(np.arange(n) % 2 == 0, NATIVE, PINE).astype("uint8")
    cls17, _ = backdate(L, a17, np.full(n, 95, "uint8"), cls24)
    maps = {
        "class40_2017": cls17,
        "class40_2024": cls24,
        "pmax40_2017": np.full(n, 95, "uint8"),
        "pmax40_2024": np.full(n, 95, "uint8"),
    }
    rng = np.random.default_rng(0)
    cols = [*analysis.STATIC, *COVER[1:5], "neigh_euc", "burned_2018_2021", "gross_gain"]
    cells = pd.DataFrame(rng.random((20, len(cols))), columns=cols)
    cells["n_base_px"] = 100
    cells["block"] = np.arange(20) % 4

    out = analysis.conversion_analysis(cells, maps, L)
    for key in ("native_to_euc_ha", "pine_to_euc_ha"):
        assert set(out[key]) == {"all_pixels", "confident"}, out[key]
    assert out["native_to_euc_ha"]["confident"] > 0


def test_brief_says_change_is_on_disturbed_pixels_by_construction():
    nat = {"all_pixels": 1477.0, "confident": 99.68}
    for txt in (_conversion_resumo(nat), _conversion_txt(nat)):
        assert "corrobor" not in txt
        assert "máis prudente" in txt
        assert "por construción" in txt
        assert "non é unha comprobación independente" in txt
    assert "99,68" in _conversion_resumo(nat)
    assert "1 477" in _conversion_txt(nat) and "99,7" in _conversion_txt(nat)
