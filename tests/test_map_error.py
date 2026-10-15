"""Map error from classified pixels, not only Gaussian noise (issue #8)."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from eucalyptus_impact.features.panel import (
    COVER_COLS,
    build_cell_panel,
    classified_cover,
    map_confusion,
)

ROOT = Path(__file__).resolve().parents[1]

# Producer's accuracies of the OSM spatial-CV species map (eucalyptus, pine, native broadleaf,
# shrub, agriculture, other); the rest of each row is spread evenly over the other classes.
DIAG = np.array([0.885, 0.655, 0.797, 0.713, 0.796, 0.888])


def _mild_confusion():
    C = np.tile(((1 - DIAG) / 5)[:, None], (1, 6))
    np.fill_diagonal(C, DIAG)
    return C


def test_map_confusion_is_row_stochastic():
    true = np.array([0, 0, 0, 1, 1, 2])
    mapped = np.array([0, 0, 1, 1, 2, 2])
    C = map_confusion(true, mapped, 4)
    assert np.allclose(C.sum(axis=1), 1)
    assert C[0, 0] == pytest.approx(2 / 3) and C[0, 1] == pytest.approx(1 / 3)
    assert C[1, 1] == pytest.approx(0.5) and C[2, 2] == 1
    assert C[3, 3] == 1  # class absent from the sample: identity row


def test_classified_cover_is_the_confusion_contraction():
    rng = np.random.default_rng(0)
    cover = rng.dirichlet(np.ones(6), size=(4, 500))
    C = _mild_confusion()
    out = classified_cover(cover, C, 625, rng)
    assert out.shape == cover.shape
    assert np.allclose(out.sum(axis=-1), 1)
    # Counts of 625 units: every fraction is a multiple of 1/625.
    assert np.allclose(out * 625, np.round(out * 625))
    # The mean error is systematic (f @ C - f), not zero as with Gaussian noise.
    bias = (out - cover).mean(axis=(0, 1))
    expected = (cover @ C - cover).mean(axis=(0, 1))
    assert np.allclose(bias, expected, atol=0.003)
    assert np.abs(expected).max() > 0.01


def _ols_class_effects(F, y):
    # Fire on eucalyptus, pine, native, shrub shares; agriculture + other is the reference.
    X = np.column_stack([np.ones(len(y)), F[:, :4]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1:]


def test_classifier_fractions_inflate_the_class_effects():
    """With pixel-count fractions the class effects move outward, not toward zero.

    Gaussian noise of the kind the synthetic panels used before attenuates the same regression,
    which is why the old check could not see this.
    """
    rng = np.random.default_rng(1)
    n = 40_000
    F = rng.dirichlet(np.full(6, 1.2), size=n)
    beta = np.array([-0.0265, -0.038, -0.049, -0.016])
    y = 0.08 + F[:, :4] @ beta + 0.01 * rng.standard_normal(n)
    b_true = _ols_class_effects(F, y)
    b_map = _ols_class_effects(classified_cover(F, _mild_confusion(), 625, rng), y)
    noisy = np.clip(F + 0.05 * rng.standard_normal(F.shape), 0, None)
    noisy /= noisy.sum(axis=1, keepdims=True)
    b_noise = _ols_class_effects(noisy, y)

    contrast = lambda b: b[0] - b[2]  # noqa: E731  eucalyptus - native broadleaf
    assert abs(b_map[0]) > 1.05 * abs(b_true[0])
    assert abs(contrast(b_map)) > 1.1 * abs(contrast(b_true))
    assert abs(b_noise[0]) < abs(b_true[0])  # classical error does attenuate


def _cfg(base, **landcover):
    cfg = copy.deepcopy(base)
    for k, v in landcover.items():
        setattr(cfg.landcover, k, v)
    return cfg


def test_panels_can_be_built_from_classified_maps(tiny_cfg, tiny_land):
    from eucalyptus_impact.pipeline import panel_landscape

    C = _mild_confusion()
    assert tiny_cfg.landcover.panel_cover == "noise"  # the default keeps the Gaussian panels
    assert panel_landscape(tiny_land, C, tiny_cfg) is tiny_land

    cfg = _cfg(tiny_cfg, panel_cover="classified", classified_units_per_cell=(100,))
    land = panel_landscape(tiny_land, C, cfg)
    assert land is not tiny_land
    assert land.cover is tiny_land.cover and land.cover_obs.shape == tiny_land.cover_obs.shape
    assert np.allclose(land.cover_obs.sum(-1), 1)
    assert np.allclose(land.cover_obs * 100, np.round(land.cover_obs * 100))
    panel = build_cell_panel(land)
    lag = land.cover_obs[: len(land.years) - 1].reshape(-1, len(COVER_COLS))
    assert np.allclose(panel[COVER_COLS].to_numpy(), lag)

    with pytest.raises(ValueError):
        panel_landscape(tiny_land, C, _cfg(tiny_cfg, panel_cover="pixels"))


def test_map_error_bias_compares_every_map_with_true_cover(tiny_cfg, tiny_land, monkeypatch):
    """The bias table re-fits the cover effects on true, noisy and classified cover."""
    import eucalyptus_impact.pipeline as pl
    from eucalyptus_impact.causal.dml import EffectEstimate

    seen = []

    def fake_cover_effects(panel, **kw):
        # Stand-in for the DML fits: the mean of each cover column, so each map is identifiable.
        seen.append(panel[COVER_COLS].to_numpy())
        return {
            c: EffectEstimate(c, float(panel[c].mean()), 0.0, "DML-PLR", len(panel))
            for c in ("f_eucalyptus", "f_pine", "f_native_broadleaf", "f_shrub")
        }

    monkeypatch.setattr(pl, "cover_effects", fake_cover_effects)
    cfg = _cfg(tiny_cfg, classified_units_per_cell=(625, 25))
    tab = pl.map_error_bias(tiny_land, _mild_confusion(), cfg)
    assert list(tab["source"]) == ["true", "noise", "classified_625", "classified_25"]
    assert len(seen) == 4
    T = len(tiny_land.years)
    assert np.allclose(seen[0], tiny_land.cover[: T - 1].reshape(-1, 6))
    assert np.allclose(seen[1], tiny_land.cover_obs[: T - 1].reshape(-1, 6))
    assert not np.allclose(seen[2], seen[0], atol=1e-3)
    assert tab.loc[0, "ratio_eucalyptus"] == 1 and tab.loc[0, "ratio_contrast"] == 1
    r = tab.iloc[2]
    assert r["ratio_eucalyptus"] == pytest.approx(r["eucalyptus"] / tab.loc[0, "eucalyptus"])
    assert r["euc_minus_native"] == pytest.approx(r["eucalyptus"] - r["native_broadleaf"])

    # A source already fitted elsewhere is reused, not fitted again.
    noise_fit = fake_cover_effects(build_cell_panel(tiny_land))
    seen.clear()
    pl.map_error_bias(tiny_land, _mild_confusion(), cfg, known={"noise": noise_fit})
    assert len(seen) == 3


# Statements that map error attenuates the effects, as they stood before issue #8. They may come
# back only with a run on classifier-derived fractions that shows it.
ATTENUATION_CLAIMS = [
    "map error attenuates every estimate",
    "map error in *every* cover fraction attenuates the effects",
    "map error attenuates dml estimates",
    "atenúa todos os efectos cara a cero",
    "ese erro atenúa os efectos cara a cero",
    "están atenuados por este erro",
]
CLAIM_FILES = [
    "README.md",
    "docs/SCOPE.md",
    "docs/synthetic_validation/report.md",
    "docs/galicia_real/informe.md",
    "src/eucalyptus_impact/real/brief.py",
    "src/eucalyptus_impact/reporting.py",
]


@pytest.mark.parametrize("rel", CLAIM_FILES)
def test_docs_do_not_claim_map_error_attenuates(rel):
    text = re.sub(r"\s+", " ", (ROOT / rel).read_text(encoding="utf-8")).lower()
    found = [c for c in ATTENUATION_CLAIMS if c in text]
    assert not found, f"{rel} still says map error attenuates: {found}"


def test_panel_docstring_no_longer_claims_real_map_error():
    from eucalyptus_impact.features import panel

    doc = re.sub(r"\s+", " ", panel.__doc__)
    assert "just as it will with real maps" not in doc
    assert "classified_cover" in doc
