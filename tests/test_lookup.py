"""Coordinate lookup: pixel mapping, risk classes, eucalyptus contribution, text output."""

import numpy as np

from eucalyptus_impact.real import lookup as LK
from eucalyptus_impact.real.common import GRID_40M


def test_to_pixel_lands_inside_the_grid_for_santiago():
    row, col, x, y = LK.to_pixel(42.8806, -8.5446)
    ny, nx = GRID_40M.shape
    assert 0 <= row < ny and 0 <= col < nx
    assert 530_000 < x < 545_000 and 4_740_000 < y < 4_755_000


def test_risk_class_thresholds():
    assert [LK.risk_class(p) for p in (0.1, 0.5, 0.85, 0.99)] == [
        "low",
        "moderate",
        "high",
        "very high",
    ]


def test_eucalyptus_contribution_scales_with_share_and_combines_ses():
    eff = {
        "cover_effects": {
            "f_eucalyptus": [-0.02, 0.01],
            "f_native_broadleaf": [-0.05, 0.01],
        }
    }
    a = LK.eucalyptus_contribution(eff, 0.5)
    assert np.isclose(a["estimate"], -0.01)
    assert np.allclose(a["ci"], [-0.01 - 1.96 * 0.005, -0.01 + 1.96 * 0.005])
    assert a["significant"]
    b = LK.eucalyptus_contribution(eff, 0.5, "native_broadleaf")
    assert np.isclose(b["estimate"], 0.015)
    assert np.isclose(b["ci"][1] - b["estimate"], 1.96 * np.hypot(0.01, 0.01) * 0.5)


def _result():
    cov = dict.fromkeys(LK.CLASSES, 0.0) | {"eucalyptus": 0.6, "shrub": 0.4}
    return {
        "lat": 42.9,
        "lon": -8.5,
        "utm29_x": 540000,
        "utm29_y": 4750000,
        "pixel_40m": {
            "class_2024": "eucalyptus",
            "class_2024_confidence_pct": 81,
            "class_2017": "eucalyptus",
            "worldcover_2021": "tree cover",
            "elevation_m": 310.0,
            "tree_cover_loss_year": 2012,
            "burnt_years": [],
        },
        "cell_1km": {
            "cover_2024": cov,
            "cover_2017": cov,
            "distance_to_sea_km": 25.0,
            "slope_deg": 8.0,
            "burnt_share_2018_2023": 0.0,
        },
        "risk": {
            "annual_fire_probability": 0.03,
            "galicia_mean": 0.02,
            "percentile": 0.9,
            "class": "high",
            "restoration_priority_percentile": 0.95,
            "in_targeted_restoration_set": True,
            "eucalyptus_contribution_vs_agriculture": {
                "estimate": -0.016,
                "ci": [-0.028, -0.005],
                "significant": True,
            },
            "eucalyptus_contribution_vs_native": {
                "estimate": 0.023,
                "ci": [-0.001, 0.047],
                "significant": False,
            },
            "model_auc": 0.8,
        },
        "ifn3_plot": {"distance_m": 400.0, "genera": ["Eucalyptus", "Ulex"]},
        "upstream": {"area_km2": 12.3, "truncated": False, "cover_2024": cov},
    }


def test_format_lookup_english_and_galician():
    en = LK.format_lookup(_result(), "en")
    assert "Fire risk" in en and "high" in en and "not distinguishable from zero" in en
    gl = LK.format_lookup(_result(), "gl")
    assert "Risco de incendio" in gl and "eucalipto 60,0%" in gl and "alto" in gl
    assert "Fire risk" not in gl
