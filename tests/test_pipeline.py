import json
import re

from eucalyptus_impact import i18n_gl
from eucalyptus_impact.data.catalog import catalog_frame
from eucalyptus_impact.pipeline import effect_table, run
from eucalyptus_impact.reporting import write_report


def test_catalog():
    df = catalog_frame()
    assert df["key"].is_unique and len(df) >= 15


def test_end_to_end(tiny_cfg, tiny_land, tmp_path):
    res = run(tiny_cfg, land=tiny_land)
    tab = effect_table(res).set_index(["effect", "method"])
    occ_dml = tab.loc[("eucalyptus -> P(burn)", "DML-PLR")]
    assert occ_dml["ci_low"] < occ_dml["truth"] < occ_dml["ci_high"]
    sev_dml = tab.loc[("eucalyptus -> dNBR", "DML-PLR")]
    sev_ols = tab.loc[("eucalyptus -> dNBR", "OLS")]
    # The naive severity regression is confounded by climate; DML lands closer to the truth.
    assert abs(sev_dml["estimate"] - sev_dml["truth"]) < abs(sev_ols["estimate"] - sev_ols["truth"])
    sm = tab.loc[("eucalyptus -> summer soil moisture", "DML-PLR")]
    assert sm["ci_low"] < sm["truth"] < sm["ci_high"]
    assert res["fire_susceptibility"]["spatial_cv_auc"] > 0.65
    assert set(res["scenarios"]["contrasts"]["scenario"]) == {
        "Cap / moratorium",
        "Targeted restoration",
        "Random restoration",
    }

    rob = res["robustness"]
    assert set(rob["gate_region"]["group"]) == {"1 coast", "2 transition", "3 interior"}
    assert (rob["sensitivity"]["rv_estimate"].between(0, 1)).all()
    assert {"soil_moisture", "severity"} <= set(rob["simex"])

    path = write_report(res, tmp_path)
    assert path.exists()
    for f in (
        "effects.png",
        "maps.png",
        "scenarios.png",
        "metrics.json",
        "effects.csv",
        "fire_gates.png",
        "simex.png",
        "restoration_priority.csv",
    ):
        assert (tmp_path / f).exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["land_cells"] == tiny_land.n
    _assert_galician_only(path.read_text())


ENGLISH = {
    "the",
    "and",
    "of",
    "with",
    "from",
    "to",
    "is",
    "are",
    "by",
    "vs",
    "truth",
    "estimate",
    "effect",
    "effects",
    "share",
    "report",
    "fire",
    "burn",
    "burned",
    "runoff",
    "eucalyptus",
    "native",
    "broadleaf",
    "shrub",
    "pine",
    "scenario",
    "restoration",
    "targeted",
    "random",
    "yes",
    "map",
    "area",
    "change",
    "water",
    "soil",
    "moisture",
    "coast",
    "low",
    "high",
    "method",
    "naive",
    "matching",
    "balance",
    "sensitivity",
    "robustness",
}


def _assert_galician_only(text: str):
    """No untranslated identifier and no common English word in the report prose."""
    assert not i18n_gl.MISSING, f"untranslated labels: {sorted(i18n_gl.MISSING)}"
    prose = re.sub(r"\(([\w./-]+\.png)\)", "", text)  # image file names
    prose = re.sub(r"`[^`]*`", "", prose)  # inline code (file paths)
    words = {w.lower() for w in re.findall(r"[^\W\d_]+", prose)}
    assert not words & ENGLISH, f"English words in report: {sorted(words & ENGLISH)}"


def test_galician_number_format():
    assert i18n_gl.num(0.02209) == "0,02209"
    assert i18n_gl.num(-35872.76) == "-35 873"
    assert i18n_gl.num(1234.5) == "1 234"
