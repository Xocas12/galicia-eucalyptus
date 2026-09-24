"""End-to-end orchestration of all workstreams (docs/SCOPE.md section 4)."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .config import StudyConfig
from .data.synthetic import (
    CLASSES,
    EUC,
    N_CLASSES,
    NATIVE,
    Landscape,
    simulate_landscape,
    simulate_spectral_pixels,
)
from .features.panel import build_catchment_panel, build_cell_cross_section, build_cell_panel
from .features.spectral import harmonic_features
from .models.conversion import attribute_loss_events, conversion_drivers, transition_matrix
from .models.fire import estimate_fire_effects, fit_susceptibility
from .models.hydrology import water_effects
from .models.landcover import olofsson_area, stratified_reference_sample, train_species_classifier
from .scenarios import run_scenarios

log = logging.getLogger("eucalyptus_impact")

CHANGE_CLASSES = ["native->eucalyptus", "other->eucalyptus", "eucalyptus stable", "other"]


def _change_class(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), 3)
    out[(a == NATIVE) & (b == EUC)] = 0
    out[(a != NATIVE) & (a != EUC) & (b == EUC)] = 1
    out[(a == EUC) & (b == EUC)] = 2
    return out


def run_landcover(land: Landscape, cfg: StudyConfig) -> dict:
    """Species mapping, area estimation and change (transition) accounting."""
    lc = cfg.landcover
    rng = np.random.default_rng(cfg.seed + 1)
    joint = land.joint.reshape(land.n, -1)

    def draw_pixels(k):
        cells = rng.integers(0, land.n, k)
        u = rng.random(k)[:, None]
        pair = (np.cumsum(joint[cells], axis=1) < u).sum(axis=1).clip(max=N_CLASSES**2 - 1)
        return cells, pair // N_CLASSES, pair % N_CLASSES

    # Training data: labelled pixels (in reality MFE polygons / IFN plots) for the last year.
    cells, _, y = draw_pixels(lc.n_train_pixels)
    series, months = simulate_spectral_pixels(y, land.block[cells], rng)
    X, feat_names = harmonic_features(series, months)
    clf = train_species_classifier(X, y, land.block[cells], cfg.causal.n_folds, cfg.seed)

    # Wall-to-wall map (a large pixel sample standing in for the full map) at both dates.
    cells, c0, c1 = draw_pixels(lc.n_map_pixels)
    s0, m = simulate_spectral_pixels(c0, land.block[cells], rng)
    s1, _ = simulate_spectral_pixels(c1, land.block[cells], rng)
    map0 = clf.model.predict(harmonic_features(s0, m)[0])
    map1 = clf.model.predict(harmonic_features(s1, m)[0])

    total_ha = land.n * land.cell_area_ha()
    true_area = land.cover[-1].sum(axis=0) * land.cell_area_ha()
    ref = stratified_reference_sample(map1, lc.reference_per_class, rng)
    area = olofsson_area(map1, map1[ref], c1[ref], N_CLASSES, total_ha)
    area.insert(1, "name", CLASSES)
    area["true_area_ha"] = true_area

    chg_map = _change_class(map0, map1)
    chg_true = _change_class(c0, c1)
    ref_c = stratified_reference_sample(chg_map, lc.reference_per_class, rng)
    change = olofsson_area(chg_map, chg_map[ref_c], chg_true[ref_c], 4, total_ha)
    change.insert(1, "name", CHANGE_CLASSES)
    J_area = land.joint.sum(axis=0) * land.cell_area_ha()
    true_change = np.array(
        [
            J_area[NATIVE, EUC],
            J_area[:, EUC].sum() - J_area[NATIVE, EUC] - J_area[EUC, EUC],
            J_area[EUC, EUC],
            0.0,
        ]
    )
    true_change[3] = total_ha - true_change[:3].sum()
    change["true_area_ha"] = true_change

    scale = total_ha / lc.n_map_pixels
    return {
        "classifier": clf,
        "feature_names": feat_names,
        "area": area,
        "change": change,
        "transition_map": transition_matrix(map0, map1) * scale,
        "transition_true": pd.DataFrame(J_area, index=CLASSES, columns=CLASSES),
    }


def run(cfg: StudyConfig, land: Landscape | None = None) -> dict:
    t0 = time.time()
    res: dict = {"config": cfg}
    land = land or simulate_landscape(cfg)
    res["land"] = land
    log.info(
        "simulated %d land cells x %d years (%.1fs)", land.n, len(land.years), time.time() - t0
    )

    res["landcover"] = run_landcover(land, cfg)
    log.info("landcover done (%.1fs)", time.time() - t0)

    panel = build_cell_panel(land)
    catch = build_catchment_panel(land)
    xsec = build_cell_cross_section(land)
    res["panel_rows"] = len(panel)

    res["loss_attribution"] = attribute_loss_events(panel)
    res["conversion"] = conversion_drivers(
        panel, cfg.causal.n_folds, cfg.seed, max_rows=cfg.causal.max_rows
    )
    log.info("conversion done (%.1fs)", time.time() - t0)

    # Map-error variance of the eucalyptus fraction, as an accuracy assessment against reference
    # plots would measure it (here: 500 cells where mapped and true cover are both known).
    rng = np.random.default_rng(cfg.seed + 2)
    ref_t = rng.integers(0, len(land.years), 500)
    ref_c = rng.integers(0, land.n, 500)
    err = land.cover_obs[ref_t, ref_c, EUC] - land.cover[ref_t, ref_c, EUC]
    res["map_error_var"] = me_var = float(np.var(err))

    kw = dict(n_folds=cfg.causal.n_folds, seed=cfg.seed, max_rows=cfg.causal.max_rows)
    res["fire_susceptibility"] = fit_susceptibility(panel, **kw)
    res["fire_effects"] = estimate_fire_effects(panel, **kw, d_error_var=me_var)
    log.info("fire done (%.1fs)", time.time() - t0)

    water = water_effects(catch, xsec, cfg.causal.n_folds, cfg.seed, d_error_var=me_var)
    water["soil_moisture_dml"].truth = land.truth.sm_euc
    if water["soil_moisture_dml_me"] is not None:
        water["soil_moisture_dml_me"].truth = land.truth.sm_euc
    water["soil_moisture_naive"].truth = land.truth.sm_euc
    t_m, c_m = water["matching_balance"].attrs["pairs"]
    sub = xsec[(xsec["f_eucalyptus"] > 0.4) | (xsec["f_eucalyptus"] < 0.1)].reset_index(drop=True)
    dose = sub["f_eucalyptus"].to_numpy()
    water["soil_moisture_matching"].truth = float(
        land.truth.sm_euc * np.mean(dose[t_m] - dose[c_m])
    )
    res["water"] = water
    res["catchment_panel"] = catch
    log.info("water done (%.1fs)", time.time() - t0)

    res["scenarios"] = run_scenarios(land, res, cfg)
    log.info("scenarios done (%.1fs)", time.time() - t0)
    res["runtime_s"] = time.time() - t0
    return res


def effect_table(res: dict) -> pd.DataFrame:
    """All naive and causal estimates side by side with the simulator truth."""
    f, w, c = res["fire_effects"], res["water"], res["conversion"]
    ests = [
        f["occurrence_naive"],
        f["occurrence_dml"],
        f.get("occurrence_dml_me"),
        f["severity_naive"],
        f["severity_dml"],
        f.get("severity_dml_me"),
        c["naive"],
        c["dml"],
        w["runoff_naive"],
        w["runoff_twfe"],
        w["runoff_budyko"],
        w["low_flow_twfe"],
        w["soil_moisture_naive"],
        w["soil_moisture_dml"],
        w.get("soil_moisture_dml_me"),
        w["soil_moisture_matching"],
        f.get("placebo"),
    ]
    ests = [e for e in ests if e is not None]
    rows = []
    for e in ests:
        lo, hi = e.ci95
        rows.append(
            {
                "effect": e.name,
                "method": e.method,
                "estimate": e.estimate,
                "se": e.se,
                "ci_low": lo,
                "ci_high": hi,
                "truth": e.truth,
                "covers_truth": e.covers_truth,
                "n": e.n,
            }
        )
    return pd.DataFrame(rows)
