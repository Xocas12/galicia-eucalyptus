"""End-to-end orchestration of all workstreams (docs/SCOPE.md section 4)."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .causal.dml import dml_plr, group_effects, reclustered_se
from .causal.sensitivity import bias_bound, robustness_value
from .causal.simex import simex
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
from .features.panel import (
    COVER_COLS,
    build_catchment_panel,
    build_cell_cross_section,
    build_cell_panel,
)
from .features.spectral import harmonic_features
from .models.conversion import attribute_loss_events, conversion_drivers, transition_matrix
from .models.fire import CONFOUNDERS, estimate_fire_effects, fit_susceptibility
from .models.hydrology import SM_COVARIATES, water_effects
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
    # Map error is corrected with SIMEX in run_robustness. The single-variance regression
    # calibration (dml_plr's d_error_var) over-corrects once the other, equally noisy, cover
    # fractions are partialled out, so it is not used for headline numbers.
    res["fire_effects"] = estimate_fire_effects(panel, **kw)
    log.info("fire done (%.1fs)", time.time() - t0)

    water = water_effects(catch, xsec, cfg.causal.n_folds, cfg.seed)
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

    res["robustness"] = run_robustness(res, xsec, cfg, me_var)
    log.info("robustness done (%.1fs)", time.time() - t0)

    res["scenarios"] = run_scenarios(land, res, cfg)
    log.info("scenarios done (%.1fs)", time.time() - t0)
    res["runtime_s"] = time.time() - t0
    return res


def _blocks(x, y, size_km: float) -> np.ndarray:
    k = size_km * 1000
    return (np.floor(x / k) * 100_000 + np.floor(y / k)).astype(np.int64)


def run_robustness(res: dict, xsec: pd.DataFrame, cfg: StudyConfig, me_var: float) -> dict:
    """Heterogeneity, unobserved-confounding sensitivity, SE block-size sensitivity and SIMEX."""
    fe = res["fire_effects"]
    occ = fe.get("occurrence_dml_me") or fe["occurrence_dml"]
    fr = fe["occurrence_frame"]
    out: dict = {}

    # Group effects: where does eucalyptus raise fire risk most?
    truth = fr["true_te_fire"] if "true_te_fire" in fr else None
    cont = pd.qcut(fr["continentality"], 3, labels=["1 coast", "2 transition", "3 interior"])
    fwi = pd.qcut(fr["fwi"], 3, labels=["1 low FWI", "2 mid FWI", "3 high FWI"])
    out["gate_region"] = group_effects(occ, cont.astype(str), truth)
    out["gate_fwi"] = group_effects(occ, fwi.astype(str), truth)

    # How strong would an unmapped confounder have to be?
    rows = []
    sev = fe.get("severity_dml_me") or fe["severity_dml"]
    sm = res["water"].get("soil_moisture_dml_me") or res["water"]["soil_moisture_dml"]
    for est in (occ, sev, sm):
        rows.append(
            {
                "effect": est.name,
                "estimate": est.estimate,
                "rv_estimate": robustness_value(est),
                "rv_ci": robustness_value(est, alpha_z=1.96),
                "max_bias_r2_0.02": bias_bound(est, 0.02, 0.02),
                "max_bias_r2_0.05": bias_bound(est, 0.05, 0.05),
            }
        )
    out["sensitivity"] = pd.DataFrame(rows)

    # Spatial autocorrelation: does the SE survive larger clusters?
    if {"x", "y"} <= set(fr.columns):
        out["se_by_block"] = pd.DataFrame(
            [
                {"block_km": b, "se": reclustered_se(occ, _blocks(fr["x"], fr["y"], b))}
                for b in (5, 10, 20, 40, 80)
            ]
        )

    # SIMEX: correct map error in *all* cover fractions, not only the treatment.
    if cfg.causal.simex:

        def sm_est(d):
            return dml_plr(
                d["soil_moisture"],
                d["f_eucalyptus"],
                d[SM_COVARIATES].to_numpy(),
                d["block"],
                cfg.causal.n_folds,
                cfg.seed,
                name="eucalyptus -> summer soil moisture",
            )

        sm_simex, sm_path = simex(sm_est, xsec, COVER_COLS, me_var, seed=cfg.seed)
        sm_simex.truth = res["land"].truth.sm_euc

        sev_df = fe["severity_frame"]

        def sev_est(d):
            return dml_plr(
                d["dnbr"],
                d["f_eucalyptus"],
                d[CONFOUNDERS].to_numpy(),
                d["block"],
                cfg.causal.n_folds,
                cfg.seed,
                name="eucalyptus -> dNBR",
            )

        sev_simex, sev_path = simex(sev_est, sev_df, COVER_COLS, me_var, seed=cfg.seed)
        sev_simex.truth = fe["severity_dml"].truth
        out["simex"] = {"soil_moisture": sm_simex, "severity": sev_simex}
        out["simex_path"] = pd.concat(
            [sm_path.assign(effect=sm_simex.name), sev_path.assign(effect=sev_simex.name)]
        )
    return out


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
    ests += list(res.get("robustness", {}).get("simex", {}).values())
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
