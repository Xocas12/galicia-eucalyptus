"""Policy scenarios to the horizon year: business as usual, cap/moratorium, targeted restoration.

Two projections are produced for each scenario:

1. **Simulated.** The synthetic world is run forward under the policy, with common random numbers
   across scenarios. On real data this does not exist; here it is the ground truth that the model
   projection is checked against.
2. **Causal-model projection.** Scenario minus BAU in cover, priced with the *estimated*
   effects: per-class DML cover effects for fire (risk-weighted) and the
   fitted Budyko curve for runoff. This is the number a real-data study can report.

Targeted restoration picks cells by a model-based priority score (predicted fire probability from
the susceptibility model, weighted by eucalyptus share and catchment aridity), and is compared with
restoring the same eucalyptus area at random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.synthetic import (
    EUC,
    NATIVE,
    PINE,
    Landscape,
    Policy,
    _sigmoid,
    budyko_w,
    conversion_logit,
    fire_logit,
    fu_et,
    neighbour_mean,
    transition_matrices,
)
from .features.panel import COVER_COLS
from .geo.raster_ops import zonal_mean


@dataclass
class ScenarioSpec:
    name: str
    conversion_multiplier: float = 1.0
    freeze: bool = False
    restore: str | None = None  # None | "targeted" | "random"


SCENARIOS = [
    ScenarioSpec("BAU"),
    ScenarioSpec("Cap / moratorium", freeze=True),
    ScenarioSpec("Targeted restoration", freeze=True, restore="targeted"),
    ScenarioSpec("Random restoration", freeze=True, restore="random"),
]


def final_year_features(land: Landscape, features: list[str]) -> pd.DataFrame:
    """Feature frame for the last observed year, in the susceptibility model's layout."""
    st = land.static
    df = pd.DataFrame({k: st[k] for k in st if np.ndim(st[k]) == 1})
    df["log_pop"] = np.log(st["pop_density"])
    for j, c in enumerate(COVER_COLS):
        df[c] = land.cover_obs[-1, :, j]
    df["fwi"] = st["fwi_base"]
    df["recent_fire"] = land.burned[-3:].any(axis=0).astype(float)
    return df[features]


def priority_score(land: Landscape, fire_model, features) -> np.ndarray:
    """Expected benefit per restored hectare: fire risk plus water stress, per unit eucalyptus."""
    p = fire_model.predict_proba(final_year_features(land, features).to_numpy())[:, 1]
    arid = land.static["pet_mean"] / land.static["precip_mean"]
    score = p / p.mean() + 0.5 * arid / arid.mean()
    return score * land.cover_obs[-1, :, EUC]


def select_cells(land: Landscape, share: float, score: np.ndarray | None, rng) -> np.ndarray:
    """Cells whose eucalyptus adds up to `share` of total eucalyptus area, best-scored first."""
    euc = land.cover[-1, :, EUC]
    order = np.argsort(-score) if score is not None else rng.permutation(land.n)
    cum = np.cumsum(euc[order])
    k = int(np.searchsorted(cum, share * euc.sum())) + 1
    sel = np.zeros(land.n, bool)
    sel[order[:k]] = True
    return sel


def project(land: Landscape, spec: ScenarioSpec, horizon: int, sel: np.ndarray | None, seed: int):
    """Run the synthetic world forward from the last observed year under one policy."""
    tp = land.truth
    rng = np.random.default_rng(seed)
    st = dict(land.static)
    lp = np.log(st["pop_density"])
    st["pop_z"] = (lp - lp.mean()) / lp.std()
    st["slope_z"] = (st["slope"] - st["slope"].mean()) / st["slope"].std()
    years = list(range(land.years[-1] + 1, horizon + 1))
    H = len(years)
    restoration = None
    if sel is not None:
        restoration = np.where(sel, min(0.95, 3.0 / H), 0.0)
    pol = Policy(spec.conversion_multiplier, spec.freeze, restoration)
    J = land.joint.copy()
    fire_hist = land.extra["fire_hist"].copy()
    from .geo.raster_ops import gaussian_field

    sig = max(1.0, 16000 / land.grid.resolution_m)
    rows = []
    area = land.cell_area_ha()
    for year in years:
        f = J.sum(axis=1)
        anom = rng.normal(0, 0.8)
        fwi = (
            st["fwi_base"]
            + 0.8 * anom
            + 0.3 * gaussian_field(land.land.shape, sig, rng).ravel()[land.land_idx]
        )
        p = _sigmoid(fire_logit(tp, f, fwi, st))
        burned = rng.random(land.n) < p
        bfrac = rng.beta(2, 2, land.n)
        ne = neighbour_mean(f[:, EUC], land.land, land.land_idx, land.grid.resolution_m)
        z = conversion_logit(
            tp,
            st,
            ne,
            fire_hist.any(axis=0).astype(float),
            year,
            tp.conv_noise * rng.standard_normal(land.n),
            pol,
        )
        M = transition_matrices(tp, f, burned, bfrac, 0.25 * _sigmoid(z), pol)
        J = np.einsum("nab,nbc->nac", J, M)
        f_new = J.sum(axis=1)
        P = st["precip_mean"] * np.exp(0.15 * rng.normal())
        pet = st["pet_mean"]
        q = P - fu_et(P, pet, budyko_w(tp, f_new))
        fire_hist = np.roll(fire_hist, 1, axis=0)
        fire_hist[0] = burned
        rows.append(
            {
                "scenario": spec.name,
                "year": year,
                "expected_burned_ha": float(np.sum(p * 0.5) * area),
                "eucalyptus_ha": float(f_new[:, EUC].sum() * area),
                "native_ha": float(f_new[:, NATIVE].sum() * area),
                "runoff_mm": float(q.mean()),
                "catchment_runoff_mm": zonal_mean(q, land.catchment, land.n_catchments),
                "f_final": f_new,
            }
        )
    return rows


def dynamic_synthetic(land: Landscape, results: dict, cfg, sel: dict) -> dict:
    """Estimate the dynamic components from a 7-year two-map window and project each scenario."""
    from .dynamic import COVER, estimate_components, project

    years = land.years
    y1 = years[-1]
    y0 = max(years[0], y1 - 7)
    t0, t1 = years.index(y0), years.index(y1)
    st = land.static
    cells = pd.DataFrame(
        {
            "elev": st["elev"],
            "slope": st["slope"],
            "continentality": st["continentality"],
            "dist_mill_km": st["dist_mill_km"],
            "dist_road_km": st["dist_road_km"],
            "log_pop": np.log(st["pop_density"]),
            "precip_mean": st["precip_mean"],
            "block": land.block,
        }
    )
    cells["neigh_euc"] = neighbour_mean(
        land.cover_obs[t0, :, EUC], land.land, land.land_idx, land.grid.resolution_m
    )
    for j, c in enumerate(COVER):
        cells[c] = land.cover_obs[t0, :, j]
        cells[c + "_end"] = land.cover_obs[t1, :, j]
    # Burned share: burn events times the mean burned share of a burning cell (0.5 here).
    cells["burned_share"] = 0.5 * land.burned[t0 + 1 : t1 + 1].sum(axis=0)
    fire_fit = results["fire_susceptibility"]
    p_base = fire_fit["model"].predict_proba(
        final_year_features(land, fire_fit["features"]).to_numpy()
    )[:, 1]
    ce = results["fire_effects"]["cover_effects"]
    theta = np.array([ce[c].estimate if c in ce else 0.0 for c in COVER])
    theta_se = np.array([ce[c].se if c in ce else 0.0 for c in COVER])
    rates = land.burned.mean(axis=1)
    weather = rates / rates.mean()
    comp = estimate_components(
        cells,
        t1 - t0,
        [
            "elev",
            "slope",
            "continentality",
            "dist_mill_km",
            "dist_road_km",
            "log_pop",
            "precip_mean",
            "neigh_euc",
        ],
        p_base,
        theta,
        0.5,
        weather,
        seed=cfg.seed,
        n_folds=cfg.causal.n_folds,
    )
    horizon = cfg.scenarios.horizon - years[-1]
    f0 = land.cover_obs[-1]
    kw = dict(cell_area_ha=land.cell_area_ha(), n_sims=20, seed=cfg.seed, theta_se=theta_se)
    trajs = {
        "BAU": project(f0, comp, horizon, "bau", **kw),
        "Cap / moratorium": project(f0, comp, horizon, "cap", **kw),
        "Targeted restoration": project(
            f0, comp, horizon, "restore", sel["targeted"].astype(float), **kw
        ),
        "Random restoration": project(
            f0, comp, horizon, "restore", sel["random"].astype(float), **kw
        ),
    }
    return {"components": comp, "trajectories": trajs}


def run_scenarios(land: Landscape, results: dict, cfg) -> dict:
    fire_fit = results["fire_susceptibility"]
    rng = np.random.default_rng(cfg.seed + 7)
    score = priority_score(land, fire_fit["model"], fire_fit["features"])
    share = cfg.scenarios.restoration_share
    sel = {
        "targeted": select_cells(land, share, score, rng),
        "random": select_cells(land, share, None, rng),
    }
    n_sims = 6
    all_rows = []
    final_f = {}
    for spec in SCENARIOS:
        fs = []
        for s in range(n_sims):  # same seeds across scenarios: common random numbers
            rows = project(
                land,
                spec,
                cfg.scenarios.horizon,
                sel[spec.restore] if spec.restore else None,
                seed=cfg.seed * 100 + s,
            )
            for r in rows:
                r["sim"] = s
            fs.append(rows[-1]["f_final"])
            all_rows += rows
        final_f[spec.name] = np.mean(fs, axis=0)
    traj = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("catchment_runoff_mm", "f_final")}
            for r in all_rows
        ]
    )
    mean_traj = (
        traj.groupby(["scenario", "year"]).mean(numeric_only=True).drop(columns="sim").reset_index()
    )

    # Scenario contrasts at the horizon: simulated (truth) vs causal-model projection.
    fe = results["fire_effects"]
    cover_theta = {COVER_COLS.index(k): e.estimate for k, e in fe["cover_effects"].items()}
    # The DML estimate is an average effect. Under a logit-type hazard the local effect scales with
    # p(1-p), so the projection re-weights it by predicted risk; otherwise targeting that works by
    # picking high-risk cells would be invisible to the projection.
    p_hat = fire_fit["model"].predict_proba(
        final_year_features(land, fire_fit["features"]).to_numpy()
    )[:, 1]
    risk_w = p_hat * (1 - p_hat) / np.mean(p_hat * (1 - p_hat))
    # Water is projected through the fitted Budyko curve, which prices each cover change by its
    # own estimated effect on the land-surface parameter w.
    bp = results["water"]["budyko_params"]
    th = np.array([bp[k][0] for k in ("w0", "w_euc", "w_pine", "w_native")])
    P_clim, PET_clim = land.static["precip_mean"], land.static["pet_mean"]

    def q_hat(F):
        w = th[0] + F[:, [EUC, PINE, NATIVE]] @ th[1:]
        return float(np.mean(P_clim - fu_et(P_clim, PET_clim, np.maximum(w, 1.01))))

    area = land.cell_area_ha()
    last = mean_traj[mean_traj["year"] == cfg.scenarios.horizon].set_index("scenario")
    base = last.loc["BAU"]
    comp = []
    for spec in SCENARIOS[1:]:
        dF = final_f[spec.name] - final_f["BAU"]
        d_p = sum(theta * dF[:, j] for j, theta in cover_theta.items())
        comp.append(
            {
                "scenario": spec.name,
                "d_eucalyptus_ha": last.loc[spec.name, "eucalyptus_ha"] - base["eucalyptus_ha"],
                "d_burned_ha_simulated": last.loc[spec.name, "expected_burned_ha"]
                - base["expected_burned_ha"],
                "d_burned_ha_model": float(0.5 * np.sum(d_p * risk_w) * area),
                "d_runoff_mm_simulated": last.loc[spec.name, "runoff_mm"] - base["runoff_mm"],
                "d_runoff_mm_model": q_hat(final_f[spec.name]) - q_hat(final_f["BAU"]),
            }
        )
    dyn = dynamic_synthetic(land, results, cfg, sel)
    comp = pd.DataFrame(comp)
    # Period means (2025 to horizon) are compared, not the horizon year alone, whose weather draw
    # differs between the simulator and the dynamic engine.
    sim_mean = mean_traj.groupby("scenario")["expected_burned_ha"].mean()
    dyn_mean = {k: v["expected_burned_ha"].mean() for k, v in dyn["trajectories"].items()}
    comp.insert(
        comp.columns.get_loc("d_burned_ha_model") + 1,
        "d_burned_ha_mean_simulated",
        [sim_mean[s] - sim_mean["BAU"] for s in comp["scenario"]],
    )
    comp.insert(
        comp.columns.get_loc("d_burned_ha_mean_simulated") + 1,
        "d_burned_ha_mean_dynamic",
        [dyn_mean[s] - dyn_mean["BAU"] for s in comp["scenario"]],
    )
    return {
        "trajectories": mean_traj,
        "contrasts": comp,
        "dynamic": dyn,
        "priority_score": score,
        "selected_targeted": sel["targeted"],
    }
