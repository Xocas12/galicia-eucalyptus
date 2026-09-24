"""Year-by-year projection of cover, fire and conversion under policy scenarios.

The static projection priced a scenario's final cover change with average fire effects. It
overstated restoration benefits because restored native stands also burn and revert to shrub.
This engine runs the fitted components forward one year at a time, so that feedback is included.

Every component is estimated from the design available on real data, namely two cover maps (t0,
t1) and the burned-area history between them:

- `conversion`: annual share of non-eucalyptus land converted to eucalyptus per cell, a
  gradient-boosting prediction from terrain, access, cover and neighbouring eucalyptus;
- `fire_boost`: DML effect of burned share on that conversion rate (the fire -> plantation loop);
- `source`: which classes the conversions come from (propensity per unit area);
- `burn_loss`: share of native / pine cover turned to shrub per unit of burned area;
- `fire`: baseline burn probability per cell (susceptibility model) plus per-class causal
  effects of cover on burn probability (DML), and year-to-year weather multipliers resampled
  from the observed record.

On the synthetic landscape the engine is checked against the simulator's own scenario runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal.dml import dml_plr, ols
from .data.synthetic import AGRI, EUC, N_CLASSES, NATIVE, OTHER, PINE, SHRUB

COVER = ["f_eucalyptus", "f_pine", "f_native_broadleaf", "f_shrub", "f_agriculture", "f_other"]


@dataclass
class Components:
    conv_rate: np.ndarray  # (n,) annual conversion share of non-eucalyptus land, no recent fire
    fire_boost: float  # added annual conversion share per unit burned share (last 3 years)
    source: np.ndarray  # (6,) relative propensity of each class to be converted
    burn_loss: np.ndarray  # (6,) share of class cover lost to shrub per unit burned share
    p_base: np.ndarray  # (n,) baseline annual burn probability at the start-year cover
    theta: np.ndarray  # (6,) causal effect of each class share on burn probability
    burn_frac: float  # mean burned share of a cell given that it burns
    weather: np.ndarray  # multipliers of burn probability, one per observed year
    diagnostics: dict = field(default_factory=dict)


def estimate_components(
    cells: pd.DataFrame,
    years: int,
    covariates: list[str],
    p_base: np.ndarray,
    theta: np.ndarray,
    burn_frac: float,
    weather: np.ndarray,
    seed: int = 0,
    n_folds: int = 5,
) -> Components:
    """Estimate the land-use components from a two-map window.

    `cells` needs start cover `COVER`, end cover `COVER` with suffix `_end`, `burned_share`
    (sum of annual burned shares in the window), `block`, and the covariates.
    """
    f0 = cells[COVER].to_numpy()
    f1 = cells[[c + "_end" for c in COVER]].to_numpy()
    b = cells["burned_share"].to_numpy()
    non_euc = np.clip(1 - f0[:, EUC], 1e-3, None)
    y = (f1[:, EUC] - f0[:, EUC]) / non_euc / years  # annual conversion share (net)

    X = cells[covariates + COVER[1:]].to_numpy()
    fire_dml = dml_plr(
        y,
        b / years,
        X,
        cells["block"].to_numpy(),
        n_folds,
        seed,
        name="burned share -> conversion rate",
    )
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, min_samples_leaf=50, random_state=seed
    )
    model.fit(np.column_stack([X, b / years]), y)
    conv0 = np.clip(model.predict(np.column_stack([X, np.zeros(len(X))])), 0, 0.2)

    # Source propensities: where the eucalyptus gain came from, per unit of each class's area.
    # Only changes above a detection floor count: map noise otherwise makes small classes look
    # like conversion sources.
    floor = 0.03
    loss = np.clip(f0 - f1, 0, None)
    loss[loss < floor] = 0
    gain = np.clip(f1[:, EUC] - f0[:, EUC], 0, None)
    gain[gain < floor] = 0
    w = gain[:, None] * loss / np.maximum(loss.sum(1, keepdims=True), 1e-9)
    src = w.sum(0) / np.maximum(f0.sum(0), 1e-9)
    src[EUC] = 0
    src = src / max(src.sum(), 1e-12)

    # Burn losses: d f_c = a_c + b_c * f0_c + (-lambda_c) * f0_c * burned_share.
    burn_loss = np.zeros(N_CLASSES)
    for c in (NATIVE, PINE):
        est = ols(
            f1[:, c] - f0[:, c],
            np.column_stack([f0[:, c] * b, f0[:, c]]),
            cells["block"].to_numpy(),
        )
        burn_loss[c] = float(np.clip(-est.estimate, 0, 1))
    return Components(
        conv0,
        max(fire_dml.estimate, 0.0),
        src,
        burn_loss,
        p_base,
        theta,
        burn_frac,
        weather,
        diagnostics={"fire_conversion": fire_dml},
    )


def project(
    f0: np.ndarray,
    comp: Components,
    horizon_years: int,
    policy: str = "bau",
    restore_cells: np.ndarray | None = None,
    cell_area_ha: float = 100.0,
    n_sims: int = 30,
    seed: int = 0,
    theta_se: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate `n_sims` weather/fire paths; returns per-year mean and 5-95% bands.

    policy: "bau", "cap" (no new eucalyptus) or "restore" (cap, plus eucalyptus in
    `restore_cells` returned to native broadleaf at 3/horizon per year). Parameter uncertainty in
    the fire effects is propagated by drawing theta from N(theta, theta_se^2) in each simulation.
    """
    rng = np.random.default_rng(seed)
    n = len(f0)
    # Local scaling of the average causal effects: under a logit-type hazard, the effect of cover
    # on burn probability grows with p(1-p). Without it, restoring a high-risk cell would count
    # the same as restoring a low-risk one.
    pb = np.clip(comp.p_base, 1e-4, 0.5)
    risk_w = (pb * (1 - pb)) / np.mean(pb * (1 - pb))
    rows = []
    for s in range(n_sims):
        th = comp.theta.copy()
        if theta_se is not None:
            th = th + rng.standard_normal(len(th)) * theta_se
        f = f0.copy()
        recent = np.zeros((3, n))
        for t in range(1, horizon_years + 1):
            m = comp.weather[rng.integers(len(comp.weather))]
            p = np.clip(comp.p_base * m + ((f - f0) @ th) * risk_w, 0, 1)
            burned = rng.random(n) < p
            bshare = burned * comp.burn_frac
            # fire damage: native / pine canopy -> shrub
            lost = f * comp.burn_loss[None, :] * bshare[:, None]
            f = f - lost
            f[:, SHRUB] += lost.sum(1)
            # planting
            if policy == "bau":
                rate = np.clip(comp.conv_rate + comp.fire_boost * recent.sum(0), 0, 0.3)
                pull = f * comp.source[None, :]
                pull_tot = pull.sum(1)
                amount = np.minimum(rate * (1 - f[:, EUC]), pull_tot)
                take = pull * (amount / np.maximum(pull_tot, 1e-12))[:, None]
                f = f - take
                f[:, EUC] += take.sum(1)
            if policy == "restore" and restore_cells is not None:
                moved = f[:, EUC] * restore_cells * min(0.95, 3.0 / horizon_years)
                f[:, EUC] -= moved
                f[:, NATIVE] += moved
            recent = np.roll(recent, 1, axis=0)
            recent[0] = bshare
            rows.append(
                {
                    "sim": s,
                    "year": t,
                    "expected_burned_ha": float(np.sum(p) * comp.burn_frac * cell_area_ha),
                    "burned_ha": float(bshare.sum() * cell_area_ha),
                    "eucalyptus_ha": float(f[:, EUC].sum() * cell_area_ha),
                    "native_ha": float(f[:, NATIVE].sum() * cell_area_ha),
                    "shrub_ha": float(f[:, SHRUB].sum() * cell_area_ha),
                }
            )
    df = pd.DataFrame(rows)
    # Per-simulation totals. Every policy draws the same random numbers in the same order (same
    # seed, same number of draws per year), so simulation s of two policies shares its weather
    # and parameter draws, and per-simulation differences give contrast uncertainty bands.
    per_sim = df.groupby("sim").agg(
        mean_burned=("expected_burned_ha", "mean"),
        final_euc=("eucalyptus_ha", "last"),
        final_native=("native_ha", "last"),
    )
    g = df.groupby("year")
    out = g.mean(numeric_only=True).drop(columns="sim")
    for col in ("expected_burned_ha", "eucalyptus_ha", "native_ha"):
        out[col + "_p05"] = g[col].quantile(0.05)
        out[col + "_p95"] = g[col].quantile(0.95)
    out["cum_burned_ha"] = out["expected_burned_ha"].cumsum()
    out = out.reset_index()
    out.attrs["per_sim"] = per_sim
    out.attrs["sim_year_burned"] = df.pivot(
        index="sim", columns="year", values="expected_burned_ha"
    )
    return out


def scenario_contrasts(trajs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scenario minus BAU: eucalyptus and native area at the horizon, and mean yearly burned
    area over the projection, with 5-95% bands from paired simulations."""
    base = trajs["BAU"]
    rows = []
    for name, tr in trajs.items():
        if name == "BAU":
            continue
        row = {
            "scenario": name,
            "d_eucalyptus_ha": tr["eucalyptus_ha"].iloc[-1] - base["eucalyptus_ha"].iloc[-1],
            "d_native_ha": tr["native_ha"].iloc[-1] - base["native_ha"].iloc[-1],
            "d_burned_ha_per_year": (
                tr["expected_burned_ha"].mean() - base["expected_burned_ha"].mean()
            ),
        }
        a, b = tr.attrs.get("per_sim"), base.attrs.get("per_sim")
        if a is not None and b is not None:
            d = a["mean_burned"] - b["mean_burned"]
            row["d_burned_p05"] = float(d.quantile(0.05))
            row["d_burned_p95"] = float(d.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "AGRI",
    "COVER",
    "OTHER",
    "Components",
    "estimate_components",
    "project",
    "scenario_contrasts",
]
