"""Cell-year and catchment-year analysis panels built from a landscape.

Treatments always use *observed* (mapped) cover, never the simulator's true cover, so
classification error flows into the estimates just as it will with real maps. Cover enters fire
models lagged by one year (cover at t-1 -> fire at t) to keep fire -> cover reverse causality out of
the treatment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.synthetic import CLASSES, EUC, NATIVE, PINE, SHRUB, Landscape
from ..geo.raster_ops import zonal_mean

STATIC_COVARIATES = [
    "elev",
    "slope",
    "dist_coast_km",
    "continentality",
    "precip_mean",
    "pet_mean",
    "summer_temp",
    "pop_density",
    "dist_road_km",
    "dist_mill_km",
]
COVER_COLS = [f"f_{c}" for c in CLASSES]


def build_cell_panel(land: Landscape) -> pd.DataFrame:
    """One row per land cell and year from the second year on, with lagged observed cover."""
    T, n = land.burned.shape
    t_idx = np.repeat(np.arange(1, T), n)
    c_idx = np.tile(np.arange(n), T - 1)
    df = pd.DataFrame(
        {
            "cell": c_idx,
            "year": np.asarray(land.years)[t_idx],
            "block": land.block[c_idx],
            "catchment": land.catchment[c_idx],
        }
    )
    for k in STATIC_COVARIATES:
        df[k] = land.static[k][c_idx]
    df["log_pop"] = np.log(df["pop_density"])
    lag = land.cover_obs[t_idx - 1, c_idx]  # (rows, 6)
    for j, col in enumerate(COVER_COLS):
        df[col] = lag[:, j]
    now = land.cover_obs[t_idx, c_idx]
    lead = t_idx + 5
    ok = lead < T
    df["f_eucalyptus_lead5"] = np.nan
    df.loc[ok, "f_eucalyptus_lead5"] = land.cover_obs[lead[ok], c_idx[ok], EUC]
    df["d_euc"] = now[:, EUC] - lag[:, EUC]
    df["d_native"] = now[:, NATIVE] - lag[:, NATIVE]
    df["d_pine"] = now[:, PINE] - lag[:, PINE]
    df["d_shrub"] = now[:, SHRUB] - lag[:, SHRUB]
    df["fwi"] = land.fwi[t_idx, c_idx]
    df["precip"] = land.precip[t_idx, c_idx]
    df["burned"] = land.burned[t_idx, c_idx].astype(int)
    df["dnbr"] = land.dnbr[t_idx, c_idx]
    df["recent_fire"] = land.extra["recent_fire"][t_idx, c_idx]
    df["neigh_euc"] = land.extra["neigh_euc"][t_idx, c_idx]
    df["tree_loss"] = sum(v[t_idx, c_idx] for v in land.loss.values())
    for k, v in land.loss.items():
        df[f"true_loss_{k}"] = v[t_idx, c_idx]
    # Conversion outcome: eucalyptus gained this year as a share of the non-eucalyptus land.
    # Net change, deliberately unclipped: clipping noisy map differences at zero biases the mean.
    df["conv_share"] = df["d_euc"] / np.clip(1 - df["f_eucalyptus"], 1e-3, None)
    for k, v in land.true_te.items():
        df[f"true_te_{k}"] = v[t_idx, c_idx]
    return df


def build_catchment_panel(land: Landscape) -> pd.DataFrame:
    """Catchment-year runoff (gauge-like, with 3% measurement noise) and cover shares."""
    rng = np.random.default_rng(12345)
    rows = []
    K = land.n_catchments
    for t, yr in enumerate(land.years):
        q = zonal_mean(land.runoff[t], land.catchment, K)
        p = zonal_mean(land.precip[t], land.catchment, K)
        pet = zonal_mean(land.pet[t], land.catchment, K)
        te = zonal_mean(land.true_te["water"][t], land.catchment, K)
        cov = [zonal_mean(land.cover_obs[t, :, j], land.catchment, K) for j in range(len(CLASSES))]
        q_obs = q * (1 + 0.03 * rng.standard_normal(K))
        # Summer low flow: a baseflow fraction depleted by deep-rooted plantations (stylised).
        low = 0.12 * q_obs * (1 - 0.45 * cov[EUC]) * (1 + 0.05 * rng.standard_normal(K))
        for k in range(K):
            rows.append(
                {
                    "catchment": k,
                    "year": yr,
                    "runoff": q_obs[k],
                    "low_flow": low[k],
                    "precip": p[k],
                    "pet": pet[k],
                    "true_te_water": te[k],
                    **{COVER_COLS[j]: cov[j][k] for j in range(len(CLASSES))},
                }
            )
    df = pd.DataFrame(rows)
    df["runoff_ratio"] = df["runoff"] / df["precip"]
    df["aridity"] = df["pet"] / df["precip"]
    return df


def build_cell_cross_section(land: Landscape) -> pd.DataFrame:
    """Final-year cross-section for soil-moisture analysis."""
    df = pd.DataFrame({"cell": np.arange(land.n), "block": land.block})
    for k in STATIC_COVARIATES:
        df[k] = land.static[k]
    df["log_pop"] = np.log(df["pop_density"])
    for j, col in enumerate(COVER_COLS):
        df[col] = land.cover_obs[-1, :, j]
    df["soil_moisture"] = land.soil_moisture
    return df
