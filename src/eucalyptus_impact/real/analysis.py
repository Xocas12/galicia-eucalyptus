"""Causal and projection analysis on the real Galicia panel.

Design:
- Unit: 1 km cell inside Galicia (at least half its area in the four provinces).
- Exposure: cover fractions in the 2017 baseline map (Nov 2016 - Sep 2017 imagery; the L2A
  archive starts in Nov 2016), fixed and
  measured before every outcome year, so fire cannot feed back into the treatment.
- Fire outcomes: EFFIS burned share and severity class, 2018-2023 (cell-years).
- Confounders: terrain, distance to the sea, location (x, y), buildings, distance to settlements,
  tree cover in 2000, forest loss 2001-2016, the other cover fractions, fire weather (station
  anomalies) and year.
- Conversion: 2017 -> 2024 map transitions (40 m), and the effect of 2018-2021 fire on
  eucalyptus gain by 2024.

Water is not analysed on real data: no streamflow record is reachable from this environment (the
Augas de Galicia / CEDEX gauges are behind blocked portals). The catchment code in
models/hydrology.py runs unchanged once gauge data is supplied.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..causal.dml import dml_plr, group_effects, ols, reclustered_se
from ..causal.sensitivity import bias_bound, robustness_value
from ..dynamic import COVER, estimate_components, project, scenario_contrasts
from ..geo.grid import block_ids
from ..geo.raster_ops import distance_to, focal_mean
from ..validation.spatial_cv import SpatialBlockKFold
from .common import GRID_1KM, INTERIM, log
from .layers import EUC, NATIVE, PINE, all_layers
from .species import CLASS_NAMES, PIXEL_HA, area_table, species_maps, transition_table

FIRE_YEARS = list(range(2018, 2024))
STATIC = [
    "elev",
    "slope",
    "dist_sea_km",
    "x_km",
    "y_km",
    "log_buildings",
    "dist_settlement_km",
    "treecover2000",
    "loss_2001_2016",
]


def cell_frame(L: dict, maps: dict) -> pd.DataFrame:
    """Static cell table for Galicia (1 km)."""
    aoi = L["aoi"]["frac1km"]
    frac17, frac24 = maps["frac_2017"], maps["frac_2024"]
    keep = (aoi >= 0.5) & np.isfinite(frac17[0]) & np.isfinite(frac24[0])
    sea = (aoi == 0) & (L["worldcover"]["wc_water"] > 0.5)
    x, y = GRID_1KM.centers()
    h = L["hansen"]
    df = pd.DataFrame(
        {
            "row": np.nonzero(keep)[0],
            "col": np.nonzero(keep)[1],
            "elev": L["dem"]["elev"][keep],
            "slope": L["dem"]["slope"][keep],
            "dist_sea_km": (distance_to(sea, 1000) / 1000)[keep],
            "x_km": x[keep] / 1000,
            "y_km": y[keep] / 1000,
            "log_buildings": np.log1p(L["buildings"]["buildings"][keep]),
            "dist_settlement_km": L["buildings"]["dist_settlement_km"][keep],
            "treecover2000": h["treecover2000"][keep],
            "loss_2001_2016": sum(h[f"loss_{yr}"] for yr in range(2001, 2017))[keep],
            "block": block_ids(GRID_1KM, 20)[keep],
        }
    )
    for j, c in enumerate(COVER):
        df[c] = frac17[j][keep]
        df[c + "_end"] = frac24[j][keep]
    if "class40_2017_independent" in maps:
        from .species import class_fractions

        ind = class_fractions(maps["class40_2017_independent"], L["aoi"]["mask40"].astype(bool))
        for j, c in enumerate(COVER):
            df[c + "_ind"] = ind[j][keep]
    df["neigh_euc"] = focal_mean(np.nan_to_num(frac17[EUC]), 3, mask=keep)[keep]
    # Gross conversion 2017 -> 2024: share of confidently mapped non-eucalyptus 40 m pixels that
    # are confidently eucalyptus in 2024. Net change would count burned eucalyptus canopy
    # (mapped as shrub afterwards) as negative "gain".
    a, b = maps["class40_2017"], maps["class40_2024"]
    conf = (maps["pmax40_2017"] >= 70) & (maps["pmax40_2024"] >= 70) & (a < 255) & (b < 255)
    base = conf & (a != EUC)
    gain = base & (b == EUC)
    from .common import block_to_1km

    n_base = block_to_1km(base.astype("float32"), "sum")
    n_gain = block_to_1km(gain.astype("float32"), "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        df["gross_gain"] = np.where(n_base > 0, n_gain / n_base, np.nan)[keep]
    df["n_base_px"] = n_base[keep]
    e = L["effis"]
    df["burned_share"] = sum(np.nan_to_num(e[f"burned_{yr}"]) for yr in FIRE_YEARS)[keep]
    df["burned_2018_2021"] = sum(np.nan_to_num(e[f"burned_{yr}"]) for yr in range(2018, 2022))[keep]
    for yr in FIRE_YEARS:
        df[f"burned_{yr}"] = np.nan_to_num(e[f"burned_{yr}"][keep])
        df[f"severity_{yr}"] = e[f"severity_{yr}"][keep]
        df[f"fwi_{yr}"] = L["weather"][f"fwi_{yr}"][keep]
    return df.reset_index(drop=True)


def cell_year_panel(cells: pd.DataFrame) -> pd.DataFrame:
    parts = []
    base = cells[["row", "col", "block", *STATIC, *COVER]]
    for yr in FIRE_YEARS:
        p = base.copy()
        p["year"] = yr
        p["fwi"] = cells[f"fwi_{yr}"].to_numpy()
        p["burned_frac"] = cells[f"burned_{yr}"].to_numpy()
        p["burned"] = (p["burned_frac"] >= 0.01).astype(int)
        p["severity"] = cells[f"severity_{yr}"].to_numpy()
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


CONF = [*STATIC, "fwi", "year"]


def fire_analysis(panel: pd.DataFrame, n_folds=5, seed=0) -> dict:
    g = panel["block"].to_numpy()
    y = panel["burned"].to_numpy().astype(float)
    out = {}
    X_all = lambda d_col: panel[CONF + [c for c in COVER[:4] if c != d_col]].to_numpy()  # noqa: E731
    out["occurrence_naive"] = ols(y, panel["f_eucalyptus"], g, name="eucalyptus -> P(burn)")
    cover_effects = {}
    for c in COVER[:4]:
        cover_effects[c] = dml_plr(
            y, panel[c].to_numpy(), X_all(c), g, n_folds, seed, name=f"{c[2:]} -> P(burn)"
        )
    out["cover_effects"] = cover_effects
    occ = cover_effects["f_eucalyptus"]
    occ.name = "eucalyptus -> P(burn)"
    out["occurrence_dml"] = occ
    out["burned_frac_dml"] = dml_plr(
        panel["burned_frac"],
        panel["f_eucalyptus"],
        X_all("f_eucalyptus"),
        g,
        n_folds,
        seed,
        name="eucalyptus -> burned share",
    )
    out["burned_frac_naive"] = ols(
        panel["burned_frac"], panel["f_eucalyptus"], g, name="eucalyptus -> burned share"
    )
    b = panel[(panel["burned"] == 1) & panel["severity"].notna()]
    Xb = b[[*CONF, "f_pine", "f_native_broadleaf", "f_shrub"]].to_numpy()
    out["severity_dml"] = dml_plr(
        b["severity"],
        b["f_eucalyptus"],
        Xb,
        b["block"],
        n_folds,
        seed,
        name="eucalyptus -> severity (EFFIS class)",
    )
    out["severity_naive"] = ols(
        b["severity"], b["f_eucalyptus"], b["block"], name="eucalyptus -> severity (EFFIS class)"
    )
    # Restricted estimate: the 100 km square holding 91% of the eucalyptus labels (easting
    # 500-600 km, northing 4800-4900 km), the only region where the eucalyptus map is validated.
    reg = panel[(panel["x_km"] >= 500) & (panel["x_km"] < 600) & (panel["y_km"] >= 4800)]
    Xr = reg[CONF + [c for c in COVER[1:4]]].to_numpy()
    out["occurrence_dml_labelled_region"] = dml_plr(
        reg["burned"].to_numpy().astype(float),
        reg["f_eucalyptus"].to_numpy(),
        Xr,
        reg["block"].to_numpy(),
        n_folds,
        seed,
        name="eucalyptus -> P(burn), labelled region",
    )
    out["labelled_region_burned_cell_years"] = int(reg["burned"].sum())
    # heterogeneity
    coast = pd.qcut(panel["dist_sea_km"], 3, labels=["1 coast", "2 transition", "3 interior"])
    fwi = pd.qcut(
        panel["fwi"].rank(method="first"), 3, labels=["1 low FWI", "2 mid FWI", "3 high FWI"]
    )
    out["gate_region"] = group_effects(occ, coast.astype(str))
    out["gate_fwi"] = group_effects(occ, fwi.astype(str))
    rows = []
    for est in (occ, out["burned_frac_dml"], out["severity_dml"]):
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
    xy = panel[["x_km", "y_km"]].to_numpy()
    out["se_by_block"] = pd.DataFrame(
        [
            {
                "block_km": k,
                "se": reclustered_se(
                    occ, (np.floor(xy[:, 0] / k) * 10_000 + np.floor(xy[:, 1] / k)).astype(int)
                ),
            }
            for k in (5, 10, 20, 40, 80)
        ]
    )
    return out


SUSC_FEATURES = [*STATIC, *COVER[:5], "fwi", "neigh_euc"]


def susceptibility(panel: pd.DataFrame, cells: pd.DataFrame, n_folds=5, seed=0) -> dict:
    p = panel.merge(cells[["row", "col", "neigh_euc"]], on=["row", "col"])
    X = p[SUSC_FEATURES].to_numpy()
    y = p["burned"].to_numpy()
    prob = np.empty(len(y))

    def make():
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, min_samples_leaf=100, random_state=seed
        )

    for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=p["block"].to_numpy()):
        prob[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    model = make().fit(X, y)
    # Baseline probability for projections: current (2024) cover, average weather.
    now = cells.copy()
    for c in COVER:
        now[c] = cells[c + "_end"]
    now["fwi"] = 0.0
    p_base = model.predict_proba(now[SUSC_FEATURES].to_numpy())[:, 1]
    # Recalibrate to the observed 2018-2023 burn rate.
    p_base *= y.mean() / max(p_base.mean(), 1e-9)
    return {
        "model": model,
        "auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "base_rate": float(y.mean()),
        "p_base": p_base,
    }


def conversion_analysis(cells: pd.DataFrame, maps: dict, L: dict, n_folds=5, seed=0) -> dict:
    out = {"transitions": transition_table(maps)}
    a, b = maps["class40_2017"], maps["class40_2024"]
    pa, pb = maps["pmax40_2017"], maps["pmax40_2024"]
    conf = (pa >= 70) & (pb >= 70)
    ly = L["hansen"]["lossyear40"]
    loss = (ly >= 18) & (ly <= 24)
    burned = np.zeros_like(loss)
    for yr in FIRE_YEARS:
        burned |= L["effis"][f"burned40_{yr}"].astype(bool)
    nat2euc = (a == NATIVE) & (b == EUC)
    out["native_to_euc_ha"] = {
        "all_pixels": float(nat2euc.sum() * PIXEL_HA),
        "confident": float((nat2euc & conf).sum() * PIXEL_HA),
        "confident_with_loss_or_fire": float((nat2euc & conf & (loss | burned)).sum() * PIXEL_HA),
    }
    pine2euc = (a == PINE) & (b == EUC)
    out["pine_to_euc_ha"] = {
        "all_pixels": float(pine2euc.sum() * PIXEL_HA),
        "confident": float((pine2euc & conf).sum() * PIXEL_HA),
        "confident_with_loss_or_fire": float((pine2euc & conf & (loss | burned)).sum() * PIXEL_HA),
    }
    # Tree-cover loss 2018-2024 attributed at 40 m.
    fire = loss & burned
    conv = loss & ~burned & np.isin(a, [NATIVE, PINE]) & (b == EUC)
    # Plantation harvest: eucalyptus or pine felled and not converted to another tree crop.
    rot = loss & ~burned & ~conv & np.isin(a, [EUC, PINE]) & (b != NATIVE)
    native_loss = loss & ~burned & ~conv & (a == NATIVE)
    other = loss & ~(fire | rot | conv | native_loss)
    out["loss_attribution"] = pd.DataFrame(
        {
            "driver": ["fire", "rotation", "conversion", "native_loss", "unattributed"],
            "attributed": [
                float(m.sum() * PIXEL_HA) for m in (fire, rot, conv, native_loss, other)
            ],
        }
    )
    out["loss_attribution"]["attributed_share"] = (
        out["loss_attribution"]["attributed"] / out["loss_attribution"]["attributed"].sum()
    )
    # Fire -> plantation: does burning in 2018-2021 raise gross conversion to eucalyptus by
    # 2024? (Gross conversion, not net change: burned eucalyptus canopy mapped as shrub in 2024
    # would otherwise count as negative gain.)
    cs = cells[cells["gross_gain"].notna() & (cells["n_base_px"] >= 50)]
    y = cs["gross_gain"]
    X = cs[[*STATIC, *COVER[1:5], "neigh_euc"]].to_numpy()
    d = cs["burned_2018_2021"].to_numpy()
    out["fire_conversion_dml"] = dml_plr(
        y, d, X, cs["block"], n_folds, seed, name="burned share 2018-2021 -> eucalyptus gain"
    )
    out["fire_conversion_naive"] = ols(
        y, d, cs["block"], name="burned share 2018-2021 -> eucalyptus gain"
    )
    feats = [*STATIC, *COVER[1:5], "neigh_euc", "burned_2018_2021"]
    m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(cs[feats], y)
    sub = cs.sample(min(8000, len(cs)), random_state=seed)
    ysub = y.loc[sub.index]
    imp = permutation_importance(m, sub[feats], ysub, n_repeats=5, random_state=seed)
    out["drivers"] = (
        pd.DataFrame({"feature": feats, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return out


def projections(
    cells: pd.DataFrame,
    fire: dict,
    susc: dict,
    horizon: int = 2040,
    share: float = 0.25,
    seed: int = 0,
) -> dict:
    ce = fire["cover_effects"]
    theta = np.array([ce[c].estimate if c in ce else 0.0 for c in COVER])
    theta_se = np.array([ce[c].se if c in ce else 0.0 for c in COVER])
    rates = np.array([(cells[f"burned_{yr}"] >= 0.01).mean() for yr in FIRE_YEARS])
    weather = rates / rates.mean()
    burned_cells = np.concatenate(
        [cells.loc[cells[f"burned_{yr}"] >= 0.01, f"burned_{yr}"] for yr in FIRE_YEARS]
    )
    comp = estimate_components(
        cells,
        7,
        [*STATIC, "neigh_euc"],
        susc["p_base"],
        theta,
        float(burned_cells.mean()),
        weather,
        seed=seed,
    )
    # Calibrate the planting rate to the observed gross conversion between confidently mapped
    # pixels. The model is fitted to noisy per-cell net changes, and clipping its predictions at
    # zero turns noise into spurious growth.
    ok = cells["gross_gain"].notna() & (cells["n_base_px"] >= 50)
    target = float(cells.loc[ok, "gross_gain"].mean()) / 7
    raw_mean = float(comp.conv_rate.mean())
    comp.conv_rate = comp.conv_rate * (target / max(raw_mean, 1e-12))
    comp.diagnostics["conv_rate_raw_mean"] = raw_mean
    comp.diagnostics["conv_rate_target"] = target
    f0 = cells[[c + "_end" for c in COVER]].to_numpy()
    euc = f0[:, EUC]
    score = susc["p_base"] * euc
    rng = np.random.default_rng(seed)

    def select(order):
        cum = np.cumsum(euc[order])
        k = int(np.searchsorted(cum, share * euc.sum())) + 1
        sel = np.zeros(len(euc))
        sel[order[:k]] = 1.0
        return sel

    targeted = select(np.argsort(-score))
    random_ = select(rng.permutation(len(euc)))
    H = horizon - 2024
    kw = dict(cell_area_ha=100.0, n_sims=40, seed=seed, theta_se=theta_se)
    trajs = {
        "BAU": project(f0, comp, H, "bau", **kw),
        "Cap / moratorium": project(f0, comp, H, "cap", **kw),
        "Targeted restoration": project(f0, comp, H, "restore", targeted, **kw),
        "Random restoration": project(f0, comp, H, "restore", random_, **kw),
    }
    for t in trajs.values():
        t["year"] = t["year"] + 2024
    # Sensitivity: BAU conversion at half the 2017-2024 average (e.g. if the post-2021
    # planting restrictions hold).
    import copy

    c2 = copy.copy(comp)
    c2.conv_rate = comp.conv_rate * 0.5
    trajs_half = {
        "BAU": project(f0, c2, H, "bau", **kw),
        "Cap / moratorium": project(f0, c2, H, "cap", **kw),
    }
    return {
        "components": comp,
        "trajectories": trajs,
        "contrasts": scenario_contrasts(trajs),
        "contrasts_half_conversion": scenario_contrasts(trajs_half),
        "targeted": targeted,
        "priority": score,
    }


def map_sensitivity(cells: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Fire-occurrence effect of eucalyptus under different exposure maps.

    The estimate moved between map versions during development, so it is reported for the
    backdated 2017 map (the main exposure), the independently classified 2017 map, and the 2024
    map (measured after the fires, so only a sensitivity check).
    """
    versions = {"2017 backdated": "", "2017 independent": "_ind", "2024 (post-fire)": "_end"}
    rows = []
    for name, suf in versions.items():
        if suf and COVER[0] + suf not in cells:
            continue
        c2 = cells.copy()
        for c in COVER:
            c2[c] = cells[c + suf] if suf else cells[c]
        est = fire_analysis(cell_year_panel(c2), seed=seed)["occurrence_dml"]
        rows.append(
            {
                "map": name,
                "estimate": est.estimate,
                "se": est.se,
                "ci_low": est.ci95[0],
                "ci_high": est.ci95[1],
            }
        )
    return pd.DataFrame(rows)


def run_real(seed: int = 0) -> dict:
    L = all_layers()
    maps = species_maps()
    cells = cell_frame(L, maps)
    panel = cell_year_panel(cells)
    log.info("real panel: %d cells, %d cell-years", len(cells), len(panel))
    res = {"cells": cells, "maps": maps, "layers": L}
    res["areas"] = {p: area_table(maps, p) for p in ("2017", "2024")}
    res["species_metrics"] = {
        p: json.loads((INTERIM / f"species_metrics_{p}.json").read_text()) for p in ("2017", "2024")
    }
    res["fire"] = fire_analysis(panel, seed=seed)
    res["fire"]["map_sensitivity"] = map_sensitivity(cells, seed=seed)
    log.info("fire done")
    res["susceptibility"] = susceptibility(panel, cells, seed=seed)
    res["conversion"] = conversion_analysis(cells, maps, L, seed=seed)
    log.info("conversion done")
    res["projections"] = projections(cells, res["fire"], res["susceptibility"], seed=seed)
    log.info("projections done")
    res["panel_summary"] = {
        "cells": len(cells),
        "cell_years": len(panel),
        "burned_cell_years": int(panel["burned"].sum()),
        "burned_km2_by_year": {yr: float(cells[f"burned_{yr}"].sum()) for yr in FIRE_YEARS},
    }
    return res


__all__ = ["CLASS_NAMES", "cell_frame", "cell_year_panel", "run_real"]
