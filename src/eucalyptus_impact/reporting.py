"""Figures, metrics JSON and a markdown report for one pipeline run.

Colour roles are fixed: causal estimates are blue, naive estimates orange and ground truth dark
ink. Magnitude maps use a single-hue light-to-dark ramp; change maps use a two-hue diverging ramp
with a neutral grey midpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .data.synthetic import EUC, NATIVE

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
CAUSAL = "#2a78d6"
NAIVE = "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SEQ_GREEN = LinearSegmentedColormap.from_list("seq", ["#f1f6ee", "#9cc98f", "#2f7d32", "#123d17"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seqo", ["#fbf3ec", "#f2b58f", "#d9612b", "#7a2c0c"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#dcdcd8", "#d9612b"])


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save(fig, path: Path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_effects(tab: pd.DataFrame, path: Path):
    """Small multiples: one panel per effect (units differ), naive vs causal with 95% CI."""
    effects = list(dict.fromkeys(tab["effect"].str.replace("f_eucalyptus", "eucalyptus")))
    tab = tab.assign(effect=tab["effect"].str.replace("f_eucalyptus", "eucalyptus"))
    ncol = 3
    nrow = -(-len(effects) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), squeeze=False)
    for ax, eff in zip(axes.ravel(), effects, strict=False):
        sub = tab[tab["effect"] == eff].reset_index(drop=True)
        for i, r in sub.iterrows():
            color = NAIVE if r["method"] == "OLS" else CAUSAL
            if "SIMEX" in r["method"] or "ME-corrected" in r["method"]:
                color = "#1baf7a"
            se = r["se"] if np.isfinite(r["se"]) else 0
            ax.errorbar(
                r["estimate"], i, xerr=1.96 * se, fmt="o", color=color, ms=6, lw=2, capsize=0
            )
            ax.annotate(
                r["method"],
                (r["estimate"], i),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK_2,
            )
        if pd.notna(sub["truth"]).any():
            ax.axvline(sub["truth"].dropna().iloc[0], color=INK, lw=1.5, ls="--")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_title(eff, fontsize=9)
    for ax in axes.ravel()[len(effects) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Effect estimates vs simulator truth (dashed): naive (orange), causal (blue), "
        "causal + map-error correction (aqua); 95% CI",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path)


def fig_maps(land, scen: dict, path: Path):
    e0 = land.to_2d(land.cover_obs[0, :, EUC])
    e1 = land.to_2d(land.cover_obs[-1, :, EUC])
    freq = land.to_2d(land.burned.mean(axis=0))
    d_native = land.to_2d(land.cover[-1, :, NATIVE] - land.cover[0, :, NATIVE])
    sel = land.to_2d(scen["selected_targeted"].astype(float))
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    panels = [
        (e0, SEQ_GREEN, f"Eucalyptus share {land.years[0]}", (0, 1)),
        (e1, SEQ_GREEN, f"Eucalyptus share {land.years[-1]}", (0, 1)),
        (e1 - e0, DIVERGING, "Change in eucalyptus share", None),
        (freq, SEQ_ORANGE, "Annual burn frequency", None),
        (d_native, DIVERGING, "Change in native broadleaf share (true)", None),
        (sel, SEQ_GREEN, "Targeted restoration cells", (0, 1)),
    ]
    for ax, (arr, cmap, title, lim) in zip(axes.ravel(), panels, strict=True):
        if cmap is DIVERGING:
            m = np.nanmax(np.abs(arr))
            lim = (-m, m)
        vmin, vmax = lim if lim else (np.nanmin(arr), np.nanpercentile(arr, 99))
        im = ax.imshow(
            np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Synthetic Galicia (stylised; not real geography)", x=0.01, ha="left", color=INK_2)
    fig.tight_layout()
    _save(fig, path)


def fig_area(area: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(area))
    ax.barh(y + 0.2, area["map_area_ha"] / 1e3, height=0.36, color=MUTED, label="Pixel count")
    ax.barh(
        y - 0.2,
        area["est_area_ha"] / 1e3,
        height=0.36,
        color=CAUSAL,
        xerr=area["ci95_ha"] / 1e3,
        error_kw={"ecolor": INK_2, "lw": 1},
        label="Stratified estimate (95% CI)",
    )
    ax.scatter(
        area["true_area_ha"] / 1e3, y, marker="|", s=250, color=INK, lw=2, zorder=5, label="Truth"
    )
    ax.set_yticks(y, area["name"])
    ax.invert_yaxis()
    ax.set_xlabel("thousand ha")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def fig_calibration(calib: pd.DataFrame, auc: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    m = max(calib["predicted"].max(), calib["observed"].max()) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    ax.plot(calib["predicted"], calib["observed"], "-o", color=CAUSAL, lw=2, ms=6)
    ax.set_xlabel("predicted P(burn), decile mean")
    ax.set_ylabel("observed burn rate")
    ax.set_title(f"Fire susceptibility calibration\nspatial-CV AUC {auc:.2f}", loc="left")
    _save(fig, path)


def fig_importance(imp: pd.DataFrame, path: Path):
    imp = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(imp["feature"], imp["importance"], color=CAUSAL, height=0.6)
    ax.set_xlabel("permutation importance (drop in R², predictive not causal)")
    ax.set_title("Drivers of conversion to eucalyptus", loc="left")
    _save(fig, path)


def fig_gates(gates: dict[str, pd.DataFrame], path: Path):
    """Group effects with 95% CI (blue) and the group's true average effect (dark tick)."""
    fig, axes = plt.subplots(1, len(gates), figsize=(5.5 * len(gates), 3), squeeze=False)
    for ax, (title, g) in zip(axes[0], gates.items(), strict=True):
        y = np.arange(len(g))
        ax.errorbar(g["estimate"], y, xerr=1.96 * g["se"], fmt="o", color=CAUSAL, ms=6, lw=2)
        if "truth" in g:
            ax.scatter(g["truth"], y, marker="|", s=250, color=INK, lw=2, zorder=5, label="truth")
            ax.legend(fontsize=8, loc="lower right")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [str(v)[2:] for v in g["group"]])
        ax.set_xlabel("dP(burn) / d eucalyptus share")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    _save(fig, path)


def fig_simex(path_df: pd.DataFrame, path: Path):
    effects = list(dict.fromkeys(path_df["effect"]))
    fig, axes = plt.subplots(1, len(effects), figsize=(5 * len(effects), 3), squeeze=False)
    for ax, eff in zip(axes[0], effects, strict=True):
        p = path_df[path_df["effect"] == eff]
        coef = np.polyfit(p["lambda"], p["estimate"], 2)
        lam = np.linspace(-1, p["lambda"].max(), 50)
        ax.plot(lam, np.polyval(coef, lam), color=MUTED, lw=1.5, ls="--")
        ax.plot(p["lambda"], p["estimate"], "o", color=CAUSAL, ms=7)
        ax.plot([-1], [np.polyval(coef, -1)], "o", color="#1baf7a", ms=8)
        ax.annotate(
            "extrapolated\n(no map error)",
            (-1, np.polyval(coef, -1)),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            va="center",
        )
        ax.set_xlabel("added map-error variance (multiples of measured)")
        ax.set_title(f"SIMEX: {eff}", loc="left")
    fig.tight_layout()
    _save(fig, path)


def export_priority(land, scen: dict, out: Path):
    """Restoration priority per cell as CSV, plus a 2-band GeoTIFF when rasterio is installed."""
    df = pd.DataFrame(
        {
            "x": land.static["x"],
            "y": land.static["y"],
            "eucalyptus_share": land.cover_obs[-1, :, EUC],
            "priority_score": scen["priority_score"],
            "selected": scen["selected_targeted"].astype(int),
        }
    )
    df.to_csv(out / "restoration_priority.csv", index=False)
    try:
        from .geo.io import write_geotiff

        stack = np.stack(
            [
                land.to_2d(scen["priority_score"]),
                land.to_2d(scen["selected_targeted"].astype(float)),
            ]
        )
        write_geotiff(stack, land.grid, out / "restoration_priority.tif")
    except ImportError:
        pass


def fig_scenarios(traj: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, col, lab in (
        (axes[0], "eucalyptus_ha", "Eucalyptus area (thousand ha)"),
        (axes[1], "expected_burned_ha", "Expected burned area (thousand ha / yr)"),
    ):
        for color, (name, g) in zip(SERIES, traj.groupby("scenario", sort=False), strict=False):
            ax.plot(g["year"], g[col] / 1e3, color=color, lw=2, label=name)
            ax.annotate(
                name,
                (g["year"].iloc[-1], g[col].iloc[-1] / 1e3),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=INK_2,
                va="center",
            )
        ax.set_title(lab, loc="left")
        ax.margins(x=0.25)
    axes[0].legend(fontsize=8, loc="lower left")
    _save(fig, path)


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "**no**"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.{nd}g}"


def _md_table(df: pd.DataFrame, nd=4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(_fmt(r[c], nd) if not isinstance(r[c], str) else r[c] for c in cols)
            + " |"
        )
    return "\n".join(lines)


def _robustness_md(rob: dict) -> str:
    if not rob:
        return ""
    parts = [
        "## 4b. Robustness",
        "",
        "**Where does eucalyptus raise fire risk?** Group effects from the same DML fit:",
        "",
        _md_table(rob["gate_region"]),
        "",
        _md_table(rob["gate_fwi"]),
        "",
        "![gates](fire_gates.png)",
        "",
        "**Unobserved confounding.** `rv_estimate` is the partial R² an unmapped confounder would",
        "need with *both* treatment and outcome to explain the whole estimate away; `rv_ci` makes",
        "the 95% CI reach zero. `max_bias` is the largest shift a confounder of the given strength",
        "could cause.",
        "",
        _md_table(rob["sensitivity"]),
    ]
    if "se_by_block" in rob:
        parts += [
            "",
            "**Spatial clustering.** Fire-occurrence SE by cluster size (km):",
            "",
            _md_table(rob["se_by_block"]),
        ]
    if "simex" in rob:
        parts += [
            "",
            "**SIMEX.** Map error in *all* cover fractions, extrapolated to zero:",
            "",
            "![simex](simex.png)",
        ]
    return "\n".join(parts)


def write_report(res: dict, out_dir: str | Path) -> Path:
    from .pipeline import effect_table

    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    land = res["land"]
    tab = effect_table(res)
    lc = res["landcover"]
    fs = res["fire_susceptibility"]
    scen = res["scenarios"]
    water = res["water"]

    fig_effects(tab, out / "effects.png")
    fig_maps(land, scen, out / "maps.png")
    fig_area(
        lc["area"], out / "area_species.png", "Species area: pixel counting vs stratified estimator"
    )
    fig_area(lc["change"], out / "area_change.png", "Change area (start → end year)")
    fig_calibration(fs["calibration"], fs["spatial_cv_auc"], out / "fire_calibration.png")
    fig_importance(res["conversion"]["importance"], out / "conversion_drivers.png")
    fig_scenarios(scen["trajectories"], out / "scenarios.png")

    tab.to_csv(out / "effects.csv", index=False)
    lc["area"].to_csv(out / "area_species.csv", index=False)
    lc["change"].to_csv(out / "area_change.csv", index=False)
    scen["trajectories"].to_csv(out / "scenario_trajectories.csv", index=False)
    scen["contrasts"].to_csv(out / "scenario_contrasts.csv", index=False)
    export_priority(land, scen, out)
    rob = res.get("robustness", {})
    if rob:
        fig_gates(
            {"By region (continentality)": rob["gate_region"], "By fire weather": rob["gate_fwi"]},
            out / "fire_gates.png",
        )
        if "simex_path" in rob:
            fig_simex(rob["simex_path"], out / "simex.png")
    res["loss_attribution"].to_csv(out / "loss_attribution.csv", index=False)

    clf = lc["classifier"]
    metrics = {
        "land_cells": land.n,
        "years": [land.years[0], land.years[-1]],
        "panel_rows": res["panel_rows"],
        "runtime_s": res.get("runtime_s"),
        "species_classifier": {
            "spatial_cv_accuracy": clf.spatial_cv_accuracy,
            "random_cv_accuracy": clf.random_cv_accuracy,
            "kappa": clf.kappa,
        },
        "fire_susceptibility": {k: fs[k] for k in ("spatial_cv_auc", "brier", "base_rate")},
        "conversion_driver_spatial_cv_r2": res["conversion"]["spatial_cv_r2"],
        "budyko_params": water["budyko_params"],
        "matching_dropped_share": water["matching_balance"].attrs.get("dropped_share"),
        "effects": tab.to_dict(orient="records"),
        "sensitivity": res.get("robustness", {})
        .get("sensitivity", pd.DataFrame())
        .to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    ha = land.cell_area_ha()
    euc0 = land.cover[0, :, EUC].sum() * ha
    euc1 = land.cover[-1, :, EUC].sum() * ha
    eff_short = tab[["effect", "method", "estimate", "se", "truth", "covers_truth"]]
    bal = water["matching_balance"]
    robustness_md = _robustness_md(rob)
    md = f"""# Galicia eucalyptus impact: pipeline report

> **SYNTHETIC DATA.** Every number below comes from the simulated landscape in
> `data/synthetic.py`, whose effects were set by hand. This report shows that the estimators
> recover known effects. It says nothing about the real Galicia.

Domain: {land.n:,} land cells at {land.grid.resolution_m:.0f} m, {land.years[0]}-{land.years[-1]}.
Eucalyptus area (truth) went from {euc0 / 1e3:,.0f}k ha to {euc1 / 1e3:,.0f}k ha.

![maps](maps.png)

## 1. Effect estimates vs truth

The naive column is what a map overlay or bivariate regression would report. The causal
column partials out climate, terrain, human pressure and the other cover types.

{_md_table(eff_short)}

![effects](effects.png)

Map-error variance of the eucalyptus fraction (from the reference sample):
{res.get("map_error_var", float("nan")):.5f}. Classifier error in the *treatment* attenuates every
effect towards zero, and every cover fraction carries error, controls included. The SIMEX rows
correct for this by adding extra simulated map error, tracing how the estimate degrades, and
extrapolating back to zero error (section 4b). A simpler single-variance regression calibration
over-corrects here, because part of the treatment's map noise is predictable from the other
fractions' noise.

Fire-occurrence effect of each cover class vs the agriculture/other reference (used to price
scenarios; truth on the logit scale: eucalyptus {land.truth.fire_euc}, pine {land.truth.fire_pine},
native {land.truth.fire_native}, shrub {land.truth.fire_shrub}):

{_md_table(pd.DataFrame([{"cover": k[2:], "dP(burn)/dshare": e.estimate, "se": e.se} for k, e in res["fire_effects"]["cover_effects"].items()]))}

## 2. Species mapping and forest-loss accounting

- Spatial-block CV accuracy **{clf.spatial_cv_accuracy:.3f}** vs random CV {clf.random_cv_accuracy:.3f}
  (the gap is the optimism of non-spatial validation). Kappa {clf.kappa:.3f}.

{_md_table(lc["area"][["name", "map_area_ha", "est_area_ha", "ci95_ha", "true_area_ha", "users_accuracy", "producers_accuracy"]], 5)}

![area](area_species.png)

Change areas (map differencing compounds two maps' errors; the stratified estimator corrects it):

{_md_table(lc["change"][["name", "map_area_ha", "est_area_ha", "ci95_ha", "true_area_ha"]], 5)}

![change](area_change.png)

Tree-cover loss attribution (cell-fraction units):

{_md_table(res["loss_attribution"])}

Conversion driver model: spatial-CV R² {res["conversion"]["spatial_cv_r2"]:.3f}.

![drivers](conversion_drivers.png)

## 3. Fire

Susceptibility: spatial-CV AUC **{fs["spatial_cv_auc"]:.3f}**, Brier {fs["brier"]:.4f},
base rate {fs["base_rate"]:.4f}.

![calibration](fire_calibration.png)

## 4. Water

Budyko (Fu) parameters (estimate, SE; truth w_euc = {land.truth.w_euc}):
`{json.dumps({k: [round(v[0], 3), round(v[1], 3)] for k, v in water["budyko_params"].items()})}`

Matching balance (share of treated dropped for lack of overlap:
{bal.attrs.get("dropped_share", float("nan")):.2f}):

{_md_table(bal)}

{robustness_md}

## 5. Policy scenarios to {res["config"].scenarios.horizon}

Horizon-year contrasts vs BAU: the simulated world (truth) next to the projection from the
estimated causal effects (per-class DML effects, risk-weighted, for fire; the fitted Budyko curve
for runoff). Small contrasts, such as the cap, are within simulation noise, so read their sign
with care.

{_md_table(scen["contrasts"], 4)}

![scenarios](scenarios.png)
"""
    path = out / "report.md"
    path.write_text(md)
    return path
