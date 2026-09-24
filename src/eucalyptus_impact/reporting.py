"""Figures, metrics JSON and a markdown report for one pipeline run.

Colour roles are fixed: causal estimates are blue, naive estimates orange and ground truth dark
ink. Magnitude maps use a single-hue light-to-dark ramp; change maps use a two-hue diverging ramp
with a neutral grey midpoint.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

from .data.synthetic import EUC, NATIVE
from .i18n_gl import num, tr, tr_frame

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


def _gl_ticks(fig):
    """Decimal commas on every numeric axis (categorical tick labels are left alone)."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(fmt)


def fig_effects(tab: pd.DataFrame, path: Path):
    """Small multiples: one panel per effect (units differ), naive vs causal with 95% CI."""
    tab = tab.assign(effect=tab["effect"].map(tr))
    effects = list(dict.fromkeys(tab["effect"]))
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
                tr(r["method"]),
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
        ax.set_title(textwrap.fill(eff, 42), fontsize=9)
    for ax in axes.ravel()[len(effects) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Estimacións dos efectos fronte ao valor real da simulación (liña descontinua): "
        "inxenuas (laranxa), causais (azul), causais con corrección do erro cartográfico "
        "(verde auga); IC 95 %",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    _gl_ticks(fig)
    _save(fig, path)


def fig_maps(land, scen: dict, path: Path):
    e0 = land.to_2d(land.cover_obs[0, :, EUC])
    e1 = land.to_2d(land.cover_obs[-1, :, EUC])
    freq = land.to_2d(land.burned.mean(axis=0))
    d_native = land.to_2d(land.cover[-1, :, NATIVE] - land.cover[0, :, NATIVE])
    sel = land.to_2d(scen["selected_targeted"].astype(float))
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    panels = [
        (e0, SEQ_GREEN, f"Fracción de eucalipto {land.years[0]}", (0, 1)),
        (e1, SEQ_GREEN, f"Fracción de eucalipto {land.years[-1]}", (0, 1)),
        (e1 - e0, DIVERGING, "Cambio na fracción de eucalipto", None),
        (freq, SEQ_ORANGE, "Frecuencia anual de queima", None),
        (d_native, DIVERGING, "Cambio na fracción de frondosas autóctonas (real)", None),
        (sel, SEQ_GREEN, "Celas de restauración dirixida", (0, 1)),
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
    fig.suptitle(
        "Galicia sintética (estilizada; non é a xeografía real)", x=0.01, ha="left", color=INK_2
    )
    fig.tight_layout()
    _gl_ticks(fig)
    _save(fig, path)


def fig_area(area: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(area))
    ax.barh(
        y + 0.2, area["map_area_ha"] / 1e3, height=0.36, color=MUTED, label="Reconto de píxeles"
    )
    ax.barh(
        y - 0.2,
        area["est_area_ha"] / 1e3,
        height=0.36,
        color=CAUSAL,
        xerr=area["ci95_ha"] / 1e3,
        error_kw={"ecolor": INK_2, "lw": 1},
        label="Estimación estratificada (IC 95 %)",
    )
    ax.scatter(
        area["true_area_ha"] / 1e3,
        y,
        marker="|",
        s=250,
        color=INK,
        lw=2,
        zorder=5,
        label="Valor real",
    )
    ax.set_yticks(y, [tr(n) for n in area["name"]])
    ax.invert_yaxis()
    ax.set_xlabel("miles de ha")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    _gl_ticks(fig)
    _save(fig, path)


def fig_calibration(calib: pd.DataFrame, auc: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    m = max(calib["predicted"].max(), calib["observed"].max()) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    ax.plot(calib["predicted"], calib["observed"], "-o", color=CAUSAL, lw=2, ms=6)
    ax.set_xlabel("P(queima) predita, media por decil")
    ax.set_ylabel("taxa de queima observada")
    ax.set_title(
        f"Calibración da susceptibilidade aos incendios\nAUC en validación cruzada espacial "
        f"{num(auc, 2)}",
        loc="left",
    )
    _gl_ticks(fig)
    _save(fig, path)


def fig_importance(imp: pd.DataFrame, path: Path):
    imp = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([tr(f) for f in imp["feature"]], imp["importance"], color=CAUSAL, height=0.6)
    ax.set_xlabel("importancia por permutación (caída do R²; predictiva, non causal)")
    ax.set_title("Factores da conversión a eucalipto", loc="left")
    _gl_ticks(fig)
    _save(fig, path)


def fig_gates(gates: dict[str, pd.DataFrame], path: Path):
    """Group effects with 95% CI (blue) and the group's true average effect (dark tick)."""
    fig, axes = plt.subplots(1, len(gates), figsize=(5.5 * len(gates), 3), squeeze=False)
    for ax, (title, g) in zip(axes[0], gates.items(), strict=True):
        y = np.arange(len(g))
        ax.errorbar(g["estimate"], y, xerr=1.96 * g["se"], fmt="o", color=CAUSAL, ms=6, lw=2)
        if "truth" in g:
            ax.scatter(
                g["truth"], y, marker="|", s=250, color=INK, lw=2, zorder=5, label="valor real"
            )
            ax.legend(fontsize=8, loc="lower right")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [tr(v) for v in g["group"]])
        ax.set_xlabel("dP(queima) / d fracción de eucalipto")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    _gl_ticks(fig)
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
            "extrapolado\n(sen erro cartográfico)",
            (-1, np.polyval(coef, -1)),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            va="center",
        )
        ax.set_xlabel("varianza engadida do erro cartográfico (múltiplos da medida)")
        ax.set_title(f"SIMEX: {tr(eff)}", loc="left")
    fig.tight_layout()
    _gl_ticks(fig)
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
        (axes[0], "eucalyptus_ha", "Superficie de eucalipto (miles de ha)"),
        (axes[1], "expected_burned_ha", "Superficie queimada esperada (miles de ha/ano)"),
    ):
        for color, (name, g) in zip(SERIES, traj.groupby("scenario", sort=False), strict=False):
            ax.plot(g["year"], g[col] / 1e3, color=color, lw=2, label=tr(name))
        ax.set_title(lab, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    axes[0].legend(fontsize=8, loc="lower left")
    _gl_ticks(fig)
    _save(fig, path)


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\u2013"
    if isinstance(x, (bool, np.bool_)):
        return "si" if x else "**non**"
    if isinstance(x, (int, np.integer)):
        return num(int(x))
    return num(float(x), nd)


def _md_table(df: pd.DataFrame, nd=4, values=()) -> str:
    """Markdown table with Galician headers, translated `values` columns and Galician numbers."""
    df = tr_frame(df, tuple(values))
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
        "## 4b. Robustez",
        "",
        "**Onde aumenta máis o eucalipto o risco de incendio?** Efectos por grupos obtidos do "
        "mesmo axuste DML:",
        "",
        _md_table(rob["gate_region"], values=("group",)),
        "",
        _md_table(rob["gate_fwi"], values=("group",)),
        "",
        "![efectos por grupos](fire_gates.png)",
        "",
        "**Confusión non observada.** O *valor de robustez* (VR) da estimación é o R² parcial "
        "que necesitaría un factor de confusión non cartografado, tanto co tratamento como co "
        "resultado, para explicar toda a estimación. O VR do IC é o que leva o intervalo de "
        "confianza do 95 % ata cero. O nesgo máximo é o maior desprazamento que podería causar "
        "un factor de confusión desa intensidade.",
        "",
        _md_table(rob["sensitivity"], values=("effect",)),
    ]
    if "se_by_block" in rob:
        parts += [
            "",
            "**Agrupamento espacial.** Erro estándar do efecto sobre a aparición de incendios "
            "segundo o tamaño do bloque (km):",
            "",
            _md_table(rob["se_by_block"]),
        ]
    if "simex" in rob:
        parts += [
            "",
            "**SIMEX.** Erro cartográfico en *todas* as fraccións de cuberta, extrapolado a cero:",
            "",
            "![SIMEX](simex.png)",
        ]
    return "\n".join(parts)


GLOSSARY = """## Siglas

| sigla | significado |
|---|---|
| AUC | área baixo a curva ROC |
| DME | diferenza de medias estandarizada |
| DML | aprendizaxe automática dobre (estimación causal con axustes cruzados) |
| dNBR | diferenza do índice normalizado de área queimada (severidade) |
| EE | erro estándar |
| ETP | evapotranspiración potencial |
| FWI | índice meteorolóxico de perigo de incendio |
| IC | intervalo de confianza |
| MCO | mínimos cadrados ordinarios (estimación inxenua) |
| SIMEX | extrapolación por simulación (corrección do erro de medida) |
| VR | valor de robustez |
"""


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
        lc["area"],
        out / "area_species.png",
        "Superficie por especie: reconto de píxeles fronte a estimador estratificado",
    )
    fig_area(
        lc["change"], out / "area_change.png", "Superficie de cambio (ano inicial → ano final)"
    )
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
            {
                "Por rexión (continentalidade)": rob["gate_region"],
                "Por meteoroloxía de incendios": rob["gate_fwi"],
            },
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
    tp = land.truth
    eff_short = tab[["effect", "method", "estimate", "se", "truth", "covers_truth"]]
    bal = water["matching_balance"]
    cover_tab = pd.DataFrame(
        [
            {"cover": k[2:], "dP(burn)/dshare": e.estimate, "se": e.se}
            for k, e in res["fire_effects"]["cover_effects"].items()
        ]
    )
    budyko_tab = pd.DataFrame(
        [
            {"parameter": tr(k), "estimate": v[0], "se": v[1]}
            for k, v in water["budyko_params"].items()
        ]
    )
    robustness_md = _robustness_md(rob)
    horizon = res["config"].scenarios.horizon
    md = f"""# Impacto do eucalipto en Galicia: informe da análise

> **DATOS SINTÉTICOS.** Todas as cifras deste informe proceden da paisaxe simulada en
> `data/synthetic.py`, cuxos efectos se fixaron a man. O informe demostra que os estimadores
> recuperan efectos coñecidos. Non di nada sobre a Galicia real.

Dominio: {num(land.n)} celas de terra de {num(land.grid.resolution_m)} m, {land.years[0]}\u2013{land.years[-1]}.
A superficie de eucalipto (valor real da simulación) pasou de {num(euc0 / 1e3, 3)} mil ha a
{num(euc1 / 1e3, 3)} mil ha.

![mapas](maps.png)

## 1. Estimacións dos efectos fronte ao valor real

As estimacións por MCO son as inxenuas: o que daría unha superposición de mapas ou unha regresión
bivariante. As causais eliminan a influencia do clima, do relevo, da presión humana e dos demais
tipos de cuberta.

{_md_table(eff_short, values=("effect", "method"))}

![efectos](effects.png)

Varianza do erro cartográfico da fracción de eucalipto (segundo a mostra de referencia):
{num(res.get("map_error_var", float("nan")), 2)}. O erro do clasificador no *tratamento* atenúa
todos os efectos cara a cero, e todas as fraccións de cuberta levan erro, tamén as que actúan como
control. As filas SIMEX corrixen isto: engaden erro cartográfico simulado, observan como se
degrada a estimación e extrapolan ata erro cero (sección 4b). Unha calibración de regresión máis
sinxela, cunha única varianza, corrixe en exceso, porque parte do ruído cartográfico do tratamento
se pode predicir a partir do ruído das outras fraccións.

Efecto de cada clase de cuberta sobre a probabilidade de incendio fronte á referencia
agricultura/outros. Úsase para valorar os escenarios. O valor real, na escala logit, é:
eucalipto {num(tp.fire_euc)}, piñeiro {num(tp.fire_pine)}, frondosas autóctonas
{num(tp.fire_native)} e mato {num(tp.fire_shrub)}.

{_md_table(cover_tab, values=("cover",))}

## 2. Cartografía de especies e contabilidade da perda forestal

- Exactitude con validación cruzada por bloques espaciais **{num(clf.spatial_cv_accuracy, 3)}**
  fronte a {num(clf.random_cv_accuracy, 3)} con validación cruzada aleatoria (a diferenza é o
  optimismo dunha validación non espacial). Kappa {num(clf.kappa, 3)}.

{_md_table(lc["area"][["name", "map_area_ha", "est_area_ha", "ci95_ha", "true_area_ha", "users_accuracy", "producers_accuracy"]], 5, values=("name",))}

![superficie por especie](area_species.png)

Superficies de cambio. A diferenza entre dous mapas acumula os erros de ambos, e o estimador
estratificado corríxeo:

{_md_table(lc["change"][["name", "map_area_ha", "est_area_ha", "ci95_ha", "true_area_ha"]], 5, values=("name",))}

![superficie de cambio](area_change.png)

Atribución da perda de cuberta arbórea (en fraccións de cela):

{_md_table(res["loss_attribution"], values=("driver",))}

Modelo de factores da conversión: R² en validación cruzada espacial
{num(res["conversion"]["spatial_cv_r2"], 3)}.

![factores da conversión](conversion_drivers.png)

## 3. Incendios

Susceptibilidade: AUC en validación cruzada espacial **{num(fs["spatial_cv_auc"], 3)}**,
puntuación de Brier {num(fs["brier"], 3)} e taxa base {num(fs["base_rate"], 3)}.

![calibración](fire_calibration.png)

## 4. Auga

Parámetros da curva de Budyko (Fu). O valor real de w eucalipto é {num(tp.w_euc)}.

{_md_table(budyko_tab)}

Equilibrio do emparellamento. Proporción de celas tratadas descartadas por falta de
solapamento: {num(bal.attrs.get("dropped_share", float("nan")), 2)}.

{_md_table(bal, values=("feature",))}

{robustness_md}

## 5. Escenarios de política ata {horizon}

Contrastes no ano horizonte fronte ao escenario tendencial: o mundo simulado (valor real) xunto á
proxección feita cos efectos causais estimados (efectos DML por clase, ponderados polo risco, para
os incendios; a curva de Budyko axustada para a escorrentía). Os contrastes pequenos, como o do
límite, quedan dentro do ruído da simulación, así que o seu signo debe lerse con cautela.

{_md_table(scen["contrasts"], 4, values=("scenario",))}

![escenarios](scenarios.png)

{GLOSSARY}"""
    path = out / "report.md"
    path.write_text(md)
    return path
