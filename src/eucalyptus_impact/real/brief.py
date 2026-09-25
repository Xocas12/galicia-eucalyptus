# ruff: noqa: RUF001  (Galician typography: en dashes in year ranges are intentional)
"""Policy brief in Galician from the real-data results (figures, tables and prose)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..i18n_gl import GL, num, tr
from ..reporting import (
    CAUSAL,
    DIVERGING,
    INK,
    INK_2,
    MUTED,
    NAIVE,
    SEQ_GREEN,
    SEQ_ORANGE,
    SERIES,
    _gl_ticks,
    _md_table,
    _save,
    _style,
    fig_gates,
)
from .common import GRID_1KM
from .layers import EUC

GL.update(
    {
        "eucalyptus -> burned share": "eucalipto → fracción queimada",
        "eucalyptus -> severity (EFFIS class)": "eucalipto → severidade (clase EFFIS)",
        "burned share 2018-2021 -> eucalyptus gain": "queimado 2018-2021 → ganancia de eucalipto",
        "pine -> P(burn)": "piñeiro → P(queima)",
        "native_broadleaf -> P(burn)": "frondosas autóctonas → P(queima)",
        "shrub -> P(burn)": "mato → P(queima)",
        "unattributed": "sen atribuír",
        "native_loss": "perda de frondosas sen conversión",
        "dist_sea_km": "distancia ao mar (km)",
        "x_km": "coordenada leste (km)",
        "y_km": "coordenada norte (km)",
        "log_buildings": "edificacións (log)",
        "dist_settlement_km": "distancia a núcleos (km)",
        "treecover2000": "cuberta arbórea 2000",
        "loss_2001_2016": "perda forestal 2001-2016",
        "burned_2018_2021": "queimado 2018-2021",
        "d_native_ha": "Δ frondosas autóctonas (ha)",
        "d_burned_ha_per_year": "Δ queimado medio (ha/ano)",
        "d_cum_burned_ha": "Δ queimado acumulado (ha)",
        "from": "de",
        "to": "a",
        "area_ha": "superficie (ha)",
        "area_confident_ha": "superficie, píxeles fiables (ha)",
        "soft_area_ha": "superficie por probabilidades (ha)",
        "class": "código",
        "measure": "medida",
        "value": "valor",
        "source": "fonte",
        "use": "uso",
        "period": "período",
        "map": "mapa",
        "ci_low": "IC 95 % inferior",
        "ci_high": "IC 95 % superior",
        "2017": "2017",
        "2024": "2024",
    }
)


def _fig_maps(res: dict, path: Path):
    cells = res["cells"]
    ny, nx = GRID_1KM.shape

    def grid(v):
        a = np.full((ny, nx), np.nan)
        a[cells["row"], cells["col"]] = v
        return a

    e17, e24 = cells["f_eucalyptus"].to_numpy(), cells["f_eucalyptus_end"].to_numpy()
    panels = [
        (grid(e17), SEQ_GREEN, "Fracción de eucalipto, 2017", (0, 1)),
        (grid(e24), SEQ_GREEN, "Fracción de eucalipto, 2024", (0, 1)),
        (grid(e24 - e17), DIVERGING, "Cambio 2017–2024", None),
        (
            grid(cells["f_native_broadleaf"].to_numpy()),
            SEQ_GREEN,
            "Fracción de frondosas autóctonas, 2017",
            (0, 1),
        ),
        (
            grid(np.clip(cells["burned_share"].to_numpy(), 0, 1)),
            SEQ_ORANGE,
            "Fracción queimada 2018–2023 (EFFIS)",
            (0, 1),
        ),
        (
            grid(res["projections"]["targeted"]),
            SEQ_GREEN,
            "Celas prioritarias para restaurar",
            (0, 1),
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (arr, cmap, title, lim) in zip(axes.ravel(), panels, strict=True):
        if lim is None:
            m = np.nanpercentile(np.abs(arr), 99)
            lim = (-m, m)
        im = ax.imshow(
            np.ma.masked_invalid(arr), cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest"
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Galicia, celas de 1 km (EPSG:25829)", x=0.01, ha="left", color=INK_2)
    fig.tight_layout()
    _gl_ticks(fig)
    _save(fig, path)


def _fig_effects(rows: list, path: Path):
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 2.6), squeeze=False)
    for ax, (title, ests) in zip(axes[0], rows, strict=True):
        for i, e in enumerate(ests):
            color = NAIVE if e.method == "OLS" else CAUSAL
            ax.errorbar(e.estimate, i, xerr=1.96 * e.se, fmt="o", color=color, ms=6, lw=2)
            ax.annotate(
                tr(e.method),
                (e.estimate, i),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK_2,
            )
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(ests) - 0.3)
        ax.set_title(title, fontsize=9)
    fig.suptitle(
        "Estimación inxenua (laranxa) fronte a causal (azul); IC 95 %",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    _gl_ticks(fig)
    _save(fig, path)


def _fig_projections(trajs: dict, path: Path):
    """Left: eucalyptus area per scenario. Right: burned area minus business as usual, per year,
    with 5-95% bands from paired simulations (same weather and parameter draws)."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for color, (name, t) in zip(SERIES, trajs.items(), strict=False):
        axes[0].plot(t["year"], t["eucalyptus_ha"] / 1e3, color=color, lw=2, label=tr(name))
    axes[0].set_title("Superficie de eucalipto (miles de ha)", loc="left")
    axes[0].legend(fontsize=8, loc="best")
    base = trajs["BAU"].attrs["sim_year_burned"]
    years = trajs["BAU"]["year"].to_numpy()
    for color, (name, t) in list(zip(SERIES, trajs.items(), strict=False))[1:]:
        d = t.attrs["sim_year_burned"] - base
        axes[1].plot(years, d.mean(axis=0).to_numpy(), color=color, lw=2, label=tr(name))
        axes[1].fill_between(
            years,
            d.quantile(0.05).to_numpy(),
            d.quantile(0.95).to_numpy(),
            color=color,
            alpha=0.15,
            lw=0,
        )
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_title("Queimado fronte á tendencia (ha/ano), bandas 5–95 %", loc="left")
    axes[1].legend(fontsize=8, loc="best")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    _gl_ticks(fig)
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    _save(fig, path)


def _ci(e) -> str:
    lo, hi = e.estimate - 1.96 * e.se, e.estimate + 1.96 * e.se
    return f"{num(e.estimate, 3)} (IC 95 %: {num(lo, 3)} a {num(hi, 3)})"


CLASS_NAMES_GL = ["eucalipto", "piñeiro", "frondosas autóctonas", "mato", "agricultura", "outros"]

SOURCES = pd.DataFrame(
    [
        (
            "Sentinel-2 L2A (Copernicus, arquivo COG de AWS)",
            "mapas de especies (NDVI, NDMI, NBR mensuais, 40 m)",
            "nov. 2016–set. 2017 (o arquivo L2A comeza en nov. 2016) e out. 2023–set. 2024",
        ),
        (
            "OpenStreetMap (vía Overture Maps)",
            "etiquetas de adestramento (tipo de folla, xénero, mato, prados)",
            "2026",
        ),
        ("ESA WorldCover", "máscara de arboredo e clases non forestais", "2021"),
        (
            "Hansen Global Forest Change v1.12",
            "perda de cuberta arbórea anual, cuberta en 2000",
            "2001–2024",
        ),
        ("EFFIS (severidade de queimados)", "área queimada e severidade", "2018–2023"),
        ("Copernicus DEM (90 m)", "altitude e pendente", "estático"),
        (
            "NOAA GHCN-Daily (6 estacións galegas)",
            "anomalías de temperatura e choiva estivais",
            "2001–2025",
        ),
        ("Overture Maps, edificacións", "presión humana (ignicións)", "2026"),
        ("Natural Earth", "límite de Galicia (catro provincias)", "estático"),
    ],
    columns=["source", "use", "period"],
)


def _sig(e) -> bool:
    """Whether the 95% CI of an estimate excludes zero."""
    return abs(e.estimate) > 1.96 * e.se


def _sensitivity_txt(fire: dict) -> str:
    ms = fire.get("map_sensitivity")
    if ms is None or ms.empty:
        return ""
    parts = ", ".join(
        f"{GL_MAP.get(r.map, r.map)}: {num(r.estimate * 10, 2)}" for r in ms.itertuples()
    )
    sig = [(r.ci_low > 0) or (r.ci_high < 0) for r in ms.itertuples()]
    same_sign = len({np.sign(r.estimate) for r in ms.itertuples()}) == 1
    if same_sign and all(sig):
        return f" As tres versións do mapa dan a mesma conclusión ({parts})."
    if same_sign:
        return (
            f" As versións do mapa coinciden no signo ({parts}), pero non todas son "
            "distinguibles de cero: o resultado é sensible ao mapa."
        )
    return (
        " **A estimación depende do mapa de eucalipto empregado** (puntos porcentuais por "
        f"cada 10 puntos de eucalipto: {parts}), así que non se debe tomar como un resultado "
        "firme."
    )


GL_MAP = {
    "2017 backdated": "2017 retrodatado",
    "2017 independent": "2017 independente",
    "2024 (post-fire)": "2024 (posterior aos lumes)",
}


def _resumo(res: dict) -> str:
    fire, conv, proj = res["fire"], res["conversion"], res["projections"]
    areas, susc = res["areas"], res["susceptibility"]
    a17 = areas["2017"].set_index("name").loc["eucalyptus"]
    a24 = areas["2024"].set_index("name").loc["eucalyptus"]
    nat = conv["native_to_euc_ha"]
    occ, shrub = fire["occurrence_dml"], fire["cover_effects"]["f_shrub"]
    sev, fc = fire["severity_dml"], conv["fire_conversion_dml"]
    rv = fire["sensitivity"].iloc[0]
    contr = proj["contrasts"].set_index("scenario")
    items = []
    items.append(
        f"- **Superficie de eucalipto (mapa).** {num(a24['map_area_ha'] / 1e3, 3)} mil ha en 2024 "
        f"e {num(a17['map_area_ha'] / 1e3, 3)} mil ha en 2017 (reconto de píxeles; "
        f"{num(a24['soft_area_ha'] / 1e3, 3)} mil ha en 2024 sumando probabilidades). **Estas "
        "cifras non están validadas** co inventario oficial (IFN, Mapa Forestal de España): "
        "compárense con el antes de citalas (sección 2)."
    )
    items.append(
        f"- **Substitución de bosque autóctono.** Entre 2017 e 2024, {num(nat['confident'])} ha "
        "pasaron de frondosas autóctonas a eucalipto en píxeles clasificados con fiabilidade nos "
        f"dous anos; {num(nat['confident_with_loss_or_fire'])} ha diso coinciden ademais cunha "
        "perda de cuberta arbórea (Hansen) ou cun incendio (EFFIS). Esta última é a cifra máis "
        "prudente; a diferenza entre mapas tende a sobreestimar o cambio."
    )
    if _sig(occ):
        occ_txt = (
            f"un aumento de 10 puntos na fracción de eucalipto cambia a probabilidade anual de "
            f"queima en {num(occ.estimate * 10, 2)} puntos porcentuais "
            f"(IC 95 %: {num((occ.estimate - 1.96 * occ.se) * 10, 2)} a "
            f"{num((occ.estimate + 1.96 * occ.se) * 10, 2)})"
        )
    else:
        occ_txt = (
            "**non se detecta un efecto do eucalipto** sobre a probabilidade anual de queima "
            "distinguible de cero, fronte a agricultura e outros usos: por cada 10 puntos de "
            f"eucalipto, {num(occ.estimate * 10, 2)} puntos porcentuais (IC 95 %: "
            f"{num((occ.estimate - 1.96 * occ.se) * 10, 2)} a "
            f"{num((occ.estimate + 1.96 * occ.se) * 10, 2)}), cunha taxa base de "
            f"{num(susc['base_rate'] * 100, 2)} % ao ano"
        )
    shrub_txt = (
        f"O mato si aumenta o risco: {num(shrub.estimate * 10, 2)} puntos porcentuais por cada "
        f"10 puntos de mato (IC 95 %: {num((shrub.estimate - 1.96 * shrub.se) * 10, 2)} a "
        f"{num((shrub.estimate + 1.96 * shrub.se) * 10, 2)})."
        if _sig(shrub)
        else "O mato non mostra un efecto distinguible de cero."
    )
    rv_txt = (
        f" O valor de robustez é {num(rv['rv_estimate'], 2)}: un factor de confusión non medido "
        "con ese R² parcial co tratamento e co resultado anularía a estimación."
        if _sig(occ)
        else ""
    )
    reg = fire["occurrence_dml_labelled_region"]
    reg_txt = (
        f" No cadro do norte, onde están as etiquetas de OpenStreetMap, o efecto é {num(reg.estimate * 10, 2)} "
        f"puntos porcentuais por cada 10 puntos de eucalipto (IC 95 %: "
        f"{num((reg.estimate - 1.96 * reg.se) * 10, 2)} a "
        f"{num((reg.estimate + 1.96 * reg.se) * 10, 2)}; "
        f"{num(fire['labelled_region_burned_cell_years'])} anos-cela queimados)"
        + (", non distinguible de cero." if not _sig(reg) else ".")
    )
    items.append(
        f"- **Incendios, 2018–2023.** Mantendo constantes o relevo, a localización, a presión "
        f"humana, a meteoroloxía e as demais cubertas, {occ_txt}. {shrub_txt}{rv_txt}{reg_txt}"
        + _sensitivity_txt(fire)
    )
    nat_e = fire["cover_effects"]["f_native_broadleaf"]
    diff = occ.estimate - nat_e.estimate
    diff_se = float(np.hypot(occ.se, nat_e.se))
    items.append(
        "- **Eucalipto fronte a frondosas autóctonas.** O efecto anterior compárase coa "
        "agricultura e outros usos, que son os que máis arden. Fronte ás frondosas autóctonas, "
        f"que son as que menos arden, 10 puntos de eucalipto no canto de frondosas cambian a "
        f"probabilidade anual de queima en {num(diff * 10, 2)} puntos porcentuais (IC 95 % "
        f"aproximado: {num((diff - 1.96 * diff_se) * 10, 2)} a "
        f"{num((diff + 1.96 * diff_se) * 10, 2)}; aproximado porque combina dúas estimacións "
        "separadas). Este é o contraste que importa para a restauración."
    )
    sev_txt = "distinguible de cero" if _sig(sev) else "non distinguible de cero"
    items.append(
        f"- **Severidade.** Entre as celas queimadas, o efecto do eucalipto sobre a clase de "
        f"severidade EFFIS é {num(sev.estimate, 2)} por unidade de fracción (EE "
        f"{num(sev.se, 2)}; {sev_txt}; n = {num(sev.n)})."
    )
    fc_txt = f"{num(fc.estimate, 2)} (EE {num(fc.se, 2)})" + (
        ", distinguible de cero" if _sig(fc) else ", non distinguible de cero"
    )
    items.append(
        f"- **Do lume á plantación.** Efecto da fracción queimada en 2018–2021 sobre a "
        f"conversión bruta a eucalipto en 2024: {fc_txt}."
    )
    t, r = contr.loc["Targeted restoration"], contr.loc["Random restoration"]
    items.append(
        "- **Proxeccións a 2040.** Restaurar o 25 % do eucalipto cambia a superficie queimada "
        f"media en {num(t['d_burned_ha_per_year'])} ha/ano se se fai nas celas prioritarias "
        f"(banda 5–95 %: {num(t['d_burned_p05'])} a {num(t['d_burned_p95'])}) e en "
        f"{num(r['d_burned_ha_per_year'])} ha/ano se se fai ao chou (banda "
        f"{num(r['d_burned_p05'])} a {num(r['d_burned_p95'])})."
        + (
            " As dúas bandas inclúen o cero: as proxeccións non permiten afirmar que restaurar "
            "reduza os incendios, nin que os aumente."
            if (t["d_burned_p05"] < 0 < t["d_burned_p95"])
            and (r["d_burned_p05"] < 0 < r["d_burned_p95"])
            else " Ao menos unha banda exclúe o cero, pero as proxeccións herdan a sensibilidade "
            "ao mapa descrita arriba."
        )
    )
    return "\n".join(items)


def write_brief(res: dict, out_dir: str | Path) -> Path:
    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fire, conv, proj = res["fire"], res["conversion"], res["projections"]
    susc, areas, sm = res["susceptibility"], res["areas"], res["species_metrics"]

    _fig_maps(res, out / "mapas.png")
    _fig_effects(
        [
            (tr("eucalyptus -> P(burn)"), [fire["occurrence_naive"], fire["occurrence_dml"]]),
            (
                tr("eucalyptus -> burned share"),
                [fire["burned_frac_naive"], fire["burned_frac_dml"]],
            ),
            (
                tr("eucalyptus -> severity (EFFIS class)"),
                [fire["severity_naive"], fire["severity_dml"]],
            ),
            (
                tr("burned share 2018-2021 -> eucalyptus gain"),
                [conv["fire_conversion_naive"], conv["fire_conversion_dml"]],
            ),
        ],
        out / "efectos.png",
    )
    fig_gates(
        {
            "Por distancia ao mar": fire["gate_region"],
            "Por meteoroloxía de incendios": fire["gate_fwi"],
        },
        out / "grupos.png",
    )
    _fig_projections(proj["trajectories"], out / "proxeccions.png")

    nat = conv["native_to_euc_ha"]
    area_cols = [
        "name",
        "map_area_ha",
        "est_area_ha",
        "ci95_ha",
        "soft_area_ha",
        "users_accuracy",
        "producers_accuracy",
    ]
    cover_tab = pd.DataFrame(
        [
            {"cover": k[2:], "dP(burn)/dshare": e.estimate, "se": e.se}
            for k, e in fire["cover_effects"].items()
        ]
    )
    effects_tab = pd.DataFrame(
        [
            {"effect": e.name, "method": e.method, "estimate": e.estimate, "se": e.se, "n": e.n}
            for e in (
                fire["occurrence_naive"],
                fire["occurrence_dml"],
                fire["burned_frac_naive"],
                fire["burned_frac_dml"],
                fire["severity_naive"],
                fire["severity_dml"],
                conv["fire_conversion_naive"],
                conv["fire_conversion_dml"],
            )
        ]
    )
    trans = conv["transitions"]
    trans_top = trans[(trans["from"] != trans["to"]) & (trans["to"] == "eucalyptus")]
    trans_top = trans_top.assign(
        **{"from": trans_top["from"].map(tr), "to": trans_top["to"].map(tr)}
    )
    ps = res["panel_summary"]
    s17, s24 = sm["2017"], sm["2024"]

    resumo = _resumo(res)
    md = f"""# Eucalipto en Galicia: informe con datos reais

> **Que é este informe.** Unha estimación con datos de satélite e rexistros públicos do
> efecto das plantacións de eucalipto sobre o bosque autóctono e os incendios en Galicia.
> Os mapas de especies adestráronse con etiquetas de OpenStreetMap, non co Mapa Forestal de
> España nin co Inventario Forestal Nacional, que non eran accesibles desde este contorno.
> **Como se validou o mapa de eucalipto.** Case todas as etiquetas de eucalipto de
> OpenStreetMap están nun cadro de 100 km do norte (A Coruña, Ferrol, Ortegal), e moitas das
> de fóra teñen un comportamento invernal de frondosa caducifolia, é dicir, están mal
> etiquetadas. Por iso as etiquetas límpanse segundo o comportamento invernal (o eucalipto é
> perennifolio) e engádense pseudoetiquetas de eucalipto en toda Galicia: arboredo verde e
> húmido no inverno en contornas con cortas a matarrasa 2001–2016. Proba de transferencia:
> adestrando sen o cadro do norte e avaliando nas súas etiquetas de OpenStreetMap, o F1 do
> eucalipto é {num(sm["2024"]["north_transfer"]["euc_f1"], 2)} en 2024 (precisión
> {num(sm["2024"]["north_transfer"]["euc_precision"], 2)}, sensibilidade
> {num(sm["2024"]["north_transfer"]["euc_recall"], 2)}) e
> {num(sm["2017"]["north_transfer"]["euc_f1"], 2)} en 2017. Sen esta corrección era
> practicamente cero. Segue sen haber unha mostra de referencia independente fóra do norte.
> Os efectos causais dependen de supostos que se explican na sección 7. O efecto sobre a auga
> **non se puido estimar** con datos reais (sección 6).

## Resumo

{resumo}

## 1. Datos empregados

{_md_table(SOURCES)}

Panel: {num(ps["cells"])} celas de 1 km e {num(ps["cell_years"])} anos-cela (2018–2023), dos cales
{num(ps["burned_cell_years"])} rexistraron queimados.

![mapas](mapas.png)

## 2. Mapas de especies

Clasificador de potenciación de gradiente sobre trazos fenolóxicos (media, amplitude e fase anual
de NDVI, NDMI e NBR). Exactitude en validación cruzada por bloques espaciais de 20 km:
**{num(s17["spatial_cv_accuracy"], 3)}** (2017) e **{num(s24["spatial_cv_accuracy"], 3)}**
(2024). Kappa {num(s17["kappa"], 3)} e {num(s24["kappa"], 3)}. A exactitude mide o acordo coas
etiquetas de OpenStreetMap, que non son unha mostra aleatoria.

Proba de transferencia (adestramento sen o cadro do norte, avaliación nas súas etiquetas de
OpenStreetMap), F1 por clase:

{_md_table(pd.DataFrame({"name": CLASS_NAMES_GL, "2017": sm["2017"]["north_transfer"]["f1"], "2024": sm["2024"]["north_transfer"]["f1"]}), 3)}

Pseudoetiquetas de eucalipto engadidas: {num(sm["2024"]["n_pseudo_eucalyptus"])} píxeles
(o mesmo modelo clasifica os dous anos).

O mapa de 2017 retrodátase desde o de 2024: as imaxes de 2017 normalízanse radiometricamente
contra as de 2024 e clasifícanse co mesmo modelo, pero nos píxeles sen perturbación entre os dous
anos (sen perda arbórea de Hansen nin queimado de EFFIS) mantense a clase de 2024. Así, o
ruído do clasificador non crea cambios falsos, e un cambio real precisa de evidencia. Retrodatouse o
{num(sm["2017"]["share_backdated"] * 100, 3)} % dos píxeles.

Superficies. A columna «superficie estimada» corrixe o mapa invertindo a matriz de confusión
da validación cruzada. Esa corrección só é fiable se as etiquetas son puras e representativas;
as de OpenStreetMap non o son, así que **tómese como unha comprobación, non como estimación**.
Se se afasta moito da superficie do mapa, a diferenza indica ruído nas etiquetas máis ca un
erro do mapa.

2024:

{_md_table(areas["2024"][area_cols], 5, values=("name",))}

2017:

{_md_table(areas["2017"][area_cols], 5, values=("name",))}

## 3. Perda de bosque autóctono

Transicións cara ao eucalipto, 2017–2024 (píxeles de 40 m):

{_md_table(trans_top[["from", "to", "area_ha", "area_confident_ha"]], 5)}

A diferenza entre dous mapas acumula os erros de ambos e sobreestima o cambio (na validación
sintética, ao redor do dobre). Por iso o informe dá tres cifras para autóctonas → eucalipto:
todos os píxeles, {num(nat["all_pixels"], 3)} ha; píxeles fiables, {num(nat["confident"], 3)} ha;
e píxeles fiables con perda arbórea ou incendio que o corroboren,
{num(nat["confident_with_loss_or_fire"], 3)} ha. A última é a máis prudente.

Atribución da perda de cuberta arbórea 2018–2024 (Hansen) a 40 m:

{_md_table(conv["loss_attribution"], values=("driver",))}

## 4. Incendios

{_md_table(effects_tab, values=("effect", "method"))}

![efectos](efectos.png)

Efecto de cada cuberta sobre a probabilidade anual de queima, fronte a agricultura e outros usos:

{_md_table(cover_tab, values=("cover",))}

Onde é maior o efecto do eucalipto:

{_md_table(fire["gate_region"], values=("group",))}

{_md_table(fire["gate_fwi"], values=("group",))}

![grupos](grupos.png)

Sensibilidade ao mapa de eucalipto (efecto sobre a probabilidade anual de queima por unidade
de fracción):

{_md_table(fire["map_sensitivity"].assign(map=fire["map_sensitivity"]["map"].map(GL_MAP)))}

Sensibilidade á confusión non observada:

{_md_table(fire["sensitivity"], values=("effect",))}

Erro estándar do efecto sobre a aparición de incendios segundo o tamaño do bloque:

{_md_table(fire["se_by_block"])}

Modelo de susceptibilidade: AUC en validación cruzada espacial {num(susc["auc"], 3)}.

## 5. Proxeccións ano a ano, 2025–2040

Motor dinámico: cada ano sortéase o lume segundo a susceptibilidade de cada cela e os efectos
causais estimados de cada cuberta; o lume converte parte das frondosas e dos piñeirais en mato;
e a plantación de eucalipto segue a taxa de conversión bruta observada en 2017–2024 entre
píxeles fiables ({num(proj["components"].diagnostics["conv_rate_target"] * 100, 2)} % ao ano da
superficie sen eucalipto), reforzada polos incendios recentes. O motor non inclúe perdas de
eucalipto agás a restauración, así que o crecemento no escenario tendencial é un límite superior. O modelo validouse na paisaxe sintética (1 km), onde sobreestimou uns 40 % os beneficios
da restauración e moito máis os do límite, porque sobreestima a plantación de referencia. As bandas son os percentís 5 e 95 de 40 simulacións que combinan a
variabilidade meteorolóxica e a incerteza dos efectos.

{_md_table(proj["contrasts"], 4, values=("scenario",))}

![proxeccións](proxeccions.png)

Sensibilidade: se a plantación futura fose a metade da observada en 2017–2024 (por exemplo,
porque se manteñen as restricións a novas plantacións):

{_md_table(proj["contrasts_half_conversion"], 4, values=("scenario",))}

## 6. Auga

Non se estimou. Os datos de caudal (Augas de Galicia, anuario de aforos do CEDEX) non eran
accesibles desde este contorno, e ningunha fonte alcanzable medía a escorrentía. O código para
os paneis de concas e a curva de Budyko está listo e validado con datos sintéticos; só precisa
os caudais diarios das estacións.

## 7. Limitacións

- **Etiquetas.** As etiquetas de especie proceden de OpenStreetMap: {num(int(s24["per_class_train"][0]))}
  píxeles de adestramento de eucalipto en 2024, a clase con menos exemplos. Supúxose que o
  bosque frondoso perennifolio de Galicia é eucalipto; as aciñeiras e sobreiras quedarían mal
  clasificadas. Hai que validar os mapas co Mapa Forestal de España ou co IFN4.
- **Erro do mapa.** O erro do mapa atenúa os efectos cara a cero. Non se aplicou SIMEX porque
  non hai unha mostra de referencia independente para medir a varianza do erro.
- **Incendios.** EFFIS rexistra sobre todo os incendios grandes; os pequenos quedan fóra. Só hai
  seis anos (2018–2023) e 2022 domina o total.
- **Causalidade.** Os efectos son causais só se non queda confusión relevante sen medir (por
  exemplo, a intencionalidade dos lumes ou a propiedade das terras). A táboa de sensibilidade
  indica canto tería que pesar ese factor.
- **Meteoroloxía.** O índice meteorolóxico procede de seis estacións; non é o FWI oficial.

## Siglas

| sigla | significado |
|---|---|
| AUC | área baixo a curva ROC |
| DML | aprendizaxe automática dobre (estimación causal con axustes cruzados) |
| EE | erro estándar |
| EFFIS | Sistema Europeo de Información sobre Incendios Forestais |
| FWI | índice meteorolóxico de perigo de incendio |
| IC | intervalo de confianza |
| MCO | mínimos cadrados ordinarios (estimación inxenua) |
| VR | valor de robustez |
"""
    path = out / "informe.md"
    path.write_text(md)
    metrics = {
        "areas": {p: a.to_dict(orient="records") for p, a in areas.items()},
        "native_to_euc_ha": nat,
        "effects": effects_tab.to_dict(orient="records"),
        "contrasts": proj["contrasts"].to_dict(orient="records"),
        "panel": ps,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return path


__all__ = ["EUC", "INK", "write_brief"]
