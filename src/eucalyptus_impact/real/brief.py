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
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, col, lab in (
        (axes[0], "eucalyptus_ha", "Superficie de eucalipto (miles de ha)"),
        (axes[1], "expected_burned_ha", "Superficie queimada esperada (miles de ha/ano)"),
    ):
        for color, (name, t) in zip(SERIES, trajs.items(), strict=False):
            ax.plot(t["year"], t[col] / 1e3, color=color, lw=2, label=tr(name))
            lo, hi = t.get(col + "_p05"), t.get(col + "_p95")
            if lo is not None:
                ax.fill_between(t["year"], lo / 1e3, hi / 1e3, color=color, alpha=0.12, lw=0)
        ax.set_title(lab, loc="left")
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    axes[0].legend(fontsize=8, loc="best")
    _gl_ticks(fig)
    _save(fig, path)


def _ci(e) -> str:
    lo, hi = e.estimate - 1.96 * e.se, e.estimate + 1.96 * e.se
    return f"{num(e.estimate, 3)} (IC 95 %: {num(lo, 3)} a {num(hi, 3)})"


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

    a17, a24 = areas["2017"], areas["2024"]
    euc17 = a17.loc[a17["name"] == "eucalyptus"].iloc[0]
    euc24 = a24.loc[a24["name"] == "eucalyptus"].iloc[0]
    nat = conv["native_to_euc_ha"]
    occ = fire["occurrence_dml"]
    rv = fire["sensitivity"].iloc[0]
    contr = proj["contrasts"].set_index("scenario")
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

    md = f"""# Eucalipto en Galicia: informe con datos reais

> **Que é este informe.** Unha estimación con datos de satélite e rexistros públicos do
> efecto das plantacións de eucalipto sobre o bosque autóctono e os incendios en Galicia.
> Os mapas de especies adestráronse con etiquetas de OpenStreetMap, non co Mapa Forestal de
> España nin co Inventario Forestal Nacional, que non eran accesibles desde este contorno.
> Os efectos causais dependen de supostos que se explican na sección 7. O efecto sobre a auga
> **non se puido estimar** con datos reais (sección 6).

## Resumo

- **Superficie de eucalipto.** O mapa de 2024 clasifica como eucalipto
  {num(euc24["map_area_ha"] / 1e3, 3)} mil ha, e a estimación corrixida polo erro do mapa é de
  {num(euc24["est_area_ha"] / 1e3, 3)} mil ha (± {num(euc24["ci95_ha"] / 1e3, 3)} mil ha). En 2017
  eran {num(euc17["est_area_ha"] / 1e3, 3)} mil ha (± {num(euc17["ci95_ha"] / 1e3, 3)} mil ha).
- **Substitución de bosque autóctono.** Entre 2017 e 2024, {num(nat["confident"], 3)} ha
  pasaron de frondosas autóctonas a eucalipto nos píxeles clasificados con fiabilidade nos
  dous anos, e {num(nat["confident_with_loss_or_fire"], 3)} ha diso coinciden cunha perda de
  cuberta arbórea (Hansen) ou cun incendio (EFFIS), o que corrobora o cambio.
- **Incendios.** Un aumento de 10 puntos na fracción de eucalipto dunha cela de 1 km cambia a
  probabilidade anual de que arda (≥ 1 % da cela) en {num(occ.estimate * 0.1 * 100, 2)} puntos
  porcentuais (IC 95 %: {num((occ.estimate - 1.96 * occ.se) * 10, 2)} a
  {num((occ.estimate + 1.96 * occ.se) * 10, 2)}), mantendo constantes o relevo, a localización,
  a presión humana, a meteoroloxía e as demais cubertas. A taxa base é
  {num(susc["base_rate"] * 100, 2)} % ao ano. O valor de robustez é
  {num(rv["rv_estimate"], 2)}: un factor de confusión non observado que explicase ese R² parcial
  do tratamento e do resultado anularía a estimación.
- **Proxeccións a 2040.** Fronte á tendencia actual, restaurar o 25 % da superficie de
  eucalipto nas celas prioritarias cambia a superficie queimada media en
  {num(contr.loc["Targeted restoration", "d_burned_ha_per_year"], 3)} ha/ano, e facelo ao
  chou en {num(contr.loc["Random restoration", "d_burned_ha_per_year"], 3)} ha/ano.

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

Superficies en 2024:

{_md_table(areas["2024"][area_cols], 5, values=("name",))}

Superficies en 2017:

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

Sensibilidade á confusión non observada:

{_md_table(fire["sensitivity"], values=("effect",))}

Erro estándar do efecto sobre a aparición de incendios segundo o tamaño do bloque:

{_md_table(fire["se_by_block"])}

Modelo de susceptibilidade: AUC en validación cruzada espacial {num(susc["auc"], 3)}.

## 5. Proxeccións ano a ano, 2025–2040

Motor dinámico: cada ano sortéase o lume segundo a susceptibilidade de cada cela e os efectos
causais estimados de cada cuberta; o lume converte parte das frondosas e dos piñeirais en mato;
e a plantación de eucalipto segue a taxa observada en 2017–2024, reforzada polos incendios
recentes. O modelo validouse na paisaxe sintética, onde sobreestimou os beneficios da
restauración nun 25–35 %. As bandas son os percentís 5 e 95 de 40 simulacións que combinan a
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
