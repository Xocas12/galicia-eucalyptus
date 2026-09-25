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
        "subset": "subconxunto",
        "epoch": "época",
        "cv_accuracy": "exactitude (validación cruzada)",
        "euc_f1": "F1 eucalipto (validación cruzada)",
        "euc_area_kha": "eucalipto (mil ha)",
        "ifn3_f1": "F1 fronte ao IFN3",
        "n_plots": "parcelas",
        "false_euc_plots": "parcelas sen eucalipto que o mapa chama eucalipto",
        "n_euc_plots": "parcelas con eucalipto",
        "precision": "precisión",
        "f1": "F1",
        "plot_type": "tipo de parcela",
        "n_euc": "puntos de eucalipto",
        "n_other_forest": "puntos doutro arboredo",
        "recall": "sensibilidade",
        "recall_ci_low": "IC 95 % inferior",
        "recall_ci_high": "IC 95 % superior",
        "recall_3x3": "sensibilidade (3×3 píxeles)",
        "false_euc_rate": "arboredo tomado por eucalipto",
        "precision_at_prior": "precisión (prevalencia do mapa)",
        "f1_at_prior": "F1 (prevalencia do mapa)",
        "genus": "xénero",
        "n": "puntos",
        "share_mapped_euc": "fracción no mapa como eucalipto",
        "estimator": "estimador",
        "change_scale": "cambio de cuberta (× o real)",
        "n_catchments": "concas",
        "true_mm_per_10pts": "efecto real (mm/ano por 10 puntos)",
        "mean_estimate": "estimación media",
        "bias": "nesgo",
        "sd_estimate": "desviación típica",
        "coverage": "cobertura IC 95 %",
        "power": "potencia",
        "mde_mm_per_10pts": "efecto mínimo detectable (mm/ano por 10 puntos)",
        "TWFE": "efectos fixos dobres",
        "TWFE + slopes": "efectos fixos dobres + pendente de choiva por conca",
        "todo": "todos",
        "fora_do_norte": "fóra do cadro do norte",
        "norte": "cadro do norte",
        "fora_das_etiquetas": "fóra dos polígonos de adestramento",
        "fora_do_norte_e_etiquetas": "fóra do norte e dos polígonos",
    }
)

REF_SUBSETS = ["todo", "fora_do_norte", "fora_do_norte_e_etiquetas", "norte"]


def _reference_table(ref: dict, period: str) -> pd.DataFrame:
    rows = []
    for k in REF_SUBSETS:
        r = ref[period].get(k)
        if not r or r.get("too_few"):
            continue
        rows.append(
            {
                "subset": k,
                "n_euc": r["n_euc"],
                "n_other_forest": r["n_other_forest"],
                "recall": r["recall"],
                "recall_ci_low": r["recall_ci"][0],
                "recall_ci_high": r["recall_ci"][1],
                "recall_3x3": r["recall_3x3"],
                "false_euc_rate": r["false_euc_rate"],
                "precision_at_prior": r["precision_at_prior"],
                "f1_at_prior": r["f1_at_prior"],
            }
        )
    return pd.DataFrame(rows)


PLOT_SUBSETS = ["todo", "fora_do_norte", "fora_das_etiquetas", "norte"]
PLOT_TYPES_ORDER = [
    "eucalipto",
    "piñeiro sen eucalipto",
    "frondosas sen eucalipto nin piñeiro",
    "só mato",
]


def _inventory_tables(inv: dict, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for k in PLOT_SUBSETS:
        r = inv[period][k]
        rows.append(
            {
                "subset": k,
                "n_plots": r["n_plots"],
                "n_euc_plots": r["n_euc_plots"],
                "recall": r["recall"],
                "precision": r["precision"],
                "f1": r["f1"],
                "false_euc_plots": r["false_euc_rate"],
            }
        )
    bt = pd.DataFrame(inv[period]["by_type"]).set_index("plot_type").reindex(PLOT_TYPES_ORDER)
    names = [*CLASS_NAMES_GL, "sen datos"]
    bt = bt.rename(columns={str(k): v for k, v in enumerate(names)})
    bt = bt.rename(columns=dict(enumerate(names)))
    bt = bt.reset_index().rename(columns={"n": "n_plots"})[
        ["plot_type", "n_plots", *CLASS_NAMES_GL]
    ]
    return pd.DataFrame(rows), bt


def _experiment_txt(ex: dict | None) -> str:
    if not ex:
        return ""
    base = ex["current_map"]["f1"]
    added = [v["f1"] for k, v in ex.items() if k.startswith("inventory_w")]
    better = max(added) > base + 0.02
    return (
        "- **Adestrar coas parcelas "
        + ("mellora o mapa" if better else "non mellora o mapa")
        + f".** Nun experimento, as parcelas da metade dos bloques de 10 km "
        f"({num(ex['n_train_plots'])} sen perturbación desde 2001) engadíronse ao adestramento "
        f"e avaliouse nas {num(ex['n_test_plots'])} da outra metade: F1 {num(base, 2)} co mapa "
        f"actual e {num(min(added), 2)}–{num(max(added), 2)} coas parcelas; adestrando só coas "
        f"parcelas, {num(ex['inventory_only']['f1'], 2)}. Unha parcela con algún eucalipto non é "
        "unha boa etiqueta para un píxel de 40 m, así que este acordo é en parte un teito da "
        "referencia, non só do mapa."
    )


def _timing_txt(li: dict | None) -> str:
    if not li:
        return ""
    l00, s24 = li["Landsat 2000"]["f1"], li["Sentinel-2 2024"]["f1"]
    return (
        f"Pero un mapa da mesma época ca o inventario (Landsat 2000, sección seguinte) non "
        f"concorda mellor coas parcelas (F1 {num(l00, 2)}, fronte a {num(s24, 2)} do mapa de "
        "2024): a maior parte do desacordo vén da propia referencia (calquera eucalipto nun "
        "círculo de 25 m) e do erro do mapa, non do cambio desde 1998."
        if l00 <= s24 + 0.02
        else f"Un mapa da mesma época (Landsat 2000) concorda mellor (F1 {num(l00, 2)})."
    )


def _landsat_section(bc: dict | None, li: dict | None) -> str:
    if not bc:
        return ""
    ep = list(bc["euc_f1"])
    tab = pd.DataFrame(
        {
            "epoch": ep,
            "cv_accuracy": [bc["cv_accuracy"][e] for e in ep],
            "euc_f1": [bc["euc_f1"][e] for e in ep],
            "euc_area_kha": [bc["euc_area_ha"][e] / 1e3 for e in ep],
            "ifn3_f1": [li[f"Landsat {e}"]["f1"] if li else np.nan for e in ep],
        }
    )
    verdict = (
        "**O mapa histórico supera a validación** e úsase na sección 6."
        if bc["passed"]
        else "**O mapa histórico non supera a validación, e non se usa.** Coas imaxes "
        "Landsat de nivel 1 (reflectancia no alto da atmosfera, sen corrección atmosférica) e "
        "poucas escenas por estación, o clasificador non separa o eucalipto o bastante: a "
        "superficie non mostra tendencia e o «cambio» entre épocas é ruído. Para facelo ben "
        "cómpren as imaxes Landsat de reflectancia de superficie (Colección 2), que non eran "
        "accesibles desde este contorno."
    )
    return f"""### Mapa histórico con Landsat, 1990–2017

Para ter un historial de cuberta máis longo (sección 6) clasificáronse compostos estacionais
Landsat 4–8 (arquivo público de Google Cloud; inverno e verán, NDVI, NDMI e NBR) en catro
épocas de tres anos. Cada época ten o seu clasificador, adestrado en píxeles sen cambios
desde 2001 (mesma clase nos dous mapas de Sentinel-2, sen perda de Hansen nin lume).

{_md_table(tab, 3)}

Comprobación de cambio: dos píxeles que pasan a eucalipto entre 2000 e 2010, o
{num(bc["gain_with_loss"] * 100, 2)} % tivo unha corta rexistrada por Hansen en 2001–2010,
fronte ao {num(bc["same_with_loss"] * 100, 2)} % dos píxeles sen cambio. Unha plantación
nova vén case sempre dunha corta, así que a proporción debería ser moito maior.

{verdict}"""


def _reference_section(
    ref: dict | None,
    inv: dict | None = None,
    transfer_f1: float = float("nan"),
    landsat_inv: dict | None = None,
    backcast: dict | None = None,
) -> str:
    parts = []
    if inv:
        i24 = inv["2024"]
        t24, b24 = _inventory_tables(inv, "2024")
        t17, _ = _inventory_tables(inv, "2017")
        t17i, _ = _inventory_tables(inv, "2017_independent")
        parts.append(f"""### Comprobación con parcelas de inventario forestal

O Mapa Forestal de España e o IFN4 non se podían descargar desde este contorno, pero o arquivo
de GBIF contén as parcelas do **Terceiro Inventario Forestal Nacional (IFN3)**, publicadas polo
Ministerio (código de institución MAGRAMA, colección IFN3, licenza CC BY-NC 4.0): unha malla
sistemática de 1 km coa lista de especies de cada parcela, sen número de pés nin data. O
traballo de campo do IFN3 en Galicia foi arredor de 1997–1998, así que as parcelas son dúas
décadas anteriores aos mapas de Sentinel-2. Quedan {num(inv["n_plots"])} parcelas. Por ser unha
mostra sistemática, dá unha precisión de deseño, non só a sensibilidade.

Unha parcela conta como «eucalipto» se a lista inclúe algún eucalipto; non se sabe se domina.
Precisión: das parcelas que o mapa chama eucalipto, fracción que ten eucalipto. Sensibilidade:
das parcelas con eucalipto, fracción que o mapa chama eucalipto.

Mapa de 2024:

{_md_table(t24, 3, values=("subset",))}

Mapa de 2017 retrodatado:

{_md_table(t17, 3, values=("subset",))}

Mapa de 2017 clasificado de forma independente:

{_md_table(t17i, 3, values=("subset",))}

Clase do mapa de 2024 segundo o tipo de parcela (fracción de parcelas):

{_md_table(b24, 3)}

Que se conclúe:

- **No agregado o mapa acerta.** O {num(i24["map_euc_share_at_forest_plots"] * 100, 3)} % das
  parcelas arboradas está no mapa como eucalipto, e o {num(i24["plot_euc_share"] * 100, 3)} %
  das parcelas arboradas ten eucalipto.
- **Parcela a parcela o acordo é baixo**: F1 {num(i24["todo"]["f1"], 2)} en toda Galicia e
  {num(i24["fora_do_norte"]["f1"], 2)} fóra do norte, lonxe do
  {num(transfer_f1, 2)} da proba de transferencia con OpenStreetMap. Esa proba era optimista.
- **O tempo explica só unha parte.** Das parcelas sen eucalipto que o mapa chama eucalipto, o
  {num(i24["disturbed_share_false_euc"] * 100, 3)} % tivo corta ou lume desde 2001 (o primeiro
  ano de Hansen), fronte ao {num(i24["disturbed_share_other"] * 100, 3)} % do resto: algunhas
  son plantacións posteriores ao inventario. {_timing_txt(landsat_inv)}
{_experiment_txt(inv.get("experiment"))}

Consecuencia: as cifras de superficie son plausibles, pero a localización do eucalipto píxel a
píxel é incerta. Os efectos estimados sobre os incendios están atenuados por este erro
(sección 7).""")
    lsec = _landsat_section(backcast, landsat_inv)
    if lsec:
        parts.append(lsec)
    if ref:
        r24 = ref["2024"]
        genus = pd.DataFrame(r24["by_genus"]).sort_values("n", ascending=False)
        genus = genus[genus["n"] >= 20]
        parts.append(f"""### Comprobación con observacións de GBIF

Observacións directas de árbores e matogueiras en Galicia (iNaturalist, Observation.org e
outras), incerteza de coordenadas de 60 m como máximo, 2019–2025, un rexistro por xénero e
píxel: {num(r24["n_points"])} puntos. Os naturalistas case non rexistran plantacións, así que hai
poucos puntos de eucalipto, e para 2017 non abondan. Precisión e F1 calculadas supoñendo que o
eucalipto é o {num(r24["prior_euc"] * 100, 3)} % do arboredo. A columna «3×3 píxeles» acepta o
acerto nun píxel veciño.

{_md_table(_reference_table(ref, "2024"), 3, values=("subset",))}

Fracción dos puntos de cada xénero que o mapa de 2024 clasifica como eucalipto:

{_md_table(genus[["genus", "n", "share_mapped_euc"]], 3)}""")
    if not parts:
        return "### Comprobación independente\n\nNon dispoñible."
    return "\n\n".join(parts)


def _water_section(w: dict | None) -> str:
    if not w:
        return "Non se estimou: faltan os datos de caudal."
    c = w["catchments"]
    mde = pd.DataFrame(w["mde"])
    mde = mde[mde["estimator"] == "TWFE"].drop(columns="estimator")
    real = mde[(mde["change_scale"] == 1.0)]
    m79 = float(real["mde_mm_per_10pts"].iloc[-1])
    err = pd.DataFrame(w["power_map_error"])
    err = err[err["estimator"] == "TWFE"][
        ["change_scale", "true_mm_per_10pts", "mean_estimate", "bias", "coverage", "power"]
    ]
    pw = pd.DataFrame(w["power"])
    pw = pw[(pw["estimator"] == "TWFE") & (pw["n_catchments"] == pw["n_catchments"].max())][
        ["change_scale", "true_mm_per_10pts", "mean_estimate", "bias", "coverage", "power"]
    ]
    e6 = err[(err["change_scale"] == err["change_scale"].max()) & (err["true_mm_per_10pts"] != 0)]
    ratio = float(e6["mean_estimate"].iloc[0] / e6["true_mm_per_10pts"].iloc[0])
    detectable = m79 <= 20
    g = w.get("gauges")
    if g and "runoff_mm_per_10pts" in g:
        e, se = g["runoff_mm_per_10pts"]
        gauge_txt = (
            f"**Estimación con aforos reais** ({g['n']} estacións, {g['n_years']} anos-estación): "
            f"un aumento de 10 puntos de eucalipto cambia a escorrentía anual en {num(e, 3)} mm "
            f"(IC 95 %: {num(e - 1.96 * se, 3)} a {num(e + 1.96 * se, 3)})."
        )
    else:
        gauge_txt = (
            "**Non hai estimación con datos reais.** Os caudais (anuario de aforos do CEDEX, "
            "Augas de Galicia, MeteoGalicia, GRDC) non eran accesibles desde este contorno. "
            "Ao copiar os ficheiros das estacións en `data/raw/gauges/` (formato do CEDEX ou "
            "CSV xenérico), o mesmo código fai a estimación."
        )
    return f"""{gauge_txt}

O que si se fixo é preparar e validar o deseño con datos reais agás os caudais:

- **Concas.** Delimitáronse desde o modelo dixital do terreo Copernicus (200 m) {c["n"]}
  concas enteiras, sen aniñar, de 30 a 1 500 km² (mediana {num(c["area_km2_median"], 3)} km²),
  que representan unha rede de aforos. Clima de cada ano hidrolóxico (outubro–setembro):
  choiva e evapotranspiración potencial (Thornthwaite) das estacións GHCN. Cuberta de cada conca e
  ano: os mapas de 2017 e 2024, co cambio datado pola perda arbórea de Hansen ou polo incendio.
- **Problema principal.** O eucalipto medio das concas en 2024 é do
  {num(c["euc_2024_mean"] * 100, 3)} %, pero dentro de cada conca só cambia
  {num(c["within_change_mean_pts"], 2)} puntos de media entre 2017 e 2024 (percentil 90:
  {num(c["within_change_p90_pts"], 2)}). Un panel de concas con efectos fixos só aprende deste
  cambio interno.
- **Proba de potencia.** Simuláronse caudais nas concas reais co clima real e un efecto
  coñecido do eucalipto (curva de Fu con parámetro propio de cada conca, choque anual común e
  erro do 8 % por conca e ano), e estimouse o efecto co mesmo modelo de efectos fixos dobres,
  200 veces por caso. O nesgo é pequeno fronte ao erro típico e a cobertura do IC 95 % está
  preto do 95 % (sen nesgo apreciable cun historial máis longo), pero coas
  {c["n"]} concas o efecto mínimo detectable (potencia do 80 %) é de
  **{num(m79, 3)} mm/ano por 10 puntos** de eucalipto. Aquí suponse que un efecto plausible, de
  substituír frondosas por eucalipto, é de 10–20 mm/ano por 10 puntos (100–200 mm/ano nunha
  conca enteira).
  «Cambio de cuberta × 3» ou «× 6» simula un historial máis longo (por exemplo, mapas desde os
  anos noventa con Landsat), que é o que faría detectable un efecto de 10–20 mm/ano.

Efecto mínimo detectable:

{_md_table(mde, 3)}

Resultados co número máximo de concas:

{_md_table(pw, 3)}

Con erro de mapa realista (caudais simulados co mapa de 2017 independente, estimación co
retrodatado):

{_md_table(err, 3)}

O erro de mapa pesa tanto coma o ruído: coa mesma conca e o mesmo caudal, cambiar de versión
do mapa multiplica a estimación por {num(ratio, 2)} (e a cobertura do IC cae). Por iso calquera
estimación con aforos debería repetirse coas dúas versións do mapa, como se fai cos incendios.

{_water_conclusion(detectable, w.get("backcast"))}"""


def _water_conclusion(detectable: bool, backcast: dict | None = None) -> str:
    tail = ""
    if backcast and not backcast["passed"]:
        tail = (
            " Intentouse ese historial con Landsat (sección 2), pero o mapa histórico non "
            "superou a validación."
        )
    return _water_conclusion_base(detectable) + tail


def _water_conclusion_base(detectable: bool) -> str:
    if detectable:
        return (
            "Conclusión: cos mapas dispoñibles, os aforos permitirían detectar un efecto "
            "plausible (arredor de 20 mm/ano por 10 puntos)."
        )
    return (
        "Conclusión: **cos mapas dispoñibles (2017 e 2024), nin sequera cos aforos se podería "
        "medir o efecto do eucalipto sobre o caudal anual**. Fai falta un historial de cuberta "
        "máis longo (Landsat desde os anos noventa) ou un deseño de concas pareadas."
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
        "cifras non están validadas** co inventario oficial descargado do Ministerio (IFN, Mapa "
        "Forestal de España): compárense con el antes de citalas (sección 2)."
        + _ref_resumo(res.get("reference"), res.get("inventory"))
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
    w = res.get("water")
    if w:
        mde = pd.DataFrame(w["mde"])
        mde = mde[(mde["estimator"] == "TWFE") & (mde["change_scale"] == 1.0)]
        items.append(
            "- **Auga.** Sen datos de caudal non hai estimación. Unha proba de potencia nas "
            f"{w['catchments']['n']} concas reais mostra que, cos mapas de 2017 e 2024, os aforos "
            "só detectarían un efecto de "
            f"{num(float(mde['mde_mm_per_10pts'].iloc[-1]), 3)} mm/ano por 10 puntos de "
            "eucalipto ou maior; fai falta un historial de cuberta máis longo (sección 6)."
        )
    return "\n".join(items)


def _ref_resumo(ref: dict | None, inv: dict | None = None) -> str:
    if inv:
        i = inv["2024"]
        f = i["fora_do_norte"]
        return (
            f" Contra {num(inv['n_plots'])} parcelas do IFN3 (arredor de 1998, publicadas en GBIF), o "
            f"mapa dá eucalipto no {num(i['map_euc_share_at_forest_plots'] * 100, 3)} % das "
            f"parcelas arboradas e as parcelas teñen eucalipto no "
            f"{num(i['plot_euc_share'] * 100, 3)} %: o total cadra. Parcela a parcela o acordo é "
            f"baixo (F1 {num(f['f1'], 2)} fóra do norte), en parte porque as parcelas son "
            "anteriores a moitas plantacións."
        )
    if not ref:
        return ""
    r = ref["2024"].get("fora_do_norte")
    if not r or r.get("too_few"):
        return ""
    return (
        f" Fóra do norte, dos {num(r['n_euc'])} puntos de eucalipto de GBIF o mapa de 2024 "
        f"recoñece o {num(r['recall'] * 100, 3)} %."
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
> practicamente cero. A comprobación independente coas parcelas do IFN3 en toda Galicia
> (sección 2) é máis severa: a superficie total cadra, pero parcela a
> parcela o acordo é baixo.
> Os efectos causais dependen de supostos que se explican na sección 7. O efecto sobre a auga
> **non se puido estimar** con datos reais; a sección 6 avalía se sería medible con aforos.

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

{_reference_section(res.get("reference"), res.get("inventory"), sm["2024"]["north_transfer"]["euc_f1"], res.get("landsat_inventory"), (res.get("water") or {}).get("backcast"))}

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

{_water_section(res.get("water"))}

## 7. Limitacións

- **Etiquetas.** As etiquetas de especie proceden de OpenStreetMap: {num(int(s24["per_class_train"][0]))}
  píxeles de adestramento de eucalipto en 2024, a clase con menos exemplos. Supúxose que o
  bosque frondoso perennifolio de Galicia é eucalipto; as aciñeiras e sobreiras quedarían mal
  clasificadas. A comprobación con GBIF (sección 2) é independente pero oportunista; a
  validación definitiva debe facerse co Mapa Forestal de España ou co IFN4.
- **Erro do mapa.** As parcelas de inventario mostran que o erro de localización do eucalipto
  é grande, e ese erro atenúa os efectos cara a cero. Non se aplicou SIMEX porque as parcelas
  (presenza de eucalipto nun círculo de 25 m, sen data) non miden o mesmo ca o píxel, así que
  non dan a varianza do erro.
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
| GBIF | Global Biodiversity Information Facility (rexistros de biodiversidade) |
| IFN3, IFN4 | Terceiro e Cuarto Inventario Forestal Nacional |
| GHCN | rede mundial de estacións meteorolóxicas da NOAA |
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
