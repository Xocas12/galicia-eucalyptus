"""Galician labels for the report.

The pipeline works with English identifiers internally. Everything a reader of the report sees,
including prose, table headers, effect and method names, classes, scenarios and figure text, goes
through `tr`, which is strict. An identifier without a translation is recorded in `MISSING`, and
the tests fail if any is left, so no English label can leak into the report unnoticed.
"""

from __future__ import annotations

import pandas as pd

MISSING: set[str] = set()

GL: dict[str, str] = {
    # Effects
    "eucalyptus -> P(burn)": "eucalipto → P(queima)",
    "eucalyptus -> dNBR": "eucalipto → severidade (dNBR)",
    "recent fire -> conversion rate": "incendio recente → taxa de conversión",
    "eucalyptus -> runoff": "eucalipto → escorrentía",
    "f_eucalyptus -> runoff": "eucalipto → escorrentía",
    "f_eucalyptus -> low_flow": "eucalipto → caudal estival",
    "eucalyptus -> summer soil moisture": "eucalipto → humidade estival do solo",
    "eucalyptus stand vs other -> soil moisture": "masa de eucalipto fronte a outras → humidade do solo",
    "reverse check: future eucalyptus gain ~ P(burn)": (
        "comprobación inversa: ganancia futura de eucalipto ~ P(queima)"
    ),
    # Methods
    "OLS": "MCO",
    "DML-PLR": "DML",
    "DML-PLR (ME-corrected)": "DML (corrección do erro de medida)",
    "DML-PLR (SIMEX)": "DML (SIMEX)",
    "TWFE": "Efectos fixos",
    "Budyko-Fu": "Budyko-Fu",
    "Matching (bias-corrected)": "Emparellamento (corrixido)",
    # Cover classes
    "eucalyptus": "eucalipto",
    "pine": "piñeiro",
    "native_broadleaf": "frondosas autóctonas",
    "shrub": "mato",
    "agriculture": "agricultura",
    "other": "outros",
    # Change classes
    "native->eucalyptus": "autóctonas → eucalipto",
    "other->eucalyptus": "outras → eucalipto",
    "eucalyptus stable": "eucalipto estable",
    # Loss drivers
    "fire": "incendio",
    "rotation": "corta de rotación",
    "conversion": "conversión",
    # Scenarios
    "BAU": "Tendencial",
    "Cap / moratorium": "Límite / moratoria",
    "Targeted restoration": "Restauración dirixida",
    "Random restoration": "Restauración aleatoria",
    # Groups
    "1 coast": "costa",
    "2 transition": "transición",
    "3 interior": "interior",
    "1 low FWI": "FWI baixo",
    "2 mid FWI": "FWI medio",
    "3 high FWI": "FWI alto",
    # Features
    "dist_mill_km": "distancia á fábrica de celulosa (km)",
    "elev": "altitude",
    "slope": "pendente",
    "neigh_euc": "eucalipto na veciñanza",
    "recent_fire": "incendio recente",
    "f_eucalyptus": "fracción de eucalipto",
    "f_native_broadleaf": "fracción de frondosas autóctonas",
    "f_shrub": "fracción de mato",
    "f_agriculture": "fracción agrícola",
    "f_pine": "fracción de piñeiro",
    "continentality": "continentalidade",
    "log_pop": "poboación (log)",
    "dist_road_km": "distancia á estrada (km)",
    "year": "ano",
    "precip_mean": "precipitación media",
    "pet_mean": "ETP media",
    "summer_temp": "temperatura estival",
    "dist_coast_km": "distancia á costa (km)",
    # Budyko parameters
    "w0": "w₀",
    "w_euc": "w eucalipto",
    "w_pine": "w piñeiro",
    "w_native": "w frondosas autóctonas",
    # Table headers
    "effect": "efecto",
    "method": "método",
    "estimate": "estimación",
    "se": "EE",
    "truth": "valor real",
    "covers_truth": "o IC contén o valor real",
    "name": "clase",
    "map_area_ha": "superficie no mapa (ha)",
    "est_area_ha": "superficie estimada (ha)",
    "ci95_ha": "IC 95 % (± ha)",
    "true_area_ha": "superficie real (ha)",
    "users_accuracy": "exactitude do usuario",
    "producers_accuracy": "exactitude do produtor",
    "driver": "causa",
    "attributed": "atribuída",
    "true": "real",
    "attributed_share": "proporción atribuída",
    "true_share": "proporción real",
    "feature": "variable",
    "smd_before": "DME antes",
    "smd_after": "DME despois",
    "group": "grupo",
    "n": "n",
    "rv_estimate": "VR da estimación",
    "rv_ci": "VR do IC",
    "max_bias_r2_0.02": "nesgo máx. (R² = 0,02)",
    "max_bias_r2_0.05": "nesgo máx. (R² = 0,05)",
    "block_km": "bloque (km)",
    "scenario": "escenario",
    "d_eucalyptus_ha": "Δ eucalipto (ha)",
    "d_burned_ha_simulated": "Δ queimado simulado (ha/ano)",
    "d_burned_ha_model": "Δ queimado modelo (ha/ano)",
    "d_runoff_mm_simulated": "Δ escorrentía simulada (mm)",
    "d_runoff_mm_model": "Δ escorrentía modelo (mm)",
    "d_burned_ha_mean_simulated": "Δ queimado medio simulado (ha/ano)",
    "d_burned_ha_mean_dynamic": "Δ queimado medio dinámico (ha/ano)",
    "cover": "cuberta",
    "dP(burn)/dshare": "dP(queima)/dfracción",
    "parameter": "parámetro",
}


def tr(key) -> str:
    """Galician label for an identifier; unknown keys are returned as-is and recorded."""
    key = str(key)
    if key in GL:
        return GL[key]
    MISSING.add(key)
    return key


def tr_frame(df: pd.DataFrame, value_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Translate the values of `value_cols`, then every column header."""
    out = df.copy()
    for c in value_cols:
        if c in out:
            out[c] = out[c].map(tr)
    return out.rename(columns={c: tr(c) for c in out.columns})


def num(x, nd: int = 4) -> str:
    """Galician number format: decimal comma, space as the thousands separator."""
    if abs(x) >= 1e4:
        s = f"{x:,.0f}"
    else:
        s = f"{x:,.{nd}g}"
    return s.replace(",", " ").replace(".", ",")
