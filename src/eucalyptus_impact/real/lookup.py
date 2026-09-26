"""Point lookup: local cover, fire history, risk scores and upstream catchment for a coordinate.

    euc lookup 42.88 -8.54            # latitude, longitude (WGS84)
    euc lookup 42.88 -8.54 --lang gl  # labels in Galician
    euc lookup 42.88 -8.54 --json     # machine-readable

Reads the cached outputs of `euc real run` (species maps, layers, cell_scores.parquet,
fire_effects.json); nothing is downloaded. Scores are the pipeline's, with their caveats:

- the 1 km fire probability is the susceptibility model's fitted annual burn probability at
  average weather, recalibrated to the 2018-2023 burn rate; its percentile is across Galicia;
- the eucalyptus contribution is the estimated cover effect times the local eucalyptus share,
  with a 95% interval, and is only as good as that estimate (sign-stable across map versions,
  not always significant; see the brief);
- the species class at 40 m has plot-level F1 ~0.45 against the IFN3 inventory, so read it as
  indicative for a single point.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .common import FINE_PER_CELL, GRID_40M, INTERIM

CLASSES = ["eucalyptus", "pine", "native_broadleaf", "shrub", "agriculture", "other"]
WORLDCOVER = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss/lichen",
}
FIRE_YEARS = range(2018, 2024)

GL = {
    "eucalyptus": "eucalipto",
    "pine": "piñeiro",
    "native_broadleaf": "frondosas autóctonas",
    "shrub": "mato",
    "agriculture": "agricultura",
    "other": "outros",
    "tree cover": "arboredo",
    "shrubland": "matogueira",
    "grassland": "pasteiro",
    "cropland": "cultivo",
    "built-up": "edificado",
    "bare": "chan espido",
    "water": "auga",
    "wetland": "zona húmida",
    "low": "baixo",
    "moderate": "medio",
    "high": "alto",
    "very high": "moi alto",
}


class OutsideGalicia(ValueError):
    pass


@lru_cache(maxsize=1)
def _data():
    sp = np.load(INTERIM / "species.npz")
    d = {
        "class24": sp["class40_2024"],
        "class17": sp["class40_2017"],
        "pmax24": sp["pmax40_2024"],
        "aoi": np.load(INTERIM / "aoi.npz")["mask40"],
        "wc": np.load(INTERIM / "worldcover.npz")["wc40"],
        "elev": np.load(INTERIM / "dem.npz")["elev40"],
        "lossyear": np.load(INTERIM / "hansen.npz")["lossyear40"],
    }
    ef = np.load(INTERIM / "effis.npz")
    d["burnt"] = {y: ef[f"burned40_{y}"] for y in FIRE_YEARS}
    cs = pd.read_parquet(INTERIM / "cell_scores.parquet")
    d["cells"] = cs.set_index(["row", "col"])
    d["effects"] = json.loads((INTERIM / "fire_effects.json").read_text())
    return d


def to_pixel(lat: float, lon: float) -> tuple[int, int, float, float]:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", GRID_40M.crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    col = int((x - GRID_40M.xmin) // GRID_40M.resolution_m)
    row = int((GRID_40M.ymax - y) // GRID_40M.resolution_m)
    return row, col, x, y


def risk_class(pct: float) -> str:
    """Percentile of the 1 km fire probability across Galicia -> class."""
    if pct >= 0.95:
        return "very high"
    if pct >= 0.80:
        return "high"
    if pct >= 0.50:
        return "moderate"
    return "low"


def eucalyptus_contribution(effects: dict, f_euc: float, other: str | None = None) -> dict:
    """Change in annual burn probability attributable to the local eucalyptus share.

    Against agriculture/other (the effects' baseline) when `other` is None; against another
    cover (e.g. native broadleaf) as the difference of the two effects otherwise, with the SEs
    combined as if independent (they come from separate fits).
    """
    e, se = effects["cover_effects"]["f_eucalyptus"]
    if other is not None:
        e2, se2 = effects["cover_effects"][f"f_{other}"]
        e, se = e - e2, float(np.hypot(se, se2))
    est, half = e * f_euc, 1.96 * se * f_euc
    return {"estimate": est, "ci": [est - half, est + half], "significant": abs(e) > 1.96 * se}


def nearest_plot(x: float, y: float, max_km: float = 1.5) -> dict | None:
    """Nearest IFN3 plot (surveyed ~1998) and the genera recorded in it."""
    from .reference import gbif_records, inventory_plots

    plots = _plots(gbif_records, inventory_plots)
    d = np.hypot(plots["x"].to_numpy() - x, plots["y"].to_numpy() - y)
    i = int(np.argmin(d))
    if d[i] > max_km * 1000:
        return None
    return {"distance_m": float(d[i]), "genera": sorted(plots["genera"].iloc[i])}


@lru_cache(maxsize=1)
def _plots(gbif_records, inventory_plots):
    return inventory_plots(gbif_records())


def upstream(row40: int, col40: int) -> dict | None:
    """Drainage area and eucalyptus share upstream of the point (200 m flow routing)."""
    from .water import AGG, GRID_HYDRO, catchment_members, routing, truncated

    rec, order, edge, acc = routing()
    ny, nx = GRID_HYDRO.shape
    r, c = row40 // AGG, col40 // AGG
    if not (0 <= r < ny and 0 <= c < nx):
        return None
    cell = r * nx + c
    if not np.isfinite(acc[cell]) or acc[cell] <= 0:
        return None
    mem = catchment_members(rec, order, np.array([cell]))[0]
    cls = _data()["class24"][: ny * AGG, : nx * AGG]
    rr, cc = np.divmod(mem, nx)
    block = cls.reshape(ny, AGG, nx, AGG)[rr, :, cc, :].reshape(len(mem), -1)
    ok = block < 6
    share = {
        k: float((block[ok] == i).mean()) if ok.any() else float("nan")
        for i, k in enumerate(CLASSES)
    }
    return {
        "area_km2": float(acc[cell]),
        "truncated": bool(truncated([mem], order, rec, edge)[0]),
        "cover_2024": share,
    }


def lookup(lat: float, lon: float, with_catchment: bool = True) -> dict:
    """Local information and risk scores for a WGS84 coordinate inside Galicia."""
    d = _data()
    row, col, x, y = to_pixel(lat, lon)
    ny, nx = GRID_40M.shape
    if not (0 <= row < ny and 0 <= col < nx) or d["aoi"][row, col] == 0:
        raise OutsideGalicia(f"({lat}, {lon}) is outside the Galicia study area")
    crow, ccol = row // FINE_PER_CELL, col // FINE_PER_CELL
    out: dict = {"lat": lat, "lon": lon, "utm29_x": round(x), "utm29_y": round(y)}
    c24, c17 = int(d["class24"][row, col]), int(d["class17"][row, col])
    ly = int(d["lossyear"][row, col])
    out["pixel_40m"] = {
        "class_2024": CLASSES[c24] if c24 < 6 else None,
        "class_2024_confidence_pct": int(d["pmax24"][row, col]),
        "class_2017": CLASSES[c17] if c17 < 6 else None,
        "worldcover_2021": WORLDCOVER.get(int(d["wc"][row, col])),
        "elevation_m": float(d["elev"][row, col]),
        "tree_cover_loss_year": 2000 + ly if ly > 0 else None,
        "burnt_years": [yr for yr in FIRE_YEARS if d["burnt"][yr][row, col]],
    }
    try:
        cs = d["cells"].loc[(crow, ccol)]
    except KeyError:
        cs = None
    if cs is not None:
        f24 = {k: float(cs[f"f_{k}_2024"]) for k in CLASSES}
        f17 = {k: float(cs[f"f_{k}_2017"]) for k in CLASSES}
        eff = d["effects"]
        out["cell_1km"] = {
            "cover_2024": f24,
            "cover_2017": f17,
            "distance_to_sea_km": float(cs["dist_sea_km"]),
            "slope_deg": float(cs["slope"]),
            "burnt_share_2018_2023": float(cs["burned_share"]),
        }
        out["risk"] = {
            "annual_fire_probability": float(cs["p_base"]),
            "galicia_mean": float(eff["base_rate"]),
            "percentile": float(cs["p_base_pct"]),
            "class": risk_class(float(cs["p_base_pct"])),
            "restoration_priority_percentile": float(cs["priority_pct"]),
            "in_targeted_restoration_set": bool(cs["targeted"]),
            "eucalyptus_contribution_vs_agriculture": eucalyptus_contribution(
                eff, f24["eucalyptus"]
            ),
            "eucalyptus_contribution_vs_native": eucalyptus_contribution(
                eff, f24["eucalyptus"], "native_broadleaf"
            ),
            "model_auc": float(eff["auc"]),
        }
    try:
        out["ifn3_plot"] = nearest_plot(x, y)
    except FileNotFoundError:
        out["ifn3_plot"] = None
    if with_catchment:
        try:
            out["upstream"] = upstream(row, col)
        except FileNotFoundError:
            out["upstream"] = None
    return out


def format_lookup(r: dict, lang: str = "en") -> str:
    t = (lambda s: GL.get(s, s)) if lang == "gl" else (lambda s: s)
    pct = lambda v: f"{100 * v:.1f}%".replace(".", "," if lang == "gl" else ".")  # noqa: E731
    pp = lambda v: f"{100 * v:+.2f} pp".replace(".", "," if lang == "gl" else ".")  # noqa: E731
    dec = lambda v, n=1: f"{v:.{n}f}".replace(".", "," if lang == "gl" else ".")  # noqa: E731
    L = (
        {
            "head": "Punto",
            "px": "Píxel de 40 m",
            "cls24": "clase 2024",
            "conf": "confianza",
            "cls17": "clase 2017",
            "wc": "WorldCover 2021",
            "elev": "altitude",
            "loss": "perda arbórea (Hansen)",
            "burnt": "queimado (EFFIS)",
            "none": "ningunha",
            "cell": "Cela de 1 km",
            "cov": "cuberta 2024",
            "cov17": "cuberta 2017",
            "sea": "distancia ao mar",
            "bshare": "fracción queimada 2018-2023 (suma dos anos)",
            "noeuc": "sen eucalipto na cela",
            "risk": "Risco de incendio",
            "prob": "probabilidade anual de queima (tempo medio)",
            "mean": "media de Galicia",
            "pctl": "percentil en Galicia",
            "class": "clase",
            "prio": "prioridade de restauración (percentil)",
            "sel": "na selección de restauración dirixida",
            "yes": "si",
            "no": "non",
            "cag": "achega do eucalipto fronte a agricultura/outros",
            "cnat": "achega do eucalipto fronte a frondosas",
            "ns": "non distinguible de cero",
            "plot": "Parcela IFN3 máis próxima (~1998)",
            "up": "Conca augas arriba",
            "area": "superficie",
            "trunc": "(sae da área de estudo: incompleta)",
            "caveat": "Estimacións do modelo, non medicións; ver o informe para os límites.",
        }
        if lang == "gl"
        else {
            "head": "Point",
            "px": "40 m pixel",
            "cls24": "class 2024",
            "conf": "confidence",
            "cls17": "class 2017",
            "wc": "WorldCover 2021",
            "elev": "elevation",
            "loss": "tree-cover loss (Hansen)",
            "burnt": "burnt (EFFIS)",
            "none": "none",
            "cell": "1 km cell",
            "cov": "cover 2024",
            "cov17": "cover 2017",
            "sea": "distance to sea",
            "bshare": "burnt share 2018-2023 (sum over years)",
            "noeuc": "no eucalyptus in the cell",
            "risk": "Fire risk",
            "prob": "annual burn probability (average weather)",
            "mean": "Galicia mean",
            "pctl": "percentile in Galicia",
            "class": "class",
            "prio": "restoration priority (percentile)",
            "sel": "in targeted-restoration set",
            "yes": "yes",
            "no": "no",
            "cag": "eucalyptus contribution vs agriculture/other",
            "cnat": "eucalyptus contribution vs native broadleaf",
            "ns": "not distinguishable from zero",
            "plot": "Nearest IFN3 plot (~1998)",
            "up": "Upstream catchment",
            "area": "area",
            "trunc": "(leaves the study area: incomplete)",
            "caveat": "Model estimates, not measurements; see the brief for their limits.",
        }
    )
    lines = [
        f"{L['head']}: {r['lat']:.5f}, {r['lon']:.5f} (ETRS89/UTM 29N {r['utm29_x']}, {r['utm29_y']})"
    ]
    p = r["pixel_40m"]
    lines += [
        f"\n{L['px']}",
        f"  {L['cls24']}: {t(p['class_2024'] or '-')} ({L['conf']} {p['class_2024_confidence_pct']}%)",
        f"  {L['cls17']}: {t(p['class_2017'] or '-')}",
        f"  {L['wc']}: {t(p['worldcover_2021'] or '-')}",
        f"  {L['elev']}: {p['elevation_m']:.0f} m",
        f"  {L['loss']}: {p['tree_cover_loss_year'] or L['none']}",
        f"  {L['burnt']}: {', '.join(map(str, p['burnt_years'])) or L['none']}",
    ]
    if "cell_1km" in r:
        c, k = r["cell_1km"], r["risk"]
        cov = lambda f: ", ".join(f"{t(n)} {pct(v)}" for n, v in f.items() if v >= 0.005)  # noqa: E731
        lines += [
            f"\n{L['cell']}",
            f"  {L['cov']}: {cov(c['cover_2024'])}",
            f"  {L['cov17']}: {cov(c['cover_2017'])}",
            f"  {L['sea']}: {dec(c['distance_to_sea_km'])} km",
            f"  {L['bshare']}: {pct(c['burnt_share_2018_2023'])}",
            f"\n{L['risk']}",
            f"  {L['prob']}: {pct(k['annual_fire_probability'])} ({L['mean']} {pct(k['galicia_mean'])})",
            f"  {L['pctl']}: {100 * k['percentile']:.0f} -> {L['class']}: {t(k['class'])}",
        ]
        if c["cover_2024"]["eucalyptus"] < 0.005:
            lines.append(f"  {L['noeuc']}")
        else:
            lines.append(
                f"  {L['prio']}: {100 * k['restoration_priority_percentile']:.0f}"
                f" ({L['sel']}: {L['yes'] if k['in_targeted_restoration_set'] else L['no']})"
            )
        for key, lab in (
            ("eucalyptus_contribution_vs_agriculture", "cag"),
            ("eucalyptus_contribution_vs_native", "cnat"),
        ):
            if c["cover_2024"]["eucalyptus"] < 0.005:
                break
            e = k[key]
            lines.append(
                f"  {L[lab]}: {pp(e['estimate'])} [{pp(e['ci'][0])}, {pp(e['ci'][1])}]"
                + ("" if e["significant"] else f" ({L['ns']})")
            )
    if r.get("ifn3_plot"):
        pl = r["ifn3_plot"]
        lines += [f"\n{L['plot']}: {pl['distance_m']:.0f} m, {', '.join(pl['genera'])}"]
    if r.get("upstream"):
        u = r["upstream"]
        cov = ", ".join(f"{t(n)} {pct(v)}" for n, v in u["cover_2024"].items() if v >= 0.005)
        lines += [
            f"\n{L['up']}: {L['area']} {dec(u['area_km2'], 2 if u['area_km2'] < 10 else 1)} km² {L['trunc'] if u['truncated'] else ''}",
            f"  {L['cov']}: {cov}",
        ]
    lines += ["", L["caveat"]]
    return "\n".join(lines)
