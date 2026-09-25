"""Water: catchments from the DEM, gauge ingestion, catchment-year panel and a power study.

No streamflow record is reachable from this environment (CEDEX, Augas de Galicia, MeteoGalicia,
GRDC and Zenodo are all blocked), so this module does two things:

1. Everything a gauge-based estimate needs except the flows: catchments delineated from the
   Copernicus DEM, water-year precipitation and PET from GHCN stations, and eucalyptus / pine /
   native fractions per catchment and year from the real species maps. Dropping gauge files in
   `data/raw/gauges/` (format in `load_gauges`) makes `water_analysis` run the fixed-effects
   and Budyko estimates in models/hydrology.py on real data.
2. A power study on the real catchments and the real 2017->2024 cover changes: flows are
   simulated with a known eucalyptus effect, the same estimators are run, and bias, coverage
   and power are reported. It says whether gauge data would detect an effect of a given size,
   and how much the map error matters. It is not an estimate of the effect.
"""

from __future__ import annotations

import heapq
import io
from pathlib import Path

import numpy as np
import pandas as pd

from ..geo.grid import Grid
from .common import BBOX, GRID_40M, INTERIM, RAW, curl, log

HYDRO_RES = 200  # m; 5 x 5 pixels of the 40 m grid
AGG = HYDRO_RES // 40
GRID_HYDRO = Grid.from_bbox(BBOX, HYDRO_RES)
WATER_YEARS = range(2001, 2025)  # water year Y = Oct Y-1 .. Sep Y
LONG_YEARS = range(1990, 2025)  # with the Landsat back-cast
GAUGE_DIR = RAW / "gauges"

# Mean day length (hours) by month at 42.5 N, for Thornthwaite PET.
DAYLENGTH = np.array([9.3, 10.4, 11.8, 13.3, 14.5, 15.2, 14.9, 13.8, 12.4, 10.9, 9.6, 8.9])
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# ---------------------------------------------------------------- flow routing


def priority_flood(z: np.ndarray):
    """Depression-filling flow routing (Barnes et al. 2014, priority-flood with directions).

    Cells are flooded inward from the grid edge and from NaN (sea) cells in order of their filled
    elevation; each cell drains to the cell that flooded it. Returns (receiver, order, edge):
    receiver is the flat index of the downstream cell (-1 at outlets), order is the flooding
    order (every cell comes after its receiver), and edge marks outlets on the grid border
    (catchments draining there may be truncated by the bounding box).
    """
    ny, nx = z.shape
    n = ny * nx
    zf = z.ravel()
    land = np.isfinite(zf)
    receiver = np.full(n, -1, np.int64)
    visited = ~land.copy()
    edge = np.zeros(n, bool)
    heap: list = []
    nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # Seeds: land cells on the border or next to sea.
    lm = land.reshape(ny, nx)
    sea_adj = np.zeros_like(lm)
    pad = np.pad(~lm, 1, constant_values=False)
    for dy, dx in nbr:
        sea_adj |= pad[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
    border = np.zeros_like(lm)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    for idx in np.flatnonzero((lm & (sea_adj | border)).ravel()):
        visited[idx] = True
        edge[idx] = border.ravel()[idx] and not sea_adj.ravel()[idx]
        heapq.heappush(heap, (float(zf[idx]), int(idx)))
    order = np.empty(int(land.sum()), np.int64)
    k = 0
    while heap:
        h, c = heapq.heappop(heap)
        order[k] = c
        k += 1
        r, q = divmod(c, nx)
        for dy, dx in nbr:
            rr, qq = r + dy, q + dx
            if 0 <= rr < ny and 0 <= qq < nx:
                m = rr * nx + qq
                if not visited[m]:
                    visited[m] = True
                    receiver[m] = c
                    heapq.heappush(heap, (max(float(zf[m]), h), m))
    return receiver, order[:k], edge


def accumulation(receiver: np.ndarray, order: np.ndarray, cell_km2: float) -> np.ndarray:
    acc = np.zeros(receiver.size)
    acc[order] = cell_km2
    for c in order[::-1]:
        r = receiver[c]
        if r >= 0:
            acc[r] += acc[c]
    return acc


def incremental_labels(receiver, order, outlets: np.ndarray) -> np.ndarray:
    """Label each cell with the first outlet downstream of it (1-based; 0 = none)."""
    lab = np.zeros(receiver.size, np.int32)
    lab[outlets] = np.arange(1, len(outlets) + 1)
    for c in order:
        if lab[c] == 0:
            r = receiver[c]
            if r >= 0:
                lab[c] = lab[r]
    return lab


def catchment_members(receiver, order, outlets: np.ndarray) -> list[np.ndarray]:
    """Flat cell indices of the full upstream catchment of each outlet (nesting handled)."""
    lab = incremental_labels(receiver, order, outlets)
    k = len(outlets)
    parent = np.array([lab[receiver[o]] if receiver[o] >= 0 else 0 for o in outlets])
    cells = [np.flatnonzero(lab == i + 1) for i in range(k)]
    children: dict[int, list[int]] = {}
    for i, p in enumerate(parent):
        if p > 0:
            children.setdefault(p - 1, []).append(i)

    def collect(i):
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(cells[j])
            stack += children.get(j, [])
        return np.concatenate(out)

    return [collect(i) for i in range(k)]


def hydro_dem() -> np.ndarray:
    """Copernicus DEM on the 200 m grid (mean of 40 m pixels), NaN over the sea."""
    L = np.load(INTERIM / "dem.npz")
    e = L["elev40"]
    ny, nx = GRID_HYDRO.shape
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = np.nanmean(e[: ny * AGG, : nx * AGG].reshape(ny, AGG, nx, AGG), axis=(1, 3))
    wc = np.load(INTERIM / "worldcover.npz")["wc40"][: ny * AGG, : nx * AGG]
    sea = (wc.reshape(ny, AGG, nx, AGG) == 80).mean(axis=(1, 3)) > 0.5
    aoi = np.load(INTERIM / "aoi.npz")["mask40"][: ny * AGG, : nx * AGG]
    offshore = (aoi.reshape(ny, AGG, nx, AGG) == 0).mean(axis=(1, 3)) > 0.5
    z[sea & offshore] = np.nan
    return z.astype("float64")


def routing():
    """Cached flow routing on the 200 m grid."""
    path = INTERIM / "routing.npz"
    if path.exists():
        z = np.load(path)
        return z["receiver"], z["order"], z["edge"], z["acc"]
    z = hydro_dem()
    rec, order, edge = priority_flood(z)
    acc = accumulation(rec, order, (HYDRO_RES / 1000) ** 2)
    np.savez_compressed(path, receiver=rec, order=order, edge=edge, acc=acc)
    return rec, order, edge, acc


def truncated(members: list[np.ndarray], order, receiver, edge) -> np.ndarray:
    """True where a catchment contains a cell draining off the bounding box (not whole)."""
    ends = set(np.flatnonzero(edge).tolist())
    return np.array([bool(ends.intersection(m.tolist())) for m in members])


# ---------------------------------------------------------------- gauges


def _cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_gauges(gauge_dir: Path = GAUGE_DIR):
    """Read gauge metadata and daily flows, or None when none were supplied.

    Two layouts are accepted:
    - generic: `stations.csv` (id, lon, lat[, area_km2][, name]) and `flows.csv`
      (id, date, q_m3s), comma-separated;
    - CEDEX Anuario de Aforos (ROEA): `estaf.csv` (indroea, xetrs89, yetrs89, suprest, ...) and
      `afliq.csv` (indroea, fecha, altura, caudal), semicolon-separated, dates dd/mm/yyyy.
    Coordinates in ETRS89/UTM 29N (x, y) or WGS84 (lon, lat).
    """
    gauge_dir = Path(gauge_dir)
    if (gauge_dir / "stations.csv").exists() and (gauge_dir / "flows.csv").exists():
        st = _cols(pd.read_csv(gauge_dir / "stations.csv", dtype={"id": str}))
        fl = _cols(pd.read_csv(gauge_dir / "flows.csv", dtype={"id": str}))
        fl["date"] = pd.to_datetime(fl["date"])
    elif (gauge_dir / "estaf.csv").exists() and (gauge_dir / "afliq.csv").exists():
        st = _cols(pd.read_csv(gauge_dir / "estaf.csv", sep=";", dtype=str, encoding="latin-1"))
        st = st.rename(
            columns={"indroea": "id", "xetrs89": "x", "yetrs89": "y", "suprest": "area_km2"}
        )
        for c in ("x", "y", "area_km2"):
            if c in st:
                st[c] = pd.to_numeric(st[c].str.replace(",", "."), errors="coerce")
        fl = _cols(pd.read_csv(gauge_dir / "afliq.csv", sep=";", dtype=str, encoding="latin-1"))
        fl = fl.rename(columns={"indroea": "id", "fecha": "date", "caudal": "q_m3s"})
        fl["date"] = pd.to_datetime(fl["date"], dayfirst=True, errors="coerce")
        fl["q_m3s"] = pd.to_numeric(fl["q_m3s"].str.replace(",", "."), errors="coerce")
    else:
        return None
    if "x" not in st or st["x"].isna().all():
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:4326", GRID_40M.crs, always_xy=True)
        st["x"], st["y"] = tr.transform(st["lon"].to_numpy(), st["lat"].to_numpy())
    return st, fl[["id", "date", "q_m3s"]].dropna()


def annual_runoff(flows: pd.DataFrame, area_km2: pd.Series, min_days: int = 330):
    """Water-year runoff (mm) and 7-day low flow (mm/day) per gauge; incomplete years dropped."""
    f = flows.copy()
    f["wy"] = f["date"].dt.year + (f["date"].dt.month >= 10).astype(int)
    out = []
    for (gid, wy), g in f.groupby(["id", "wy"]):
        if g["q_m3s"].notna().sum() < min_days:
            continue
        a = float(area_km2.get(gid, np.nan))
        mm_day = g.set_index("date")["q_m3s"].sort_index() * 86400 / (a * 1e6) * 1000
        out.append(
            {
                "catchment": gid,
                "year": int(wy),
                "runoff": float(mm_day.mean() * 365.25),
                "low_flow": float(mm_day.rolling(7).mean().min()),
            }
        )
    return pd.DataFrame(out)


def snap_gauges(st: pd.DataFrame, acc: np.ndarray, radius_m: float = 1000) -> pd.DataFrame:
    """Move each gauge to the highest-accumulation cell within `radius_m`, preferring the cell
    whose drainage area best matches the reported one when it is given."""
    ny, nx = GRID_HYDRO.shape
    A = acc.reshape(ny, nx)
    k = int(radius_m // HYDRO_RES)
    rows = []
    for _, g in st.iterrows():
        r = int((GRID_HYDRO.ymax - g["y"]) // HYDRO_RES)
        c = int((g["x"] - GRID_HYDRO.xmin) // HYDRO_RES)
        r0, r1, c0, c1 = max(r - k, 0), min(r + k + 1, ny), max(c - k, 0), min(c + k + 1, nx)
        win = A[r0:r1, c0:c1]
        rep = g.get("area_km2", np.nan)
        score = -np.abs(np.log(win / rep)) if np.isfinite(rep) and rep > 0 else win
        i = np.unravel_index(np.nanargmax(np.where(win > 0, score, -np.inf)), win.shape)
        rr, cc = r0 + i[0], c0 + i[1]
        rows.append({"id": g["id"], "outlet": rr * nx + cc, "area_dem_km2": float(A[rr, cc])})
    out = st.merge(pd.DataFrame(rows), on="id")
    if "area_km2" in out:
        out["area_ratio"] = out["area_dem_km2"] / out["area_km2"]
    else:
        out["area_km2"] = out["area_dem_km2"]
    return out


# ---------------------------------------------------------------- climate


def ghcn_monthly() -> pd.DataFrame:
    """Monthly precipitation (mm) and mean temperature (C) per GHCN station, 1985-2024."""
    from .layers import GHCN_STATIONS

    path = INTERIM / "ghcn_monthly_1985.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for sid in GHCN_STATIONS:
        txt = curl(f"https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{sid}.csv", 300)
        df = pd.read_csv(io.StringIO(txt), dtype={"ID": str}, low_memory=False)
        df.columns = [c.upper() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")
        df = df[df["ELEMENT"].isin(["TMAX", "TMIN", "PRCP"]) & (df["DATE"].dt.year >= 1984)]
        df["v"] = df["DATA_VALUE"] / 10.0
        df["ym"] = df["DATE"].dt.to_period("M")
        p = df[df["ELEMENT"] == "PRCP"].groupby("ym")["v"].agg(["sum", "size"])
        tx = df[df["ELEMENT"] == "TMAX"].groupby("ym")["v"].mean()
        tn = df[df["ELEMENT"] == "TMIN"].groupby("ym")["v"].mean()
        m = pd.DataFrame({"prcp": p["sum"], "n_p": p["size"], "tmean": (tx + tn) / 2})
        m = m[m["n_p"] >= 25].drop(columns="n_p")
        m["station"] = sid
        m["year"] = m.index.year
        m["month"] = m.index.month
        rows.append(m.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def thornthwaite(tmean: np.ndarray, heat_index: float) -> np.ndarray:
    """Monthly PET (mm) from monthly mean temperatures (12 values, Jan..Dec)."""
    t = np.clip(tmean, 0, None)
    a = 6.75e-7 * heat_index**3 - 7.71e-5 * heat_index**2 + 1.792e-2 * heat_index + 0.49239
    return 16 * (10 * t / heat_index) ** a * (DAYLENGTH / 12) * (DAYS / 30)


def station_water_years(m: pd.DataFrame) -> pd.DataFrame:
    """Water-year P and Thornthwaite PET per station (years with all 12 months only)."""
    out = []
    for sid, g in m.groupby("station"):
        clim = g.groupby("month")["tmean"].mean().reindex(range(1, 13))
        hi = float(((np.clip(clim.to_numpy(), 0, None) / 5) ** 1.514).sum())
        g = g.assign(wy=g["year"] + (g["month"] >= 10).astype(int))
        for wy, w in g.groupby("wy"):
            if len(w) < 12 or w["tmean"].isna().any():
                continue
            w = w.set_index("month").reindex(range(1, 13))
            pet = thornthwaite(w["tmean"].to_numpy(), hi).sum()
            out.append({"station": sid, "year": int(wy), "precip": w["prcp"].sum(), "pet": pet})
    return pd.DataFrame(out)


def catchment_climate(centroids: np.ndarray, years=WATER_YEARS) -> pd.DataFrame:
    """Inverse-distance-weighted station P and PET at catchment centroids (x, y in m)."""
    from pyproj import Transformer

    from .layers import GHCN_STATIONS

    sw = station_water_years(ghcn_monthly())
    tr = Transformer.from_crs("EPSG:4326", GRID_40M.crs, always_xy=True)
    xy = {s: tr.transform(lon, lat) for s, (lat, lon) in GHCN_STATIONS.items()}
    rows = []
    for yr in years:
        sub = sw[sw["year"] == yr]
        if sub.empty:
            continue
        sx = np.array([xy[s][0] for s in sub["station"]])
        sy = np.array([xy[s][1] for s in sub["station"]])
        for i, (cx, cy) in enumerate(centroids):
            w = 1.0 / (np.hypot(sx - cx, sy - cy) / 1000 + 10) ** 2
            rows.append(
                {
                    "catchment": i,
                    "year": yr,
                    "precip": float(np.sum(w * sub["precip"]) / w.sum()),
                    "pet": float(np.sum(w * sub["pet"]) / w.sum()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cover paths


def cover_paths(members: list[np.ndarray], version: str = "backdated", years=WATER_YEARS):
    """Eucalyptus, pine and native fraction per catchment and water year.

    2017 and 2024 are the mapped years. A pixel whose class differs between them switches in its
    Hansen loss year or EFFIS burn year when one falls in 2018-2024, otherwise in 2021. Before
    2017 the 2017 map is held fixed (no earlier map), so identification rests on 2017-2024.
    version: "backdated" (2017 backdated from 2024) or "independent" (2017 classified alone).
    """
    sp = np.load(INTERIM / "species.npz")
    c24 = sp["class40_2024"]
    c17 = sp["class40_2017_independent" if version == "independent" else "class40_2017"]
    ly = np.load(INTERIM / "hansen.npz")["lossyear40"].astype(int) + 2000
    ef = np.load(INTERIM / "effis.npz")
    burn = np.zeros(c24.shape, np.int16)
    for y in range(2023, 2017, -1):
        burn[ef[f"burned40_{y}"]] = y
    ny, nx = GRID_HYDRO.shape
    sl = (slice(0, ny * AGG), slice(0, nx * AGG))
    c24, c17, ly, burn = c24[sl], c17[sl], ly[sl], burn[sl]
    when = np.where((ly >= 2018) & (ly <= 2024), ly, np.where(burn > 0, burn, 2021))
    # Hydro cell of each 40 m pixel.
    hr = (np.arange(ny * AGG) // AGG)[:, None] * nx + (np.arange(nx * AGG) // AGG)[None, :]
    lab = np.zeros(ny * nx, np.int32)
    for i, m in enumerate(members):
        lab[m] = i + 1
    plab = lab[hr]
    ok = (plab > 0) & (c24 < 6) & (c17 < 6)
    pl, a, b, w = plab[ok] - 1, c17[ok], c24[ok], when[ok]
    k = len(members)
    n = np.bincount(pl, minlength=k).astype(float)
    rows = []
    for yr in years:
        if yr <= 2017:
            cls = a
        elif yr >= 2024:
            cls = b
        else:
            cls = np.where(w <= yr, b, a)
        f = {
            name: np.bincount(pl[cls == c], minlength=k) / np.maximum(n, 1)
            for name, c in (("f_eucalyptus", 0), ("f_pine", 1), ("f_native_broadleaf", 2))
        }
        rows.append(pd.DataFrame({"catchment": np.arange(k), "year": yr, **f}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- catchments for the study


def candidate_catchments(
    min_km2: float = 30, max_km2: float = 1500, seed: int = 0
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Non-nested whole catchments inside Galicia, standing in for a gauge network.

    Outlets are chosen greedily from large to small drainage area among cells in the target
    range, skipping any cell inside an already chosen catchment and any catchment draining off
    the bounding box.
    """
    rec, order, edge, acc = routing()
    ny, nx = GRID_HYDRO.shape
    aoi = np.load(INTERIM / "aoi.npz")["mask40"][: ny * AGG, : nx * AGG]
    gal = (aoi.reshape(ny, AGG, nx, AGG) > 0).mean(axis=(1, 3)).ravel() > 0.5
    # Candidate outlets: river cells whose receiver crosses the upper size bound or reaches the
    # sea, i.e. the most downstream cell of each tributary within the size range.
    ok = (acc >= min_km2) & (acc <= max_km2) & gal
    down = np.where(rec >= 0, acc[np.maximum(rec, 0)], np.inf)
    cand = np.flatnonzero(ok & ((down > max_km2) | (rec < 0)))
    cand = cand[np.argsort(-acc[cand])]
    members = catchment_members(rec, order, cand)
    trunc = truncated(members, order, rec, edge)
    taken = np.zeros(rec.size, bool)
    keep = []
    for i in np.argsort(-acc[cand]):
        if trunc[i] or taken[members[i]].any():
            continue
        # Mostly inside Galicia (the maps and fire data stop at the border).
        if gal[members[i]].mean() < 0.9:
            continue
        taken[members[i]] = True
        keep.append(i)
    X, Y = (a.ravel() for a in GRID_HYDRO.centers())
    info = pd.DataFrame(
        {
            "catchment": np.arange(len(keep)),
            "outlet": cand[keep],
            "area_km2": acc[cand[keep]],
            "cx": [X[members[i]].mean() for i in keep],
            "cy": [Y[members[i]].mean() for i in keep],
        }
    )
    return info, [members[i] for i in keep]


# ---------------------------------------------------------------- power study


def simulate_runoff(
    panel: pd.DataFrame,
    effect_mm_per_10pts: float,
    true_f: np.ndarray,
    rng: np.random.Generator,
    noise_cv: float = 0.08,
):
    """Runoff from Fu's curve with a catchment-specific w, plus an additive eucalyptus effect.

    w0 ~ N(1.9, 0.25) gives runoff ratios around 0.5-0.6, typical of Galician rivers. The
    eucalyptus effect is additive in mm (`effect_mm_per_10pts` per 10 points of cover; negative
    means less water), so the true effect is linear and known exactly; Fu's curve only shapes
    the baseline. Noise: a year shock common to all catchments plus independent log-normal error
    (CV `noise_cv`) for gauge error and unmodelled catchment-year variation.
    """
    from ..data.synthetic import fu_et

    P, PET = panel["precip"].to_numpy(), panel["pet"].to_numpy()
    units = panel["catchment"].to_numpy()
    w0 = rng.normal(1.9, 0.25, units.max() + 1)[units]
    q = P - fu_et(P, PET, np.maximum(w0, 1.05))
    years = panel["year"].to_numpy()
    uy = np.unique(years)
    shock = rng.normal(0, 0.05, len(uy))[np.searchsorted(uy, years)]
    q = q * np.exp(shock + rng.normal(0, noise_cv, len(q)))
    return np.maximum(q + effect_mm_per_10pts * 10 * true_f, 0.0)


def twfe_unit_slopes(p: pd.DataFrame, y: str, x: str, controls=("pet",)):
    """TWFE with a separate precipitation slope per catchment.

    Runoff responds to rainfall with a catchment-specific slope (roughly its runoff ratio at the
    margin), which a single linear precip control misses; the leftover inflates the error. Here
    y, x and the controls are first residualised on (1, precip) within each catchment, then
    year-demeaned; SEs clustered by catchment.
    """
    from ..causal.dml import ols

    cols = [y, x, *controls]
    r = p[cols].astype(float).copy()
    for _u, idx in p.groupby("catchment").groups.items():
        Z = np.column_stack([np.ones(len(idx)), p.loc[idx, "precip"].to_numpy()])
        V = r.loc[idx].to_numpy()
        beta, *_ = np.linalg.lstsq(Z, V, rcond=None)
        r.loc[idx] = V - Z @ beta
    r = r - r.groupby(p["year"]).transform("mean")
    e = ols(r[y].to_numpy(), r[[x, *controls]].to_numpy(), p["catchment"].to_numpy(), name=x)
    n, k = len(p), 2 * p["catchment"].nunique() + p["year"].nunique() + len(controls)
    e.se *= np.sqrt((n - 1) / max(n - k - 1, 1))
    e.method = "TWFE, precipitation slope per catchment"
    return e


def _stretch(p: pd.DataFrame, cols, scale: float) -> pd.DataFrame:
    """Scale each catchment's deviations from its own mean cover by `scale` (clipped to [0, 1]).

    Used to ask what a longer cover history (larger within-catchment change) would buy.
    """
    if scale == 1:
        return p
    p = p.copy()
    for c in cols:
        m = p.groupby("catchment")[c].transform("mean")
        p[c] = np.clip(m + scale * (p[c] - m), 0, 1)
    return p


def power_study(
    panel: pd.DataFrame,
    f_true_col: str = "f_eucalyptus",
    n_catchments=(20, 40, 79),
    effects=(0.0, -10.0, -20.0, -40.0),
    change_scale=(1.0,),
    reps: int = 200,
    noise_cv: float = 0.08,
    seed: int = 0,
) -> pd.DataFrame:
    """Bias, 95% coverage and power of catchment-panel runoff estimates on the real catchments.

    Real catchments, real station climate and the real 2017-2024 cover paths; simulated flows
    with a known effect. `f_true_col` is the path flows are simulated from, while estimates use
    `f_eucalyptus`, so passing the other map version's path adds realistic map error.
    `change_scale` > 1 stretches within-catchment cover change (a stand-in for a longer mapped
    history, e.g. a Landsat back-cast to the 1990s).
    """
    from ..models.hydrology import twfe

    rng = np.random.default_rng(seed)
    all_units = panel["catchment"].unique()
    ctrl = ["precip", "pet", "f_pine", "f_native_broadleaf"]
    rows = []
    for sc in change_scale:
        base = _stretch(
            panel, sorted({"f_eucalyptus", f_true_col, "f_pine", "f_native_broadleaf"}), sc
        )
        for n in n_catchments:
            n = min(n, len(all_units))
            for eff in effects:
                res = {"TWFE": ([], []), "TWFE + slopes": ([], [])}
                for _ in range(reps):
                    units = rng.choice(all_units, n, replace=False)
                    p = base[base["catchment"].isin(units)].copy()
                    p["catchment"] = pd.factorize(p["catchment"])[0]
                    p["runoff"] = simulate_runoff(
                        p, eff, p[f_true_col].to_numpy(), rng, noise_cv=noise_cv
                    )
                    for name, e in (
                        ("TWFE", twfe(p, "runoff", "f_eucalyptus", "catchment", "year", ctrl)),
                        (
                            "TWFE + slopes",
                            twfe_unit_slopes(
                                p, "runoff", "f_eucalyptus", ["pet", "f_pine", "f_native_broadleaf"]
                            ),
                        ),
                    ):
                        res[name][0].append(e.estimate * 0.1)  # per 10 points
                        res[name][1].append(e.se * 0.1)
                for name, (est, se) in res.items():
                    est, se = np.array(est), np.array(se)
                    rows.append(
                        {
                            "estimator": name,
                            "change_scale": sc,
                            "n_catchments": n,
                            "true_mm_per_10pts": eff,
                            "mean_estimate": float(np.mean(est)),
                            "bias": float(np.mean(est) - eff),
                            "sd_estimate": float(np.std(est)),
                            "mean_se": float(np.mean(se)),
                            "coverage": float(np.mean(np.abs(est - eff) <= 1.96 * se)),
                            "power": float(np.mean(np.abs(est) > 1.96 * se)),
                        }
                    )
    return pd.DataFrame(rows)


def minimum_detectable(ps: pd.DataFrame, power: float = 0.8) -> pd.DataFrame:
    """Smallest effect (mm per 10 points) detected with the given power, from the SE:
    MDE = (1.96 + z_power) * sd of the estimate under no effect."""
    from scipy.stats import norm

    z = 1.96 + norm.ppf(power)
    null = ps[ps["true_mm_per_10pts"] == 0]
    return null.assign(mde_mm_per_10pts=z * null["sd_estimate"])[
        ["estimator", "change_scale", "n_catchments", "mde_mm_per_10pts"]
    ]


def build_panel(info: pd.DataFrame, members, version: str = "backdated") -> pd.DataFrame:
    clim = catchment_climate(info[["cx", "cy"]].to_numpy())
    cov = cover_paths(members, version)
    return clim.merge(cov, on=["catchment", "year"])


def landsat_fractions(members: list[np.ndarray]) -> pd.DataFrame:
    """Eucalyptus, pine and native share per catchment in each Landsat epoch (classified pixels
    only)."""
    from .landsat import EPOCHS, landsat_maps

    lm = landsat_maps()
    ny, nx = GRID_HYDRO.shape
    hr = (np.arange(ny * AGG) // AGG)[:, None] * nx + (np.arange(nx * AGG) // AGG)[None, :]
    lab = np.zeros(ny * nx, np.int32)
    for i, m in enumerate(members):
        lab[m] = i + 1
    plab = lab[hr]
    k = len(members)
    rows = []
    for e in EPOCHS:
        cls = lm[f"class40_{e}"][: ny * AGG, : nx * AGG]
        ok = (plab > 0) & (cls < 6)
        pl, cl = plab[ok] - 1, cls[ok]
        n = np.maximum(np.bincount(pl, minlength=k), 1).astype(float)
        rows.append(
            pd.DataFrame(
                {
                    "catchment": np.arange(k),
                    "epoch": int(e),
                    "f_eucalyptus": np.bincount(pl[cl == 0], minlength=k) / n,
                    "f_pine": np.bincount(pl[cl == 1], minlength=k) / n,
                    "f_native_broadleaf": np.bincount(pl[cl == 2], minlength=k) / n,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def cover_paths_long(members: list[np.ndarray], years=LONG_YEARS) -> pd.DataFrame:
    """Cover per catchment and water year 1990-2024.

    2017-2024 is the Sentinel-2 path (cover_paths). Before 2017 each catchment keeps its
    Sentinel-2 2017 level and adds the Landsat change relative to the Landsat 2017 epoch,
    interpolated linearly between epochs (1990, 2000, 2010, 2017). Anchoring on the change, not
    the level, avoids a step where the source switches from one sensor's map to the other's.
    """
    recent = cover_paths(members, "backdated", range(2017, 2025))
    base = recent[recent["year"] == 2017].set_index("catchment")
    lf = landsat_fractions(members)
    cols = ["f_eucalyptus", "f_pine", "f_native_broadleaf"]
    rows = []
    for c, g in lf.groupby("catchment"):
        g = g.sort_values("epoch")
        ref = g[g["epoch"] == 2017][cols].to_numpy()[0]
        for yr in years:
            if yr >= 2017:
                continue
            vals = {
                col: float(
                    np.clip(base.loc[c, col] + np.interp(yr, g["epoch"], g[col]) - ref[i], 0, 1)
                )
                for i, col in enumerate(cols)
            }
            rows.append({"catchment": c, "year": yr, **vals})
    return pd.concat([pd.DataFrame(rows), recent], ignore_index=True).sort_values(
        ["catchment", "year"]
    )


def long_history_study(info, members, seed: int = 0, reps: int = 200) -> dict:
    """Power study with the 1990-2024 cover paths from the Landsat back-cast."""
    clim = catchment_climate(info[["cx", "cy"]].to_numpy(), LONG_YEARS)
    panel = clim.merge(cover_paths_long(members), on=["catchment", "year"])
    d = panel.groupby("catchment")["f_eucalyptus"].agg(lambda s: s.max() - s.min())
    first = panel[panel["year"] == panel["year"].min()]["f_eucalyptus"].mean()
    last = panel[panel["year"] == 2024]["f_eucalyptus"].mean()
    ps = power_study(panel, change_scale=(1.0,), reps=reps, seed=seed + 2)
    return {
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "n_catchment_years": len(panel),
        "euc_first_mean": float(first),
        "euc_2024_mean": float(last),
        "within_change_mean_pts": float(100 * d.mean()),
        "within_change_p90_pts": float(100 * d.quantile(0.9)),
        "power": ps.to_dict(orient="records"),
        "mde": minimum_detectable(ps).to_dict(orient="records"),
    }


def water_analysis(seed: int = 0, reps: int = 200) -> dict:
    """Real gauge estimate when gauges are supplied; power study on real catchments always."""
    from ..models.hydrology import fit_budyko, twfe

    out: dict = {}
    info, members = candidate_catchments(seed=seed)
    panel = build_panel(info, members, "backdated")
    alt = cover_paths(members, "independent").rename(columns={"f_eucalyptus": "f_euc_ind"})
    panel = panel.merge(alt[["catchment", "year", "f_euc_ind"]], on=["catchment", "year"])
    d = panel.groupby("catchment")["f_eucalyptus"].agg(lambda s: s.max() - s.min())
    out["catchments"] = {
        "n": len(info),
        "area_km2_median": float(info["area_km2"].median()),
        "euc_2024_mean": float(panel[panel["year"] == 2024]["f_eucalyptus"].mean()),
        "within_change_mean_pts": float(100 * d.mean()),
        "within_change_p90_pts": float(100 * d.quantile(0.9)),
    }
    log.info("water: %d catchments, power study", len(info))
    ps = power_study(panel, change_scale=(1.0, 3.0, 6.0), reps=reps, seed=seed)
    ps_err = power_study(
        panel,
        f_true_col="f_euc_ind",
        n_catchments=(79,),
        effects=(0.0, -20.0),
        change_scale=(1.0, 6.0),
        reps=reps,
        seed=seed + 1,
    )
    out["power"] = ps.to_dict(orient="records")
    out["mde"] = minimum_detectable(ps).to_dict(orient="records")
    out["power_map_error"] = ps_err.to_dict(orient="records")
    if (INTERIM / "landsat_maps_v2.npz").exists():
        out["long"] = long_history_study(info, members, seed=seed, reps=reps)

    g = load_gauges()
    if g is None:
        out["gauges"] = None
        return out
    st, fl = g
    rec, order, _edge, acc = routing()
    st = snap_gauges(st, acc)
    mem = catchment_members(rec, order, st["outlet"].to_numpy())
    X, Y = (a.ravel() for a in GRID_HYDRO.centers())
    st["cx"] = [X[m].mean() for m in mem]
    st["cy"] = [Y[m].mean() for m in mem]
    gp = build_panel(st.reset_index(drop=True).assign(catchment=range(len(st))), mem)
    gp["catchment"] = st["id"].to_numpy()[gp["catchment"].to_numpy()]
    q = annual_runoff(fl, st.set_index("id")["area_km2"])
    gp = gp.merge(q, on=["catchment", "year"])
    if gp["catchment"].nunique() < 3:
        out["gauges"] = {"n": int(gp["catchment"].nunique()), "note": "too few gauges"}
        return out
    ctrl = ["precip", "pet", "f_pine", "f_native_broadleaf"]
    fe = twfe(gp, "runoff", "f_eucalyptus", "catchment", "year", ctrl)
    fs = twfe_unit_slopes(gp, "runoff", "f_eucalyptus", ["pet", "f_pine", "f_native_broadleaf"])
    lo = twfe(gp, "low_flow", "f_eucalyptus", "catchment", "year", ctrl)
    out["gauges"] = {
        "n": int(gp["catchment"].nunique()),
        "n_years": len(gp),
        "runoff_mm_per_10pts": [fe.estimate * 0.1, fe.se * 0.1],
        "runoff_slopes_mm_per_10pts": [fs.estimate * 0.1, fs.se * 0.1],
        "low_flow_per_10pts": [lo.estimate * 0.1, lo.se * 0.1],
        "budyko": fit_budyko(gp),
        "area_ratio_median": float(st.get("area_ratio", pd.Series([np.nan])).median()),
    }
    return out
