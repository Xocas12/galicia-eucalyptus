"""Synthetic Galicia-like landscape with known causal effects.

Every estimator in this package is validated here before it touches real data: the simulator
plants effects of known size, and the pipeline has to recover them. The landscape is *stylised*.
Coastline, relief and city positions are rough approximations placed in ETRS89 / UTM 29N, and all
effect sizes in `TruthParams` are assumptions chosen to be plausible, **not** estimates about
Galicia.

Mechanisms simulated (docs/SCOPE.md section 5):

- Atlantic-to-continental gradient. The coast is wet and mild, the interior (Ourense) is drier
  with hot summers and higher fire weather.
- Plantations favour low elevation (frost), proximity to the pulp mill, neighbouring plantations
  (contagion) and recently burned land (the fire -> plantation feedback).
- Fire probability depends on lagged cover (eucalyptus, pine and shrub raise it; native
  broadleaf lowers it), fire weather, ignition pressure (population, roads) and slope.
- Fire converts native broadleaf and pine to shrub. Eucalyptus resprouts. Eucalyptus is
  clear-felled on a ~12-year rotation, which registers as tree-cover loss without land-use change.
- Water balance follows Fu's Budyko curve, with the land-surface parameter w raised by forest
  cover and most by eucalyptus.

Cover is tracked per cell as a joint matrix J[c0, c] (area fraction that was class c0 at the start
year and is class c now), updated each year by a per-cell row-stochastic transition matrix. This
gives exact gross transition areas, not just net change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import StudyConfig
from ..geo.grid import Grid, block_ids
from ..geo.raster_ops import distance_to, focal_mean, gaussian_field

CLASSES = ["eucalyptus", "pine", "native_broadleaf", "shrub", "agriculture", "other"]
EUC, PINE, NATIVE, SHRUB, AGRI, OTHER = range(6)
N_CLASSES = len(CLASSES)
FOREST = (EUC, PINE, NATIVE)

# Approximate positions (ETRS89 / UTM 29N, metres) of the main cities and of the pulp mill near
# Pontevedra. They are used only to place population and plantation attractors on the synthetic
# landscape.
CITIES = {
    "A Coruna": (548_000, 4_801_000, 1.0),
    "Vigo": (523_000, 4_677_000, 1.0),
    "Santiago": (537_000, 4_747_000, 0.7),
    "Ourense": (594_000, 4_689_000, 0.6),
    "Lugo": (617_000, 4_763_000, 0.55),
    "Pontevedra": (530_000, 4_698_000, 0.5),
    "Ferrol": (562_000, 4_815_000, 0.45),
}
PULP_MILL = (526_000, 4_696_000)


@dataclass
class TruthParams:
    """Ground-truth mechanism parameters (assumptions, not estimates)."""

    # Fire occurrence, logit scale, per unit cover fraction (reference = agriculture/other).
    fire_b0: float = -4.9
    fire_euc: float = 0.9
    fire_pine: float = 0.7
    fire_shrub: float = 1.1
    fire_native: float = -0.6
    fire_fwi: float = 0.85
    fire_pop: float = 0.45
    fire_slope: float = 0.25
    # Fire severity (dNBR units), exactly partially linear.
    sev_base: float = 260.0
    sev_euc: float = 120.0
    sev_pine: float = 60.0
    sev_shrub: float = 80.0
    sev_native: float = -100.0
    sev_fwi: float = 70.0
    sev_noise: float = 60.0
    # Share of burned native / pine canopy that reverts to shrub.
    burn_native_loss: float = 0.5
    burn_pine_loss: float = 0.6
    # Conversion to eucalyptus, logit scale.
    conv_c0: float = -2.4
    conv_mill_per_50km: float = -1.1
    conv_elev_per_km: float = -2.5
    conv_neigh: float = 3.0
    conv_recent_fire: float = 1.2
    conv_policy: float = -1.5
    conv_policy_year: int = 2021
    conv_noise: float = 0.3
    # Relative susceptibility of each source class to being planted.
    conv_source: tuple = (0.0, 0.4, 0.25, 1.0, 0.5, 0.0)
    abandonment: float = 0.012  # agriculture -> shrub per year
    shrub_to_native: float = 0.010
    shrub_to_pine: float = 0.006
    rotation_years: float = 12.0
    pine_rotation_years: float = 35.0
    # Budyko (Fu) land-surface parameter w = w0 + sum(a_c * f_c).
    w0: float = 2.0
    w_euc: float = 1.2
    w_pine: float = 0.5
    w_native: float = 0.4
    # Summer soil moisture index, exactly partially linear in eucalyptus fraction.
    sm_euc: float = -0.06
    sm_native: float = 0.02
    sm_noise: float = 0.015
    # Classifier-style measurement error on observed cover fractions.
    cover_obs_noise: float = 0.02


@dataclass
class Policy:
    """Knobs used by the scenario engine; the defaults reproduce the historical simulation."""

    conversion_multiplier: float = 1.0
    freeze_eucalyptus: bool = False
    restoration_rate: np.ndarray | None = None  # per-cell euc -> native flow per year


@dataclass
class Landscape:
    """Everything the simulator produced. Land cells are stored as 1-D vectors of length n."""

    grid: Grid
    years: list[int]
    land: np.ndarray  # (ny, nx) bool
    land_idx: np.ndarray  # flat indices of land cells
    static: dict[str, np.ndarray]  # name -> (n,)
    block: np.ndarray  # (n,) spatial block id
    catchment: np.ndarray  # (n,) catchment id
    n_catchments: int
    cover: np.ndarray  # (T, n, 6) true fractions
    cover_obs: np.ndarray  # (T, n, 6) observed fractions (with map error)
    joint: np.ndarray  # (n, 6, 6) start-year x end-year class fractions
    burned: np.ndarray  # (T, n) bool
    dnbr: np.ndarray  # (T, n) severity, NaN if unburned
    fwi: np.ndarray  # (T, n) standardised fire weather
    precip: np.ndarray  # (T, n) mm
    pet: np.ndarray  # (T, n) mm
    runoff: np.ndarray  # (T, n) mm
    soil_moisture: np.ndarray  # (n,) final-year summer index
    loss: dict[str, np.ndarray]  # driver -> (T, n) tree-cover-loss fraction
    true_te: dict[str, np.ndarray]  # per-row true effects for validation
    truth: TruthParams
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.land_idx)

    def to_2d(self, vec: np.ndarray, fill=np.nan) -> np.ndarray:
        out = np.full(self.land.size, fill, dtype=float)
        out[self.land_idx] = vec
        return out.reshape(self.land.shape)

    def cell_area_ha(self) -> float:
        return self.grid.cell_area_ha


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _smooth_1d(n: int, rng, scale: float, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    f = gaussian_filter1d(rng.standard_normal(n), sigma, mode="reflect")
    f = (f - f.min()) / (np.ptp(f) + 1e-12)
    return f * scale


def _rasterize_segment(mask, grid: Grid, a, b):
    n = int(np.hypot(b[0] - a[0], b[1] - a[1]) / (grid.resolution_m / 2)) + 2
    ny, nx = mask.shape
    for x, y in zip(np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n), strict=True):
        r, c = grid.index_of(x, y)
        if 0 <= r < ny and 0 <= c < nx:
            mask[r, c] = True


def build_static(grid: Grid, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Terrain, climate normals, population and access layers on the full grid (2-D)."""
    ny, nx = grid.shape
    x, y = grid.centers()
    u = (x - grid.xmin) / (grid.xmax - grid.xmin)  # 0 west -> 1 east
    v = (grid.ymax - y) / (grid.ymax - grid.ymin)  # 0 north -> 1 south
    sig = max(1.0, 6000 / grid.resolution_m)

    # Coastline: west and north coasts with ria-like indentations; Portugal (Minho) to the south
    # and Asturias/Leon to the east are treated as land borders by clipping the domain.
    coast_w = 0.03 + _smooth_1d(ny, rng, 0.10, sigma=max(1, ny / 25))[:, None]
    coast_n = 0.02 + _smooth_1d(nx, rng, 0.09, sigma=max(1, nx / 25))[None, :]
    rias = 0.03 * gaussian_field((ny, nx), sig / 2, rng)
    land = (u - coast_w + rias > 0) & (v - coast_n + rias > 0)
    land &= (u < 0.97) & (v < 0.97) & ~((u > 0.80) & (v < 0.10))
    for cx, cy, _ in CITIES.values():
        r, c = grid.index_of(cx, cy)
        land[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2] = True
    r, c = grid.index_of(*PULP_MILL)
    land[r, c] = True

    dist_coast = distance_to(~land, grid.resolution_m) / 1000.0
    cont = np.clip(dist_coast / 90.0, 0, 1) * 0.7 + 0.3 * u  # continentality 0..1

    mountains = (
        900 * np.exp(-(((u - 0.92) / 0.10) ** 2 + ((v - 0.38) / 0.18) ** 2))  # Ancares/Courel
        + 700 * np.exp(-(((u - 0.78) / 0.12) ** 2 + ((v - 0.86) / 0.10) ** 2))  # Macizo Central
        + 500 * np.exp(-(((u - 0.52) / 0.10) ** 2 + ((v - 0.95) / 0.06) ** 2))  # Xures
    )
    elev = 60 + 520 * cont + mountains + 180 * gaussian_field((ny, nx), sig * 2, rng)
    elev = np.clip(elev, 0, 2000) * land
    gy, gx = np.gradient(elev, grid.resolution_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))) + 4 * np.abs(gaussian_field((ny, nx), sig, rng))

    precip = 1850 - 1000 * cont + 0.45 * elev + 120 * gaussian_field((ny, nx), sig * 2, rng)
    precip = np.clip(precip, 600, 2800)
    summer_temp = 18.5 + 7.5 * cont * np.clip(0.4 + v, 0, 1) - 0.004 * elev
    pet = 560 + 280 * cont + 25 * (summer_temp - 20)

    pop = np.zeros((ny, nx))
    for cx, cy, w in CITIES.values():
        d = np.hypot(x - cx, y - cy) / 1000.0
        pop += w * 2500 * np.exp(-d / 6.0)
    pop += 40 * np.exp(-cont * 1.5) * (1 + 0.5 * gaussian_field((ny, nx), sig, rng))
    pop = np.clip(pop, 1, None)

    roads = np.zeros((ny, nx), bool)
    names = list(CITIES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = CITIES[a][:2], CITIES[b][:2]
            if np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 110_000:
                _rasterize_segment(roads, grid, pa, pb)
    dist_road = distance_to(roads, grid.resolution_m) / 1000.0
    dist_mill = np.hypot(x - PULP_MILL[0], y - PULP_MILL[1]) / 1000.0

    fwi_base = (summer_temp - summer_temp[land].mean()) / summer_temp[land].std()
    fwi_base -= 0.6 * (precip - precip[land].mean()) / precip[land].std()
    fwi_base += 1.2 * (cont - cont[land].mean()) / cont[land].std()
    fwi_base = (fwi_base - fwi_base[land].mean()) / fwi_base[land].std()

    static = {
        "x": x,
        "y": y,
        "elev": elev,
        "slope": slope,
        "dist_coast_km": dist_coast,
        "continentality": cont,
        "precip_mean": precip,
        "pet_mean": pet,
        "summer_temp": summer_temp,
        "pop_density": pop,
        "dist_road_km": dist_road,
        "dist_mill_km": dist_mill,
        "fwi_base": fwi_base,
    }
    return land, static


def _catchments(land, grid, n_catch, rng) -> np.ndarray:
    """Voronoi catchments over land cells (a stand-in for gauged river basins)."""
    ys, xs = np.nonzero(land)
    seeds = rng.choice(len(ys), size=n_catch, replace=False)
    sy, sx = ys[seeds], xs[seeds]
    d = (ys[:, None] - sy[None, :]) ** 2 + (xs[:, None] - sx[None, :]) ** 2
    return np.argmin(d, axis=1).astype(np.int64)


def initial_cover(s: dict, rng: np.random.Generator, noise: np.ndarray) -> np.ndarray:
    """Cover fractions at the start year from a softmax of site suitability scores."""
    elev_km = s["elev"] / 1000
    cont = s["continentality"]
    slope_n = s["slope"] / 30
    pop_n = np.log(s["pop_density"]) / 8
    scores = np.stack(
        [
            1.1 - 2.5 * elev_km - 0.012 * s["dist_mill_km"] + 1.6 * (1 - cont),  # eucalyptus
            0.2 + 1.2 * cont + 0.6 * slope_n - 0.5 * elev_km,  # pine
            0.1 + 1.6 * elev_km + 0.3 * (s["precip_mean"] / 1000) - 0.4 * cont,  # native
            -0.3 + 1.6 * cont + 1.2 * elev_km + 0.4 * slope_n,  # shrub
            0.8 - 1.0 * slope_n + 0.6 * pop_n - 0.4 * elev_km,  # agriculture
            -2.8 + 3.5 * pop_n,  # other / urban
        ],
        axis=-1,
    )
    z = 1.6 * scores + 0.5 * noise + 0.35 * rng.standard_normal(scores.shape)
    z -= z.max(axis=-1, keepdims=True)
    f = np.exp(z)
    return f / f.sum(axis=-1, keepdims=True)


def fire_logit(tp: TruthParams, f: np.ndarray, fwi: np.ndarray, st: dict) -> np.ndarray:
    return (
        tp.fire_b0
        + tp.fire_euc * f[:, EUC]
        + tp.fire_pine * f[:, PINE]
        + tp.fire_shrub * f[:, SHRUB]
        + tp.fire_native * f[:, NATIVE]
        + tp.fire_fwi * fwi
        + tp.fire_pop * st["pop_z"]
        + tp.fire_slope * st["slope_z"]
    )


def fu_et(p, pet, w):
    """Fu (1981) Budyko curve: actual ET (mm) from precipitation, PET and parameter w."""
    phi = pet / p
    return p * (1 + phi - (1 + phi**w) ** (1 / w))


def budyko_w(tp: TruthParams, f: np.ndarray) -> np.ndarray:
    return tp.w0 + tp.w_euc * f[:, EUC] + tp.w_pine * f[:, PINE] + tp.w_native * f[:, NATIVE]


def conversion_logit(tp, st, neigh_euc, recent_fire, year, noise, policy: Policy) -> np.ndarray:
    z = (
        tp.conv_c0
        + tp.conv_mill_per_50km * st["dist_mill_km"] / 50
        + tp.conv_elev_per_km * st["elev"] / 1000
        + tp.conv_neigh * neigh_euc
        + tp.conv_recent_fire * recent_fire
        + noise
    )
    if year >= tp.conv_policy_year:
        z = z + tp.conv_policy
    return z


def transition_matrices(
    tp: TruthParams,
    f: np.ndarray,
    burned: np.ndarray,
    bfrac: np.ndarray,
    conv_rate: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    """Per-cell row-stochastic 6x6 transition matrices for one year."""
    n = len(f)
    M = np.zeros((n, N_CLASSES, N_CLASSES))
    b = burned * bfrac
    r = conv_rate * policy.conversion_multiplier
    if policy.freeze_eucalyptus:
        r = np.zeros_like(r)
    src = np.asarray(tp.conv_source)
    # fire damage
    M[:, NATIVE, SHRUB] += tp.burn_native_loss * b
    M[:, PINE, SHRUB] += tp.burn_pine_loss * b
    # planting
    for c in range(N_CLASSES):
        if src[c] > 0:
            M[:, c, EUC] += r * src[c]
    # succession and abandonment
    M[:, AGRI, SHRUB] += tp.abandonment
    M[:, SHRUB, NATIVE] += tp.shrub_to_native
    M[:, SHRUB, PINE] += tp.shrub_to_pine
    if policy.restoration_rate is not None:
        M[:, EUC, NATIVE] += policy.restoration_rate
    off = M.sum(axis=2)
    scale = np.where(off > 0.95, 0.95 / np.maximum(off, 1e-12), 1.0)
    M *= scale[:, :, None]
    idx = np.arange(N_CLASSES)
    M[:, idx, idx] = 1.0 - M.sum(axis=2)
    return M


def simulate_landscape(cfg: StudyConfig, truth: TruthParams | None = None) -> Landscape:
    """Run the synthetic landscape from cfg.years[0] to cfg.years[1]."""
    tp = truth or TruthParams()
    rng = np.random.default_rng(cfg.seed)
    grid = Grid.from_bbox(cfg.grid.bbox_utm29n, cfg.grid.resolution_m)
    land, static2d = build_static(grid, rng)
    land_idx = np.flatnonzero(land)
    st = {k: v.ravel()[land_idx] for k, v in static2d.items()}
    st["pop_z"] = (np.log(st["pop_density"]) - np.log(st["pop_density"]).mean()) / np.log(
        st["pop_density"]
    ).std()
    st["slope_z"] = (st["slope"] - st["slope"].mean()) / st["slope"].std()
    n = len(land_idx)
    shape = land.shape
    sig = max(1.0, 8000 / grid.resolution_m)

    def field_vec(sigma):
        return gaussian_field(shape, sigma, rng).ravel()[land_idx]

    blocks = block_ids(grid, cfg.grid.block_size_km).ravel()[land_idx]
    catch = _catchments(land, grid, cfg.n_catchments, rng)

    years = cfg.year_list
    T = len(years)
    cover_noise = np.stack([field_vec(sig) for _ in range(N_CLASSES)], axis=-1)
    f0 = initial_cover(st, rng, cover_noise)
    J = np.zeros((n, N_CLASSES, N_CLASSES))
    J[:, np.arange(N_CLASSES), np.arange(N_CLASSES)] = f0

    # Year-level weather: a stylised sequence with two extreme fire years (2006 and 2017 are well
    # known bad fire years in Galicia; the magnitudes here are invented).
    anom = rng.standard_normal(T) * 0.8
    for yr, val in ((2006, 2.0), (2017, 2.4)):
        if yr in years:
            anom[years.index(yr)] = val
    wet = -0.5 * anom + np.sqrt(0.75) * rng.standard_normal(T)

    cover = np.zeros((T, n, N_CLASSES))
    burned = np.zeros((T, n), bool)
    dnbr = np.full((T, n), np.nan)
    fwi = np.zeros((T, n))
    precip = np.zeros((T, n))
    pet = np.zeros((T, n))
    runoff = np.zeros((T, n))
    loss = {k: np.zeros((T, n)) for k in ("fire", "rotation", "conversion")}
    te_fire = np.zeros((T, n))
    te_sev = np.full((T, n), tp.sev_euc)
    te_conv = np.zeros((T, n))
    te_water = np.zeros((T, n))
    recent_fire = np.zeros((T, n))
    neigh = np.zeros((T, n))
    conv_rate_true = np.zeros((T, n))
    pol = Policy()
    fire_hist = np.zeros((3, n), bool)  # rolling window of the last three years

    for t, year in enumerate(years):
        f_prev = J.sum(axis=1)
        fwi[t] = st["fwi_base"] + 0.8 * anom[t] + 0.3 * field_vec(sig * 2)
        eta = fire_logit(tp, f_prev, fwi[t], st)
        p = _sigmoid(eta)
        burned[t] = rng.random(n) < p
        te_fire[t] = p * (1 - p) * tp.fire_euc
        sev = (
            tp.sev_base
            + tp.sev_euc * f_prev[:, EUC]
            + tp.sev_pine * f_prev[:, PINE]
            + tp.sev_shrub * f_prev[:, SHRUB]
            + tp.sev_native * f_prev[:, NATIVE]
            + tp.sev_fwi * np.maximum(fwi[t], 0)
            + tp.sev_noise * rng.standard_normal(n)
        )
        dnbr[t, burned[t]] = np.clip(sev[burned[t]], 0, None)
        bfrac = rng.beta(2, 2, n)

        # Conversion pressure uses fire in the previous three years (lagged).
        rf = fire_hist.any(axis=0).astype(float)
        recent_fire[t] = rf
        ne = neighbour_mean(f_prev[:, EUC], land, land_idx, grid.resolution_m)
        neigh[t] = ne
        cnoise = tp.conv_noise * rng.standard_normal(n)
        z = conversion_logit(tp, st, ne, rf, year, cnoise, pol)
        rate = 0.25 * _sigmoid(z)
        conv_rate_true[t] = rate
        src = np.asarray(tp.conv_source)
        avail = f_prev @ src
        rate1 = 0.25 * _sigmoid(conversion_logit(tp, st, ne, np.ones(n), year, cnoise, pol))
        rate0 = 0.25 * _sigmoid(conversion_logit(tp, st, ne, np.zeros(n), year, cnoise, pol))
        te_conv[t] = (rate1 - rate0) * avail / np.maximum(1 - f_prev[:, EUC], 1e-6)

        if t > 0:  # the first year is the baseline map; transitions start after it
            M = transition_matrices(tp, f_prev, burned[t], bfrac, rate, pol)
            J = np.einsum("nab,nbc->nac", J, M)
            b = burned[t] * bfrac
            loss["fire"][t] = b * (
                f_prev[:, EUC]
                + tp.burn_native_loss * f_prev[:, NATIVE]
                + tp.burn_pine_loss * f_prev[:, PINE]
            )
            loss["conversion"][t] = M[:, NATIVE, EUC] * f_prev[:, NATIVE] + (
                M[:, PINE, EUC] * f_prev[:, PINE]
            )
            harvest = f_prev[:, EUC] * rng.binomial(1, 1 / tp.rotation_years, n) * (1 - b)
            harvest += f_prev[:, PINE] * rng.binomial(1, 1 / tp.pine_rotation_years, n) * (1 - b)
            loss["rotation"][t] = harvest
        f_now = J.sum(axis=1)
        cover[t] = f_now

        precip[t] = st["precip_mean"] * np.exp(0.15 * wet[t] + 0.05 * field_vec(sig * 2))
        pet[t] = st["pet_mean"] * (1 + 0.04 * anom[t])
        w = budyko_w(tp, f_now)
        runoff[t] = precip[t] - fu_et(precip[t], pet[t], w)
        dq = (fu_et(precip[t], pet[t], w + tp.w_euc * 0.01) - fu_et(precip[t], pet[t], w)) / 0.01
        te_water[t] = -dq

        fire_hist = np.roll(fire_hist, 1, axis=0)
        fire_hist[0] = burned[t]

    f_last = cover[-1]
    sm = (
        0.18
        + 0.12 * (st["precip_mean"] - 700) / 1100
        - 0.05 * st["continentality"]
        + tp.sm_euc * f_last[:, EUC]
        + tp.sm_native * f_last[:, NATIVE]
        + tp.sm_noise * rng.standard_normal(n)
    )

    obs = cover + tp.cover_obs_noise * rng.standard_normal(cover.shape)
    obs = np.clip(obs, 0, None)
    obs /= obs.sum(axis=-1, keepdims=True)

    return Landscape(
        grid=grid,
        years=years,
        land=land,
        land_idx=land_idx,
        static=st,
        block=blocks,
        catchment=catch,
        n_catchments=cfg.n_catchments,
        cover=cover,
        cover_obs=obs,
        joint=J,
        burned=burned,
        dnbr=dnbr,
        fwi=fwi,
        precip=precip,
        pet=pet,
        runoff=runoff,
        soil_moisture=sm,
        loss=loss,
        true_te={
            "fire": te_fire,
            "severity": te_sev,
            "conversion": te_conv,
            "water": te_water,
        },
        truth=tp,
        extra={
            "recent_fire": recent_fire,
            "neigh_euc": neigh,
            "conv_rate": conv_rate_true,
            "weather_anomaly": anom,
            "wet_anomaly": wet,
            "fire_hist": fire_hist,
            "rng_state": rng.bit_generator.state,
        },
    )


def neighbour_mean(vals, land, land_idx, resolution_m, radius_m: float = 3000) -> np.ndarray:
    """Mean of a land-cell vector within ~radius_m (plantation contagion)."""
    full = np.zeros(land.size)
    full[land_idx] = vals
    r = max(1, round(radius_m / resolution_m))
    return focal_mean(full.reshape(land.shape), radius=r, mask=land).ravel()[land_idx]


# ------------------------------------------------------------------------------------------------
# Pixel-level spectral time series for species mapping
# ------------------------------------------------------------------------------------------------

# Per class: (mean, amplitude, peak month) for NDVI, NDMI and NBR. Evergreen eucalyptus has a high,
# flat NDVI; deciduous native broadleaf has a strong seasonal cycle; pine is evergreen but drier
# (lower NDMI). Values are stylised.
SPECTRAL = {
    EUC: ((0.78, 0.04, 5), (0.34, 0.03, 4), (0.55, 0.04, 5)),
    PINE: ((0.72, 0.05, 6), (0.26, 0.04, 5), (0.49, 0.05, 6)),
    NATIVE: ((0.66, 0.19, 7), (0.26, 0.13, 7), (0.46, 0.14, 7)),
    SHRUB: ((0.50, 0.10, 5), (0.11, 0.07, 4), (0.30, 0.08, 5)),
    AGRI: ((0.56, 0.21, 4), (0.15, 0.14, 4), (0.35, 0.15, 4)),
    OTHER: ((0.20, 0.03, 6), (-0.05, 0.03, 6), (0.05, 0.03, 6)),
}
INDICES = ("ndvi", "ndmi", "nbr")


def simulate_spectral_pixels(
    classes: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
    n_months: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Monthly NDVI/NDMI/NBR series with cloud gaps for pixels of the given classes.

    Returns (series, months): series is (n, n_months, 3) with NaN where cloudy; months is
    (n_months,) month index starting at January. Per-block site offsets make spatial CV matter.
    """
    classes = np.asarray(classes)
    n = len(classes)
    months = np.arange(n_months)
    moy = months % 12
    ub, binv = np.unique(blocks, return_inverse=True)
    site = rng.normal(0, 0.06, (len(ub), N_CLASSES, 3))
    out = np.empty((n, n_months, 3))
    for c, params in SPECTRAL.items():
        m = classes == c
        k = int(m.sum())
        if k == 0:
            continue
        for j, (mu, amp, peak) in enumerate(params):
            mu_i = mu + rng.normal(0, 0.045, k) + site[binv[m], c, j]
            amp_i = np.clip(amp + rng.normal(0, 0.03, k), 0, None)
            pk = peak + rng.normal(0, 1.0, k)
            cyc = np.cos(2 * np.pi * (moy[None, :] - pk[:, None]) / 12)
            out[m, :, j] = mu_i[:, None] + amp_i[:, None] * cyc
        if c == EUC:  # clear-fell in the series for some plantation pixels
            hit = np.flatnonzero(m)[rng.random(k) < 0.08]
            for i in hit:
                s = rng.integers(0, n_months - 6)
                out[i, s:, :] -= (
                    np.array([0.35, 0.3, 0.4]) * np.exp(-(months[s:] - s) / 10)[:, None]
                )
    out += rng.normal(0, 0.035, out.shape)
    cloud_p = 0.45 + 0.3 * np.cos(2 * np.pi * moy / 12)  # cloudy winters, clearer summers
    cloudy = rng.random((n, n_months)) < cloud_p[None, :]
    out[cloudy] = np.nan
    return out, months
