# galicia-eucalyptus

Machine learning and geospatial intelligence to measure what eucalyptus plantations in Galicia do
to **native forest**, **wildfire** and **water**, and what alternative forest plans would change.

- **Scope, estimands, data inventory and threats to validity:** [`docs/SCOPE.md`](docs/SCOPE.md)
- **Latest synthetic validation report (1 km, 2000–2024), in Galician:** [`docs/synthetic_validation/report.md`](docs/synthetic_validation/report.md)

- **Real-data brief for Galicia (in Galician):** [`docs/galicia_real/informe.md`](docs/galicia_real/informe.md)

> **Read the real-data results with their caveats.** The species maps are trained on
> OpenStreetMap labels, not the official forest map or inventory (unreachable from the build
> environment), and 91% of the eucalyptus labels sit in one 100 km square in the north. Outside
> it, the eucalyptus layer does not validate, and that limits every estimate that uses it.
> Numbers from the synthetic runs are properties of the methods, not facts about Galicia.

Self-contained subproject: it shares nothing with `gosplan/` at the repository root and has its own
`pyproject.toml`, tests and virtual environment.

## Quickstart

```bash
cd galicia-eucalyptus
uv venv && uv pip install -e '.[dev]'          # add '.[geo]' for real-data ingestion
.venv/bin/pytest                                # 43 tests, ~4 min
.venv/bin/euc run --config configs/fast.yaml    # 4 km smoke run, ~1 min  -> outputs/fast/
.venv/bin/euc run --config configs/default.yaml # 1 km full run, ~6 min   -> outputs/default/
.venv/bin/euc catalog                           # the real data sources
```

Each run writes `report.md` (in Galician: prose, tables and figures; enforced by a test),
`metrics.json`, CSV tables, figures and a restoration-priority
map (`restoration_priority.csv`, plus a GeoTIFF in EPSG:25829 when the `geo` extra is installed)
to its `output_dir`.

## What the pipeline does

| Workstream | Question | Method |
|---|---|---|
| Species mapping | Where is eucalyptus, and how many hectares? | Harmonic (phenology) features from cloudy NDVI/NDMI/NBR series → gradient boosting; spatial-block CV; Olofsson et al. (2014) stratified area estimates with 95% CIs |
| Forest loss | Is native forest being converted? | Change-class area estimation (native→eucalyptus, other→eucalyptus); tree-cover-loss attribution into fire / rotation harvest / conversion |
| Conversion drivers | What drives planting, and does fire feed it? | Gradient-boosting driver model with permutation importance; DML effect of recent fire on conversion |
| Fire | Does eucalyptus raise fire occurrence and severity? | Susceptibility model (spatial-CV AUC, calibration); partially linear DML with spatial cross-fitting and block-clustered SEs; per-class cover effects; reverse-causality check |
| Water | Does eucalyptus reduce runoff and soil moisture? | Catchment two-way fixed effects; Budyko (Fu) fit with a cover-dependent land-surface parameter; DML and bias-corrected matching with balance diagnostics for soil moisture |
| Robustness | Where is the effect largest, and how fragile is it? | Group effects (coast vs interior, by fire weather); omitted-variable-bias bounds and robustness values; SE sensitivity to cluster size; SIMEX correction for map error in all cover fractions |
| Policy | What do a cap and restoration buy? | Forward projection of BAU / cap / targeted vs random restoration to 2040; model-based projection checked against the simulated outcome |

Every causal estimate is reported next to a **naive** one, so the size of the confounding bias is
itself a result.

## What the synthetic validation shows (1 km run)

These are properties of the *methods*, not facts about Galicia:

- **Naive analysis gets the sign wrong.** Plantations sit on the wet, mild coast where fire
  weather is low. A bivariate regression says eucalyptus *reduces* fire occurrence and severity.
  DML recovers the planted positive effects: severity 122 ± 14 dNBR against a true 120, and
  occurrence 0.028 ± 0.012 against a true 0.022.
- **The same happens for water.** Plantations are in the wettest catchments, so naive runoff
  regressions come out positive. Catchment fixed effects (−109 mm per unit share) and the Budyko
  fit (−106) recover the true −114.
- **Map-based change is badly inflated.** Differencing two classified maps overstates
  "other→eucalyptus" conversion about 2×. The stratified estimator corrects this: 167k ± 60k ha
  against a true 156k ha.
- **Two biases hid inside "causal ML", and both are now fixed:**
  - Tree-only nuisance models approximate near-linear relationships in steps, which cost about
    17% of the soil-moisture effect even with perfect maps. The default nuisance learner is now
    ridge plus boosted residuals.
  - Map error in *every* cover fraction attenuates the effects. A single-variance correction
    over-corrects, while SIMEX recovers the truth: soil moisture −0.058 vs −0.06, severity 127
    vs 120.
- **The effect is heterogeneous.** Eucalyptus raises fire probability most in the interior and
  under high fire weather. The group effects rank the same way as the truth, but the low-risk
  groups (coast, low fire weather) are overestimated by roughly 2× and shrink toward the pooled
  average. Treat them as a ranking, not as calibrated local effects.
- **The fire-occurrence effect is fragile to hidden confounding.** An unmapped confounder
  explaining about 1% of the residual variance of both cover and fire would erase it. Severity
  and soil moisture need about 10% and 17%. On real data this is the table to read before any
  causal claim about fire frequency.
- **Targeting pays, and the static projection overstates it.** Restoring the same eucalyptus area
  in model-prioritised cells cuts expected burned area more than random restoration, and the
  causal-model projection ranks them correctly. It overstates the size of the benefit (about 1.5 to
  1.9×) because it holds restored stands fixed, while in the simulation they burn and revert to
  shrub. Next step: a dynamic projection that feeds the fitted conversion and fire models forward.

## Layout

```
configs/                 fast (4 km) and default (1 km) run configurations
docs/SCOPE.md            research design
src/eucalyptus_impact/
  data/catalog.py        real data source catalogue (Sentinel-2, MFE, IFN, EFFIS, FIRMS, gauges, ...)
  data/sources.py        real-data ingestion adapters (STAC, rasterisation, FIRMS API, gauges)
  data/synthetic.py      synthetic landscape with known effects
  geo/                   UTM 29N grid, spatial blocks, raster ops, GeoTIFF export
  features/              spectral indices + harmonic features; cell-year / catchment-year panels
  models/                landcover, conversion, fire, hydrology
  causal/                DML (partially linear, clustered SEs), learners, matching,
                         sensitivity (OVB bounds), SIMEX
  validation/            spatial block k-fold
  scenarios.py           policy projections and restoration prioritisation (synthetic)
  dynamic.py             year-by-year projection engine (synthetic and real)
  real/                  real Galicia pipeline: layers, Sentinel-2 composites, species maps,
                         analysis, Galician brief
  pipeline.py, reporting.py, cli.py
tests/                   unit tests per estimator + end-to-end run
```

## Real data (Galicia)

```bash
uv pip install -e '.[geo,dev]' pyarrow
.venv/bin/euc real fetch   # ~1.5 h: layers + Sentinel-2 monthly composites for 2017 and 2024
.venv/bin/euc real run     # ~1 h: species maps, fire and conversion analysis, projections, brief
```

Everything comes from public object storage: Sentinel-2 L2A COGs, ESA WorldCover, the
Copernicus DEM, Hansen Global Forest Change v1.12, EFFIS burn severity 2018–2023, Overture Maps
(OpenStreetMap land, land-use and building layers), NOAA GHCN stations and Natural Earth.
Downloads are cached under `data/` (git-ignored).

What the real-data run established, and what it did not:

- **Imagery.** Monthly NDVI/NDMI/NBR composites at 40 m. Each acquisition date is mosaicked
  across tiles after removing per-tile radiometric offsets estimated on tile overlaps (the
  archive's per-tile atmospheric correction left seams up to ~0.02).
- **Species maps.** A gradient-boosting classifier on harmonic phenology and gap-filled
  monthly indices, trained on 2024 imagery. OSM labels are cleaned by winter behaviour
  (eucalyptus stays green and wet in winter; deciduous natives drop), and eucalyptus
  pseudo-labels are added across Galicia from evergreen, winter-moist pixels in areas with a
  history of Hansen clear-cut harvests. Tested by training without the northern 100 km square
  and predicting its OSM labels, eucalyptus F1 is 0.72 (it was about 0 before the cleaning and
  pseudo-labels). The 2017 image is quantile-normalised to 2024 on stable pixels, and 2017 classes
  are backdated from 2024 wherever no harvest or fire happened in between (an independent 2017
  classifier transfers with F1 0.65 and is kept as a sensitivity check). Mapped eucalyptus:
  about 440k ha in 2024 and 489k ha in 2017, with the difference on harvested or burnt pixels.
- **Independent map check.** The official downloads (MFE, IFN4) are blocked here, but the GBIF
  archive on AWS holds the Ministry's **IFN3** plots (MAGRAMA, collection IFN3; Galicia surveyed
  around 1997-1998): a systematic 1 km grid with the species present in each plot, but no
  counts or dates (about 6,900 plots in Galicia). The eucalyptus area is right in aggregate
  (31-34% of forested plots mapped as eucalyptus vs 27.5% listing it in 1998), but plot-level
  agreement is low: F1 0.45 overall, 0.44 outside the north, well below the OSM transfer test.
  A date-matched Landsat 2000 map agrees no better (F1 0.42), so most of the gap is the
  reference (any eucalyptus in a 25 m plot) plus map error, not change since 1998. Adding the
  plots to training (block-split experiment) did not help. Opportunistic GBIF sightings are a
  secondary check. `real/reference.py`.
- **Landsat back-cast (1990-2017).** Landsat 4-8 Collection 1 from Google's public archive,
  seasonal NDVI/NDMI/NBR composites for four epochs, one classifier per epoch trained on pixels
  unchanged since 2001 (`real/landsat.py`). It **fails validation**: eucalyptus F1 0.53-0.65,
  no area trend (597, 551, 573, 572 kha), and pixels turning eucalyptus 2000-2010 show Hansen
  loss no more often than unchanged pixels (1.8% vs 1.6%). It is gated out of the water study.
  Doing it properly needs Landsat Collection 2 surface reflectance (Planetary Computer, or the
  requester-pays `usgs-landsat` bucket with AWS credentials).
- **Fire (EFFIS 2018–2023, 29,565 cells × 6 years).** Relative to agriculture and other cover,
  10 more points of eucalyptus lower annual burn probability by 0.27 pp (95% CI −0.46 to
  −0.08; robustness value 0.018, so a weak confounder could explain it). Against native
  broadleaf, eucalyptus raises it by 0.39 pp, but the interval touches zero. The three map
  versions (2017 backdated, 2017 independent, 2024) agree on the sign, but not all are
  significant: the result depends on the map. No severity effect is detectable.
- **Native forest.** 2017→2024 native-to-eucalyptus conversion is reported three ways (all
  pixels 1,548 ha, confident pixels 563 ha, confident pixels corroborated by Hansen loss or fire
  484 ha), since map differencing inflates change.
- **Projections.** A year-by-year engine (validated on the simulator, where it overstates
  restoration benefits by about 40%) projects the scenarios to 2040 with paired uncertainty
  bands. Restoring 25% of eucalyptus in priority cells cuts mean burnt area by about 2,500
  ha/yr (5–95% band 870–3,600); the projections inherit the map sensitivity above.
- **Water.** No gauge record is reachable (CEDEX, Augas de Galicia, MeteoGalicia, GRDC and
  Zenodo are blocked), so there is no estimate. `real/water.py` builds everything else:
  priority-flood routing on the Copernicus DEM, 79 whole non-nested catchments (30–1,500 km²),
  water-year precipitation and Thornthwaite PET from GHCN stations, catchment cover paths from
  the 2017/2024 maps dated by Hansen loss or fire, and loaders for CEDEX `afliq.csv`/`estaf.csv`
  or a generic `stations.csv` + `flows.csv` dropped in `data/raw/gauges/`. A power study on the
  real catchments (simulated flows with a known effect) shows the estimator is unbiased with
  correct coverage, but eucalyptus changes by only ~2 points per catchment over 2017–2024, so
  the minimum detectable effect is ~140 mm/yr per 10 points, far above plausible effects
  (10–20). The published gauge datasets for Spain (CAMELS-ES, EStreams, GRDC-Caravan) are on
  Zenodo, which this environment's network policy blocks. A cover history six times longer (e.g. a working Landsat back-cast) brings it to ~21, and
  swapping map versions roughly doubles the estimate, so map error matters as much as noise.

Next steps that would change the conclusions: a dated, pixel-level reference (the Mapa
Forestal de España polygons or IFN4 plots with dominance), gauge data plus a longer (Landsat) cover history for the water question, and EFFIS perimeters
before 2018 for more fire years.
