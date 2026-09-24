# galicia-eucalyptus

Machine learning and geospatial intelligence to measure what eucalyptus plantations in Galicia do
to **native forest**, **wildfire** and **water**, and what alternative forest plans would change.

- **Scope, estimands, data inventory and threats to validity:** [`docs/SCOPE.md`](docs/SCOPE.md)
- **Latest synthetic validation report (1 km, 2000–2024):** [`docs/synthetic_validation/report.md`](docs/synthetic_validation/report.md)

> **No real-data result exists yet.** The pipeline currently runs on a *synthetic* Galicia-like
> landscape with hand-set causal effects, to prove the estimators recover known answers before
> they touch real data. The real-data adapters (`data/sources.py`) are written but have not been
> run. Do not cite any number from this repo as a finding about Galicia.

Self-contained subproject: it shares nothing with `gosplan/` at the repository root and has its own
`pyproject.toml`, tests and virtual environment.

## Quickstart

```bash
cd galicia-eucalyptus
uv venv && uv pip install -e '.[dev]'          # add '.[geo]' for real-data ingestion
.venv/bin/pytest                                # 23 tests, ~1 min
.venv/bin/euc run --config configs/fast.yaml    # 4 km smoke run, ~1 min  -> outputs/fast/
.venv/bin/euc run --config configs/default.yaml # 1 km full run, ~3.5 min -> outputs/default/
.venv/bin/euc catalog                           # the real data sources
```

Each run writes `report.md`, `metrics.json`, CSV tables and figures to its `output_dir`.

## What the pipeline does

| Workstream | Question | Method |
|---|---|---|
| Species mapping | Where is eucalyptus, and how many hectares? | Harmonic (phenology) features from cloudy NDVI/NDMI/NBR series → gradient boosting; spatial-block CV; Olofsson et al. (2014) stratified area estimates with 95% CIs |
| Forest loss | Is native forest being converted? | Change-class area estimation (native→eucalyptus, other→eucalyptus); tree-cover-loss attribution into fire / rotation harvest / conversion |
| Conversion drivers | What drives planting, and does fire feed it? | Gradient-boosting driver model with permutation importance; DML effect of recent fire on conversion |
| Fire | Does eucalyptus raise fire occurrence and severity? | Susceptibility model (spatial-CV AUC, calibration); partially linear DML with spatial cross-fitting and block-clustered SEs; per-class cover effects; reverse-causality check |
| Water | Does eucalyptus reduce runoff and soil moisture? | Catchment two-way fixed effects; Budyko (Fu) fit with a cover-dependent land-surface parameter; DML and bias-corrected matching with balance diagnostics for soil moisture |
| Policy | What do a cap and restoration buy? | Forward projection of BAU / cap / targeted vs random restoration to 2040; model-based projection checked against the simulated outcome |

Every causal estimate is reported next to a **naive** one, so the size of the confounding bias is
itself a result.

## What the synthetic validation shows (1 km run)

These are properties of the *methods*, not facts about Galicia:

- **Naive analysis gets the sign wrong.** Plantations sit on the wet, mild coast where fire
  weather is low. A bivariate regression says eucalyptus *reduces* fire occurrence and severity.
  DML recovers the planted positive effect: severity 122 ± 16 dNBR against a true 120.
- **The same happens for water.** Plantations are in the wettest catchments, so naive runoff
  regressions come out positive. Catchment fixed effects (−109 mm per unit share) and the Budyko
  fit (−106) recover the true −114.
- **Map-based change is badly inflated.** Differencing two classified maps overstates
  "other→eucalyptus" conversion about 2×. The stratified estimator corrects this: 167k ± 60k ha
  against a true 156k ha.
- **Random CV modestly overstates map accuracy** compared with spatial-block CV (0.939 vs 0.934 here; the gap grows with stronger site effects).
- **Classifier error attenuates effects.** Error in mapped cover biases every effect toward
  zero, and conditioning on the other cover fractions makes it worse. Regression calibration using
  the map-error variance fixes most of it, but soil moisture is still about 15% attenuated. This is
  a known open issue to resolve before the real-data phase.
- **Targeting pays.** Restoring the same eucalyptus area in cells chosen by the model-based
  priority score cuts expected burned area by more than random restoration. The causal-model
  projection lands within about a third of the simulated restoration effects.

## Layout

```
configs/                 fast (4 km) and default (1 km) run configurations
docs/SCOPE.md            research design
src/eucalyptus_impact/
  data/catalog.py        real data source catalogue (Sentinel-2, MFE, IFN, EFFIS, FIRMS, gauges, ...)
  data/sources.py        real-data ingestion adapters (STAC, rasterisation, FIRMS API, gauges)
  data/synthetic.py      synthetic landscape with known effects
  geo/                   UTM 29N grid, spatial blocks, raster neighbourhood ops
  features/              spectral indices + harmonic features; cell-year / catchment-year panels
  models/                landcover, conversion, fire, hydrology
  causal/                DML (partially linear, clustered SEs, ME correction), matching
  validation/            spatial block k-fold
  scenarios.py           policy projections and restoration prioritisation
  pipeline.py, reporting.py, cli.py
tests/                   unit tests per estimator + end-to-end run
```

## Next step: real data (milestone M2)

1. Build Sentinel-2 monthly index cubes with `sources.search_sentinel2` and
   `sentinel2_monthly_indices`. Label pixels from MFE polygons (`load_mfe_labels`) and keep IFN
   plots as independent reference data.
2. Rasterise EFFIS perimeters to the 1 km grid (`burned_panel`) and add FIRMS counts, ERA5-Land
   and CEMS FWI.
3. Delineate gauged basins on the Copernicus DEM and compute annual runoff and low flow from the
   Anuario de Aforos (`gauge_annual_runoff`).
4. Assemble the same panel columns as `features/panel.py` and run the models unchanged.
