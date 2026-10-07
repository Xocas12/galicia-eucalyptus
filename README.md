# galicia-eucalyptus

Machine learning and geospatial intelligence to measure what eucalyptus plantations in Galicia do
to **native forest**, **wildfire** and **water**, and what alternative forest plans would change.

- **Scope, estimands, data inventory and threats to validity:** [`docs/SCOPE.md`](docs/SCOPE.md)
- **Latest synthetic validation report (1 km, 2000–2024), in Galician:** [`docs/synthetic_validation/report.md`](docs/synthetic_validation/report.md)

- **Real-data brief for Galicia (in Galician):** [`docs/galicia_real/informe.md`](docs/galicia_real/informe.md)

> **Read the real-data results with their caveats.** The species maps are trained on cleaned
> OpenStreetMap labels, harvest-history pseudo-labels and the Mapa Forestal de España (MFE50,
> about 1998), and are checked against the MFE50 and the IFN3 inventory plots. The eucalyptus
> area is plausible, but pixel- and plot-level agreement is modest (F1 about 0.5), and map error
> attenuates every estimate that uses the map. Numbers from the synthetic runs are properties of
> the methods, not facts about Galicia.

Self-contained: one `pyproject.toml`, one test suite, one virtual environment. The synthetic
end-to-end run is pure numpy and scikit-learn; the real-data ingestion lives behind the
optional `geo` extra, so the pipeline is runnable without any geospatial stack installed.

## Quickstart

```bash
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

## Point lookup

```bash
.venv/bin/euc lookup 42.6187 -8.7500            # latitude, longitude (WGS84)
.venv/bin/euc lookup 42.6187 -8.7500 --lang gl  # Galician labels
.venv/bin/euc lookup 42.6187 -8.7500 --json     # machine-readable
```

For any point in Galicia (after `euc real run`): the 40 m species class in 2017 and 2024 with
its confidence, WorldCover, elevation, Hansen loss year and EFFIS burnt years; the 1 km cell's
cover, calibrated annual burn probability with its Galicia percentile and risk class,
restoration priority, and the estimated eucalyptus effect with a 95% interval; the nearest
IFN3 plot; and the upstream catchment's area and cover. These are model estimates with the
limits described below (the 40 m class has plot-level F1 ~0.45).

## Real data (Galicia)

```bash
uv pip install -e '.[geo,dev]' pyarrow
.venv/bin/euc real fetch   # ~1.5 h: layers + Sentinel-2 monthly composites for 2017 and 2024
.venv/bin/euc real run     # ~1 h: species maps, fire and conversion analysis, projections, brief
```

Everything comes from public object storage: Sentinel-2 L2A COGs, ESA WorldCover, the
Copernicus DEM, Hansen Global Forest Change v1.12, EFFIS burn severity 2018–2023, Overture Maps
(OpenStreetMap land, land-use and building layers), NOAA GHCN stations, Natural Earth, the GBIF
occurrence archive (IFN3 plots), Landsat Collection 2 (Planetary Computer), CAMELS-ES (Zenodo)
and the MFE50 (MITECO).
Downloads are cached under `data/` (git-ignored).

What the real-data run established, and what it did not:

- **Imagery.** Monthly NDVI/NDMI/NBR composites at 40 m. Each acquisition date is mosaicked
  across tiles after removing per-tile radiometric offsets estimated on tile overlaps (the
  archive's per-tile atmospheric correction left seams up to ~0.02).
- **Species maps.** A gradient-boosting classifier on harmonic phenology and gap-filled
  monthly indices, trained on 2024 imagery. Labels: OSM polygons cleaned by winter behaviour
  (eucalyptus stays green and wet in winter; deciduous natives drop), eucalyptus pseudo-labels
  from evergreen, winter-moist pixels in areas with a history of Hansen clear-cuts, and the
  Mapa Forestal de España (MFE50, about 1998) on pixels with no recorded loss or fire since 2001.
  Trained without the northern 100 km square and tested on its OSM labels, eucalyptus F1 is
  0.77 (0.72 without the MFE labels, about 0 before cleaning and pseudo-labels). The 2017 image
  is quantile-normalised to 2024 on stable pixels and its classes are backdated from 2024
  wherever no harvest or fire happened in between (an independent 2017 classifier, F1 0.68, is
  kept as a sensitivity check). Mapped eucalyptus: about 442k ha in 2024 and 483k ha in 2017.
- **Independent map checks.** Two official references, both from around 1998:
  - *MFE50* (MITECO, 1:50,000, IFN3 base; wall to wall, rasterised by forest formation, mixed
    formations excluded): the 2024 map scores overall accuracy 0.61, eucalyptus precision 0.47,
    recall 0.68, F1 0.55; pine F1 0.47, native F1 0.59. A block-split experiment (train on half
    of the 10 km blocks, test on the other half) showed that adding MFE labels raises held-out
    accuracy from 0.55 to 0.61 and IFN3-plot F1 from 0.45 to 0.47, so the maps now use them.
    The newer MFE25 (IFN4 base, 2011) sits behind an anti-bot challenge on the download server;
    put its shapefile in `data/raw/mfe25/` to use it.
  - *IFN3 plots* (on GBIF, MAGRAMA collection IFN3; species present per plot, no counts or
    dates; about 6,900 plots): the eucalyptus share is right in aggregate (30% of forested plots
    mapped as eucalyptus vs 27.5% listing it), plot-level F1 0.46.

  Pixel- and plot-level agreement stays modest: partly real change since 1998, partly map
  error. Opportunistic GBIF sightings are a secondary check. `real/reference.py`.
- **Landsat back-cast (1990-2017).** Landsat 4-8 Collection 2 surface reflectance from
  Planetary Computer: monthly NDVI/NDMI/NBR cubes per three-year epoch from all clear scenes,
  the Sentinel-2 features, and one classifier per epoch trained on pixels unchanged since 2001
  (`real/landsat.py`). Much better than the earlier Level-1 attempt (eucalyptus F1 0.67, 0.71,
  0.71, 0.77 for 1990, 2000, 2010, 2017; against MFE50 the 2000 map scores F1 0.57), but it
  still **fails the change test**: the area does not grow (571, 570, 552, 543 kha, against the
  known expansion), and pixels turning eucalyptus in 2000-2010 had a Hansen clear-cut only 1.75
  times as often as unchanged pixels (the gate requires 2). It is gated out of the water study.
- **Fire (EFFIS 2018–2023, 29,565 cells × 6 years).** Relative to agriculture and other cover,
  10 more points of eucalyptus lower annual burn probability by 0.26 pp (95% CI −0.46 to
  −0.07; robustness value 0.017, so a weak confounder could explain it). Against native
  broadleaf, eucalyptus raises it by 0.23 pp, but the interval includes zero (−0.12 to 0.57).
  All three map versions now agree and are significant (2017 backdated −0.26, 2017 independent
  −0.16, 2024 −0.46). No severity effect is detectable.
- **Ground check of the fire result.** IFN3 plots record where eucalyptus was around 1998,
  with no map error. Plot-level burn rates 2018-2023 (EFFIS): 0.14%/yr on eucalyptus plots vs
  0.38% elsewhere; with the same controls the difference is -0.05 pp (95% CI -0.16 to 0.05),
  and eucalyptus-only plots vs pine or native plots are also negative but not significant.
  Only 122 plots burned, so it is weak, but it agrees in sign with the map-based estimate:
  no sign that eucalyptus burns more. `analysis.plot_fire_check`.
- **Red-edge bands.** Adding Sentinel-2 B05/B07/B8A indices moves eucalyptus F1 only within
  noise (north transfer 0.721 -> 0.727, IFN3 plots 0.453 -> 0.465), so the maps were not
  rebuilt (`species.red_edge_pilot`, `s2.build_period(..., product="re")`).
- **Native forest.** 2017→2024 native-to-eucalyptus conversion is reported three ways (all
  pixels 1,477 ha, confident pixels 100 ha, confident pixels corroborated by Hansen loss or
  fire 89 ha), since map differencing inflates change. With the MFE-trained maps the confident
  figures fell from 563/484 ha: most of the earlier "conversion" was classification noise.
- **Projections.** A year-by-year engine (validated on the simulator, where it overstates
  restoration benefits by about 40%) projects the scenarios to 2040 with paired uncertainty
  bands. Restoring 25% of eucalyptus in priority cells changes mean burnt area by about −960
  ha/yr (5–95% band −1,740 to +280; random placement about −330): the bands now include zero.
  Baseline burn probabilities are cross-fitted by spatial block and isotonic-calibrated.
- **Water.** CAMELS-ES (Zenodo 15040948, CC BY 4.0) gives daily flows, EMO-1 precipitation and
  reference ET for 33 catchments mostly inside Galicia, 1992–2020. With two-way fixed effects
  the estimate is uninformative: −1,009 mm/yr per 10 points of eucalyptus (95% CI −2,236 to
  218), because eucalyptus barely changes within catchments in the gauge years (the maps start
  in 2017 and the records end in 2020). A power study on 79 DEM catchments confirms it: with
  the 2017/2024 maps the minimum detectable effect is ~120 mm/yr per 10 points, far above
  plausible effects (10–20); a cover history six times longer would bring it to ~20, which is
  what the Landsat back-cast was for. `real/water.py` also reads CEDEX `afliq.csv`/`estaf.csv`
  or a generic `stations.csv` + `flows.csv` in `data/raw/gauges/`.

Next steps that would change the conclusions: a current pixel-level reference (the MFE25 for
Galicia, 2011, downloadable by hand from MITECO), a cover history that passes the change test
(e.g. a dedicated change-detection approach on Landsat), and EFFIS perimeters before 2018 for
more fire years.
