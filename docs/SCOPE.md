# Galicia Eucalyptus Plan: project scope

**Goal.** Measure, with machine learning and geospatial data, how eucalyptus plantations in
Galicia (NW Spain) affect three things:

1. **Native forest loss.** Is eucalyptus expansion replacing native broadleaf forest (oak, chestnut,
   birch) and shrubland, directly or through a fire → shrub → plantation pathway?
2. **Wildfire.** Does eucalyptus cover *cause* more frequent or more severe fire, once climate,
   terrain and human ignition pressure are held fixed? And does fire in turn open land to
   eucalyptus (the feedback loop)?
3. **Water.** Does eucalyptus cover reduce catchment water yield, summer low flows and soil
   moisture relative to the native vegetation it replaces?

The answers feed a **policy layer**: projections of fire and water outcomes under alternative
plans (business as usual, an area cap or moratorium, targeted native restoration) and a
data-driven map of where restoration buys the most risk reduction.

> **Status.** Synthetic validation: [`synthetic_validation/report.md`](synthetic_validation/report.md)
> (hand-set effects, not findings about Galicia). Real-data results: the Galician brief
> [`galicia_real/informe.md`](galicia_real/informe.md). The real species map has not been
> checked against an independent inventory sample, and the fire estimate depends on which map
> version is used. Plot-level agreement with inventory plots is low, and no river-gauge data
> was reachable.

---

## 1. Why this needs causal ML and not just maps

Mapping eucalyptus and overlaying fire perimeters shows *co-location*, not effect. In Galicia the
confounding is severe and runs in both directions:

- Plantations cluster on the wet Atlantic coast (A Coruña, Pontevedra, north Lugo), at low
  elevation (*E. globulus* is frost-sensitive), near the pulp industry and on abandoned farmland.
- Fire concentrates in the drier, hotter interior (Ourense, south Lugo) with pine and shrub, and
  follows human ignition pressure, rural depopulation and fire weather.

So a naive regression of "burned" on "eucalyptus share" can get the **sign** wrong. The same holds
for water: plantations sit in the wettest catchments. The design therefore separates
*prediction* (susceptibility models, validated with spatial cross-validation) from *effect
estimation* (double/debiased ML, matching, panel fixed effects), and validates both on a simulator
where the true answer is known.

## 2. Estimands

| Workstream | Unit | Treatment *D* | Outcome *Y* | Estimand |
|---|---|---|---|---|
| Forest loss | 1 km cell-year | native → eucalyptus transition | area converted | Transition matrix and area with confidence intervals (Olofsson et al. 2014 stratified estimator); share of tree-cover loss due to conversion vs. rotation harvest vs. fire |
| Conversion drivers | cell-year (native/shrub cells) | fire in previous 3 years | P(convert to eucalyptus) | Average effect of recent fire on conversion probability (fire → plantation feedback), DML |
| Fire occurrence | cell-year | eucalyptus cover fraction | burned (0/1) | ∂P(burn)/∂eucalyptus share, partially linear DML, cluster-robust by spatial block |
| Fire severity | burned cell-year | eucalyptus cover fraction | dNBR | Partially linear DML coefficient |
| Water yield | catchment-year | eucalyptus share of catchment | runoff (mm) and runoff ratio | Two-way fixed-effects panel (catchment + year) and Budyko (Fu) parameter shift |
| Soil moisture | cell | eucalyptus fraction | summer soil moisture index | DML plus propensity matching with balance diagnostics |

Every estimand gets a **naive** estimate (bivariate or simple OLS) next to the causal one, so the
size of the confounding bias is itself a reported quantity.

## 3. Data inventory (real-data phase)

| Layer | Source | Resolution / period | Role |
|---|---|---|---|
| Multispectral time series | Sentinel-2 L2A (Copernicus; STAC via Earth Search or Planetary Computer) | 10–20 m, 2017– | Species mapping (harmonic phenology features), NBR/dNBR, NDMI |
| Long history | Landsat 5/7/8/9 Collection 2 | 30 m, 1985– | Back-cast plantation expansion before Sentinel-2 |
| SAR | Sentinel-1 GRD | 10 m, 2015– | Cloud-free structure signal (Galicia winters are cloudy); soil-moisture proxy |
| Forest map | Mapa Forestal de España (MFE25/MFE50), Ministerio para la Transición Ecológica | 1:25k–1:50k | Training labels, species polygons |
| Field plots | Inventario Forestal Nacional (IFN3, IFN4) | plot network | Independent reference data for accuracy and area estimates |
| Land cover | CORINE Land Cover, SIOSE, ESA WorldCover | 10–100 m, several epochs | Covariates, cross-check |
| Tree cover loss | Hansen Global Forest Change | 30 m, 2001– | Loss events to be attributed (fire / rotation / conversion) |
| Burned area | EFFIS burnt areas; MODIS MCD64A1 | perimeters / 500 m | Fire outcome |
| Active fire | NASA FIRMS (MODIS, VIIRS) | 375 m–1 km, daily | Fire timing, ignition density |
| Fire weather | Copernicus CEMS Fire Weather Index; ERA5-Land | ~0.1–0.25°, daily | Confounder (weather) |
| Terrain | Copernicus DEM GLO-30 | 30 m | Elevation, slope, aspect, TWI |
| Streamflow | Augas de Galicia / CEDEX Anuario de Aforos gauging stations | daily | Water yield outcome |
| Evapotranspiration | MODIS MOD16A2; ERA5-Land | 500 m / 0.1° | ET mechanism check |
| Soil moisture | Sentinel-1 derived, ESA CCI SM, SMAP | 1–25 km | Soil-moisture outcome |
| Human pressure | INE population grid, OSM roads, cadastre (Catastro) parcels | – | Ignition-pressure and ownership confounders |
| Policy | Plan Forestal de Galicia (2021 revision) and subsequent eucalyptus planting restrictions | – | Scenario definitions (verify current legal status before citing) |

`src/eucalyptus_impact/data/catalog.py` holds this table as code, and `data/sources.py`
has the ingestion adapters. The adapters are written but **have not been run from this sandbox**
(outbound access to those portals is blocked here).

## 4. Methods by workstream

### 4.1 Species mapping and forest-loss accounting
- Features: per-pixel harmonic regression (mean, amplitude, phase of the annual cycle) of NDVI,
  NDMI and NBR, robust to cloud gaps. Evergreen eucalyptus with low amplitude separates from
  deciduous oak/chestnut, which has a high amplitude. Pine separates on NDMI level.
- Model: gradient-boosted trees (`HistGradientBoostingClassifier`), with **spatial block
  cross-validation** (random k-fold overstates accuracy under spatial autocorrelation).
- Area: map counts are biased by misclassification, so areas and transition areas are reported
  with the **stratified estimator of Olofsson et al. (2014)** and 95% CIs.
- Loss attribution: every tree-cover-loss event is labelled *fire*, *plantation rotation*
  (eucalyptus before and after) or *conversion* (native before, eucalyptus after). Without this
  split, eucalyptus harvest cycles inflate "deforestation" statistics.

### 4.2 Conversion drivers
- Gradient-boosted model of P(native/shrub → eucalyptus) with permutation importance and partial
  dependence on distance to the pulp mill, elevation, slope, neighbouring eucalyptus share
  (contagion), recent fire and farmland abandonment.
- Causal: DML effect of *fire in the previous 3 years* on conversion. This is the feedback loop.

### 4.3 Fire
- **Susceptibility (prediction):** cell-year gradient boosting on cover fractions, fire weather,
  terrain and human pressure; spatial-block CV; AUC, Brier score and calibration.
- **Effect (causal):** partially linear DML, *Y = θ·D + g(X) + ε*, with cross-fitting on spatial
  folds and standard errors clustered by 20 km block. Confounders *X* are climate, weather, terrain,
  human pressure and the other cover fractions. Outcomes are occurrence and dNBR severity.
- Robustness: propensity-matched comparison, placebo outcome (fire in the year *before* the
  plantation was established), sensitivity to block size.

### 4.4 Water
- **Catchment panel:** two-way fixed effects (catchment and year) of runoff and runoff ratio on
  eucalyptus share, controlling for precipitation and PET. SEs are clustered by catchment. Fixed
  effects absorb the "plantations sit in wet catchments" confounding.
- **Budyko:** fit Fu's equation *ET/P = 1 + φ − (1 + φ^w)^{1/w}* with *w = w₀ + a·euc + b·forest*,
  so the plantation effect is a shift in the land-surface parameter, then convert to mm of yield.
- **Pixel level:** DML and matching on summer soil moisture.

### 4.5 Policy scenarios (2025–2040)
- *BAU:* the conversion model keeps running.
- *Cap / moratorium:* eucalyptus area frozen at current extent.
- *Targeted restoration:* a share of eucalyptus area returned to native broadleaf, **prioritised by
  a model-based score** (predicted fire-risk reduction plus water gain per hectare).
- Outputs: expected burned area per year and change in water yield per catchment, with the
  scenario difference attributed using the causal estimates (not the predictive model's
  associations).

## 5. Validation strategy

1. **Simulator recovery (implemented).** `data/synthetic.py` generates a Galicia-shaped landscape
   (Atlantic–continental climate gradient, terrain, a pulp-mill attractor, contagious plantation
   expansion, fire–cover feedback, Budyko water balance) with *known* effects. The pipeline reports
   estimate vs. truth, and the tests assert recovery within tolerance while the naive estimate is
   biased.
2. **Spatial CV** for every predictive model.
3. **Independent reference data** (IFN plots) for map accuracy in the real-data phase.
4. **Falsification tests:** placebo timing and negative-control outcomes.

### What the simulator has already taught us

- Confounding is strong enough to flip signs: naive fire and runoff associations have the wrong
  sign, and the causal estimators recover the planted effects.
- Differencing two maps roughly doubles the apparent conversion area. Stratified estimation
  is mandatory.
- Map error attenuates DML estimates. Because *every* cover fraction is noisy, controls
  included, a single-variance regression calibration mis-corrects. SIMEX (extra simulated map
  error, extrapolated back to zero) recovers the truth and is the correction used.
- Tree-only nuisance learners leave regularization bias on near-linear structure (about 17% on
  soil moisture with perfect maps). The default nuisance learner is now ridge plus boosted
  residuals.
- The fire-occurrence effect has a low robustness value (about 1% partial R²). Real-data
  conclusions on fire *frequency* must be read through the sensitivity table. Severity and soil
  moisture are much sturdier.
- Static scenario pricing overstates restoration benefits (about 1.5 to 1.9×) because restored stands
  also burn and revert to shrub. **Open item before M5:** a dynamic model-based projection
  using the fitted conversion and fire models.
- Small scenario contrasts (the cap) are within simulation noise. Only the large ones
  (restoration) are resolved.

## 6. Threats to validity

- *Unobserved confounding* (arson, land-ownership disputes, community forests (montes vecinais)):
  mitigated by rich covariates and panel FE, and reported with sensitivity analysis. It is never
  "solved".
- *Interference / spillover:* fire spreads across cells. Neighbourhood cover is added as a
  covariate, and SEs are clustered at block scale.
- *Measurement error in treatment:* species maps are imperfect. Propagate classifier uncertainty
  (re-estimate with posterior class probabilities).
- *Reverse causality:* fire → plantation is modelled explicitly with lagged cover (cover at t−1
  → fire at t).
- *Scale:* 1 km cells mix land uses. Fractions rather than dominant class are used for all effects.

## 7. Deliverables and milestones

| # | Deliverable | Status |
|---|---|---|
| M0 | Scope (this document), repo skeleton, data catalog | done |
| M1 | Synthetic simulator with known effects, full pipeline end to end, tests | done |
| M1b | Robustness: group effects, OVB sensitivity, SIMEX, cluster-size SEs; adapter tests on fixtures | done |
| M2 | Real-data ingestion → 1 km panel | done from object storage (Sentinel-2, WorldCover, Copernicus DEM, Hansen GFC, EFFIS severity, Overture/OSM, GHCN); MFE/IFN labels and river gauges unreachable |
| M3 | Species maps 2017 and 2024, areas, 2017→2024 conversion | done: cleaned OSM labels plus harvest-history pseudo-labels, held-out-north eucalyptus F1 0.72; 2017 normalised and backdated from 2024 (≈489k ha 2017, 440k ha 2024); checked against ~6,900 inventory-design plots from GBIF (area right in aggregate, plot-level F1 0.45); Landsat back-cast not done |
| M4 | Causal estimates on real data with robustness | done for fire (eucalyptus burns less than agriculture/other, −0.27 pp per 10 points, sign stable across map versions but not always significant; no severity effect); water not estimable without gauges; gauge-ready pipeline and power study on 79 real DEM catchments (MDE ~140 mm/yr per 10 points with 2017–2024 cover change) |
| M5 | Year-by-year projections, restoration priority map, policy brief | done: dynamic engine, brief in Galician (`docs/galicia_real/informe.md`) |
