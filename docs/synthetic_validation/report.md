# Galicia eucalyptus impact: pipeline report

> **SYNTHETIC DATA.** Every number below comes from the simulated landscape in
> `data/synthetic.py`, whose effects were set by hand. This report shows that the estimators
> recover known effects. It says nothing about the real Galicia.

Domain: 38,581 land cells at 1000 m, 2000-2024.
Eucalyptus area (truth) went from 669k ha to 857k ha.

![maps](maps.png)

## 1. Effect estimates vs truth

The naive column is what a map overlay or bivariate regression would report. The causal
column partials out climate, terrain, human pressure and the other cover types.

| effect | method | estimate | se | truth | covers_truth |
|---|---|---|---|---|---|
| eucalyptus -> P(burn) | OLS | -0.01678 | 0.007853 | 0.02209 | **no** |
| eucalyptus -> P(burn) | DML-PLR | 0.01501 | 0.005517 | 0.02209 | yes |
| eucalyptus -> P(burn) | DML-PLR (ME-corrected) | 0.01684 | 0.006193 | 0.02209 | yes |
| eucalyptus -> dNBR | OLS | -71.13 | 21.38 | 120 | **no** |
| eucalyptus -> dNBR | DML-PLR | 105.3 | 7.693 | 120 | yes |
| eucalyptus -> dNBR | DML-PLR (ME-corrected) | 122 | 8.307 | 120 | yes |
| recent fire -> conversion rate | OLS | 0.007231 | 0.001187 | 0.006126 | yes |
| recent fire -> conversion rate | DML-PLR | 0.006484 | 0.00111 | 0.006126 | yes |
| eucalyptus -> runoff | OLS | 115.2 | 105.3 | -113.9 | **no** |
| f_eucalyptus -> runoff | TWFE | -108.8 | 63.11 | -113.9 | yes |
| eucalyptus -> runoff | Budyko-Fu | -105.5 | - | -113.9 | - |
| f_eucalyptus -> low_flow | TWFE | -37.15 | 15.15 | - | - |
| eucalyptus -> summer soil moisture | OLS | -0.03394 | 0.008257 | -0.06 | **no** |
| eucalyptus -> summer soil moisture | DML-PLR | -0.04391 | 0.002878 | -0.06 | **no** |
| eucalyptus -> summer soil moisture | DML-PLR (ME-corrected) | -0.05014 | 0.003395 | -0.06 | **no** |
| eucalyptus stand vs other -> soil moisture | Matching (bias-corrected) | -0.02293 | 0.001437 | -0.02424 | yes |
| reverse check: future eucalyptus gain ~ P(burn) | DML-PLR | 0.3673 | 0.04089 | - | - |

![effects](effects.png)

Map-error variance of the eucalyptus fraction (from the reference sample):
0.00042. Classifier error in the *treatment* attenuates every
effect towards zero, and partialling out the other cover fractions makes it worse (it removes
signal but not noise). The ME-corrected rows apply regression calibration.

Fire-occurrence effect of each cover class vs the agriculture/other reference (used to price
scenarios; truth on the logit scale: eucalyptus 0.9, pine 0.7,
native -0.6, shrub 1.1):

| cover | dP(burn)/dshare | se |
|---|---|---|
| eucalyptus | 0.01684 | 0.006193 |
| pine | 0.01432 | 0.008039 |
| native_broadleaf | -0.008951 | 0.006105 |
| shrub | 0.04344 | 0.009612 |

## 2. Species mapping and forest-loss accounting

- Spatial-block CV accuracy **0.934** vs random CV 0.939
  (the gap is the optimism of non-spatial validation). Kappa 0.916.

| name | map_area_ha | est_area_ha | ci95_ha | true_area_ha | users_accuracy | producers_accuracy |
|---|---|---|---|---|---|---|
| eucalyptus | 9.0646e+05 | 8.4114e+05 | 60,550 | 8.5677e+05 | 0.84 | 0.90523 |
| pine | 4.7828e+05 | 5.436e+05 | 60,550 | 5.2349e+05 | 0.83333 | 0.7332 |
| native_broadleaf | 1.0718e+06 | 1.068e+06 | 15,495 | 1.0692e+06 | 0.99333 | 0.99683 |
| shrub | 8.5457e+05 | 8.4708e+05 | 17,094 | 8.5559e+05 | 0.98667 | 0.99538 |
| agriculture | 5.0753e+05 | 5.1931e+05 | 23,048 | 5.081e+05 | 0.98667 | 0.9643 |
| other | 39,481 | 38,955 | 727.12 | 44,904 | 0.98667 | 1 |

![area](area_species.png)

Change areas (map differencing compounds two maps' errors; the stratified estimator corrects it):

| name | map_area_ha | est_area_ha | ci95_ha | true_area_ha |
|---|---|---|---|---|
| native->eucalyptus | 32,215 | 27,490 | 1,830 | 30,921 |
| other->eucalyptus | 3.1328e+05 | 1.6718e+05 | 59,715 | 1.5644e+05 |
| eucalyptus stable | 5.6097e+05 | 6.3365e+05 | 30,018 | 6.6941e+05 |
| other | 2.9516e+06 | 3.0298e+06 | 61,924 | 3.0013e+06 |

![change](area_change.png)

Tree-cover loss attribution (cell-fraction units):

| driver | attributed | true | attributed_share | true_share |
|---|---|---|---|---|
| fire | 4,713 | 4,456 | 0.1975 | 0.1867 |
| rotation | 1.878e+04 | 1.887e+04 | 0.7868 | 0.7909 |
| conversion | 375.9 | 535.6 | 0.01575 | 0.02244 |

Conversion driver model: spatial-CV R² 0.129.

![drivers](conversion_drivers.png)

## 3. Fire

Susceptibility: spatial-CV AUC **0.813**, Brier 0.0252,
base rate 0.0275.

![calibration](fire_calibration.png)

## 4. Water

Budyko (Fu) parameters (estimate, SE; truth w_euc = 1.2):
`{"w0": [2.04, 0.041], "w_euc": [1.089, 0.064], "w_pine": [0.43, 0.123], "w_native": [0.293, 0.067]}`

Matching balance (share of treated dropped for lack of overlap:
0.97):

| feature | smd_before | smd_after |
|---|---|---|
| elev | -2.358 | 0.06848 |
| slope | -0.3863 | 0.04156 |
| continentality | -1.984 | 0.1302 |
| precip_mean | 0.4187 | 0.04002 |
| pet_mean | -1.161 | 0.1334 |
| summer_temp | -0.01642 | 0.1247 |
| log_pop | 0.617 | -0.1291 |
| dist_coast_km | -0.9331 | 0.1918 |

## 5. Policy scenarios to 2040

Horizon-year contrasts vs BAU: the simulated world (truth) next to the projection from the
estimated causal effects (per-class DML effects, risk-weighted, for fire; the fitted Budyko curve
for runoff). Small contrasts, such as the cap, are within simulation noise, so read their sign
with care.

| scenario | d_eucalyptus_ha | d_burned_ha_simulated | d_burned_ha_model | d_runoff_mm_simulated | d_runoff_mm_model |
|---|---|---|---|---|---|
| Cap / moratorium | -3.587e+04 | -102.3 | 152.6 | 0.674 | 0.6627 |
| Targeted restoration | -2.424e+05 | -2,437 | -3,208 | 3.646 | 3.924 |
| Random restoration | -2.423e+05 | -1,899 | -1,922 | 3.877 | 4.116 |

![scenarios](scenarios.png)
