"""Real-data pipeline for Galicia: ingestion, species mapping, causal analysis and projections.

Sources reachable from object storage (no portal logins): Sentinel-2 L2A COGs (AWS), ESA
WorldCover 2021 (AWS), Copernicus DEM (AWS), Hansen Global Forest Change v1.12 (Google Cloud
Storage), EFFIS burn-severity rasters 2018-2023 (EFFIS S3), Overture Maps land, land-use and
building layers (OpenStreetMap-derived, AWS), NOAA GHCN-Daily stations (AWS) and Natural Earth
boundaries. Everything is cached under data/ so a rerun reads locally.
"""
