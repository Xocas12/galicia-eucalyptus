"""Raster/table export of grid results (GeoTIFF needs the optional `geo` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import Grid


def write_geotiff(arr: np.ndarray, grid: Grid, path: str | Path, nodata: float = -9999.0) -> Path:
    """Write a (ny, nx) or (bands, ny, nx) array as a float32 GeoTIFF in the grid's CRS."""
    import rasterio
    from rasterio.transform import from_origin

    a = np.asarray(arr, dtype="float32")
    if a.ndim == 2:
        a = a[None]
    a = np.where(np.isfinite(a), a, nodata)
    path = Path(path)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=a.shape[1],
        width=a.shape[2],
        count=a.shape[0],
        dtype="float32",
        crs=grid.crs,
        nodata=nodata,
        compress="deflate",
        transform=from_origin(grid.xmin, grid.ymax, grid.resolution_m, grid.resolution_m),
    ) as dst:
        dst.write(a)
    return path
