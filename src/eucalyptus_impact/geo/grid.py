"""Analysis grid in ETRS89 / UTM 29N, spatial blocks and aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CRS = "EPSG:25829"  # ETRS89 / UTM zone 29N, the official projection for Galicia.


@dataclass(frozen=True)
class Grid:
    """Regular grid over the study bounding box. Row 0 is the northern edge."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    resolution_m: float
    crs: str = CRS

    @classmethod
    def from_bbox(cls, bbox, resolution_m: float) -> Grid:
        return cls(*map(float, bbox), resolution_m=float(resolution_m))

    @property
    def shape(self) -> tuple[int, int]:
        ny = round((self.ymax - self.ymin) / self.resolution_m)
        nx = round((self.xmax - self.xmin) / self.resolution_m)
        return ny, nx

    @property
    def cell_area_ha(self) -> float:
        return self.resolution_m**2 / 1e4

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates (x, y) as 2-D arrays."""
        ny, nx = self.shape
        xs = self.xmin + (np.arange(nx) + 0.5) * self.resolution_m
        ys = self.ymax - (np.arange(ny) + 0.5) * self.resolution_m
        return np.meshgrid(xs, ys)

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.xmin) // self.resolution_m)
        row = int((self.ymax - y) // self.resolution_m)
        return row, col


def block_ids(grid: Grid, block_size_km: float) -> np.ndarray:
    """Integer id of the square spatial block (for CV and clustered SEs) each cell falls in."""
    ny, nx = grid.shape
    k = max(1, round(block_size_km * 1000 / grid.resolution_m))
    rows = np.arange(ny)[:, None] // k
    cols = np.arange(nx)[None, :] // k
    n_col_blocks = -(-nx // k)
    return (rows * n_col_blocks + cols).astype(np.int64)


def aggregate(arr: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    """Block-aggregate a 2-D array by an integer factor (edges trimmed), ignoring NaNs."""
    ny, nx = arr.shape
    ny2, nx2 = ny // factor, nx // factor
    a = arr[: ny2 * factor, : nx2 * factor].reshape(ny2, factor, nx2, factor)
    fn = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}[how]
    return fn(a, axis=(1, 3))
