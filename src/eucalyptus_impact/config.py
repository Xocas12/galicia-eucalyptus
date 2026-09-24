"""Study configuration: grid, years, model and scenario settings loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GridConfig:
    resolution_m: float = 1000.0
    # ETRS89 / UTM 29N (EPSG:25829): xmin, ymin, xmax, ymax.
    bbox_utm29n: tuple[float, float, float, float] = (470000, 4625000, 690000, 4850000)
    block_size_km: float = 20.0


@dataclass
class CausalConfig:
    n_folds: int = 5
    # Cell-year rows passed to DML; larger panels are subsampled for speed.
    max_rows: int = 250_000
    # SIMEX map-error correction for the soil-moisture and severity effects (re-fits DML ~9 times).
    simex: bool = True


@dataclass
class LandcoverConfig:
    n_train_pixels: int = 12_000
    n_map_pixels: int = 60_000
    reference_per_class: int = 150


@dataclass
class ScenarioConfig:
    horizon: int = 2040
    restoration_share: float = 0.25


@dataclass
class StudyConfig:
    """Top-level configuration (see configs/default.yaml)."""

    seed: int = 0
    grid: GridConfig = field(default_factory=GridConfig)
    years: tuple[int, int] = (2000, 2024)
    n_catchments: int = 60
    causal: CausalConfig = field(default_factory=CausalConfig)
    landcover: LandcoverConfig = field(default_factory=LandcoverConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    output_dir: str = "outputs/default"

    @property
    def year_list(self) -> list[int]:
        return list(range(self.years[0], self.years[1] + 1))


def config_from_dict(d: dict) -> StudyConfig:
    d = dict(d)
    grid = GridConfig(**d.pop("grid", {}))
    grid.bbox_utm29n = tuple(grid.bbox_utm29n)
    return StudyConfig(
        grid=grid,
        causal=CausalConfig(**d.pop("causal", {})),
        landcover=LandcoverConfig(**d.pop("landcover", {})),
        scenarios=ScenarioConfig(**d.pop("scenarios", {})),
        years=tuple(d.pop("years", (2000, 2024))),
        **d,
    )


def load_config(path: str | Path) -> StudyConfig:
    with open(path) as fh:
        return config_from_dict(yaml.safe_load(fh) or {})
