"""Study configuration: grid, years, model and scenario settings loaded from YAML."""

from dataclasses import dataclass


@dataclass
class StudyConfig:
    """Top-level configuration (see configs/default.yaml)."""


def load_config(path):
    raise NotImplementedError("M1")
