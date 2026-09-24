"""Spatial block cross-validation splitters."""


class SpatialBlockKFold:
    """K-fold over spatial blocks so test cells are not neighbours of train cells."""
