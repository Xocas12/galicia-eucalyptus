"""Raster neighbourhood operations: focal means, distance transforms, zonal statistics."""


def focal_mean(arr, radius, mask=None):
    raise NotImplementedError("M1")


def distance_to(mask, resolution_m):
    raise NotImplementedError("M1")


def zonal_mean(values, zones, n_zones):
    raise NotImplementedError("M1")
