import numpy as np

from eucalyptus_impact.geo.grid import Grid, aggregate, block_ids
from eucalyptus_impact.geo.raster_ops import distance_to, focal_mean, zonal_mean
from eucalyptus_impact.validation.spatial_cv import SpatialBlockKFold


def test_grid_shape_and_blocks():
    g = Grid.from_bbox((0, 0, 100_000, 50_000), 1000)
    assert g.shape == (50, 100)
    assert g.cell_area_ha == 100
    b = block_ids(g, 10)
    assert b.shape == g.shape
    assert len(np.unique(b)) == 5 * 10
    assert g.index_of(500, 49_500) == (0, 0)


def test_aggregate_and_focal():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert aggregate(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    ones = np.ones((5, 5))
    assert np.allclose(focal_mean(ones, 1), 1.0)


def test_distance_and_zonal():
    m = np.zeros((3, 3), bool)
    m[0, 0] = True
    d = distance_to(m, 10)
    assert d[0, 0] == 0 and np.isclose(d[2, 2], np.hypot(20, 20))
    z = np.array([0, 0, 1, -1])
    assert zonal_mean(np.array([1.0, 3.0, 5.0, 100.0]), z, 2).tolist() == [2.0, 5.0]


def test_spatial_block_kfold_separates_blocks():
    groups = np.repeat(np.arange(20), 7)
    cv = SpatialBlockKFold(4, seed=1)
    seen = []
    for tr, te in cv.split(groups=groups):
        assert not set(groups[tr]) & set(groups[te])
        seen.append(te)
    assert sorted(np.concatenate(seen).tolist()) == list(range(len(groups)))
