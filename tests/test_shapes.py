import sys
import os
import numpy as np
from src.fdtd2d import Grid2D
from src.shape import radial_polygon, save_shape_json, load_shape_json, validate_polygon
import tempfile
import os


def test_add_ellipse_polygon_rotated_rect():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    grid = Grid2D(nx=100, ny=120, dx=1.0, c=1.0)
    # initial obstacles none
    assert np.count_nonzero(grid.obstacle) == 0
    grid.add_ellipse_obstacle(cx=40, cy=50, rx=8, ry=15, angle=15, value=0.0)
    grid.add_polygon_obstacle(vertices=[(20, 10), (25, 40), (50, 20)], value=0.0)
    grid.add_rotated_rect_obstacle(cx=70, cy=80, width=18, height=10, angle=30, value=0.0)
    # check obstacles were set in mask
    assert np.count_nonzero(grid.obstacle) > 0
    # ensure values 0 correspond
    assert (grid.rho[grid.obstacle] == 0.0).all()


def test_save_and_load_shape_json():
    center = (30, 40)
    radii = [10, 12, 11, 9, 10, 12]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'shape.json')
        save_shape_json(p, center=center, radii=radii, metadata={'test': True})
        c, r, meta, verts = load_shape_json(p)
        assert tuple(c) == tuple(center)
        assert list(r) == list(radii)
        assert meta.get('test', False) is True
        # vertices optional, but should be present
        assert verts is not None
        # polygon validation
        valid, msg = validate_polygon(verts, min_area=5.0, max_aspect=10.0, min_edge=1.0)
        assert valid, msg
        # apply to grid
        grid = Grid2D(nx=100, ny=100, dx=1.0, c=1.0)
        grid.add_polygon_obstacle(verts, value=0.0)
        # some obstacle must exist
        assert np.count_nonzero(grid.obstacle) > 0
