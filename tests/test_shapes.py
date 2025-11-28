import sys
import os
import numpy as np
from src.fdtd2d import Grid2D


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
