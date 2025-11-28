import os
import sys
import numpy as np
from pathlib import Path

# Ensure project root in sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from src.shape import load_shape_json, radial_polygon, validate_polygon
from src.fdtd2d import Grid2D


def test_load_and_apply_shape_jsons():
    shapes_dir = Path(__file__).resolve().parent.parent / 'examples' / 'shapes_json'
    assert shapes_dir.exists()
    files = list(shapes_dir.glob('*.json'))
    assert len(files) > 0
    for f in files:
        center, radii, metadata, verts = load_shape_json(str(f))
        # either vertices or radii/center should exist
        assert (verts is not None) or (center is not None and radii is not None)
        if verts is None:
            verts = radial_polygon(center, radii)
        valid, msg = validate_polygon(verts, min_area=1.0, max_aspect=20.0, min_edge=1.0)
        assert valid, f'{f.name} invalid: {msg}'
        grid = Grid2D(nx=100, ny=120, dx=1.0, c=1.0)
        grid.add_polygon_obstacle(verts, value=0.0)
        # ensure something is marked as obstacle
        assert np.count_nonzero(grid.obstacle) > 0
