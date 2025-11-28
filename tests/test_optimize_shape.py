from src.fdtd2d import Grid2D
from examples.optimize_shape import run_sweep_circle, run_sweep_rotrect, run_sweep_ellipse, run_optimize_shape_json
from src.shape import radial_polygon, validate_polygon
from examples.optimize_shape import run_optimize_shape_json_file
from pathlib import Path
import tempfile
import numpy as np


def test_circle_sweep_returns_best_and_keeps_source_out():
    best_r, radii, metrics = run_sweep_circle(nx=80, ny=100)
    # best_r may be None if all skipped
    if best_r is not None:
        grid = Grid2D(nx=80, ny=100, dx=1.0, c=1.0)
        grid.add_circular_obstacle(cx=40, cy=50, radius=best_r, value=0.0)
        # Source/receiver positions are hard-coded in script as (20, ny//2) and (24, ny//2)
        assert not grid.obstacle[20, 50]
        assert not grid.obstacle[24, 50]


def test_rotated_rect_sweep_returns_best_angle():
    best_angle, angles, metrics = run_sweep_rotrect(nx=80, ny=100)
    assert best_angle in angles


def test_ellipse_sweep_returns_best_and_keeps_source_out():
    best, rx_values, ry_values, res_matrix = run_sweep_ellipse(nx=80, ny=100)
    m, rx, ry = best
    grid = Grid2D(nx=80, ny=100, dx=1.0, c=1.0)
    grid.add_ellipse_obstacle(cx=40, cy=50, rx=rx, ry=ry, angle=15, value=0.0)
    assert not grid.obstacle[20, 50]
    assert not grid.obstacle[24, 50]


def test_optimize_shape_json_finds_valid_shape():
    # small run for CI-speed
    best_metric, center, best_radii, history = run_optimize_shape_json(nx=60, ny=80, nspokes=8, iterations=30, seed=42, save_best=False)
    # Ensure we got a metric and it's finite
    assert np.isfinite(best_metric)
    # convert to polygon and validate
    cx, cy = center
    verts = radial_polygon((cx, cy), best_radii)
    valid, msg = validate_polygon(verts, min_area=5.0, max_aspect=15.0, min_edge=1.0)
    assert valid, f'Shape validation failed: {msg}'
    # ensure source and receiver positions aren't inside obstacle
    grid = Grid2D(nx=60, ny=80, dx=1.0, c=1.0)
    grid.add_polygon_obstacle(verts, value=0.0)
    assert not grid.obstacle[20, 80//2]
    assert not grid.obstacle[24, 80//2]


def test_optimize_shape_json_file_works():
    # Use the circle.json from examples/shapes_json
    shapes_dir = Path(__file__).resolve().parent.parent / 'examples' / 'shapes_json'
    json_path = shapes_dir / 'circle.json'
    assert json_path.exists()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        out_path, best_metric = run_optimize_shape_json_file(str(json_path), nx=60, ny=80, iterations=30, step_scale=2.0, seed=123, save_best=True, save_format='auto', out_dir=out_dir)
        assert out_path is not None
        saved = Path(out_path)
        assert saved.exists()
        # load and validate
        from src.shape import load_shape_json
        c, r, meta, verts = load_shape_json(str(saved))
        assert (verts is not None) or (r is not None)
        # metric present
        assert 'metric' in meta
