"""
Run a set of shape JSONs and compute a simple backscatter metric per shape.

Usage:
    python examples/run_shape_jsons.py

This script loads all JSON files from `examples/shapes_json/`, creates a Grid2D
with each shape, runs a short FDTD simulation, computes a metric at a fixed
receiver and prints a summary. It's a small driver for regression/visual
inspection and demonstration of using the JSON shapes across the codebase.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root on sys.path so `src` module is importable
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from src.shape import load_shape_json, radial_polygon
from src.fdtd2d import Grid2D, FDWave2D, GaussianPulseSource, Receiver

SHAPES_DIR = Path(__file__).resolve().parent / 'shapes_json'
OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(exist_ok=True)


def apply_shape_to_grid(grid: Grid2D, center, radii, vertices):
    """Given center/radii/vertices data loaded from JSON, apply as polygon.

    The JSON may provide either `vertices` explicitly, or it may provide `center` + `radii`.
    We convert to polygon vertices as needed then apply to the grid as an obstacle.
    """
    if vertices is None:
        # convert radial to vertices
        verts = radial_polygon(tuple(center), radii)
    else:
        verts = [tuple(v) for v in vertices]
    grid.add_polygon_obstacle(verts, value=0.0)
    return verts


def simulate_shape_json(path, nx=140, ny=200):
    center, radii, metadata, vertices = load_shape_json(str(path))
    grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
    # if nothing at all provided, load defaults
    if center is None and radii is None and vertices is None:
        raise RuntimeError('Shape JSON contains no recognized geometry fields')
    verts = apply_shape_to_grid(grid, center, radii, vertices)

    # simple check: source/receiver positions as in examples
    source_ix, source_iy = 20, ny // 2
    rec_ix, rec_iy = 24, ny // 2
    if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
        print(f'{path.name}: source or receiver inside obstacle; skipping')
        return None, None, verts

    grid.set_damping_layer(pad=20, amplitude=0.04)
    source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
    fd = FDWave2D(grid, dt=0.4, source=source)
    rec = Receiver(ixs=[rec_ix], iys=[rec_iy])
    steps = 200
    for t in range(steps):
        fd.step()
        rec.capture(grid)
    from examples.optimize_shape import compute_metric
    m = compute_metric(rec.get_signal(), metric='max')
    return m, grid, verts


if __name__ == '__main__':
    shapes = list(SHAPES_DIR.glob('*.json'))
    results = []
    for s in shapes:
        print('Running', s.name)
        try:
            m, grid, verts = simulate_shape_json(s, nx=140, ny=200)
            if m is not None:
                print(f'{s.name}: metric={m:.4f}')
                results.append((s.name, m))
                # save mask for manual inspection
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(grid.obstacle.astype(int), cmap='gray')
                plt.title(s.name)
                plt.axis('off')
                plt.savefig(OUT_DIR / f'{s.stem}_mask.png')
            else:
                print(f'{s.name}: skipped')
        except Exception as e:
            print('Error running', s.name, ':', e)

    print('\nSummary:')
    for name, m in results:
        print(f'{name}: {m:.4f}')

    # Save summary CSV
    import csv
    csv_path = OUT_DIR / 'shape_json_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'metric'])
        for name, m in results:
            w.writerow([name, m])
    print('Saved summary to', csv_path)
