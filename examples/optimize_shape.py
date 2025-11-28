"""
Optimization demo (pedagogical): sweep simple parametric shapes to find which
one minimizes the backscatter measured by a receiver.

NOTES:
- This uses the 2D scalar wave FDTD solver in `src/fdtd2d.py` and is purely
  educational. Real-world RCS/stealth design involves Maxwell vector fields,
  detailed material models and engineering constraints beyond the scope here.

Usage:
    python examples/optimize_shape.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

# Ensure project root is importable
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.fdtd2d import Grid2D, FDWave2D, GaussianPulseSource, Receiver
from src.shape import radial_polygon, save_shape_json, load_shape_json, validate_polygon

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Simulation helper
def simulate_with(grid: Grid2D, dt: float, steps: int, source: GaussianPulseSource, rec_coords):
    fd = FDWave2D(grid, dt=dt, source=source)
    rec = Receiver(ixs=[rec_coords[0]], iys=[rec_coords[1]])
    for t in range(steps):
        fd.step()
        rec.capture(grid)
    return rec.get_signal(), grid


def compute_metric(signal: np.ndarray, metric='max'):
    # metric options: 'max' => max abs, 'energy' => sum of squares
    if metric == 'max':
        return float(np.max(np.abs(signal)))
    elif metric == 'energy':
        return float(np.sum(signal**2))
    else:
        raise ValueError('Unknown metric: ' + metric)


def run_sweep_circle(nx=160, ny=220):
    """Sweep circle radius and return the best radius with metric values.
    The source/receiver positions are fixed so we optimize the shape only.
    """
    dt = 0.4
    steps = 300
    source_ix, source_iy = 20, ny // 2
    rec_ix, rec_iy = 24, ny // 2

    radii = list(range(5, 61, 5))
    metrics = []

    for r in radii:
        grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
        grid.add_circular_obstacle(cx=nx // 2, cy=ny // 2, radius=r, value=0.0)
        # skip if receiver or source is inside obstacle
        if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
            print('radius', r, 'skipped: source or receiver inside obstacle')
            metrics.append(np.inf)
            continue
        grid.set_damping_layer(pad=20, amplitude=0.04)
        source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
        signal, _ = simulate_with(grid, dt, steps, source, (rec_ix, rec_iy))
        m = compute_metric(signal, metric='max')
        metrics.append(m)
        print('radius', r, 'metric', m)

    # select best valid radius
    valid_indices = [i for i, v in enumerate(metrics) if np.isfinite(v)]
    if not valid_indices:
        print('No valid radii (all skipped)')
        return None, radii, metrics
    best_idx = valid_indices[int(np.argmin([metrics[i] for i in valid_indices]))]
    best_r = radii[best_idx]
    print('\nBest circle radius (min max-reflection):', best_r)

    # Re-run and save outputs for best
    grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
    grid.add_circular_obstacle(cx=nx // 2, cy=ny // 2, radius=best_r, value=0.0)
    grid.set_damping_layer(pad=20, amplitude=0.04)
    source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
    rec = Receiver(ixs=[rec_ix], iys=[rec_iy])
    frames = []
    fd = FDWave2D(grid, dt=dt, source=source)
    for t in range(steps):
        fd.step()
        rec.capture(grid)
        if t % 3 == 0:
            frame = (grid.u - grid.u.min()) / (grid.u.max() - grid.u.min() + 1e-9)
            img = (plt.cm.seismic(frame)[:, :, :3] * 255).astype(np.uint8)
            frames.append(img)

    gif_path = OUT_DIR / f'opt_circle_best_r_{best_r}.gif'
    imageio.mimsave(gif_path, frames, fps=20)
    print('Saved best circle animation:', gif_path)

    # save static image of best obstacle mask
    plt.figure()
    plt.imshow(grid.obstacle.astype(int), cmap='gray')
    plt.title(f'Best circle obstacle (r={best_r})')
    plt.axis('off')
    plt.savefig(OUT_DIR / f'opt_circle_best_mask_r_{best_r}.png')
    print('Saved best circle obstacle mask')

    plt.figure()
    plt.plot(radii, metrics, '-o')
    plt.xlabel('Circle radius (grid points)')
    plt.ylabel('Metric (max abs at receiver)')
    plt.title('Circle sweep — metric vs radius')
    plt.grid(True)
    plot_path = OUT_DIR / 'opt_circle_sweep.png'
    plt.savefig(plot_path)
    print('Saved circle sweep plot:', plot_path)

    return best_r, radii, metrics


def run_sweep_rotrect(nx=160, ny=220):
    dt = 0.4
    steps = 300
    source_ix, source_iy = 20, ny//2
    rec_ix, rec_iy = 24, ny//2

    angles = list(range(0, 180, 15))
    metrics = []

    for ang in angles:
        grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
        grid.add_rotated_rect_obstacle(cx=nx//2, cy=ny//2, width=70, height=12, angle=ang, value=0.0)
        if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
            print('angle', ang, 'skipped: source or receiver inside obstacle')
            metrics.append(np.inf)
            continue
        grid.set_damping_layer(pad=20, amplitude=0.04)
        source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=12)
        signal, _ = simulate_with(grid, dt, steps, source, (rec_ix, rec_iy))
        m = compute_metric(signal, metric='max')
        metrics.append(m)
        print('angle', ang, 'metric', m)

    best_idx = int(np.argmin(metrics))
    best_angle = angles[best_idx]
    print('\nBest angle (min max-reflection):', best_angle)

    # Save plot
    plt.figure()
    plt.plot(angles, metrics, '-o')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Metric (max abs at receiver)')
    plt.title('Rotated rect sweep — metric vs angle')
    plt.grid(True)
    plot_path = OUT_DIR / 'opt_rotrect_sweep.png'
    plt.savefig(plot_path)
    print('Saved rotated rect sweep plot:', plot_path)

    return best_angle, angles, metrics


def run_sweep_ellipse(nx=160, ny=220):
    dt = 0.4
    steps = 300
    source_ix, source_iy = 20, ny//2
    rec_ix, rec_iy = 24, ny//2

    rx_values = [10, 15, 20, 25]
    ry_values = [5, 8, 12, 18]

    best = None
    all_configs = []

    for rx in rx_values:
        for ry in ry_values:
            grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
            grid.add_ellipse_obstacle(cx=nx//2, cy=ny//2, rx=rx, ry=ry, angle=15, value=0.0)
            grid.set_damping_layer(pad=20, amplitude=0.04)
            if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
                print('ellipse rx,ry', (rx, ry), 'skipped: source or receiver inside obstacle')
                all_configs.append(((rx, ry), np.inf))
                continue
            source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
            signal, _ = simulate_with(grid, dt, steps, source, (rec_ix, rec_iy))
            m = compute_metric(signal, metric='max')
            all_configs.append(((rx, ry), m))
            print('ellipse rx,ry', (rx, ry), 'metric', m)
            if best is None or m < best[0]:
                best = (m, rx, ry)

    print('\nBest ellipse (min max-reflection): rx, ry =', (best[1], best[2]), 'metric', best[0])
    # Save a plot of all metrics as a heatmap
    res_matrix = np.zeros((len(rx_values), len(ry_values)))
    for i, rx in enumerate(rx_values):
        for j, ry in enumerate(ry_values):
            for (rxy, val) in all_configs:
                if rxy == (rx, ry):
                    res_matrix[i, j] = val
    plt.figure()
    plt.imshow(res_matrix, origin='lower', cmap='viridis', aspect='auto', extent=(min(ry_values), max(ry_values), min(rx_values), max(rx_values)))
    plt.colorbar(label='Metric (max abs)')
    plt.xlabel('ry')
    plt.ylabel('rx')
    plt.title('Ellipse sweep - metric heatmap')
    plot_path = OUT_DIR / 'opt_ellipse_heatmap.png'
    plt.savefig(plot_path)
    print('Saved ellipse heatmap:', plot_path)

    return best, rx_values, ry_values, res_matrix


def run_optimize_shape_json(nx=160, ny=220, nspokes=12, iterations=200, step_scale=3.0, seed=None, save_best: bool = False):
    """Optimize a radial polygon (center + radii) by random local search.

    - nspokes: number of radial spokes (resolution of the shape)
    - iterations: number of iterations for the search
    - step_scale: typical amplitude (grid units) for random perturbations
    - save_best: if True save the best shape JSON in outputs/
    Returns: (best_metric, best_center, best_radii, history)
    """
    import random
    dt = 0.4
    steps = 300
    source_ix, source_iy = 20, ny//2
    rec_ix, rec_iy = 24, ny//2

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cx, cy = nx // 2, ny // 2
    # initial radii: uniform moderate radius (avoid touching source/rec)
    base_r = min(nx, ny) // 6
    radii = [base_r for _ in range(nspokes)]

    def make_grid_from_radii(radii_list):
        grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
        verts = radial_polygon((cx, cy), radii_list, start_angle=0.0)
        # validate polygon and skip if invalid
        valid, msg = validate_polygon(verts, min_area=8.0, max_aspect=12.0, min_edge=1.0)
        if not valid:
            return None, None, msg
        grid.add_polygon_obstacle(verts, value=0.0)
        grid.set_damping_layer(pad=20, amplitude=0.04)
        return grid, verts, msg

    # evaluation metric for radii
    def eval_radii(radii_list):
        grid, verts, msg = make_grid_from_radii(radii_list)
        if grid is None:
            return np.inf, None
        # don't allow source or receiver inside obstacle
        if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
            return np.inf, None
        source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
        signal, _ = simulate_with(grid, dt, steps, source, (rec_ix, rec_iy))
        m = compute_metric(signal, metric='max')
        return m, grid

    history = []
    best_radii = radii.copy()
    best_metric, _ = eval_radii(best_radii)
    print('Init metric', best_metric)
    history.append(best_metric)

    for it in range(iterations):
        # random gaussian perturbation around the current best
        cand = [max(1.0, r + np.random.normal(0, step_scale)) for r in best_radii]
        # limit radii to reasonable range (1 to min(nx,ny)//2 - margin)
        max_r = min(nx, ny) // 2 - 2
        cand = [min(max_r, max(1.0, rr)) for rr in cand]
        m, _ = eval_radii(cand)
        history.append(m)
        if m < best_metric:
            best_metric = m
            best_radii = cand
            print(f'it {it} new best metric: {best_metric:.3f}')

    # Save best shape json and visualization
    if save_best:
        path = OUT_DIR / f'opt_shape_best_{nspokes}_spokes.json'
        save_shape_json(str(path), center=(cx, cy), radii=best_radii, metadata={'metric': float(best_metric)}, vertices=verts)
        print('Saved best shape JSON:', path)
        # also save mask
        grid, verts, _ = make_grid_from_radii(best_radii)
        if grid is not None:
            plt.figure()
            plt.imshow(grid.obstacle.astype(int), cmap='gray')
            plt.title('Best optimized obstacle')
            plt.axis('off')
            img_path = OUT_DIR / f'opt_shape_best_mask_{nspokes}.png'
            plt.savefig(img_path)
            print('Saved best mask:', img_path)

    return best_metric, (cx, cy), best_radii, history


def run_optimize_shape_json_file(json_path: str, nx=160, ny=220, iterations=200, step_scale=3.0, seed=None, save_best: bool = True, save_format: str = 'auto', out_dir: Path | None = None):
    """Load a shape JSON and optimize it.

    - If the JSON provides `radii`, we optimize radii (radial parametric).
    - If the JSON provides `vertices`, we optimize vertex positions.
    - save_format: 'auto' (keep input), 'radial', or 'vertices' — determines JSON format of output.
    - out_dir: directory to save resulting JSON (default: examples/outputs)

    Returns: path to saved JSON (or None if failed) and best metric.
    """
    import random
    if out_dir is None:
        out_dir = OUT_DIR
    out_dir.mkdir(exist_ok=True)

    center, radii, metadata, vertices = load_shape_json(str(json_path))
    # determine optimization mode
    mode = 'radial' if radii is not None else ('vertices' if vertices is not None else None)
    if mode is None:
        raise ValueError('JSON must contain either radii or vertices')

    if save_format == 'auto':
        save_format = mode

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # helpers
    def compute_centroid(verts):
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys))))

    # eval for radii representation
    def eval_radii(inner_center, rads):
        # transform to polygon and evaluate metric
        grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
        verts = radial_polygon(inner_center, rads)
        # validate
        valid, msg = validate_polygon(verts, min_area=5.0, max_aspect=20.0, min_edge=1.0)
        if not valid:
            return np.inf, None
        grid.add_polygon_obstacle(verts, value=0.0)
        if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
            return np.inf, None
        grid.set_damping_layer(pad=20, amplitude=0.04)
        source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
        signal, _ = simulate_with(grid, dt, steps, source, (rec_ix, rec_iy))
        return compute_metric(signal, metric='max'), grid

    # eval for vertices representation
    def eval_vertices(verts):
        grid = Grid2D(nx=nx, ny=ny, dx=1.0, c=1.0)
        valid, msg = validate_polygon(verts, min_area=5.0, max_aspect=20.0, min_edge=1.0)
        if not valid:
            return np.inf, None
        grid.add_polygon_obstacle(verts, value=0.0)
        if grid.obstacle[source_ix, source_iy] or grid.obstacle[rec_ix, rec_iy]:
            return np.inf, None
        grid.set_damping_layer(pad=20, amplitude=0.04)
        source = GaussianPulseSource(ix=source_ix, iy=source_iy, amplitude=3.0, freq=0.03, t0=50, width=10)
        signal, _ = simulate_with(grid, dt, steps, source, (rec_ix, rec_iy))
        return compute_metric(signal, metric='max'), grid

    # use the same sim params as previous helpers
    dt = 0.4
    steps = 300
    source_ix, source_iy = 20, ny // 2
    rec_ix, rec_iy = 24, ny // 2

    # radial-mode optimization
    if mode == 'radial':
        # ensure center available
        if center is None:
            raise ValueError('radial JSON must contain center')
        cx, cy = center
        best_radii = list(radii)
        best_metric, _ = eval_radii((cx, cy), best_radii)
        print('Initial metric:', best_metric)
        for it in range(iterations):
            cand = [max(1.0, r + np.random.normal(0, step_scale)) for r in best_radii]
            max_r = min(nx, ny) // 2 - 2
            cand = [min(max_r, rr) for rr in cand]
            m, _ = eval_radii((cx, cy), cand)
            if m < best_metric:
                best_metric = m
                best_radii = cand
                print(f'it {it} radial new best {best_metric:.3f}')

        # prepare output JSON options
        if save_format == 'radial':
            out_path = out_dir / f'optimized_{Path(json_path).stem}_radial.json'
            # compute vertices also for convenience
            verts = radial_polygon((cx, cy), best_radii)
            save_shape_json(str(out_path), center=(cx, cy), radii=best_radii, metadata={'metric': float(best_metric)}, vertices=verts)
        else:
            out_path = out_dir / f'optimized_{Path(json_path).stem}_vertices.json'
            verts = radial_polygon((cx, cy), best_radii)
            save_shape_json(str(out_path), center=(cx, cy), radii=best_radii, metadata={'metric': float(best_metric)}, vertices=verts)
        return str(out_path), best_metric

    # vertices-mode optimization
    else:
        # copy initial verts
        best_verts = [tuple(v) for v in vertices]
        # optionally compute centroid
        if center is None:
            c0 = compute_centroid(best_verts)
            cx, cy = c0
        else:
            cx, cy = center
        best_metric, _ = eval_vertices(best_verts)
        print('Initial metric:', best_metric)
        for it in range(iterations):
            cand = []
            for (vx, vy) in best_verts:
                dx = int(round(np.random.normal(0, step_scale)))
                dy = int(round(np.random.normal(0, step_scale)))
                nxv = max(1, min(nx - 2, vx + dx))
                nyv = max(1, min(ny - 2, vy + dy))
                cand.append((nxv, nyv))
            m, _ = eval_vertices(cand)
            if m < best_metric:
                best_metric = m
                best_verts = cand
                print(f'it {it} verts new best {best_metric:.3f}')

        # Output
        if save_format == 'vertices':
            out_path = out_dir / f'optimized_{Path(json_path).stem}_vertices.json'
            save_shape_json(str(out_path), center=(cx, cy), radii=[float(0) for _ in best_verts], metadata={'metric': float(best_metric)}, vertices=best_verts)
        else:
            # convert to radial representation using centroid
            radii_out = []
            for (vx, vy) in best_verts:
                r = float(np.hypot(vx - cx, vy - cy))
                radii_out.append(r)
            out_path = out_dir / f'optimized_{Path(json_path).stem}_radial.json'
            save_shape_json(str(out_path), center=(cx, cy), radii=radii_out, metadata={'metric': float(best_metric)}, vertices=best_verts)
        return str(out_path), best_metric


if __name__ == '__main__':
    print('Running circle sweep...')
    best_r, radii, metrics = run_sweep_circle(nx=140, ny=200)

    print('\nRunning rotated-rectangle sweep...')
    best_angle, angles, rect_metrics = run_sweep_rotrect(nx=140, ny=200)

    print('\nRunning ellipse sweep...')
    best_ellipse, rx_values, ry_values, res_matrix = run_sweep_ellipse(nx=140, ny=200)

    # Save best ellipse mask
    if best_ellipse is not None:
        _, brx, bry = best_ellipse
        grid = Grid2D(nx=140, ny=200, dx=1.0, c=1.0)
        grid.add_ellipse_obstacle(cx=140//2, cy=200//2, rx=brx, ry=bry, angle=15, value=0.0)
        plt.figure()
        plt.imshow(grid.obstacle.astype(int), cmap='gray')
        plt.title(f'Best ellipse (rx={brx}, ry={bry})')
        plt.axis('off')
        plt.savefig(OUT_DIR / 'opt_ellipse_best_mask.png')
        print('Saved ellipse best mask')

    print('\nDone. Optimizations are purely educational and use a scalar-wave FDTD model.')
    # Run JSON shape optimizer for demo
    print('\nRunning radial polygon optimizer demo (JSON output)...')
    best_metric, center, best_radii, history = run_optimize_shape_json(nx=140, ny=200, nspokes=12, iterations=120, seed=0, save_best=True)
    print('Best metric (radial JSON optimizer):', best_metric)
