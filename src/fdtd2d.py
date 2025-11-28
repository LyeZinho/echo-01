"""
Minimal, pedagogical 2D scalar wave (FDTD) solver in Python.

This code is intentionally simple and educational: it solves the 2D scalar wave
equation for a homogeneous medium with obstacles (as regions with different
velocity or density) using a 2nd-order finite-difference time-domain (FDTD)
leapfrog scheme. Designed to illustrate wave propagation, reflection and
diffraction qualitatively.

NOTES:
- This is NOT a full EM solver. It uses scalar waves (like acoustics) to
  illustrate scattering ideas.
- Boundaries: simple absorbing boundary via a damping layer ("sponge") is used.
- For far-field / backscatter qualitative response we measure reflection at a
  small receiver region and compute FFT.

Author: Copilot (Raptor mini Preview)
"""

import numpy as np

class Grid2D:
    def __init__(self, nx, ny, dx=1.0, dy=None, c=1.0):
        """Create a 2D grid for FDTD wave simulation.

        nx, ny: grid points in x/y
        dx, dy: grid spacing (meters) — if dy None set equal to dx
        c: wave speed (m/s) — default 1 for non-dimensional runs
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx if dy is None else dy
        self.c = c
        self.u = np.zeros((nx, ny), dtype=float)  # wavefield at time n
        self.u_prev = np.zeros_like(self.u)       # at time n-1
        self.u_next = np.zeros_like(self.u)       # at time n+1
        self.rho = np.ones_like(self.u)           # material parameter (for inhomog.)
        # damping sponge (simple 1D along edges)
        self.damping = np.zeros_like(self.u)
        # obstacle mask (True=obstacle where field is zero)
        self.obstacle = np.zeros_like(self.u, dtype=bool)

    def add_circular_obstacle(self, cx, cy, radius, value=0.0):
        """Add obstacle as region where wavespeed is different (value of c).
        Here `value` is a local wave speed.
        Coordinates in grid points (indices), not physical units.
        """
        X, Y = np.meshgrid(np.arange(self.ny), np.arange(self.nx))
        dist2 = (X - cy)**2 + (Y - cx)**2
        mask = dist2 <= radius**2
        self.rho[mask] = value
        # also mark as obstacle if value==0
        if value == 0.0:
            self.obstacle[mask] = True

    def add_rect_obstacle(self, x0, y0, x1, y1, value=0.0):
        """Add rectangular obstacle by grid indices"""
        ix0, iy0 = x0, y0
        ix1, iy1 = x1, y1
        self.rho[ix0:ix1, iy0:iy1] = value
        if value == 0.0:
            self.obstacle[ix0:ix1, iy0:iy1] = True

    def add_ellipse_obstacle(self, cx, cy, rx, ry, angle=0.0, value=0.0):
        """Add an ellipse obstacle centered at (cx, cy) with radii rx, ry.
        Angle is in degrees, counter-clockwise.
        Coordinates are grid indices.
        """
        # Convert angle to radians and precompute sin/cos
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        X, Y = np.meshgrid(np.arange(self.ny), np.arange(self.nx))
        # Translate
        x = X - cy
        y = Y - cx
        # Rotate coordinates by -theta to align with ellipse axes
        xr = cos_t * x + sin_t * y
        yr = -sin_t * x + cos_t * y
        mask = (xr / rx)**2 + (yr / ry)**2 <= 1.0
        self.rho[mask] = value
        if value == 0.0:
            self.obstacle[mask] = True

    def add_polygon_obstacle(self, vertices, value=0.0):
        """Add a polygonal obstacle given by a list of vertices
        `vertices` are [(ix0, iy0), (ix1, iy1), ...] in grid indices.
        """
        try:
            from matplotlib.path import Path
        except Exception:
            raise RuntimeError("matplotlib is required for polygon point-in-polygon tests")

        # Build path in (x,y) corresponding to (row, col) = (ix, iy) coordinates
        # But Path expects (x,y) in the usual cartesian order. We'll use (iy, ix) as x,y.
        xy = [(iy, ix) for (ix, iy) in vertices]
        path = Path(xy)
        X, Y = np.meshgrid(np.arange(self.ny), np.arange(self.nx))
        pts = np.vstack((X.ravel(), Y.ravel())).T  # shape (N, 2) with columns (x=col, y=row)
        mask_flat = path.contains_points(pts)
        mask = mask_flat.reshape((self.nx, self.ny))
        self.rho[mask] = value
        if value == 0.0:
            self.obstacle[mask] = True

    def add_rotated_rect_obstacle(self, cx, cy, width, height, angle=0.0, value=0.0):
        """Add a rotated rectangle centered at (cx, cy); width and height in grid points.
        Angle is in degrees (CCW)."""
        theta = np.deg2rad(angle)
        w2 = width / 2.0
        h2 = height / 2.0
        # Rectangle corners in local coords
        corners_local = [(-h2, -w2), (-h2, w2), (h2, w2), (h2, -w2)]
        # Rotate and translate to grid coordinates (ix, iy)
        corners = []
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for (dx, dy) in corners_local:
            rx = cos_t * dx - sin_t * dy
            ry = sin_t * dx + cos_t * dy
            # rx, ry correspond to (row offset, col offset) = (ix offset, iy offset)
            corners.append((int(round(cx + rx)), int(round(cy + ry))))
        self.add_polygon_obstacle(corners, value=value)

    def set_damping_layer(self, pad, amplitude=0.02):
        """Simple linear damping in border layers to absorb waves.
        pad: thickness in grid points
        amplitude: damping factor multiplier
        """
        nx, ny = self.nx, self.ny
        d = np.zeros_like(self.damping)
        for i in range(nx):
            for j in range(ny):
                dist = min(i, j, nx - 1 - i, ny - 1 - j)
                if dist < pad:
                    d[i, j] = amplitude * (1 - dist / pad)
        self.damping = d


class FDWave2D:
    """2D FDTD wave solver (scalar) — central differences, leapfrog in time."""

    def __init__(self, grid: Grid2D, dt, source=None):
        self.grid = grid
        self.dt = dt
        self.time = 0.0
        self.source = source
        # CFL stability check
        cmax = grid.c
        stability_limit = 1.0 / np.sqrt((1.0 / grid.dx**2) + (1.0 / grid.dy**2))
        if dt * cmax >= stability_limit:
            print("Warning: dt may violate CFL condition for stability.")

    def step(self):
        g = self.grid
        nx, ny = g.nx, g.ny
        u = g.u
        u_prev = g.u_prev
        u_next = g.u_next
        c = g.c
        dx = g.dx
        dy = g.dy

        # 5-point Laplacian (with proper dx/dy)
        lap_x = (np.roll(u, 1, axis=0) - 2 * u + np.roll(u, -1, axis=0)) / (dx*dx)
        lap_y = (np.roll(u, 1, axis=1) - 2 * u + np.roll(u, -1, axis=1)) / (dy*dy)
        lap = lap_x + lap_y

        # Update: u_next = 2u - u_prev + (c*dt)^2 * Laplacian(u)
        factor = (c * self.dt)**2
        u_next[:] = 2 * u - u_prev + factor * lap

        # Damping (simple sponge)
        u_next *= (1.0 - g.damping)

        # Apply sources
        if self.source is not None:
            self.source.apply(u_next, self.time)

        # enforce obstacle Dirichlet condition (fields zero inside obstacle)
        if np.any(g.obstacle):
            u_next[g.obstacle] = 0.0

        # step forward
        g.u_prev[:], g.u[:] = g.u.copy(), u_next.copy()
        self.time += self.dt

    def run(self, steps, callback=None):
        """Run simulation for given number of time steps.
        callback(grid, tstep) called optionally every iteration.
        """
        for t in range(steps):
            self.step()
            if callback is not None:
                callback(self.grid, t)


class GaussianPulseSource:
    """A point source with a Gaussian-modulated sine or single pulse.

    The source adds amplitude to a grid point (or small region) at each step.
    """
    def __init__(self, ix, iy, amplitude=1.0, freq=0.05, t0=20.0, width=6.0):
        self.ix = ix
        self.iy = iy
        self.amp = amplitude
        self.freq = freq
        self.t0 = t0
        self.width = width

    def apply(self, field, time):
        t = time
        val = self.amp * np.exp(-((t - self.t0) / self.width)**2) * np.sin(2*np.pi*self.freq*t)
        field[self.ix, self.iy] += val


class PlaneWaveSource:
    """Implements a plane wave by forcing a band of grid points (left edge) to a
    prescribed time dependency approximating a continuous plane wave.
    """
    def __init__(self, iy_range, amplitude=1.0, freq=0.05, phase=0.0):
        self.iy_range = iy_range
        self.amp = amplitude
        self.freq = freq
        self.phase = phase

    def apply(self, field, time):
        t = time
        val = self.amp * np.sin(2*np.pi*self.freq*t + self.phase)
        for iy in self.iy_range:
            field[:, iy] += val


# Simple util: measure time signal at a set of grid points
class Receiver:
    def __init__(self, ixs, iys):
        self.ixs = np.array(ixs)
        self.iys = np.array(iys)
        self.record = []

    def capture(self, grid):
        vals = grid.u[self.ixs, self.iys]
        self.record.append(np.mean(vals))

    def get_signal(self):
        return np.array(self.record)


if __name__ == "__main__":
    # Quick demo runner for dev/testing
    import matplotlib.pyplot as plt
    grid = Grid2D(nx=150, ny=200, dx=1.0, c=1.0)
    grid.add_circular_obstacle(cx=70, cy=100, radius=15, value=0.0)
    grid.set_damping_layer(pad=15, amplitude=0.02)
    source = GaussianPulseSource(ix=20, iy=100, amplitude=2.0, freq=0.02, t0=30, width=8)
    fd = FDWave2D(grid, dt=0.5, source=source)

    rec = Receiver(ixs=[10], iys=[100])
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    im = ax.imshow(grid.u, cmap="RdBu", vmin=-0.5, vmax=0.5)

    def animate(t):
        fd.step()
        rec.capture(grid)
        im.set_data(grid.u)
        return [im]

    ani = animation.FuncAnimation(fig, animate, frames=300, interval=30, blit=True)
    plt.show()
