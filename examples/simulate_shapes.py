"""
Example: simulate scattering for multiple geometric shapes: circle, ellipse,
rotated rectangle and a polygon (triangle). Saves animation and receiver data.

Usage:
    python examples/simulate_shapes.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import imageio

# Ensure the project root is on sys.path when running this script directly
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.fdtd2d import Grid2D, FDWave2D, GaussianPulseSource, Receiver

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Grid
nx, ny = 250, 320
dx = 1.0
c = 1.0

grid = Grid2D(nx=nx, ny=ny, dx=dx, c=c)
# Add several obstacles
grid.add_circular_obstacle(cx=60, cy=80, radius=20, value=0.0)
grid.add_ellipse_obstacle(cx=140, cy=90, rx=30, ry=12, angle=20, value=0.0)
grid.add_rotated_rect_obstacle(cx=160, cy=220, width=60, height=18, angle=35, value=0.0)
# Triangle polygon
tri = [(40, 220), (80, 260), (90, 210)]
grid.add_polygon_obstacle(vertices=tri, value=0.0)

grid.set_damping_layer(pad=25, amplitude=0.04)

source = GaussianPulseSource(ix=30, iy=160, amplitude=3.0, freq=0.03, t0=60, width=12)
fd = FDWave2D(grid, dt=0.4, source=source)

rec = Receiver(ixs=[32], iys=[160])

frames = []
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(grid.u, cmap='RdBu', vmin=-1.0, vmax=1.0)
ax.set_title('Wavefield')


def save_frame(tstep):
    frame = (grid.u - grid.u.min()) / (grid.u.max() - grid.u.min() + 1e-9)
    img = (plt.cm.seismic(frame)[:, :, :3] * 255).astype(np.uint8)
    frames.append(img)


def step_and_record(grid, tstep):
    rec.capture(grid)
    if tstep % 2 == 0:
        save_frame(tstep)


nsteps = 500
fd.run(nsteps, callback=step_and_record)

# Save animation
gif_path = OUT_DIR / 'wave_scatter_shapes.gif'
imageio.mimsave(gif_path, frames, fps=25)
print('Saved animation to:', gif_path)

# Plot receiver signal
signal = rec.get_signal()
plt.figure()
plt.plot(signal)
plt.title('Receiver signal (time)')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / 'receiver_signal_shapes.png')
print('Saved receiver signal plot')

# FFT
from scipy.fftpack import fft, fftfreq
sig = signal - np.mean(signal)
N = sig.size
dt = fd.dt
freqs = fftfreq(N, d=dt)
S = np.abs(fft(sig))
mask = freqs >= 0
plt.figure()
plt.plot(freqs[mask], 20*np.log10(S[mask] + 1e-12))
plt.title('Receiver - Frequency (qualitative backscatter)')
plt.xlabel('Frequency')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / 'receiver_spectrum_shapes.png')
print('Saved receiver spectrum plot')

print('Done')
