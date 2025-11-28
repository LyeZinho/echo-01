"""
Example: simulate scattering off a circular obstacle using scalar 2D FDTD solver.

Usage:
    python examples/simulate_circle.py

This script runs a short simulation and saves a GIF animation and plots the
receiver signal and its frequency spectrum (qualitative backscatter measurement).
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

# Simulation setup
nx, ny = 200, 300
dx = 1.0
c = 1.0

grid = Grid2D(nx=nx, ny=ny, dx=dx, c=c)
grid.add_circular_obstacle(cx=100, cy=150, radius=25, value=0.0)
grid.set_damping_layer(pad=20, amplitude=0.04)

# Source on the left side (not exactly a plane wave, but a wide band of points)
source = GaussianPulseSource(ix=20, iy=150, amplitude=3.0, freq=0.03, t0=40, width=10)

fd = FDWave2D(grid, dt=0.4, source=source)

rec = Receiver(ixs=[22], iys=[150])  # receiver slightly near the source to measure reflection

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
    if tstep % 3 == 0:
        save_frame(tstep)


# Run simulation
nsteps = 400
fd.run(nsteps, callback=step_and_record)

# Save animation
gif_path = OUT_DIR / 'wave_scatter_circle.gif'
imageio.mimsave(gif_path, frames, fps=20)
print('Saved animation to:', gif_path)

# Plot time signal at receiver
signal = rec.get_signal()
plt.figure()
plt.plot(signal)
plt.title('Receiver signal (time) — reflected')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / 'receiver_signal.png')
print('Saved receiver signal plot')

# Frequency content (FFT) - qualitative
from scipy.fftpack import fft, fftfreq
sig = signal - np.mean(signal)
N = sig.size
dt = fd.dt
freqs = fftfreq(N, d=dt)
S = np.abs(fft(sig))

# only positive freqs
mask = freqs >= 0
plt.figure()
plt.plot(freqs[mask], 20*np.log10(S[mask] + 1e-12))
plt.title('Receiver - Frequency (qualitative backscatter)')
plt.xlabel('Frequency')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / 'receiver_spectrum.png')
print('Saved receiver spectrum plot')

print('Done')
