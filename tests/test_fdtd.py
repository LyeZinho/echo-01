import sys
import os
import pytest
import numpy as np

# Ensure the repository root is in sys.path for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from src.fdtd2d import Grid2D, FDWave2D, GaussianPulseSource


def test_wave_propagation_basic():
    grid = Grid2D(nx=60, ny=80, dx=1.0, c=1.0)
    # small damping
    grid.set_damping_layer(pad=6, amplitude=0.01)
    source = GaussianPulseSource(ix=10, iy=40, amplitude=2.0, freq=0.04, t0=10, width=4)
    fd = FDWave2D(grid, dt=0.4, source=source)

    # run a few steps
    fd.run(50)
    # after stepping, fields should not all be exactly zero
    assert np.abs(grid.u).max() > 1e-6
