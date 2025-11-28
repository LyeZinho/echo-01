"""
Simple GUI for the pedagogical 2D FDTD wave simulator in `src/fdtd2d.py`.

Usage:
    python examples/gui_simulator.py

This tool opens a Tkinter window with Matplotlib widgets to run a short
simulation, pause/play, and modify a few parameters interactively: source
frequency and obstacle radius.

Note: The GUI is intentionally minimal for educational use.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import numpy as np
import matplotlib
# Use TkAgg backend
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path when running this script directly
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.fdtd2d import Grid2D, FDWave2D, GaussianPulseSource, Receiver


class SimulatorGUI:
    def __init__(self, master):
        self.master = master
        master.title('echo-01 — 2D FDTD Simulator (Pedagógico)')

        # Default simulation parameters
        self.nx, self.ny = 200, 300
        self.dx = 1.0
        self.c = 1.0
        self.dt = 0.4
        self.obstacle_radius = 25
        self.source_freq = 0.03
        self.running = False
        self.delay_ms = 50  # update interval in ms
        self.steps_per_update = 4

        # Build UI elements
        self._build_controls()
        self._build_canvas()

        # Create initial simulation objects
        self.reset_simulation()

    def _build_controls(self):
        frm = ttk.Frame(self.master)
        frm.pack(side=tk.TOP, fill=tk.X)

        # Play/Pause
        self.btn_start = ttk.Button(frm, text='Start', command=self.toggle_run)
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_step = ttk.Button(frm, text='Step', command=self.step_once)
        self.btn_step.grid(row=0, column=1, padx=5, pady=5)

        self.btn_reset = ttk.Button(frm, text='Reset', command=self.reset_simulation)
        self.btn_reset.grid(row=0, column=2, padx=5, pady=5)

        # Slider: frequency
        ttk.Label(frm, text='Freq').grid(row=0, column=3)
        self.freq_var = tk.DoubleVar(value=self.source_freq)
        self.freq_slider = ttk.Scale(frm, from_=0.005, to=0.2, orient=tk.HORIZONTAL,
                                     variable=self.freq_var, command=self.on_freq_change)
        self.freq_slider.grid(row=0, column=4, padx=4)

        # Slider: obstacle radius
        ttk.Label(frm, text='Radius').grid(row=0, column=5)
        self.radius_var = tk.IntVar(value=self.obstacle_radius)
        self.radius_slider = ttk.Scale(frm, from_=0, to=min(self.nx, self.ny)/3,
                                       orient=tk.HORIZONTAL, variable=self.radius_var,
                                       command=self.on_radius_change)
        self.radius_slider.grid(row=0, column=6, padx=4)

        # Speed control
        ttk.Label(frm, text='Speed').grid(row=0, column=7)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_slider = ttk.Scale(frm, from_=0.1, to=5.0, orient=tk.HORIZONTAL,
                                      variable=self.speed_var, command=self.on_speed_change)
        self.speed_slider.grid(row=0, column=8, padx=4)

    def _build_canvas(self):
        # Create matplotlib figure and embed in Tkinter
        self.fig, (self.ax_field, self.ax_signal) = plt.subplots(1, 2, figsize=(8, 3))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # field image
        self.im = self.ax_field.imshow(np.zeros((self.nx, self.ny)), cmap='RdBu', vmin=-1, vmax=1)
        self.ax_field.set_title('Wavefield')

        # receiver signal
        self.signal_line, = self.ax_signal.plot([], [])
        self.ax_signal.set_title('Receiver signal')
        self.ax_signal.set_xlabel('Time step')
        self.ax_signal.set_ylabel('Amplitude')
        self.ax_signal.grid(True)

    def reset_simulation(self):
        # Initialize grid and solver
        self.grid = Grid2D(nx=self.nx, ny=self.ny, dx=self.dx, c=self.c)
        self.grid.add_circular_obstacle(cx=self.nx//2, cy=self.ny//2, radius=self.obstacle_radius, value=0.0)
        self.grid.set_damping_layer(pad=20, amplitude=0.04)

        # Source on the left
        self.source = GaussianPulseSource(ix=20, iy=self.ny//2, amplitude=3.0,
                                          freq=self.source_freq, t0=40, width=10)
        self.fd = FDWave2D(self.grid, dt=self.dt, source=self.source)
        self.receiver = Receiver(ixs=[22], iys=[self.ny//2])

        # Reset visual elements
        self.im.set_data(self.grid.u)
        self.ax_field.set_xlim(0, self.ny-1)
        self.ax_field.set_ylim(self.nx-1, 0)

        self.receiver.record = []
        self.signal_line.set_data([], [])
        self.ax_signal.relim()
        self.ax_signal.autoscale_view()
        self.canvas.draw()

    def on_freq_change(self, event=None):
        val = float(self.freq_var.get())
        self.source.freq = val if hasattr(self, 'source') else val

    def on_radius_change(self, event=None):
        val = int(self.radius_var.get())
        self.obstacle_radius = val
        # update obstacle: re-create object geometry without resetting time
        self.grid.rho[:] = 1.0
        self.grid.obstacle[:] = False
        self.grid.add_circular_obstacle(cx=self.nx//2, cy=self.ny//2, radius=self.obstacle_radius, value=0.0)
        # zero fields inside obstacle
        self.grid.u[self.grid.obstacle] = 0.0
        self.grid.u_prev[self.grid.obstacle] = 0.0

    def on_speed_change(self, event=None):
        v = float(self.speed_var.get())
        # adjust delay and steps per update
        self.delay_ms = int(max(10, 200 / v))
        self.steps_per_update = max(1, int(round(4 * v)))

    def toggle_run(self):
        self.running = not self.running
        self.btn_start.config(text='Pause' if self.running else 'Start')
        if self.running:
            self.master.after(self.delay_ms, self._run_step)

    def step_once(self):
        # Run a few internal steps and update plot
        for _ in range(self.steps_per_update):
            self.fd.step()
            self.receiver.capture(self.grid)
        self._update_plots()

    def _run_step(self):
        if not self.running:
            return
        self.step_once()
        self.master.after(self.delay_ms, self._run_step)

    def _update_plots(self):
        self.im.set_data(self.grid.u)
        # update receiver signal
        sig = np.array(self.receiver.record)
        if sig.size > 0:
            self.signal_line.set_data(np.arange(sig.size), sig)
            self.ax_signal.relim()
            self.ax_signal.autoscale_view()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
