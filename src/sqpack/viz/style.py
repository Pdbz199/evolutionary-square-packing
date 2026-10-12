"""Shared colours and Matplotlib styling for every figure.

Colours follow a validated categorical palette (fixed slot order) and a single
blue sequential ramp. In packings, a square's colour encodes its *tilt*: pale
blue = axis-aligned, deep blue = turned 45 degrees.
"""

import math

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots, always assigned in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BEST, MEDIAN, WORST = SERIES[0], SERIES[1], SERIES[2]

# sequential blue ramp (light -> dark)
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
TILT_CMAP = LinearSegmentedColormap.from_list("tilt", BLUE_RAMP[1:12])
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", [SURFACE] + BLUE_RAMP)
OVERLAP = "#d03b3b"  # status "critical" - only ever used to flag an invalid overlap


def tilt_fraction(theta):
    """0 for an axis-aligned square, 1 for a square turned 45 degrees."""
    t = np.asarray(theta, dtype=float)
    t = t - (math.pi / 2) * np.floor((t + math.pi / 4) / (math.pi / 2))
    return np.abs(t) / (math.pi / 4)


def tilt_colors(theta):
    return TILT_CMAP(tilt_fraction(theta))


def apply():
    """Install the house style into Matplotlib's rcParams."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PAGE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": PAGE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Ubuntu Sans", "Ubuntu", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
