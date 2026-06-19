import gzip
import json
import numpy as np
import matplotlib as mpl


def setup_style(fontsize=15, scale=1.0, grid=True):
    fig_width_pt = 246.0
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "figure.figsize": [fig_width, fig_height],
        "axes.titlesize": fontsize,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
    })


def load_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)
