#!/usr/bin/env python3
"""
plot_small_n_solver_comparison.py — statistically fair solver comparison at
small system sizes on the 1D TFIM critical point (h=1.0).

Matched-config protocol: for N in {4, 8, 16}, the exact same RBM architecture
(fully connected, n_hidden=N), learning rate (0.1), regularization (1e-5),
n_samples (1000) and iteration budget (100) is compared across three solvers:

    custom/metropolis        classical Metropolis-Hastings
    dimod/simulated_annealing neal-based simulated annealing
    dimod/tabu                tabu search

Only the sampler differs between the three groups at each N — every other
hyperparameter and the physics instance (h=1.0, the hardest point of the 1D
TFIM) are held fixed. All available seeds per cell are used (no best-of); the
full per-seed distribution of final relative error |E_final - E_exact|/|E_exact|
is plotted as a box (median + IQR) with individual seeds overlaid as jittered
points, so seed-to-seed variance is visible rather than hidden.

Output: plots/small_n_solver_comparison.{png,pdf}
"""

import glob
import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results/tfim_1d"
OUT_DIR = "plots"

SIZES = [4, 8, 16]
SOLVERS = [
    ("custom/metropolis", "Metropolis (custom)"),
    ("dimod/simulated_annealing", "Simulated annealing (dimod)"),
    ("dimod/tabu", "Tabu search (dimod)"),
]
COLORS = {
    "custom/metropolis": "#2a78d6",
    "dimod/simulated_annealing": "#008300",
    "dimod/tabu": "#e87ba4",
}

CONFIG_TAG = "h1.0_rbmfull_nh{n}_lr0.1_reg1e-05_ns1000"


def load_rel_errors(n, solver_path):
    pattern = os.path.join(
        RESULTS_DIR, str(n), solver_path,
        f"result_1d_{CONFIG_TAG.format(n=n)}_seed*_iter*",
    )
    rel_errors = []
    for f in sorted(glob.glob(pattern)):
        with gzip.open(f) as fh:
            d = json.load(fh)
        rel_errors.append(abs(d["final_energy"] - d["exact_energy"]) / abs(d["exact_energy"]))
    return rel_errors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    group_width = 0.8
    n_solvers = len(SOLVERS)
    box_width = group_width / n_solvers
    rng = np.random.default_rng(0)

    positions = []
    box_data = []
    box_colors = []
    counts = []

    for gi, n in enumerate(SIZES):
        for si, (solver_path, _label) in enumerate(SOLVERS):
            errs = load_rel_errors(n, solver_path)
            if not errs:
                continue
            pos = gi + (si - (n_solvers - 1) / 2) * box_width
            positions.append(pos)
            box_data.append(errs)
            box_colors.append(COLORS[solver_path])
            counts.append(len(errs))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width * 0.75,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#0b0b0b", linewidth=1.5),
        whiskerprops=dict(color="#52514e"),
        capprops=dict(color="#52514e"),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    for pos, errs, color in zip(positions, box_data, box_colors):
        jitter = rng.uniform(-box_width * 0.15, box_width * 0.15, size=len(errs))
        ax.scatter(
            [pos] * len(errs) + jitter, errs,
            color=color, edgecolor="white", linewidth=0.4,
            s=22, zorder=3, alpha=0.85,
        )

    for pos, n_seeds in zip(positions, counts):
        ax.text(pos, ax.get_ylim()[1], f"n={n_seeds}", ha="center", va="bottom",
                fontsize=7, color="#898781")

    ax.set_yscale("log")
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([f"N={n}" for n in SIZES])
    ax.set_xlabel("System size")
    ax.set_ylabel(r"Final relative error $|E_\mathrm{final} - E_\mathrm{exact}| / |E_\mathrm{exact}|$")
    ax.set_title(
        "Solver comparison at fixed hyperparameters, TFIM critical point (h=1.0)\n"
        "100 SR iterations, all available seeds per cell (no best-of)",
        fontsize=11,
    )
    ax.grid(axis="y", which="major", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[sp],
                   markeredgecolor=COLORS[sp], markersize=10, alpha=0.7, label=lbl)
        for sp, lbl in SOLVERS
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=9, borderaxespad=0,
    )

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"small_n_solver_comparison.{ext}")
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight",
                    bbox_extra_artists=(legend,))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
