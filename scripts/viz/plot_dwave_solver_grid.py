#!/usr/bin/env python3
"""
2x2 grid, one subplot per D-Wave solver (pegasus, pegasus_fast, zephyr,
zephyr_fast), each subplot overlaying all 20 seeds' energy-convergence
curves for tfim_1d N=16 h=0.5, plus a dashed line at the exact ground
state energy. One figure per cem value (0 = plain beta_x heuristic,
1 = online CEM-tracked beta_x) -- see scripts/exper/dwave_matched_sweep.py.

Usage:
    python scripts/viz/plot_dwave_solver_grid.py
    python scripts/viz/plot_dwave_solver_grid.py --cem 0 1 --size 16 --h 0.5
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import setup_style, load_json

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "tfim_1d"
PLOTS_DIR = ROOT / "plots" / "dwave_solver_grid"

SOLVERS = ["pegasus", "pegasus_fast", "zephyr", "zephyr_fast"]


def plot_pegasus_cem_comparison(size, h):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for col, cem in zip(range(2), [0, 1]):
        ax_energy, ax_beta = axes[0, col], axes[1, col]
        solver_dir = RESULTS_DIR / str(size) / "dimod" / "pegasus"
        files = sorted(solver_dir.glob(f"result_1d_h{h}_*_cem{cem}_*.json.gz"))

        exact_energy = None
        for f in files:
            data = load_json(f)
            energy = data["history"]["energy"]
            beta_x = data["history"]["beta_x"]
            ax_energy.plot(range(1, len(energy) + 1), energy, linewidth=1, alpha=0.6)
            ax_beta.plot(range(1, len(beta_x) + 1), beta_x, linewidth=1, alpha=0.6)
            exact_energy = data["exact_energy"]

        if exact_energy is not None:
            ax_energy.axhline(exact_energy, color="black", linestyle="--", linewidth=1.5,
                               label=f"Exact: {exact_energy:.4f}")
            ax_energy.legend(fontsize=9, loc="best")
        ax_beta.axhline(1.0, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_beta.set_yscale("log")

        label = "with CEM" if cem else "without CEM"
        ax_energy.set_title(f"pegasus {label} ({len(files)} seeds)")
        ax_beta.set_xlabel("Iteration")
        ax_energy.grid(True, alpha=0.3)
        ax_beta.grid(True, alpha=0.3)

    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[0, 0].set_ylabel("Energy")
    axes[1, 0].set_ylabel(r"$\beta_x$ (effective temperature scale)")
    fig.suptitle(f"pegasus: CEM on vs off — N={size}, h={h}", fontweight="bold")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = PLOTS_DIR / f"pegasus_cem_comparison_N{size}_h{h}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_grid(size, h, cem):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for ax, solver in zip(axes.flat, SOLVERS):
        solver_dir = RESULTS_DIR / str(size) / "dimod" / solver
        files = sorted(solver_dir.glob(f"result_1d_h{h}_*_cem{cem}_*.json.gz"))

        exact_energy = None
        for f in files:
            data = load_json(f)
            energy = data["history"]["energy"]
            ax.plot(range(1, len(energy) + 1), energy, linewidth=1, alpha=0.6)
            exact_energy = data["exact_energy"]

        if exact_energy is not None:
            ax.axhline(exact_energy, color="black", linestyle="--", linewidth=1.5,
                        label=f"Exact: {exact_energy:.4f}")
            ax.legend(fontsize=9, loc="best")

        ax.set_title(f"{solver} ({len(files)} seeds)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"D-Wave convergence — N={size}, h={h}, cem={cem}", fontweight="bold")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = PLOTS_DIR / f"grid_N{size}_h{h}_cem{cem}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--cem", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    setup_style()
    for cem in args.cem:
        plot_grid(args.size, args.h, cem)
    plot_pegasus_cem_comparison(args.size, args.h)
