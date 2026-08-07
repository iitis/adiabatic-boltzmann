#!/usr/bin/env python3
"""
D-Wave grid plotting: two figure types, generated for every (solver, size)
combination that actually has data under results/tfim_1d/*/dimod/{solver}/
(auto-discovered -- solvers differ in embeddable range: pegasus up to
N=128, zephyr up to N=64, pegasus_fast/zephyr_fast only run at N=16 so
far, so no size list is hardcoded).

plot_grid(size, h, cem) -- one figure per (size, cem): 2x2, one subplot
per solver (pegasus, pegasus_fast, zephyr, zephyr_fast), overlaying all
seeds' energy-convergence curves plus a dashed exact-ground-state line.

plot_solver_cem_comparison(size, h, solver) -- one figure per (solver,
size): 2x2, cem off (left column) vs cem on (right column) -- top row
energy convergence, bottom row beta_x (the sampler's effective inverse
temperature; see scripts/exper/dwave_matched_sweep.py). cem=0 is the
plain beta_x heuristic, cem=1 is online CEM-tracked beta_x.

Usage:
    python scripts/viz/plot_dwave_solver_grid.py
    python scripts/viz/plot_dwave_solver_grid.py --cem 0 1 --h 0.5
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import setup_style, load_json

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "tfim_1d"
PLOTS_DIR = ROOT / "plots" / "dwave_solver_grid"

SOLVERS = ["pegasus", "pegasus_fast", "zephyr", "zephyr_fast"]


def plot_solver_cem_comparison(size, h, solver):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for col, cem in zip(range(2), [0, 1]):
        ax_energy, ax_beta = axes[0, col], axes[1, col]
        solver_dir = RESULTS_DIR / str(size) / "dimod" / solver
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
        ax_energy.set_title(f"{solver} {label} ({len(files)} seeds)")
        ax_beta.set_xlabel("Iteration")
        ax_energy.grid(True, alpha=0.3)
        ax_beta.grid(True, alpha=0.3)

    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[0, 0].set_ylabel("Energy")
    axes[1, 0].set_ylabel(r"$\beta_x$ (effective temperature scale)")
    fig.suptitle(f"{solver}: CEM on vs off — N={size}, h={h}", fontweight="bold")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = PLOTS_DIR / f"{solver}_cem_comparison_N{size}_h{h}.png"
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


def _discover_sizes(h):
    """solver -> sorted list of sizes with at least one result file at this h."""
    available = defaultdict(list)
    if not RESULTS_DIR.is_dir():
        return available
    for size_dir in sorted(RESULTS_DIR.iterdir(),
                            key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not size_dir.name.isdigit():
            continue
        size = int(size_dir.name)
        for solver in SOLVERS:
            solver_dir = size_dir / "dimod" / solver
            if solver_dir.is_dir() and next(solver_dir.glob(f"result_1d_h{h}_*.json.gz"), None):
                available[solver].append(size)
    return available


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--cem", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    setup_style()
    available = _discover_sizes(args.h)
    all_sizes = sorted(set(s for sizes in available.values() for s in sizes))

    for size in all_sizes:
        for cem in args.cem:
            plot_grid(size, args.h, cem)

    for solver, sizes in available.items():
        for size in sizes:
            plot_solver_cem_comparison(size, args.h, solver)
