#!/usr/bin/env python3
"""
Classical-solver grid plotting: Metropolis, Gibbs, LSB, at the same matched
cell as Figure 10 (lr=0.08, reg=0.05, ns=200, iter=100, h=0.5 -- see
scripts/exper/mcmc_matched_sweep.py). results/tfim_1d/*/custom/{solver}/
also holds many unrelated hyperparameter-search runs, so every glob below
pins the exact matched-cell filename, not a wildcard over the whole solver
directory.

plot_grid(size, h) -- one figure per size: subplots per solver, overlaying
all seeds' energy-convergence curves plus a dashed exact-ground-state line.

plot_lsb_cem_comparison(size, h) -- one figure per size: 2x2, cem off (left)
vs cem on (right) for LSB -- top row energy convergence, bottom row beta_x.
Metropolis/Gibbs have no cem variant (beta_x fixed at 1.0 for them --
Trainer._beta_fixed in src/encoder.py), so no comparison figure for those.

Usage:
    python scripts/viz/plot_mcmc_solver_grid.py
    python scripts/viz/plot_mcmc_solver_grid.py --h 0.5
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import setup_style, load_json

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "tfim_1d"
PLOTS_DIR = ROOT / "plots" / "mcmc_solver_grid"

SOLVERS = ["metropolis", "gibbs", "lsb"]
_MATCHED_CELL = "lr0.08_reg0.05_ns200_seed*_iter100"


def _matched_files(size, h, solver, cem):
    solver_dir = RESULTS_DIR / str(size) / "custom" / solver
    return sorted(solver_dir.glob(
        f"result_1d_h{h}_rbmfull_nh{size}_{_MATCHED_CELL}_cem{cem}_sigma1.0.json.gz"
    ))


def plot_lsb_cem_comparison(size, h):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for col, cem in zip(range(2), [0, 1]):
        ax_energy, ax_beta = axes[0, col], axes[1, col]
        files = _matched_files(size, h, "lsb", cem)

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
        ax_energy.set_title(f"lsb {label} ({len(files)} seeds)")
        ax_beta.set_xlabel("Iteration")
        ax_energy.grid(True, alpha=0.3)
        ax_beta.grid(True, alpha=0.3)

    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[0, 0].set_ylabel("Energy")
    axes[1, 0].set_ylabel(r"$\beta_x$ (effective temperature scale)")
    fig.suptitle(f"LSB: CEM on vs off — N={size}, h={h}", fontweight="bold")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = PLOTS_DIR / f"lsb_cem_comparison_N{size}_h{h}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_grid(size, h):
    fig, axes = plt.subplots(1, len(SOLVERS), figsize=(5 * len(SOLVERS), 5), sharex=True)

    for ax, solver in zip(axes, SOLVERS):
        # LSB has both cem0/cem1 at this cell -- plain (cem0) here, matching
        # the "LSB" (not "LSB (+CEM)") series in Figure 10.
        files = _matched_files(size, h, solver, cem=0)

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

    fig.suptitle(f"Classical solvers convergence — N={size}, h={h}", fontweight="bold")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = PLOTS_DIR / f"grid_N{size}_h{h}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def _discover_sizes(h):
    """Sizes where at least one solver has the matched cell at this h."""
    sizes = []
    if not RESULTS_DIR.is_dir():
        return sizes
    for size_dir in sorted(RESULTS_DIR.iterdir(),
                            key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not size_dir.name.isdigit():
            continue
        n = int(size_dir.name)
        if any(_matched_files(n, h, solver, cem=0) for solver in SOLVERS):
            sizes.append(n)
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=0.5)
    args = parser.parse_args()

    setup_style()
    for size in _discover_sizes(args.h):
        plot_grid(size, args.h)
        if _matched_files(size, args.h, "lsb", cem=1):
            plot_lsb_cem_comparison(size, args.h)
