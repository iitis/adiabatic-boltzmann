#!/usr/bin/env python3
"""
Compare results across RBM types (full, pegasus, zephyr).

For each (model, N, h, sampler/method) combination that has data for more than
one RBM, produces a side-by-side plot of energy convergence and absolute error,
with one curve per RBM.  This lets you directly compare how much the topology
constraint costs or helps.

Figures are saved to figures/fig_rbm_topology/{model}/.

Usage (from src/):
    python plots/fig_rbm_topology.py
    python plots/fig_rbm_topology.py --results ../results --method custom/metropolis
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures" / "fig_rbm_topology"

RBM_COLORS = {
    "full":    "#1f77b4",
    "pegasus": "#ff7f0e",
    "zephyr":  "#2ca02c",
}

RBM_MARKERS = {
    "full":    "o",
    "pegasus": "s",
    "zephyr":  "^",
}

# Reference energies per spin for 2D TFIM (thermodynamic limit)
EXACT_ENERGY_2D_PER_SPIN = {
    0.5: -2.0555,
    1.0: -2.1276,
    2.0: -2.4549,
    3.044: -3.0440,
}


def compute_exact_energy(model, N, h):
    if model == "2d":
        if h not in EXACT_ENERGY_2D_PER_SPIN:
            print(f"No 2D reference energy for h={h}.")
            return None
        return EXACT_ENERGY_2D_PER_SPIN[h]

    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "exact_diag_ising_analytical.py"), "-N", str(N), "-g", str(h)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[-1]
            return float(line.split(": ")[-1])
        print(f"Error computing exact energy: {result.stderr}")
        return None
    except Exception as e:
        print(f"Failed to compute exact energy for N={N}, h={h}: {e}")
        return None


def load_results(results_dir: Path) -> dict:
    """
    Returns a dict keyed by (model, N, h, method) where each value is a dict
    keyed by rbm type containing a list of runs.
    """
    results = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict) or "config" not in data:
            continue

        config = data["config"]
        try:
            model  = config["model"]
            N      = config["size"]
            h      = config["h"]
            rbm    = config["rbm"]
            sampler         = config["sampler"]
            sampling_method = config["sampling_method"]
            lr       = config["learning_rate"]
            n_hidden = config["n_hidden"]
        except KeyError:
            continue

        n_visible = N if model == "1d" else N * N

        if lr != 0.1:
            continue
        if n_hidden != n_visible:
            continue
        if config.get("cem", False):
            continue

        method = f"{sampler}/{sampling_method}"
        results[(model, N, h, method)][rbm].append({
            "data":      data,
            "config":    config,
            "n_visible": n_visible,
        })

    return results


def plot_rbm_comparison(results: dict, method_filter: str = None):
    """
    One figure per (model, N, h, method) that has data for >= 2 RBM types.
    Each figure: left = energy per spin, right = |E - E_exact| on log scale.
    """
    figs = []

    for (model, N, h, method), rbm_data in sorted(results.items()):
        if method_filter and method != method_filter:
            continue
        if len(rbm_data) < 2:
            continue

        exact_E = compute_exact_energy(model, N, h)
        if exact_E is None:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for rbm in sorted(rbm_data.keys()):
            runs = rbm_data[rbm]
            all_energies = []
            min_len = float("inf")
            n_visible = runs[0]["n_visible"]

            for run in runs:
                energy = run["data"]["history"]["energy"]
                all_energies.append(energy)
                min_len = min(min_len, len(energy))

            if not all_energies or min_len == 0:
                continue

            all_energies = [e[:min_len] for e in all_energies]
            mean_energy = np.mean(all_energies, axis=0)
            iterations = np.arange(len(mean_energy))

            mean_per_spin = mean_energy / n_visible
            delta = np.abs(mean_per_spin - exact_E)

            color  = RBM_COLORS.get(rbm)
            marker = RBM_MARKERS.get(rbm)
            label  = f"RBM={rbm}  (n={len(runs)} runs)"

            ax1.plot(iterations, mean_per_spin, label=label,
                     color=color, linewidth=2, alpha=0.85,
                     marker=marker, markevery=max(1, len(iterations)//10), markersize=5)
            ax2.semilogy(iterations, delta, label=label,
                         color=color, linewidth=2, alpha=0.85,
                         marker=marker, markevery=max(1, len(iterations)//10), markersize=5)

        ax1.axhline(y=exact_E, color="black", linestyle="--", linewidth=2,
                    label=f"Exact: {exact_E:.6f}", zorder=10)

        title = f"model={model}  N={N}  h={h}  method={method}"
        for ax, ylabel, suffix in [
            (ax1, "Energy per spin", "Convergence"),
            (ax2, "Δ E per spin = |E − E_exact|", "Error"),
        ]:
            ax.set_xscale("log")
            ax.set_xlabel("# iterations", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{title}\n{suffix}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=10, loc="best")

        plt.tight_layout()

        out_dir = FIGURES_DIR / model
        out_dir.mkdir(parents=True, exist_ok=True)
        method_slug = method.replace("/", "_")
        filename = out_dir / f"rbm_compare_N{N}_h{h}_{method_slug}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
        figs.append(filename)
        plt.close(fig)

    return figs


def plot_summary(results: dict, method_filter: str = None):
    """
    One summary figure per (model, method) — grid of (N, h) cells, each cell
    showing the RBM comparison error curve.
    """
    # Group (N, h) combos by (model, method)
    groups = defaultdict(list)
    for model, N, h, method in sorted(results.keys()):
        if method_filter and method != method_filter:
            continue
        if len(results[(model, N, h, method)]) >= 2:
            groups[(model, method)].append((N, h))

    figs = []

    for (model, method), nh_list in sorted(groups.items()):
        n = len(nh_list)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig = plt.figure(figsize=(10 * ncols, 5 * nrows))
        method_slug = method.replace("/", "_")
        fig.suptitle(f"RBM comparison — model={model}  method={method}",
                     fontsize=14, fontweight="bold")

        subfigs = np.array(fig.subfigures(nrows, ncols, wspace=0.04, hspace=0.08)).reshape(nrows, ncols)

        for idx, (N, h) in enumerate(sorted(nh_list)):
            row, col = divmod(idx, ncols)
            subfig = subfigs[row, col]
            subfig.patch.set_facecolor("#f5f5f5")
            subfig.patch.set_edgecolor("#aaaaaa")
            subfig.patch.set_linewidth(1.2)

            exact_E = compute_exact_energy(model, N, h)
            if exact_E is None:
                subfig.set_visible(False)
                continue

            ax1, ax2 = subfig.subplots(1, 2)
            rbm_data = results[(model, N, h, method)]

            for rbm in sorted(rbm_data.keys()):
                runs = rbm_data[rbm]
                all_energies = []
                min_len = float("inf")
                n_visible = runs[0]["n_visible"]

                for run in runs:
                    energy = run["data"]["history"]["energy"]
                    all_energies.append(energy)
                    min_len = min(min_len, len(energy))

                if not all_energies or min_len == 0:
                    continue

                all_energies = [e[:min_len] for e in all_energies]
                mean_per_spin = np.mean(all_energies, axis=0)[:min_len] / n_visible
                delta = np.abs(mean_per_spin - exact_E)
                iters = np.arange(min_len)

                color  = RBM_COLORS.get(rbm)
                marker = RBM_MARKERS.get(rbm)
                kw = dict(color=color, linewidth=1.5, alpha=0.85,
                          marker=marker, markevery=max(1, min_len//8), markersize=4)
                ax1.plot(iters, mean_per_spin, label=f"RBM={rbm}", **kw)
                ax2.semilogy(iters, delta, label=f"RBM={rbm}", **kw)

            ax1.axhline(y=exact_E, color="black", linestyle="--", linewidth=1.5,
                        label=f"Exact: {exact_E:.4f}", zorder=10)

            subfig.suptitle(f"N={N}, h={h}", fontsize=11, fontweight="bold")
            for ax, ylabel, title in [
                (ax1, "Energy per spin", "Convergence"),
                (ax2, "Δ E per spin",   "Error"),
            ]:
                ax.set_xscale("log")
                ax.set_xlabel("# iterations", fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(title, fontsize=9)
                ax.grid(True, alpha=0.3, which="both")
                ax.legend(fontsize=7, loc="best")

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            subfigs[row, col].set_visible(False)

        out_dir = FIGURES_DIR / model
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"summary_{method_slug}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved summary: {filename}")
        figs.append(filename)
        plt.close(fig)

    return figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RBM types across runs")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR,
        help="Path to results directory (default: results/)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Only plot this sampler/method (e.g. 'custom/metropolis'). Default: all.",
    )
    args = parser.parse_args()

    print("Loading results...")
    results = load_results(args.results)
    print(f"Found {len(results)} (model, N, h, method) combinations")

    print("\n" + "=" * 70)
    print("Generating per-combo RBM comparison plots...")
    print("=" * 70)
    figs1 = plot_rbm_comparison(results, method_filter=args.method)

    print("\n" + "=" * 70)
    print("Generating summary pages...")
    print("=" * 70)
    figs2 = plot_summary(results, method_filter=args.method)

    print("\n" + "=" * 70)
    print(f"Done! {len(figs1)} comparison plots, {len(figs2)} summary pages.")
    print("=" * 70)
