#!/usr/bin/env python3
"""
Per-size learning-rate overview plot.

For each (model, size) combination found in the results directory, produces one
figure whose grid is:

    rows  = distinct h values  (sorted ascending)
    cols  = distinct learning rates  (sorted ascending)

Each cell shows the |ΔE per spin| convergence curves for every sampling method
present, averaged over seeds.  A dashed black line marks the exact ground-state
energy.

Saved to:
    plots/lr_overview/{model}/lr_overview_N{size}.png

Usage:
    python scripts/plot_lr_overview.py
    python scripts/plot_lr_overview.py --model 1d
    python scripts/plot_lr_overview.py --model 1d --size 8
    python scripts/plot_lr_overview.py --results path/to/results
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "lr_overview"

METHOD_COLORS = {
    "custom/metropolis":         "#1f77b4",
    "custom/sbm":                "#e377c2",
    "custom/lsb":                "#17becf",
    "dimod/pegasus":             "#ff7f0e",
    "dimod/simulated_annealing": "#2ca02c",
    "dimod/zephyr":              "#d62728",
    "dimod/tabu":                "#8c564b",
    "velox/velox":               "#9467bd",
}


# ---------------------------------------------------------------------------
# Exact energy helper (reuses the analytical script for 1D, literature for 2D)
# ---------------------------------------------------------------------------

EXACT_ENERGY_2D_PER_SPIN = {
    0.5: -2.0555,
    1.0: -2.1276,
    2.0: -2.4549,
    3.044: -3.0440,
}


def compute_exact_energy(model: str, N: int, h: float) -> float | None:
    if model == "2d":
        val = EXACT_ENERGY_2D_PER_SPIN.get(h)
        if val is None:
            print(f"No 2D reference energy for h={h}")
        return val

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "exact_diag_ising_analytical.py"),
                "-N", str(N),
                "-g", str(h),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(f"Error computing exact energy: {result.stderr}")
            return None
        line = result.stdout.strip().split("\n")[-1]
        return float(line.split(": ")[-1])
    except Exception as e:
        print(f"Failed to compute exact energy for N={N}, h={h}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, model_filter: str | None, size_filter: int | None):
    """
    Returns a nested dict:
        results[(model, size, h, lr)][method_name] = list of run dicts
    where each run dict has keys: data, config, seed, n_visible.
    No filtering on n_hidden or learning rate — we want all LRs.
    """
    results = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

        cfg = data.get("config", {})
        model = cfg.get("model")
        N = cfg.get("size")
        h = cfg.get("h")
        rbm = cfg.get("rbm")
        sampler = cfg.get("sampler")
        sampling_method = cfg.get("sampling_method")
        lr = cfg.get("learning_rate")
        seed = cfg.get("seed")
        n_hidden = cfg.get("n_hidden")

        if None in (model, N, h, rbm, sampler, sampling_method, lr, seed, n_hidden):
            continue
        if cfg.get("cem", False):
            continue
        if model_filter and model != model_filter:
            continue
        if size_filter and N != size_filter:
            continue

        n_visible = N if model == "1d" else N * N
        method_name = f"{sampler}/{sampling_method}"

        results[(model, N, h, lr)][method_name].append(
            {
                "data": data,
                "config": cfg,
                "seed": seed,
                "n_visible": n_visible,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _mean_energy_curve(runs: list, n_visible: int):
    """Return (iterations, mean_energy_per_spin) averaged over seeds."""
    arrays = [run["data"]["history"]["energy"] for run in runs]
    min_len = min(len(a) for a in arrays)
    if min_len == 0:
        return None, None
    truncated = np.array([a[:min_len] for a in arrays], dtype=float)
    mean = np.mean(truncated, axis=0) / n_visible
    return np.arange(min_len), mean


def plot_lr_overview(results, size_key):
    """
    Build the overview figure for one (model, size) key.
    size_key = (model, N)
    """
    model, N = size_key

    # Collect distinct h and lr values present for this size
    h_values = sorted({key[2] for key in results if key[0] == model and key[1] == N})
    lr_values = sorted({key[3] for key in results if key[0] == model and key[1] == N})

    if not h_values or not lr_values:
        return

    n_rows = len(h_values)
    n_cols = len(lr_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Learning-rate overview — model={model}, N={N}",
        fontsize=13, fontweight="bold",
    )

    # Pre-compute exact energies (one per h)
    exact_per_h = {}
    for h in h_values:
        exact_per_h[h] = compute_exact_energy(model, N, h)

    for row_idx, h in enumerate(h_values):
        for col_idx, lr in enumerate(lr_values):
            ax = axes[row_idx][col_idx]
            key = (model, N, h, lr)

            if key not in results:
                ax.set_visible(False)
                continue

            methods_data = results[key]
            exact_E = exact_per_h.get(h)

            for method_name in sorted(methods_data.keys()):
                runs = methods_data[method_name]
                n_visible = runs[0]["n_visible"]
                iters, mean_energy = _mean_energy_curve(runs, n_visible)
                if iters is None:
                    continue

                color = METHOD_COLORS.get(method_name)
                n_seeds = len(runs)
                label = f"{method_name}" + (f" [n={n_seeds}]" if n_seeds > 1 else "")

                if exact_E is not None:
                    delta = np.abs(mean_energy - exact_E)
                    # Guard against zero before log
                    delta = np.where(delta > 0, delta, np.nan)
                    ax.semilogy(iters, delta, label=label, color=color,
                                linewidth=1.8, alpha=0.85)
                else:
                    ax.plot(iters, mean_energy, label=label, color=color,
                            linewidth=1.8, alpha=0.85)

            ax.set_xscale("log")
            ax.set_xlabel("# iterations", fontsize=9)
            if exact_E is not None:
                ax.set_ylabel("|ΔE per spin|", fontsize=9)
            else:
                ax.set_ylabel("Energy per spin", fontsize=9)
            ax.set_title(f"h={h},  lr={lr}", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=7, loc="best")

    plt.tight_layout()

    out_dir = PLOTS_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lr_overview_N{N}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-size learning-rate overview plots")
    parser.add_argument("--model", choices=["1d", "2d"], default=None,
                        help="Restrict to this model (default: all)")
    parser.add_argument("--size", type=int, default=None,
                        help="Restrict to this system size (default: all)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help="Path to results directory")
    args = parser.parse_args()

    print("Loading results...")
    results = load_results(args.results, args.model, args.size)

    if not results:
        print("No results found. Check --results path and filters.")
        return

    # Collect distinct (model, N) pairs
    size_keys = sorted({(key[0], key[1]) for key in results})
    print(f"Found {len(results)} (model, size, h, lr) combos across {len(size_keys)} sizes.")

    for size_key in size_keys:
        model, N = size_key
        print(f"Plotting model={model}, N={N} ...")
        plot_lr_overview(results, size_key)

    print("Done.")


if __name__ == "__main__":
    main()
