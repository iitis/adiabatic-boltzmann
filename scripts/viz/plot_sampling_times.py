#!/usr/bin/env python3
"""
Sampling-time scaling plot.

For each model (1d / 2d) produces one figure showing how mean per-iteration
sampling time scales with instance size N, broken down by sampler/method.

Only runs that actually recorded `sampling_time_s` in their history are used;
runs without the field are silently skipped.

Each point = mean over all iterations and all seeds at that (method, N).
Error bars = std over those same values.

Saved to:
    plots/sampling_times/{model}_sampling_times.png

Usage:
    python scripts/plot_sampling_times.py
    python scripts/plot_sampling_times.py --model 1d
    python scripts/plot_sampling_times.py --results path/to/results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "sampling_times"

METHOD_COLORS = {
    "custom/metropolis":         "#1f77b4",
    "custom/simulated_annealing":"#aec7e8",
    "custom/gibbs":              "#ffbb78",
    "custom/sbm":                "#e377c2",
    "custom/lsb":                "#17becf",
    "dimod/pegasus":             "#ff7f0e",
    "dimod/simulated_annealing": "#2ca02c",
    "dimod/zephyr":              "#d62728",
    "dimod/tabu":                "#8c564b",
    "velox/velox":               "#9467bd",
}

METHOD_MARKERS = {
    "custom/metropolis":         "o",
    "custom/simulated_annealing":"s",
    "custom/gibbs":              "^",
    "custom/sbm":                "D",
    "custom/lsb":                "v",
    "dimod/pegasus":             "P",
    "dimod/simulated_annealing": "X",
    "dimod/zephyr":              "*",
    "dimod/tabu":                "h",
    "velox/velox":               "p",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_timing(results_dir: Path, model_filter: str | None):
    """
    Returns:
        timing[(model, method)][(N)] = list of per-iteration sampling times
                                       (all seeds concatenated)
    Only runs that have a non-empty `sampling_time_s` history list are included.
    """
    timing: dict = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {json_file.name}: {e}")
            continue

        cfg = data.get("config", {})
        history = data.get("history", {})

        model = cfg.get("model")
        N = cfg.get("size")
        sampler = cfg.get("sampler")
        method = cfg.get("sampling_method")

        if None in (model, N, sampler, method):
            continue
        if model_filter and model != model_filter:
            continue

        times = history.get("sampling_time_s")
        if not times:
            continue  # field absent or empty — skip silently

        method_key = f"{sampler}/{method}"
        timing[(model, method_key)][N].extend(times)

    return timing


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model(model: str, timing: dict, out_dir: Path):
    """One figure per model: mean sampling time vs N, one line per method."""

    # Collect all (method_key, N) data for this model
    model_data = {
        method_key: size_map
        for (m, method_key), size_map in timing.items()
        if m == model
    }

    if not model_data:
        print(f"No timing data for model={model}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for method_key in sorted(model_data.keys()):
        size_map = model_data[method_key]
        sizes = sorted(size_map.keys())
        means = [float(np.mean(size_map[N])) for N in sizes]
        stds  = [float(np.std(size_map[N]))  for N in sizes]

        color  = METHOD_COLORS.get(method_key, None)
        marker = METHOD_MARKERS.get(method_key, "o")

        ax.errorbar(
            sizes, means, yerr=stds,
            label=method_key,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            capsize=3,
            alpha=0.85,
        )

    ax.set_xlabel("Instance size N", fontsize=12)
    ax.set_ylabel("Mean sampling time per iteration (s)", fontsize=12)
    ax.set_title(
        f"Sampling time vs instance size — model={model}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_sampling_times.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot sampling time scaling per model")
    parser.add_argument("--model", choices=["1d", "2d"], default=None,
                        help="Restrict to this model (default: all)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help="Path to results directory")
    args = parser.parse_args()

    print("Loading results...")
    timing = load_timing(args.results, args.model)

    if not timing:
        print("No timing data found. Check --results path.")
        return

    models = sorted({m for (m, _) in timing})
    print(f"Found timing data for models: {models}")

    for model in models:
        print(f"Plotting model={model} ...")
        plot_model(model, timing, PLOTS_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
