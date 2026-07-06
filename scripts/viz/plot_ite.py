#!/usr/bin/env python3
"""
Iterations/Time/Energy-to-Epsilon (ITE) scaling plot.

Reuses the existing results/ tree (no new runs, no results/hparam_search/
dependency — those are tuning trials, not the data we want here) and
applies the same causal rolling-mean convergence criterion as
scripts/ite/ite_run.py's compute_ite(): the first iteration whose
rolling-mean energy (window W) is within relative error epsilon of the
exact ground state.

results/ pools many different hyperparameter configs per (method, N) —
short hparam-search probes, learning-rate sweeps, production multi-seed
runs, all mixed together, at whatever total iteration budget each was
configured with. Total iteration budget doesn't matter here: the
epsilon-crossing criterion only cares whether and when a run's energy
trajectory crosses epsilon, however long that trajectory is — a run
capped at 30 iterations that never crosses just counts as unconverged,
exactly like a 300-iteration run that never crosses.

What does need to be controlled for is hyperparameters: runs still span
multiple combos (e.g. 17 different learning rates under the same seed).
select_best_configs() groups by (learning_rate, regularization,
n_samples, n_hidden, sigma, cem, cem_interval), scores each combo by its
mean tail relative error across seeds, and keeps only the runs matching
the single best-scoring combo per (method, N). What's left is genuine
seed-to-seed variance for one fixed config — not a blend of seed noise
and hyperparameter-quality variance.

Three panels per plot, all vs instance size N:
  1. ITE          — iterations to reach epsilon
  2. Time-to-ITE  — cumulative sampling time (s) to reach that iteration
  3. Energy-to-ITE — watt-hours to reach that iteration:
       - measured: gpu_energy_wh from the result JSON (src/energy.py),
         scaled by the fraction of total sampling time elapsed at the ITE
         iteration. Only available for runs collected after that
         instrumentation was added.
       - assumed: for hardware this repo cannot self-measure (FPGA
         accelerator, Velox cloud GPU), a constant power draw (see
         ASSUMED_POWER_W below) times time-to-ITE. Rough estimates, not
         calibrated telemetry — marked "(assumed)" in the legend, never
         blended with measured watt-hours as if equally trustworthy.
       - methods with neither a measurement nor an assumption (e.g. D-Wave
         QPU, whose dilution-refrigerator draw is ~constant regardless of
         job) are omitted from this panel rather than given a fabricated
         value.

Runs that never reach epsilon are excluded from all three panels and
counted as non-converged in the printed data table.

Saved to:
    plots/ite/{model}_ite.png

Usage:
    python scripts/viz/plot_ite.py --model tfim_1d --h 0.5
    python scripts/viz/plot_ite.py --model heisenberg_j1j2_1d --j2 0.5
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import setup_style
from plot_ttc import (
    KNOWN_MODELS,
    METHOD_COLORS,
    METHOD_MARKERS,
    aggregate,
    load_runs,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "ite"

# Constant power draw (Watts) assumed for hardware this machine cannot
# self-measure via nvidia-smi. Rough estimates for order-of-magnitude
# comparison only — see module docstring. Keyed by base "sampler/method"
# (before any "@sweepsNNN" campaign suffix — see plot_ttc.load_runs); the
# power draw is a hardware property, not a function of which campaign a
# run came from.
ASSUMED_POWER_W = {
    "fpga/fpga": 45.0,
    "velox/velox": 380.0,
    "velox/simulated_annealing": 380.0,
}


def _base_method(method_key: str) -> str:
    return method_key.split("@")[0]

# Hyperparameter fields that define "the same config" for best-config
# selection within a (method, N) group.
CONFIG_KEYS = (
    "learning_rate", "regularization", "n_samples", "n_hidden", "sigma",
    "cem", "cem_interval",
)


def _config_key(run: dict) -> tuple:
    return tuple(run.get(k) for k in CONFIG_KEYS)


def _tail_rel_error(run: dict) -> float:
    energies = run["energies"]
    exact = run["exact_energy"]
    tail_start = max(0, int(0.8 * len(energies)))
    tail = energies[tail_start:]
    tail_mean = sum(tail) / len(tail)
    return abs(tail_mean - exact) / abs(exact) if exact else float("inf")


def select_best_configs(runs: list[dict]) -> list[dict]:
    """
    Within each (method_key, N) group, keep only the runs whose
    hyperparameter combo (CONFIG_KEYS) has the lowest mean tail relative
    error — collapses hyperparameter-sweep noise to a single config so the
    remaining spread across runs is genuine seed variance.
    """
    by_group: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        by_group[(run["method_key"], run["N"])][_config_key(run)].append(run)

    selected = []
    for (mk, N), by_config in by_group.items():
        best_key, best_score = None, float("inf")
        for key, group_runs in by_config.items():
            errs = [_tail_rel_error(r) for r in group_runs]
            score = sum(errs) / len(errs)
            if score < best_score:
                best_key, best_score = key, score
        selected.extend(by_config[best_key])

    return selected


# ---------------------------------------------------------------------------
# ITE convergence criterion (must match scripts/ite/ite_run.py:compute_ite)
# ---------------------------------------------------------------------------

def compute_ite_iter(energies: list[float], exact_energy: float,
                      epsilon: float, window: int) -> int | None:
    """
    First iteration index t (0-based) where the causal rolling-mean relative
    error drops below epsilon. Window grows naturally from iteration 0.
    Returns None if never reached within the recorded curve.
    """
    for t in range(len(energies)):
        w_start = max(0, t - window + 1)
        mean_e = sum(energies[w_start:t + 1]) / (t - w_start + 1)
        if abs(mean_e - exact_energy) / abs(exact_energy) < epsilon:
            return t
    return None


# ---------------------------------------------------------------------------
# Per-run ITE / time / energy computation
# ---------------------------------------------------------------------------

def compute_ite_metrics(run: dict, epsilon: float, window: int) -> dict | None:
    """
    Returns {ite, time_s, energy_wh, energy_is_assumed} for the first
    epsilon-crossing, or None if the run never reaches epsilon.
    """
    energies = run["energies"]
    times = run["times_per_iter"]
    exact = run["exact_energy"]

    t = compute_ite_iter(energies, exact, epsilon, window)
    if t is None:
        return None

    cum_times = np.cumsum(times)
    time_s = float(cum_times[t])

    energy_wh = None
    energy_is_assumed = False
    measured_wh = run.get("gpu_energy_wh")
    total_time = float(cum_times[-1]) if len(cum_times) else 0.0
    if measured_wh is not None and total_time > 0:
        energy_wh = measured_wh * (time_s / total_time)
    else:
        power_w = ASSUMED_POWER_W.get(_base_method(run["method_key"]))
        if power_w is not None:
            energy_wh = power_w * (time_s / 3600.0)
            energy_is_assumed = True

    return {
        "ite": t + 1,  # 1-indexed, matches ite_run.py's convention
        "time_s": time_s,
        "energy_wh": energy_wh,
        "energy_is_assumed": energy_is_assumed,
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_convergence_table(bucket: dict) -> None:
    all_methods = sorted(bucket.keys())
    all_sizes = sorted({N for mk in bucket for N in bucket[mk]})

    col_w = max(len(mk) for mk in all_methods) + 2
    size_w = max(7, *(len(str(N)) + 4 for N in all_sizes))

    header = f"{'Sampler':<{col_w}}" + "".join(f"  N={N:<{size_w - 4}}" for N in all_sizes)
    print("\n" + "=" * len(header))
    print("Convergence table  (converged / total runs, after best-config selection)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for mk in all_methods:
        row = f"{mk:<{col_w}}"
        for N in all_sizes:
            vals = bucket[mk].get(N, {}).get("ite", [])
            total = len(vals)
            converged = sum(1 for v in vals if v is not None)
            cell = "-" if total == 0 else f"{converged}/{total}"
            row += f"  {cell:<{size_w - 2}}"
        print(row)

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ite(model: str, runs: list[dict], epsilon: float, window: int,
             out_dir: Path) -> None:
    bucket: dict = defaultdict(lambda: defaultdict(
        lambda: {"ite": [], "time_s": [], "energy_wh": []}
    ))
    any_assumed = False

    for run in runs:
        m = compute_ite_metrics(run, epsilon, window)
        mk, N = run["method_key"], run["N"]
        if m is None:
            bucket[mk][N]["ite"].append(None)
            continue
        bucket[mk][N]["ite"].append(m["ite"])
        bucket[mk][N]["time_s"].append(m["time_s"])
        bucket[mk][N]["energy_wh"].append(m["energy_wh"])
        if m["energy_is_assumed"]:
            any_assumed = True

    if not bucket:
        print(f"No data for model={model}")
        return

    print_convergence_table(bucket)

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
    ax_ite, ax_time, ax_energy = axes

    fig.suptitle(
        f"Iterations/Time/Energy to reach $\\epsilon$={epsilon:g}  —  model={model}\n"
        f"(best config per method/N, causal rolling-mean window={window})",
        fontsize=13, fontweight="bold",
    )

    for mk in sorted(bucket.keys()):
        size_data = bucket[mk]
        sizes = sorted(size_data.keys())

        color = METHOD_COLORS.get(mk)
        marker = METHOD_MARKERS.get(mk, "o")
        base_kw = dict(color=color, marker=marker, markersize=6,
                        linewidth=1.8, capsize=3, alpha=0.85)

        for ax, key, label in (
            (ax_ite, "ite", mk),
            (ax_time, "time_s", mk),
            (ax_energy, "energy_wh", f"{mk} (assumed)" if _base_method(mk) in ASSUMED_POWER_W else mk),
        ):
            meds, los, his, xs = [], [], [], []
            for N in sizes:
                vals = size_data[N][key]
                m, lo, hi = aggregate(vals)
                if np.isfinite(m):
                    xs.append(N)
                    meds.append(m)
                    los.append(lo)
                    his.append(hi)
            if not xs:
                continue
            yerr = [[m - lo for m, lo in zip(meds, los)],
                    [hi - m for m, hi in zip(meds, his)]]
            ax.errorbar(xs, meds, yerr=yerr, label=label, **base_kw)

    def _legend_if_any(ax):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="upper left")

    ax_ite.set_ylabel("ITE (iterations)", fontsize=11)
    ax_ite.set_xscale("log")
    ax_ite.set_yscale("log")
    ax_ite.grid(True, alpha=0.3, which="both")
    _legend_if_any(ax_ite)
    ax_ite.set_title("Iterations to epsilon", fontsize=11)

    ax_time.set_ylabel("Time-to-ITE (s)", fontsize=11)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.grid(True, alpha=0.3, which="both")
    _legend_if_any(ax_time)
    ax_time.set_title("Cumulative sampling time to epsilon", fontsize=11)

    ax_energy.set_xlabel("Instance size N", fontsize=11)
    ax_energy.set_ylabel("Energy-to-ITE (Wh)", fontsize=11)
    ax_energy.set_xscale("log")
    ax_energy.set_yscale("log")
    ax_energy.grid(True, alpha=0.3, which="both")
    _legend_if_any(ax_energy)
    title = "Energy consumed to epsilon"
    if any_assumed:
        title += "  (methods marked \"assumed\" use a constant power draw, not measured telemetry)"
    ax_energy.set_title(title, fontsize=10)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_ite.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()

    parser = argparse.ArgumentParser(
        description="Plot Iterations/Time/Energy-to-Epsilon (ITE) scaling vs instance size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=KNOWN_MODELS, default="tfim_1d")
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Relative-error threshold for the epsilon-crossing criterion",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Rolling-mean window for the epsilon-crossing criterion",
    )
    parser.add_argument(
        "--h", type=float, default=None,
        help="[TFIM] Filter runs to a specific transverse field value (e.g. 0.5). "
             "Default: aggregate over all h values.",
    )
    parser.add_argument(
        "--j2", type=float, default=None,
        help="[Heisenberg] Filter runs to a specific J2 value (e.g. 0.5). "
             "Default: aggregate over all J2 values.",
    )
    parser.add_argument(
        "--results", type=Path, default=RESULTS_DIR,
        help="Root results directory",
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results / args.model}")
    runs = load_runs(args.results, args.model, args.h, args.j2)
    print(f"{len(runs)} total runs")
    if not runs:
        print("No runs found. Check --results/--model/--h/--j2.")
        return

    runs = select_best_configs(runs)

    param_str = ""
    if args.h is not None:
        param_str = f" h={args.h}"
    if args.j2 is not None:
        param_str += f" J2={args.j2}"
    print(f"{len(runs)} runs after best-config selection for model={args.model}{param_str}.")
    methods = sorted({r["method_key"] for r in runs})
    sizes = sorted({r["N"] for r in runs})
    print(f"Methods: {methods}")
    print(f"Sizes  : {sizes}")

    plot_ite(
        model=args.model,
        runs=runs,
        epsilon=args.epsilon,
        window=args.window,
        out_dir=PLOTS_DIR,
    )

    print("Done.")


if __name__ == "__main__":
    main()
