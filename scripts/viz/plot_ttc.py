#!/usr/bin/env python3
"""
Time-to-Convergence (TTC) scaling plot.

For each solver/sampler, plots how long it takes to reach a convergence
criterion as a function of instance size N, alongside the relative energy
error at that point.

Two convergence modes (--convergence):
  rolling  (default)
      Declare convergence at the first iteration t where
      std(energy[t-W+1 : t+1]) < tol * |E_exact|.
      W = --window (default 10), tol = --tol (default 0.01).
      Runs that never meet the criterion are excluded from the TTC panel
      and counted as non-converged in the data table.

  fixed
      Read energy/time at iteration --fixed-iter (default: last available).
      Useful for comparing "how good is each solver after the same budget".

Time metric: cumulative sum of total_sampling_time_s up to the convergence
iteration (per-iteration wall-clock time recorded in the result JSON).

Error metric: |E_achieved - E_exact| / |E_exact|

Multiple seeds at the same (method, N) are aggregated: median + IQR (p25–p75).
Before plotting, a table is printed showing converged/total runs per sampler and N.

Saved to:
    plots/ttc/{model}_ttc_{convergence}.png

Usage:
    python scripts/viz/plot_ttc.py
    python scripts/viz/plot_ttc.py --convergence rolling --window 10 --tol 0.01
    python scripts/viz/plot_ttc.py --convergence fixed --fixed-iter 50
    python scripts/viz/plot_ttc.py --model tfim_1d --h 0.5
    python scripts/viz/plot_ttc.py --results path/to/results
    python scripts/viz/plot_ttc.py --model tfim_1d --h 0.5 --include-hparam
"""

import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import setup_style, load_json

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "ttc"

KNOWN_MODELS = [
    "tfim_1d", "tfim_2d", "lr_tfim_1d",
    "j1j2_1d", "heisenberg_j1j2_1d", "heisenberg_xxz_1d",
    "heisenberg_xxz_2d", "heisenberg_xy_1d",
]

_HEISENBERG_MODELS = {
    "j1j2_1d", "heisenberg_j1j2_1d", "heisenberg_xxz_1d",
    "heisenberg_xxz_2d", "heisenberg_xy_1d",
}

EXCLUDED_METHODS = {"dimod/zephyr_ra", "dimod/pegasus_mh"}

METHOD_COLORS = {
    "custom/metropolis":          "#1f77b4",
    "custom/simulated_annealing": "#aec7e8",
    "custom/gibbs":               "#ffbb78",
    "custom/sbm":                 "#e377c2",
    "custom/exchange":            "#2ca02c",
    "dimod/pegasus":              "#ff7f0e",
    "dimod/pegasus_fast":         "#ffa040",
    "dimod/simulated_annealing":  "#2ca02c",
    "dimod/zephyr":               "#d62728",
    "dimod/tabu":                 "#8c564b",
    "velox/velox":                "#9467bd",
    "fpga/fpga":                  "#bcbd22",
    "fpga/fpga@sweeps100":            "#e6d800",
    "fpga/fpga@sweeps2000":           "#8a8600",
    "velox/simulated_annealing@sweeps100":  "#c9a6f5",
    "velox/simulated_annealing@sweeps2000": "#6a3fa0",
}

METHOD_MARKERS = {
    "custom/metropolis":          "o",
    "custom/simulated_annealing": "s",
    "custom/gibbs":               "^",
    "custom/sbm":                 "D",
    "custom/exchange":            "D",
    "dimod/pegasus":              "P",
    "dimod/pegasus_fast":         "p",
    "dimod/simulated_annealing":  "X",
    "dimod/zephyr":               "*",
    "dimod/tabu":                 "h",
    "velox/velox":                "p",
    "fpga/fpga":                  "H",
    "fpga/fpga@sweeps100":            "8",
    "fpga/fpga@sweeps2000":           "H",
    "velox/simulated_annealing@sweeps100":  "d",
    "velox/simulated_annealing@sweeps2000": "p",
}


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def _rolling_convergence_iter(energies: list[float], exact: float,
                               window: int, tol: float) -> int | None:
    """
    Return first iteration index t (0-based) where
    std(energy[t-W+1 : t+1]) < tol * |E_exact|.

    Returns None if the criterion is never met.
    """
    threshold = tol * abs(exact)
    arr = np.array(energies)
    for t in range(window - 1, len(arr)):
        if np.std(arr[t - window + 1: t + 1]) < threshold:
            return t
    return None


def _fixed_convergence_iter(energies: list[float], fixed_iter: int) -> int:
    """Return min(fixed_iter, len-1), clamped to valid range."""
    return min(fixed_iter, len(energies) - 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path, model_filter: str,
              h_filter: float | None, j2_filter: float | None = None,
              include_hparam: bool = False):
    """
    Scan results_dir/model_filter/**/*.json AND results_dir/sweeps*/model_filter/**/*.json
    (sibling campaign directories — e.g. sweeps100/, sweeps2000/ — that hold
    FPGA/Velox runs under a different, hardware-specific setting not captured
    in the JSON config; not nested under results_dir/model_filter, so a plain
    results_dir/model_filter scan misses them entirely). Runs from a sweeps*
    directory get "@sweeps100"/"@sweeps2000" appended to method_key so they
    plot as distinct series rather than silently merging with runs of the
    same sampler/method from the base directory or from each other.

    include_hparam=True additionally scans
    results_dir/hparam_search/model_filter/**/*.json.gz — Optuna trial runs
    (written by scripts/hparam/hparam_optuna.py in the same save_results()
    format). These are search probes, not production multi-seed sweeps: many
    distinct hyperparameter configs, often 1-2 seeds each, sometimes short
    iteration budgets. Off by default for that reason; when on, they get
    "@hparam" appended to method_key so they never silently merge with
    production runs of the same sampler/method.

    Returns a list of run dicts:
        {method_key, N, h, j2, energies, times_per_iter, exact_energy, seed,
         gpu_energy_wh, n_iterations_cfg, learning_rate, regularization,
         n_samples, n_hidden, sigma, cem, cem_interval}

    gpu_energy_wh is the measured GPU energy for the whole run (see
    src/energy.py), or None for runs predating that instrumentation.

    n_iterations_cfg and the hyperparameter fields are the configured
    values (config.iterations / config.learning_rate / etc.), not derived
    from the recorded history — a diverged run's history can be shorter
    than n_iterations_cfg.

    For TFIM models: filters by h (transverse field); requires exact_energy.
    For Heisenberg models: filters by J2 (frustration); skips runs without
    exact_energy (e.g. large N where exact diagonalisation is unavailable).
    """
    is_heisenberg = model_filter in _HEISENBERG_MODELS
    runs = []

    campaigns = [("", results_dir / model_filter)]
    for campaign_dir in sorted(results_dir.glob("sweeps*")):
        root = campaign_dir / model_filter
        if root.exists():
            campaigns.append((f"@{campaign_dir.name}", root))
    if include_hparam:
        hparam_root = results_dir / "hparam_search" / model_filter
        if hparam_root.exists():
            campaigns.append(("@hparam", hparam_root))

    for suffix, search_root in campaigns:
        if not search_root.exists():
            continue

        for json_file in sorted(chain(search_root.rglob("*.json"), search_root.rglob("*.json.gz"))):
            try:
                data = load_json(json_file)
            except Exception as e:
                print(f"  [skip] {json_file.name}: {e}")
                continue

            cfg = data.get("config", {})
            history = data.get("history", {})

            N = cfg.get("size")
            sampler = cfg.get("sampler")
            method = cfg.get("sampling_method")
            seed = cfg.get("seed", 0)
            exact = data.get("exact_energy")

            if None in (N, sampler, method):
                continue
            if exact is None:
                continue
            if f"{sampler}/{method}" in EXCLUDED_METHODS:
                continue

            if is_heisenberg:
                j2 = cfg.get("J2")
                if j2 is None:
                    continue
                if j2_filter is not None and abs(j2 - j2_filter) > 1e-6:
                    continue
                h = float(cfg.get("h", 0.0))
            else:
                h = cfg.get("h")
                j2 = float(cfg.get("J2", 0.0))
                if h is None:
                    continue
                if h_filter is not None and abs(h - h_filter) > 1e-6:
                    continue

            energies = history.get("energy")
            times = history.get("total_sampling_time_s")
            if not energies or not times or len(energies) != len(times):
                continue
            runs.append(
                dict(
                    method_key=f"{sampler}/{method}{suffix}",
                    N=int(N),
                    h=float(h),
                    j2=float(j2),
                    energies=energies,
                    times_per_iter=times,
                    exact_energy=float(exact),
                    seed=seed,
                    gpu_energy_wh=data.get("gpu_energy_wh"),
                    n_iterations_cfg=cfg.get("iterations"),
                    learning_rate=cfg.get("learning_rate"),
                    regularization=cfg.get("regularization"),
                    n_samples=cfg.get("n_samples"),
                    n_hidden=cfg.get("n_hidden"),
                    sigma=cfg.get("sigma"),
                    cem=cfg.get("cem"),
                    cem_interval=cfg.get("cem_interval"),
                )
            )

    return runs


# ---------------------------------------------------------------------------
# Per-run TTC and error computation
# ---------------------------------------------------------------------------

def compute_ttc_and_error(run: dict, mode: str, window: int, tol: float,
                           fixed_iter: int) -> tuple[float | None, float | None]:
    """
    Returns (ttc_seconds, relative_error) or (None, None) on failure.

    ttc_seconds : cumulative sampling time up to (and including) convergence iter
    relative_error : |E_achieved - E_exact| / |E_exact|
    """
    energies = run["energies"]
    times = run["times_per_iter"]
    exact = run["exact_energy"]
    cum_times = np.cumsum(times)

    if mode == "rolling":
        t = _rolling_convergence_iter(energies, exact, window, tol)
        if t is None:
            return None, None
    else:  # fixed
        t = _fixed_convergence_iter(energies, fixed_iter)

    ttc = float(cum_times[t])
    e_achieved = float(energies[t])
    rel_err = abs(e_achieved - exact) / abs(exact) if abs(exact) > 1e-12 else float("nan")
    return ttc, rel_err


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(values: list[float]):
    """Return (median, p25, p75) ignoring NaN."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return med, p25, p75


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def print_timing_overview(runs: list[dict]):
    """
    Print per-solver timing statistics across all runs.

    Columns: solver | N | runs | median iter time (ms) | median total time (s) | min–max total (s)
    """
    # Bucket: method_key -> N -> list of (per_iter_median_ms, total_s)
    timing: dict = defaultdict(lambda: defaultdict(list))
    for run in runs:
        times = run["times_per_iter"]
        per_iter_ms = float(np.median(times)) * 1e3
        total_s = float(np.sum(times))
        timing[run["method_key"]][run["N"]].append((per_iter_ms, total_s))

    all_methods = sorted(timing.keys())
    col_w = max(len(mk) for mk in all_methods) + 2

    header = (
        f"{'Solver':<{col_w}}  {'N':>5}  {'runs':>5}  "
        f"{'med iter (ms)':>14}  {'med total (s)':>14}  {'min–max total (s)':>20}"
    )
    sep = "=" * len(header)
    print("\n" + sep)
    print("Timing overview  (per-iteration and total sampling time)")
    print(sep)
    print(header)
    print("-" * len(header))

    for mk in all_methods:
        n_sizes = sorted(timing[mk].keys())
        for i, N in enumerate(n_sizes):
            entries = timing[mk][N]
            iter_meds = [e[0] for e in entries]
            totals    = [e[1] for e in entries]
            label = mk if i == 0 else ""
            print(
                f"{label:<{col_w}}  {N:>5}  {len(entries):>5}  "
                f"{np.median(iter_meds):>14.3f}  "
                f"{np.median(totals):>14.3f}  "
                f"{np.min(totals):>9.3f}–{np.max(totals):<9.3f}"
            )
        if len(n_sizes) > 1:
            print("-" * len(header))

    print(sep + "\n")


def print_convergence_table(bucket: dict, mode: str):
    """
    Print a table of converged/total datapoints per (sampler, N).

    In rolling mode a None TTC means the run never converged.
    In fixed mode all runs produce a value, so converged == total.
    """
    all_methods = sorted(bucket.keys())
    all_sizes = sorted({N for mk in bucket for N in bucket[mk]})

    col_w = max(len(mk) for mk in all_methods) + 2
    size_w = max(7, *(len(str(N)) + 4 for N in all_sizes))

    header = f"{'Sampler':<{col_w}}" + "".join(f"  N={N:<{size_w - 4}}" for N in all_sizes)
    print("\n" + "=" * len(header))
    print("Convergence table  (converged / total runs)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for mk in all_methods:
        row = f"{mk:<{col_w}}"
        for N in all_sizes:
            vals = bucket[mk].get(N, {}).get("ttc", [])
            total = len(vals)
            converged = sum(1 for v in vals if v is not None)
            if total == 0:
                cell = "-"
            elif mode == "fixed":
                cell = str(total)
            else:
                cell = f"{converged}/{total}"
            row += f"  {cell:<{size_w - 2}}"
        print(row)

    print("=" * len(header) + "\n")


def plot_ttc(model: str, runs: list[dict], mode: str, window: int, tol: float,
             fixed_iter: int, out_dir: Path):

    # Bucket by (method_key, N)
    bucket: dict = defaultdict(lambda: defaultdict(lambda: {"ttc": [], "err": []}))
    for run in runs:
        ttc, err = compute_ttc_and_error(run, mode, window, tol, fixed_iter)
        mk = run["method_key"]
        N = run["N"]
        bucket[mk][N]["ttc"].append(ttc)
        bucket[mk][N]["err"].append(err)

    if not bucket:
        print(f"No data for model={model}")
        return

    print_convergence_table(bucket, mode)
    print_timing_overview(runs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax_ttc, ax_err = axes

    mode_label = (
        f"rolling std < {tol:.0%}·|E_exact|, window={window}"
        if mode == "rolling"
        else f"fixed iter={fixed_iter}"
    )
    fig.suptitle(
        f"Time-to-Convergence  —  model={model}\n({mode_label})",
        fontsize=13, fontweight="bold",
    )

    for mk in sorted(bucket.keys()):
        size_data = bucket[mk]
        sizes = sorted(size_data.keys())

        ttc_med, ttc_lo, ttc_hi = [], [], []
        err_med, err_lo, err_hi = [], [], []

        for N in sizes:
            tm, tl, th = aggregate(size_data[N]["ttc"])
            em, el, eh = aggregate(size_data[N]["err"])
            ttc_med.append(tm); ttc_lo.append(tl); ttc_hi.append(th)
            err_med.append(em); err_lo.append(el); err_hi.append(eh)

        color = METHOD_COLORS.get(mk)
        marker = METHOD_MARKERS.get(mk, "o")
        kw = dict(color=color, marker=marker, markersize=6,
                  linewidth=1.8, capsize=3, alpha=0.85, label=mk)

        # TTC panel — skip sizes where median is NaN (never converged)
        valid_ttc = [(N, m, lo, hi)
                     for N, m, lo, hi in zip(sizes, ttc_med, ttc_lo, ttc_hi)
                     if np.isfinite(m)]
        if valid_ttc:
            xs, ms, los, his = zip(*valid_ttc)
            yerr = [
                [m - lo for m, lo in zip(ms, los)],
                [hi - m for m, hi in zip(ms, his)],
            ]
            ax_ttc.errorbar(xs, ms, yerr=yerr, **kw)

        # Error panel — skip NaN
        valid_err = [(N, m, lo, hi)
                     for N, m, lo, hi in zip(sizes, err_med, err_lo, err_hi)
                     if np.isfinite(m)]
        if valid_err:
            xs, ms, los, his = zip(*valid_err)
            yerr = [
                [m - lo for m, lo in zip(ms, los)],
                [hi - m for m, hi in zip(ms, his)],
            ]
            ax_err.errorbar(xs, ms, yerr=yerr, **kw)

    ax_ttc.set_ylabel("TTC (s) — cumulative sampling time", fontsize=11)
    ax_ttc.set_xscale("log")
    ax_ttc.set_yscale("log")
    ax_ttc.grid(True, alpha=0.3, which="both")
    ax_ttc.legend(fontsize=8, loc="lower right")
    ax_ttc.set_title("Time to convergence", fontsize=11)

    ax_err.set_xlabel("Instance size N", fontsize=11)
    ax_err.set_ylabel("Relative energy error  |E − E_exact| / |E_exact|", fontsize=11)
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.grid(True, alpha=0.3, which="both")
    ax_err.legend(fontsize=8, loc="lower right")
    ax_err.set_title("Energy error at convergence point", fontsize=11)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_ttc_{mode}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()

    parser = argparse.ArgumentParser(
        description="Plot Time-to-Convergence (TTC) scaling vs instance size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=KNOWN_MODELS, default="tfim_1d")
    parser.add_argument(
        "--convergence", choices=["rolling", "fixed"], default="rolling",
        help="rolling: std over last W iters < tol*|E_exact|; "
             "fixed: read energy at --fixed-iter",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="[rolling] number of iterations in the rolling std window",
    )
    parser.add_argument(
        "--tol", type=float, default=0.01,
        help="[rolling] convergence threshold as fraction of |E_exact|",
    )
    parser.add_argument(
        "--fixed-iter", type=int, default=999999,
        help="[fixed] iteration index to read (clamped to run length). "
             "Default: last available iteration.",
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
    parser.add_argument(
        "--include-hparam", action="store_true",
        help="Also include Optuna trial runs from results/hparam_search/{model}/ "
             "(tagged '@hparam' in method_key). Off by default — these are "
             "hyperparameter search probes, not production multi-seed sweeps.",
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results / args.model}")
    runs = load_runs(args.results, args.model, args.h, args.j2,
                      include_hparam=args.include_hparam)
    if not runs:
        print("No runs found. Check --results path and --model.")
        return

    param_str = ""
    if args.h is not None:
        param_str = f" h={args.h}"
    if args.j2 is not None:
        param_str += f" J2={args.j2}"
    print(
        f"Found {len(runs)} runs for model={args.model}{param_str}. "
        f"Convergence mode: {args.convergence}."
    )
    methods = sorted({r["method_key"] for r in runs})
    sizes = sorted({r["N"] for r in runs})
    print(f"Methods: {methods}")
    print(f"Sizes  : {sizes}")

    plot_ttc(
        model=args.model,
        runs=runs,
        mode=args.convergence,
        window=args.window,
        tol=args.tol,
        fixed_iter=args.fixed_iter,
        out_dir=PLOTS_DIR,
    )

    print("Done.")


if __name__ == "__main__":
    main()
