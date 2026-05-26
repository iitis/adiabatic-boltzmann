#!/usr/bin/env python3
"""
Time-to-Energy (TTE) scaling plot.

For each solver/sampler, plots how long it takes to reach a convergence
criterion as a function of instance size N, alongside the relative energy
error at that point.

Two convergence modes (--convergence):
  rolling  (default)
      Declare convergence at the first iteration t where
      std(energy[t-W+1 : t+1]) < tol * |E_exact|.
      W = --window (default 10), tol = --tol (default 0.01).

  fixed
      Read energy/time at iteration --fixed-iter (default: last available).
      Useful for comparing "how good is each solver after the same budget".

Time metric: cumulative sum of total_sampling_time_s up to the convergence
iteration (per-iteration wall-clock time recorded in the result JSON).

Error metric: |E_achieved - E_exact| / |E_exact|

Multiple seeds at the same (method, N) are aggregated: median + IQR (p25–p75).
Runs that never reach the rolling threshold are reported as "did not converge"
and excluded from that panel.

Saved to:
    plots/tte/{model}_tte_{convergence}.png

Usage:
    python scripts/viz/plot_tte.py
    python scripts/viz/plot_tte.py --convergence rolling --window 10 --tol 0.01
    python scripts/viz/plot_tte.py --convergence fixed --fixed-iter 50
    python scripts/viz/plot_tte.py --model tfim_1d --h 0.5
    python scripts/viz/plot_tte.py --results path/to/results
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "tte"

KNOWN_MODELS = ["tfim_1d", "tfim_2d", "heisenberg_xxz_1d", "lr_tfim_1d"]

METHOD_COLORS = {
    "custom/metropolis":          "#1f77b4",
    "custom/simulated_annealing": "#aec7e8",
    "custom/gibbs":               "#ffbb78",
    "custom/sbm":                 "#e377c2",
    "custom/lsb":                 "#17becf",
    "dimod/pegasus":              "#ff7f0e",
    "dimod/pegasus_fast":         "#ffa040",
    "dimod/pegasus_mh":           "#c05000",
    "dimod/simulated_annealing":  "#2ca02c",
    "dimod/zephyr":               "#d62728",
    "dimod/tabu":                 "#8c564b",
    "velox/velox":                "#9467bd",
    "fpga/fpga":                  "#bcbd22",
}

METHOD_MARKERS = {
    "custom/metropolis":          "o",
    "custom/simulated_annealing": "s",
    "custom/gibbs":               "^",
    "custom/sbm":                 "D",
    "custom/lsb":                 "v",
    "dimod/pegasus":              "P",
    "dimod/pegasus_fast":         "p",
    "dimod/pegasus_mh":           "X",
    "dimod/simulated_annealing":  "X",
    "dimod/zephyr":               "*",
    "dimod/tabu":                 "h",
    "velox/velox":                "p",
    "fpga/fpga":                  "H",
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

def load_runs(results_dir: Path, model_filter: str, h_filter: float | None):
    """
    Scan results_dir/model_filter/**/*.json and return a list of run dicts:
        {method_key, N, h, energies, times_per_iter, exact_energy, seed}

    Skips files where exact_energy is absent or times are missing/empty.
    """
    runs = []
    search_root = results_dir / model_filter
    if not search_root.exists():
        return runs

    for json_file in sorted(search_root.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [skip] {json_file.name}: {e}")
            continue

        cfg = data.get("config", {})
        history = data.get("history", {})

        N = cfg.get("size")
        h = cfg.get("h")
        sampler = cfg.get("sampler")
        method = cfg.get("sampling_method")
        seed = cfg.get("seed", 0)
        exact = data.get("exact_energy")

        if None in (N, h, sampler, method, exact):
            continue
        if h_filter is not None and abs(h - h_filter) > 1e-6:
            continue

        energies = history.get("energy")
        times = history.get("total_sampling_time_s")
        if not energies or not times or len(energies) != len(times):
            continue

        runs.append(
            dict(
                method_key=f"{sampler}/{method}",
                N=int(N),
                h=float(h),
                energies=energies,
                times_per_iter=times,
                exact_energy=float(exact),
                seed=seed,
            )
        )

    return runs


# ---------------------------------------------------------------------------
# Per-run TTE and error computation
# ---------------------------------------------------------------------------

def compute_tte_and_error(run: dict, mode: str, window: int, tol: float,
                           fixed_iter: int) -> tuple[float | None, float | None]:
    """
    Returns (tte_seconds, relative_error) or (None, None) on failure.

    tte_seconds : cumulative sampling time up to (and including) convergence iter
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

    tte = float(cum_times[t])
    e_achieved = float(energies[t])
    rel_err = abs(e_achieved - exact) / abs(exact) if abs(exact) > 1e-12 else float("nan")
    return tte, rel_err


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

def plot_tte(model: str, runs: list[dict], mode: str, window: int, tol: float,
             fixed_iter: int, out_dir: Path):

    # Bucket by (method_key, N)
    bucket: dict = defaultdict(lambda: defaultdict(lambda: {"tte": [], "err": []}))
    for run in runs:
        tte, err = compute_tte_and_error(run, mode, window, tol, fixed_iter)
        mk = run["method_key"]
        N = run["N"]
        bucket[mk][N]["tte"].append(tte)
        bucket[mk][N]["err"].append(err)

    if not bucket:
        print(f"No data for model={model}")
        return

    # Warn about non-converging runs
    for mk, size_data in bucket.items():
        for N, vals in size_data.items():
            n_none = sum(1 for v in vals["tte"] if v is None)
            n_total = len(vals["tte"])
            if n_none:
                print(
                    f"  [warn] {mk} N={N}: {n_none}/{n_total} seeds did not converge"
                    " — excluded from TTE panel"
                )

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
    ax_tte, ax_err = axes

    mode_label = (
        f"rolling std < {tol:.0%}·|E_exact|, window={window}"
        if mode == "rolling"
        else f"fixed iter={fixed_iter}"
    )
    fig.suptitle(
        f"Time-to-Energy  —  model={model}\n({mode_label})",
        fontsize=13, fontweight="bold",
    )

    for mk in sorted(bucket.keys()):
        size_data = bucket[mk]
        sizes = sorted(size_data.keys())

        tte_med, tte_lo, tte_hi = [], [], []
        err_med, err_lo, err_hi = [], [], []

        for N in sizes:
            tm, tl, th = aggregate(size_data[N]["tte"])
            em, el, eh = aggregate(size_data[N]["err"])
            tte_med.append(tm); tte_lo.append(tl); tte_hi.append(th)
            err_med.append(em); err_lo.append(el); err_hi.append(eh)

        color = METHOD_COLORS.get(mk)
        marker = METHOD_MARKERS.get(mk, "o")
        kw = dict(color=color, marker=marker, markersize=6,
                  linewidth=1.8, capsize=3, alpha=0.85, label=mk)

        # TTE panel — skip sizes where median is NaN (never converged)
        valid_tte = [(N, m, lo, hi)
                     for N, m, lo, hi in zip(sizes, tte_med, tte_lo, tte_hi)
                     if np.isfinite(m)]
        if valid_tte:
            xs, ms, los, his = zip(*valid_tte)
            yerr = [
                [m - lo for m, lo in zip(ms, los)],
                [hi - m for m, hi in zip(ms, his)],
            ]
            ax_tte.errorbar(xs, ms, yerr=yerr, **kw)

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

    # -- TTE panel formatting
    ax_tte.set_ylabel("TTE (s) — cumulative sampling time", fontsize=11)
    ax_tte.set_xscale("log")
    ax_tte.set_yscale("log")
    ax_tte.grid(True, alpha=0.3, which="both")
    ax_tte.legend(fontsize=8, loc="upper left")
    ax_tte.set_title("Time to convergence", fontsize=11)

    # -- Error panel formatting
    ax_err.set_xlabel("Instance size N", fontsize=11)
    ax_err.set_ylabel("Relative energy error  |E − E_exact| / |E_exact|", fontsize=11)
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.grid(True, alpha=0.3, which="both")
    ax_err.legend(fontsize=8, loc="upper left")
    ax_err.set_title("Energy error at convergence point", fontsize=11)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_tte_{mode}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Time-to-Energy (TTE) scaling vs instance size",
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
        help="Filter runs to a specific transverse field value (e.g. 0.5). "
             "Default: aggregate over all h values.",
    )
    parser.add_argument(
        "--results", type=Path, default=RESULTS_DIR,
        help="Root results directory",
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results / args.model}")
    runs = load_runs(args.results, args.model, args.h)
    if not runs:
        print("No runs found. Check --results path and --model.")
        return

    h_str = f" h={args.h}" if args.h is not None else ""
    print(
        f"Found {len(runs)} runs for model={args.model}{h_str}. "
        f"Convergence mode: {args.convergence}."
    )
    methods = sorted({r["method_key"] for r in runs})
    sizes = sorted({r["N"] for r in runs})
    print(f"Methods: {methods}")
    print(f"Sizes  : {sizes}")

    plot_tte(
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
