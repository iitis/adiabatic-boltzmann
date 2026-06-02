#!/usr/bin/env python3
"""
plot_j1j2_analysis.py — Analysis plots for J1-J2 benchmark results.

Supports two distinct datasets that must NOT be mixed:

  heisenberg_j1j2_1d — Heisenberg J1-J2 chain (results/heisenberg_j1j2_1d/)
                        N ∈ {8, 12}, methods: exchange / gibbs / lsb / SA / pegasus_fast
  j1j2_1d            — J1-J2 model (results/j1j2_1d/)
                        N ∈ {8, 16, 32}, methods: metropolis / zephyr

Use --model to select which dataset to plot (default: heisenberg_j1j2_1d).

Four figures per run:

  fig_tte_vs_j2.*            — TTE (cumulative sampling time to convergence)
                               vs J₂ for each system size, one line per sampler.

  fig_error_vs_j2.*          — Relative energy error |ΔE|/|E_exact| vs J₂,
                               aggregated over seeds (median + IQR shading).

  fig_energy_vs_j2.*         — Converged E/N per spin vs J₂ (one panel per N),
                               with exact E₀/N shown for reference.

  fig_convergence_curves_N*  — Raw energy/N curves per J₂ row, coloured by sampler.

Convergence criterion (same as plot_tte.py rolling mode):
    std(energy[t-W+1 : t+1]) < tol * |E_exact|
    W = --window (default 20), tol = --tol (default 0.01).
Runs that never converge contribute to the non-converged count but are
excluded from TTE aggregation.  Energy at convergence is the mean of the
last W iterations.

Usage:
    python scripts/plot_j1j2_analysis.py
    python scripts/plot_j1j2_analysis.py --model j1j2_1d
    python scripts/plot_j1j2_analysis.py --model heisenberg_j1j2_1d --window 20 --tol 0.005
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

KNOWN_MODELS = ["heisenberg_j1j2_1d", "j1j2_1d"]
DEFAULT_MODEL = "heisenberg_j1j2_1d"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "figure.dpi":      150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

METHOD_COLOR = {
    "exchange":           "#1f77b4",
    "gibbs":              "#d62728",
    "lsb":                "#9467bd",
    "simulated_annealing":"#ff7f0e",
    "metropolis":         "#2ca02c",
    "zephyr":             "#e377c2",
    "pegasus_fast":       "#8c564b",
    "zephyr_fast":        "#f7b6d2",
}
METHOD_MARKER = {
    "exchange":            "o",
    "gibbs":               "s",
    "lsb":                 "^",
    "simulated_annealing": "D",
    "metropolis":          "v",
    "zephyr":              "*",
    "pegasus_fast":        "P",
    "zephyr_fast":         "X",
}
# Colour cycle fallback for unknown methods
_PALETTE = ["#17becf", "#bcbd22", "#8c564b", "#e377c2"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_runs(results_dir: Path) -> list[dict]:
    """
    Scan results_dir/**/*.json and return list of run dicts:
        {N, J2, method, energies, times_per_iter, exact_energy, seed}
    Files without exact_energy or timing data are skipped.
    """
    runs = []
    for p in sorted(results_dir.rglob("result_*.json")):
        try:
            d = json.load(open(p))
        except Exception as exc:
            print(f"  [skip] {p.name}: {exc}")
            continue

        cfg     = d.get("config", {})
        history = d.get("history", {})

        N      = cfg.get("size")
        J2     = cfg.get("J2")
        method = cfg.get("sampling_method")
        seed   = cfg.get("seed", 0)
        exact  = d.get("exact_energy")

        if None in (N, J2, method, exact):
            continue

        energies = history.get("energy")
        times    = history.get("total_sampling_time_s")
        if not energies or not times or len(energies) != len(times):
            continue
        if not np.all(np.isfinite(energies)):
            continue

        runs.append(dict(
            N=int(N), J2=round(float(J2), 6),
            method=method, seed=int(seed),
            energies=list(energies), times=list(times),
            exact=float(exact),
        ))
    return runs


# ── Convergence helpers ───────────────────────────────────────────────────────

def _rolling_converge(energies: list[float], exact: float,
                      window: int, tol: float) -> int | None:
    """First iteration index t where std(energy[t-W+1:t+1]) < tol*|E_exact|."""
    threshold = tol * abs(exact)
    arr = np.array(energies, dtype=float)
    for t in range(window - 1, len(arr)):
        if np.std(arr[t - window + 1 : t + 1]) < threshold:
            return t
    return None


def compute_metrics(run: dict, window: int, tol: float) -> tuple:
    """
    Returns (tte_s, rel_error, e_per_spin) or (None, None, None) if not converged.

    tte_s       — cumulative sampling time at convergence
    rel_error   — |E_mean_tail - E_exact| / |E_exact|
    e_per_spin  — E_mean_tail / N
    """
    energies = run["energies"]
    times    = run["times"]
    exact    = run["exact"]
    N        = run["N"]

    t = _rolling_converge(energies, exact, window, tol)
    if t is None:
        return None, None, None

    cum_t = float(np.cumsum(times)[t])
    tail  = np.array(energies[max(0, t - window + 1) : t + 1], dtype=float)
    e_mean = float(tail.mean())
    rel_err = abs(e_mean - exact) / abs(exact) if abs(exact) > 1e-12 else float("nan")
    return cum_t, rel_err, e_mean / N


# ── Aggregation ───────────────────────────────────────────────────────────────

def _agg(values: list) -> tuple:
    """(median, p25, p75) over finite values, or (nan, nan, nan)."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def bucket_runs(runs: list[dict], window: int, tol: float,
                exclude_j2_zero: bool = True) -> dict:
    """
    Returns nested dict:
        bucket[N][method][J2] = {"tte": [...], "err": [...], "eps": [...],
                                  "total": int}
    J2=0 runs are excluded by default (pure Heisenberg, no frustration).
    """
    bucket: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"tte": [], "err": [], "eps": [], "total": 0})
        )
    )
    for run in runs:
        if exclude_j2_zero and run["J2"] == 0.0:
            continue
        tte, err, eps = compute_metrics(run, window, tol)
        entry = bucket[run["N"]][run["method"]][run["J2"]]
        entry["total"] += 1
        if tte is not None:
            entry["tte"].append(tte)
            entry["err"].append(err)
            entry["eps"].append(eps)
    return bucket


# ── Colour / marker helpers ───────────────────────────────────────────────────

_extra_color_idx = 0

def _method_style(method: str) -> tuple[str, str]:
    global _extra_color_idx
    color  = METHOD_COLOR.get(method)
    marker = METHOD_MARKER.get(method, "o")
    if color is None:
        color = _PALETTE[_extra_color_idx % len(_PALETTE)]
        _extra_color_idx += 1
    return color, marker


def _shade(ax, xs, meds, p25s, p75s, color, alpha=0.15):
    """Fill between p25 and p75 where all three are finite."""
    valid = [(x, m, lo, hi)
             for x, m, lo, hi in zip(xs, meds, p25s, p75s)
             if all(np.isfinite([m, lo, hi]))]
    if len(valid) < 2:
        return
    xv, _, lov, hiv = zip(*valid)
    ax.fill_between(xv, lov, hiv, color=color, alpha=alpha, linewidth=0)


def _save(fig, out_dir: Path, stem: str, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {out_dir / stem}.pdf")


# ── Figure 1: TTE vs J₂ ──────────────────────────────────────────────────────

def fig_tte_vs_j2(bucket: dict, out: Path, dpi: int, tol: float, window: int,
                  model_label: str = ""):
    sizes   = sorted(bucket.keys())
    methods = sorted({m for n in bucket.values() for m in n})
    nrows   = len(sizes)

    fig, axes = plt.subplots(nrows, 1, figsize=(6.0, nrows * 3.0), squeeze=False)
    fig.suptitle(
        f"Time-to-Convergence vs J₂  ({model_label})\n"
        f"rolling std < {tol:.1%}·|E_exact|, window={window}",
        fontsize=10, y=1.01,
    )

    for ri, N in enumerate(sizes):
        ax = axes[ri, 0]
        n_conv_info = []

        for method in methods:
            if method not in bucket[N]:
                continue
            j2_map = bucket[N][method]
            j2s    = sorted(j2_map)
            meds, p25s, p75s = [], [], []
            conv_fracs = []

            for j2 in j2s:
                e = j2_map[j2]
                m, lo, hi = _agg(e["tte"])
                meds.append(m); p25s.append(lo); p75s.append(hi)
                total = e["total"]
                conv  = len(e["tte"])
                conv_fracs.append(f"{conv}/{total}")

            color, marker = _method_style(method)
            valid = [(j2, m, lo, hi)
                     for j2, m, lo, hi in zip(j2s, meds, p25s, p75s)
                     if np.isfinite(m)]
            if not valid:
                continue

            xv, mv, lov, hiv = zip(*valid)
            ax.plot(xv, mv, color=color, marker=marker, markersize=5,
                    linewidth=1.6, label=method, zorder=3)
            _shade(ax, xv, mv, lov, hiv, color)

        ax.set_yscale("log")

        # Frustration marker at J₂ = 0.5 — y in axes-fraction coords (0-1)
        ax.axvline(0.5, color="#999", linestyle=":", linewidth=0.9, zorder=0)
        ax.text(0.502, 0.97, "J₂=0.5\n(frustrated)", fontsize=6, color="#888",
                va="top", ha="left", transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"N={N}\nTTE (s)", fontsize=8)
        ax.set_xlabel("J₂" if ri == nrows - 1 else "")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25, which="both")

        ax.set_title(f"N={N}", fontsize=8, pad=3)

    fig.tight_layout()
    _save(fig, out, "fig_tte_vs_j2", dpi)


# ── Figure 2: Relative error vs J₂ ───────────────────────────────────────────

def fig_error_vs_j2(bucket: dict, out: Path, dpi: int, tol: float, window: int,
                    model_label: str = ""):
    sizes   = sorted(bucket.keys())
    methods = sorted({m for n in bucket.values() for m in n})
    nrows   = len(sizes)

    fig, axes = plt.subplots(nrows, 1, figsize=(6.0, nrows * 3.0), squeeze=False)
    fig.suptitle(
        f"Relative energy error vs J₂  —  {model_label}",
        fontsize=10, y=1.01,
    )

    for ri, N in enumerate(sizes):
        ax = axes[ri, 0]
        for method in methods:
            if method not in bucket[N]:
                continue
            j2_map = bucket[N][method]
            j2s    = sorted(j2_map)
            meds, p25s, p75s, conv_k, totals = [], [], [], [], []

            for j2 in j2s:
                e = j2_map[j2]
                # Only converged runs contribute to e["err"] (set in bucket_runs).
                m, lo, hi = _agg(e["err"])
                meds.append(m); p25s.append(lo); p75s.append(hi)
                conv_k.append(len(e["err"]))
                totals.append(e["total"])

            color, marker = _method_style(method)

            # Build per-point alpha based on convergence fraction
            # (full = 0.9, none = not plotted, partial = proportionally dimmed)
            valid = [
                (j2, m, lo, hi, k, tot)
                for j2, m, lo, hi, k, tot in zip(j2s, meds, p25s, p75s, conv_k, totals)
                if np.isfinite(m) and k > 0
            ]
            if not valid:
                continue

            xv, mv, lov, hiv, kv, tv = zip(*valid)
            frac = [k / t if t > 0 else 0.0 for k, t in zip(kv, tv)]

            # Plot line with uniform style; dim individual markers by frac
            ax.plot(xv, mv, color=color, linewidth=1.6,
                    linestyle="-", alpha=0.7, zorder=2, label=method)
            for x, m_val, k, t, f in zip(xv, mv, kv, tv, frac):
                alpha = max(0.25, f)   # never fully invisible
                ax.plot(x, m_val, marker=marker, markersize=6,
                        color=color, alpha=alpha, zorder=4,
                        markeredgewidth=0.5,
                        markeredgecolor="white" if f >= 1.0 else color)
                if k < t:
                    ax.annotate(f"{k}/{t}", xy=(x, m_val),
                                xytext=(0, 5), textcoords="offset points",
                                fontsize=5.5, color=color, ha="center",
                                alpha=0.85)

            _shade(ax, xv, mv, lov, hiv, color,
                   alpha=0.12 * min(frac) if frac else 0.12)

        ax.axvline(0.5, color="#999", linestyle=":", linewidth=0.9, zorder=0)
        ax.axhline(0.01, color="#bbb", linestyle="--", linewidth=0.75, zorder=0,
                   label="1% target")

        ax.set_yscale("log")
        ax.set_ylabel(f"N={N}\n|ΔE|/|E_exact|", fontsize=8)
        ax.set_xlabel("J₂" if ri == nrows - 1 else "")
        ax.set_title(f"N={N}", fontsize=8, pad=3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    _save(fig, out, "fig_error_vs_j2", dpi)


# ── Figure 3: Converged E/N per spin vs J₂ ───────────────────────────────────

def fig_energy_vs_j2(runs: list[dict], bucket: dict, out: Path, dpi: int,
                     window: int, tol: float, model_label: str = ""):
    """
    Converged ⟨E⟩/N vs J₂, one panel per N.
    Median over seeds + IQR shading, exact E₀/N as dashed reference.
    Also shows all individual seed values as small dots.
    J2=0 runs are excluded (consistent with bucket_runs filtering).
    """
    runs    = [r for r in runs if r["J2"] != 0.0]
    sizes   = sorted(bucket.keys())
    methods = sorted({m for n in bucket.values() for m in n})
    nrows   = len(sizes)

    # Collect exact energies per (N, J2) — they should be consistent across seeds
    exact_map: dict[tuple, float] = {}
    for run in runs:
        exact_map[(run["N"], run["J2"])] = run["exact"] / run["N"]

    fig, axes = plt.subplots(nrows, 1, figsize=(6.0, nrows * 3.0), squeeze=False)
    fig.suptitle(
        f"Converged ⟨E⟩/N per spin vs J₂  ({model_label})\n"
        "Median ± IQR over seeds",
        fontsize=10, y=1.01,
    )

    for ri, N in enumerate(sizes):
        ax = axes[ri, 0]

        # Plot exact E₀/N reference
        exact_j2 = sorted((j2, e) for (n, j2), e in exact_map.items() if n == N)
        if exact_j2:
            xex, yex = zip(*exact_j2)
            ax.plot(xex, yex, color="#333", linestyle="--", linewidth=1.1,
                    zorder=0, label="exact E₀/N")

        for method in methods:
            if method not in bucket[N]:
                continue
            j2_map = bucket[N][method]
            j2s    = sorted(j2_map)
            meds, p25s, p75s = [], [], []
            all_dots_x, all_dots_y = [], []

            for j2 in j2s:
                e = j2_map[j2]
                vals = [v for v in e["eps"] if v is not None and np.isfinite(v)]
                m, lo, hi = _agg(vals)
                meds.append(m); p25s.append(lo); p75s.append(hi)
                all_dots_x.extend([j2] * len(vals))
                all_dots_y.extend(vals)

            color, marker = _method_style(method)

            # Individual seed dots
            if all_dots_x:
                ax.scatter(all_dots_x, all_dots_y, color=color, alpha=0.25,
                           s=12, zorder=2)

            # Median line
            valid = [(j2, m, lo, hi)
                     for j2, m, lo, hi in zip(j2s, meds, p25s, p75s)
                     if np.isfinite(m)]
            if not valid:
                continue
            xv, mv, lov, hiv = zip(*valid)
            ax.plot(xv, mv, color=color, marker=marker, markersize=5,
                    linewidth=1.6, label=method, zorder=3)
            _shade(ax, xv, mv, lov, hiv, color)

        ax.axvline(0.5, color="#999", linestyle=":", linewidth=0.9, zorder=0)
        ax.set_ylabel(f"N={N}\n⟨E⟩/N", fontsize=8)
        ax.set_xlabel("J₂" if ri == nrows - 1 else "")
        ax.set_title(f"N={N}", fontsize=8, pad=3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, out, "fig_energy_vs_j2", dpi)


# ── Figure 4: Convergence curves per J₂ (one row per J₂, per N) ──────────────

def fig_convergence_curves(runs: list[dict], out: Path, dpi: int,
                           model_label: str = ""):
    """
    One figure per N.  One row per J₂.  All seed curves shown, coloured by method.
    Exact E₀/N dashed.  Y-axis = E/N per spin.
    J2=0 runs are excluded.
    """
    runs  = [r for r in runs if r["J2"] != 0.0]
    sizes = sorted({r["N"] for r in runs})

    for N in sizes:
        n_runs = [r for r in runs if r["N"] == N]
        j2_vals = sorted({r["J2"] for r in n_runs})
        methods = sorted({r["method"] for r in n_runs})

        nrows = len(j2_vals)
        fig, axes = plt.subplots(nrows, 1, figsize=(6.0, nrows * 2.2), squeeze=False)
        fig.suptitle(
            f"Energy convergence per J₂ — N={N}  ({model_label})\n"
            "Each line = one seed, coloured by sampler",
            fontsize=10, y=1.01,
        )

        for ri, j2 in enumerate(j2_vals):
            ax = axes[ri, 0]
            group = [r for r in n_runs if r["J2"] == j2]
            exact_per_spin = group[0]["exact"] / N if group else None
            n_plotted = 0

            for method in methods:
                color, _ = _method_style(method)
                for run in group:
                    if run["method"] != method:
                        continue
                    eng = np.array(run["energies"], dtype=float) / N
                    ax.plot(eng, color=color, linewidth=0.7, alpha=0.55)
                    n_plotted += 1

            if exact_per_spin is not None:
                ax.axhline(exact_per_spin, color="#333", linestyle="--",
                           linewidth=0.9, zorder=0, label="E₀/N")
                ax.legend(frameon=False, fontsize=6.5, loc="upper right")

            ax.set_ylim(exact_per_spin * 1.5 if exact_per_spin < 0 else None,
                        abs(exact_per_spin) * 0.3 if exact_per_spin is not None else None)
            ax.set_ylabel(f"J₂={j2:.3g}\n⟨E⟩/N", fontsize=8)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if n_plotted:
                ax.annotate(f"n={n_plotted}", xy=(0.97, 0.97),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=6.5, color="#555")
            if ri == nrows - 1:
                ax.set_xlabel("iteration")

        # Shared legend for methods
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=_method_style(m)[0], linewidth=2,
                          label=m)
                   for m in methods]
        fig.legend(handles=handles, loc="lower center", ncol=len(methods),
                   fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        _save(fig, out, f"fig_convergence_curves_N{N}", dpi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analysis plots for J1-J2 benchmark results. "
                    "Use --model to select which dataset to plot; "
                    "heisenberg_j1j2_1d and j1j2_1d are distinct models "
                    "and must not be mixed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=KNOWN_MODELS, default=DEFAULT_MODEL,
        help="Which J1-J2 dataset to plot. "
             "heisenberg_j1j2_1d: Heisenberg chain, N∈{8,12}, exchange/gibbs/lsb/SA. "
             "j1j2_1d: J1-J2 model, N∈{8,16,32}, metropolis/zephyr.",
    )
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Override results directory (default: results/<model>/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: plots/j1j2_analysis/<model>/)",
    )
    parser.add_argument("--dpi",    type=int,   default=150)
    parser.add_argument("--window", type=int,   default=20,
                        help="Rolling std window for convergence criterion")
    parser.add_argument("--tol",    type=float, default=0.01,
                        help="Convergence threshold as fraction of |E_exact|")
    parser.add_argument("--include-j2-zero", action="store_true", default=False,
                        help="Include J₂=0 points (excluded by default)")
    args = parser.parse_args()

    results_dir = args.results or (ROOT / "results" / args.model)
    out_dir     = args.output_dir or (ROOT / "plots" / "j1j2_analysis" / args.model)
    model_label = args.model

    print(f"Model   : {model_label}")
    print(f"Loading : {results_dir}")
    runs = load_runs(results_dir)
    if not runs:
        print("No runs found. Check --results path and --model.")
        return

    sizes   = sorted({r["N"] for r in runs})
    j2s     = sorted({r["J2"] for r in runs})
    methods = sorted({r["method"] for r in runs})
    print(f"  {len(runs)} runs  |  N={sizes}  |  J₂={j2s}  |  methods={methods}")

    exclude_j2_zero = not args.include_j2_zero
    bucket = bucket_runs(runs, args.window, args.tol,
                         exclude_j2_zero=exclude_j2_zero)

    # Print convergence summary table
    print(f"\nConvergence summary (window={args.window}, tol={args.tol}):")
    for N in sizes:
        for method in methods:
            if method not in bucket[N]:
                continue
            parts = []
            for j2 in sorted(bucket[N][method]):
                e = bucket[N][method][j2]
                parts.append(f"J₂={j2:.3g}: {len(e['tte'])}/{e['total']}")
            print(f"  N={N:2d}  {method:<22}  " + "  ".join(parts))

    print(f"\nSaving figures to: {out_dir}")

    fig_tte_vs_j2(bucket, out_dir, args.dpi, args.tol, args.window,
                  model_label=model_label)
    fig_error_vs_j2(bucket, out_dir, args.dpi, args.tol, args.window,
                    model_label=model_label)
    fig_energy_vs_j2(runs, bucket, out_dir, args.dpi, args.window, args.tol,
                     model_label=model_label)
    fig_convergence_curves(runs, out_dir, args.dpi, model_label=model_label)

    print("\nDone.")


if __name__ == "__main__":
    main()
