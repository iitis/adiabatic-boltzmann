"""
plot_results.py — supervisor-ready grid plots from VMC training results.

Generates:
  fig_convergence_grid.*         — convergence per (N, J2), all samplers
  fig_error_vs_J2.*              — final relative error vs J2, one line per N
  fig_method_comparison.*        — exchange / lsb / gibbs on heisenberg_xxz_1d
  fig_sample_quality.*           — ESS and unique-sample ratio over training
  fig_error_vs_N.*               — error scaling with system size
  fig_solver_metropolis.*  ┐
  fig_solver_gibbs.*       │  per-solver hyperparam sensitivity grids
  fig_solver_lsb.*         │  (j1j2_1d, includes Optuna-searched runs)
  fig_solver_zephyr.*      ┘

Usage (from project root):
    python scripts/plot_results.py
    python scripts/plot_results.py --results-dir results --dpi 150
"""

import argparse
import json
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plot_style import setup_style, load_json

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
})

# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

_METHOD_COLOR = {
    "metropolis":        "#1f77b4",
    "simulated_annealing": "#ff7f0e",
    "exchange":          "#2ca02c",
    "gibbs":             "#d62728",
    "lsb":               "#9467bd",
    "zephyr":            "#8c564b",
    "zephyr_ra":         "#e377c2",
    "tabu":              "#7f7f7f",
}


# ── Data loader ───────────────────────────────────────────────────────────────

def load_results(results_dir: Path, include_hparam: bool = False) -> list[dict]:
    """Load every result JSON under results_dir.

    include_hparam=True also loads runs saved inside hparam_search/
    (Optuna trial results with varied hyperparams).
    """
    records = []
    for p in chain(results_dir.rglob("*.json"), results_dir.rglob("*.json.gz")):
        if not include_hparam and "hparam" in p.parts:
            continue
        if "sample_quality" in p.name or p.name == "index.jsonl":
            continue
        try:
            d = load_json(p)
            if "config" not in d or "history" not in d:
                continue
            records.append({"path": p, "config": d["config"], "history": d["history"],
                             "exact_energy": d.get("exact_energy"),
                             "final_energy": d.get("final_energy"),
                             "error": d.get("error")})
        except Exception:
            pass
    return records


def _smooth(arr, w=15):
    """Centred rolling mean; clips window at array length."""
    if len(arr) < 2 * w:
        w = max(1, len(arr) // 4)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def _converged_energy(arr: np.ndarray, window: int = 20,
                      rel_tol: float = 1e-3) -> float:
    """
    Mean energy over the converged tail of a training curve.

    Convergence is declared at the first iteration t ≥ 2·W where the
    relative change between consecutive W-point window means satisfies:

        |mean(E[t-W : t]) - mean(E[t-2W : t-W])| / |mean(E[t-2W : t-W])| < rel_tol

    The returned value is the mean of all remaining samples from that
    iteration onward — not just the endpoint — reducing noise.

    Falls back to the last-20 % tail mean if no convergence is detected
    within the run.
    """
    n = len(arr)
    if n < 2 * window:
        return float(arr.mean())
    for t in range(2 * window, n):
        prev = float(arr[t - 2 * window : t - window].mean())
        curr = float(arr[t - window : t].mean())
        if prev != 0 and abs(curr - prev) / abs(prev) < rel_tol:
            return float(arr[t - window :].mean())
    return float(arr[max(0, int(0.8 * n)) :].mean())


def _rel_err(energy, exact):
    if exact is None or exact == 0:
        return None
    return abs(energy - exact) / abs(exact)


# ── Figure 1: convergence grid (j1j2_1d, all samplers) ───────────────────────

def fig_convergence_grid(records: list[dict], out: Path, dpi: int):
    target_J2   = [0.0, 0.1, 0.45, 0.5, 0.55, 0.7]
    target_N    = [8, 16, 32]
    snap_tol    = 0.03

    # Group: (N, method, J2_snapped) → list of runs
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        c = r["config"]
        if c.get("model") != "j1j2_1d":
            continue
        N      = int(c.get("size", -1))
        method = c.get("sampling_method", "")
        J2     = float(c.get("J2", -1))
        if N not in target_N:
            continue
        snap = min(target_J2, key=lambda x: abs(x - J2))
        if abs(snap - J2) > snap_tol:
            continue
        groups[(N, method, snap)].append(r)

    # Which methods actually appear?
    all_methods = sorted({k[1] for k in groups})
    method_color = {m: _METHOD_COLOR.get(m, _PALETTE[i % len(_PALETTE)])
                    for i, m in enumerate(all_methods)}

    nrows, ncols = len(target_N), len(target_J2)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.5, nrows * 2.1),
                              sharex=False, sharey=False)
    fig.suptitle("VMC energy convergence — J1–J2 frustrated Ising chain (all samplers)",
                 fontsize=11, y=1.01)

    for ri, N in enumerate(target_N):
        for ci, J2 in enumerate(target_J2):
            ax = axes[ri, ci]
            ax.set_title(f"N={N},  J₂/J₁={J2}", fontsize=7.5, pad=3)

            # Gather exact energy from any run in this (N, J2) cell
            exact = None
            for m in all_methods:
                runs = groups.get((N, m, J2), [])
                if runs and runs[0].get("exact_energy") is not None:
                    exact = runs[0]["exact_energy"]
                    break

            has_any = False
            for method in all_methods:
                runs = groups.get((N, method, J2), [])
                if not runs:
                    continue
                color = method_color[method]
                all_sm = []
                for run in runs:
                    eng = np.array([e for e in run["history"]["energy"] if e is not None],
                                   dtype=float)
                    if eng.size == 0 or not np.all(np.isfinite(eng)):
                        continue
                    all_sm.append(_smooth(eng / N))   # ← energy per spin

                if not all_sm:
                    continue
                has_any = True

                min_len = min(len(s) for s in all_sm)
                stack   = np.array([s[:min_len] for s in all_sm])
                mean_   = stack.mean(0)
                std_    = stack.std(0)
                xs      = np.arange(min_len)

                ax.plot(xs, mean_, color=color, linewidth=1.4, label=method)
                ax.fill_between(xs, mean_ - std_, mean_ + std_, color=color, alpha=0.12)

                # Per-method relative error badge at the right edge
                final = float(mean_[-1])
                if exact is not None:
                    re = _rel_err(final, exact)
                    if re is not None:
                        col = "#2ca02c" if re < 0.01 else ("#ff7f0e" if re < 0.05 else "#d62728")
                        ax.annotate(f"{method[0].upper()}:{re:.1%}",
                                    xy=(1.01, mean_[-1]),
                                    xycoords=("axes fraction", "data"),
                                    va="center", fontsize=5.5, color=col)

            if not has_any:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#bbbbbb", fontsize=7)

            if exact is not None:
                ax.axhline(exact / N, color="#333333", linestyle="--",
                           linewidth=0.8, zorder=0)

            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

            if ri == nrows - 1:
                ax.set_xlabel("iteration", fontsize=7)
            if ci == 0:
                ax.set_ylabel("⟨E⟩ / N", fontsize=8)

    # One shared legend — sampler colours + exact line
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=method_color[m], linewidth=1.8, label=m)
               for m in all_methods]
    handles.append(Line2D([0], [0], color="#333333", linestyle="--",
                          linewidth=0.9, label="exact E₀/N"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    path = out / "fig_convergence_grid.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 2: final relative error vs J2 ─────────────────────────────────────

def fig_error_vs_J2(records: list[dict], out: Path, dpi: int):
    target_N = [8, 16, 32]
    colors    = {8: _PALETTE[0], 16: _PALETTE[1], 32: _PALETTE[2]}

    # (N, J2) → list of relative errors across seeds
    data: dict[tuple, list] = defaultdict(list)
    for r in records:
        c = r["config"]
        if c.get("model") != "j1j2_1d":
            continue
        if c.get("sampling_method") != "metropolis":
            continue
        N   = int(c.get("size", -1))
        J2  = float(c.get("J2", -1))
        exact = r.get("exact_energy")
        hist  = r["history"]["energy"]
        if N not in target_N or exact is None or not hist:
            continue
        tail  = np.array([e for e in hist[int(0.8 * len(hist)):] if e is not None], dtype=float)
        if len(tail) == 0:
            continue
        re = _rel_err(float(tail.mean()), exact)
        if re is not None and re < 5:            # drop catastrophic divergences
            data[(N, J2)].append(re)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for N in target_N:
        J2s   = sorted(set(j for n, j in data if n == N))
        means = [np.mean(data[(N, j)]) for j in J2s]
        stds  = [np.std(data[(N, j)])  for j in J2s]
        ax.errorbar(J2s, means, yerr=stds, fmt="o-", color=colors[N],
                    label=f"N={N}", capsize=3, markersize=4)

    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8, label="J₂/J₁=0.5 (max frustration)")
    ax.set_xlabel("J₂ / J₁")
    ax.set_ylabel("relative error  |⟨E⟩ − E₀| / |E₀|")
    ax.set_title("Final relative energy error vs. frustration (metropolis, mean ± std over seeds)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    fig.tight_layout()

    path = out / "fig_error_vs_J2.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 3: method comparison (heisenberg_xxz_1d, N=16) ─────────────────────

def fig_method_comparison(records: list[dict], out: Path, dpi: int):
    methods = ["exchange", "lsb", "gibbs", "metropolis"]

    # Group by method
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        c = r["config"]
        if c.get("model") != "heisenberg_xxz_1d":
            continue
        if int(c.get("size", 0)) != 16:
            continue
        delta = float(c.get("delta", -1))
        if abs(delta - 1.0) > 0.05:           # XXZ Heisenberg (isotropic)
            continue
        groups[c.get("sampling_method", "?")].append(r)

    present = [m for m in methods if groups[m]]
    if not present:
        print("  ⚠  No heisenberg_xxz_1d N=16 data found — skipping fig 3")
        return

    metrics = [("energy", "⟨E⟩"), ("ess", "ESS"), ("n_unique_ratio", "unique sample ratio")]
    ncols = len(metrics)
    nrows = len(present)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.0),
                              sharex=False)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Sampler comparison — Heisenberg XXZ chain  N=16, δ=1 (isotropic)",
                 fontsize=10, y=1.01)

    for ri, method in enumerate(present):
        runs = groups[method]
        color = _METHOD_COLOR.get(method, "#333333")
        exact = runs[0].get("exact_energy")

        for ci, (key, label) in enumerate(metrics):
            ax = axes[ri, ci]
            all_sm = []
            for run in runs:
                arr = run["history"].get(key, [])
                vals = np.array([v for v in arr if v is not None], dtype=float)
                if vals.size == 0:
                    continue
                if not np.all(np.isfinite(vals)):
                    continue                        # skip diverged runs for this panel
                sm = _smooth(vals)
                all_sm.append(sm)
                ax.plot(sm, color=color, alpha=0.3, linewidth=0.7)

            if all_sm:
                min_len = min(len(s) for s in all_sm)
                stack = np.array([s[:min_len] for s in all_sm])
                mean_ = stack.mean(0)
                std_  = stack.std(0)
                xs = np.arange(min_len)
                ax.plot(xs, mean_, color=color, linewidth=1.6)
                ax.fill_between(xs, mean_ - std_, mean_ + std_, color=color, alpha=0.15)
            else:
                ax.text(0.5, 0.5, "all runs\ndiverged", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=7)

            if key == "energy" and exact is not None:
                ax.axhline(exact, color="#d62728", linestyle="--",
                           linewidth=0.9, label=f"exact")
                if all_sm:
                    final = float(mean_[-1])
                    re = _rel_err(final, exact)
                    if re is not None:
                        col = "#2ca02c" if re < 0.01 else ("#ff7f0e" if re < 0.05 else "#d62728")
                        ax.annotate(f"ε={re:.2%}", xy=(0.97, 0.06),
                                    xycoords="axes fraction", ha="right",
                                    fontsize=7, color=col,
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
                else:
                    n_div = sum(1 for run in runs
                                if not np.all(np.isfinite(
                                    np.array([v for v in run["history"].get("energy",[]) if v is not None], dtype=float)
                                )))
                    ax.annotate(f"{n_div}/{len(runs)} diverged", xy=(0.97, 0.06),
                                xycoords="axes fraction", ha="right",
                                fontsize=7, color="#d62728",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

            if ri == 0:
                ax.set_title(label, fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(method, fontsize=8, color=color, fontweight="bold")
            if ri == nrows - 1:
                ax.set_xlabel("iteration")

            ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    fig.tight_layout()
    path = out / "fig_method_comparison.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 4: sample quality dashboard ────────────────────────────────────────

def fig_sample_quality(records: list[dict], out: Path, dpi: int):
    """ESS + unique-sample ratio over training for several representative runs."""
    # Pick j1j2_1d metropolis N=8, a range of J2 values, seed=1
    candidates = []
    for r in records:
        c = r["config"]
        if c.get("model") != "j1j2_1d":
            continue
        if c.get("sampling_method") != "metropolis":
            continue
        if int(c.get("size", 0)) != 8:
            continue
        if int(c.get("seed", 0)) != 1:
            continue
        J2 = float(c.get("J2", -1))
        if J2 in [0.0, 0.25, 0.5, 0.7, 1.0] or J2 in [0.45, 0.55]:
            candidates.append((J2, r))

    # Keep one run per J2
    per_J2: dict[float, dict] = {}
    for J2, r in candidates:
        if J2 not in per_J2:
            per_J2[J2] = r
    per_J2 = dict(sorted(per_J2.items()))

    if not per_J2:
        print("  ⚠  No sample quality data found — skipping fig 4")
        return

    cmap  = plt.cm.viridis_r
    J2s   = list(per_J2.keys())
    colors = {j: cmap(i / max(1, len(J2s) - 1)) for i, j in enumerate(J2s)}

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    fig.suptitle("Sample quality over training — J1–J2 Ising N=8, metropolis (seed 1)",
                 fontsize=10)

    panels = [("ess", "effective sample size (ESS)"),
              ("n_unique_ratio", "unique samples / batch"),
              ("kl_exact", "KL divergence from exact")]

    for ax, (key, label) in zip(axes, panels):
        for J2, run in per_J2.items():
            arr = run["history"].get(key, [])
            vals = np.array([v for v in arr if v is not None], dtype=float)
            if vals.size == 0:
                continue
            ax.plot(_smooth(vals, w=10), color=colors[J2], linewidth=1.1,
                    label=f"J₂={J2}")

        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("iteration")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)

    fig.tight_layout()
    path = out / "fig_sample_quality.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 5: error vs system size N (scaling) ───────────────────────────────

def fig_error_vs_N(records: list[dict], out: Path, dpi: int):
    """Relative error vs N for j1j2_1d metropolis across J2 values."""
    target_J2 = [0.0, 0.25, 0.45, 0.5, 0.55, 0.7, 1.0]
    snap_tol  = 0.04

    data: dict[float, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        c = r["config"]
        if c.get("model") != "j1j2_1d":
            continue
        if c.get("sampling_method") != "metropolis":
            continue
        exact = r.get("exact_energy")
        if exact is None:
            continue
        N  = int(c.get("size", -1))
        J2 = float(c.get("J2", -1))
        snap = min(target_J2, key=lambda x: abs(x - J2))
        if abs(snap - J2) > snap_tol:
            continue
        hist = r["history"]["energy"]
        tail = [e for e in hist[int(0.8 * len(hist)):] if e is not None]
        if not tail:
            continue
        re = _rel_err(float(np.mean(tail)), exact)
        if re is not None and re < 5:
            data[snap][N].append(re)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    cmap = plt.cm.plasma

    j2_list = sorted(j for j in data if data[j])
    for i, J2 in enumerate(j2_list):
        color = cmap(i / max(1, len(j2_list) - 1))
        ns    = sorted(data[J2])
        means = [np.mean(data[J2][n]) for n in ns]
        stds  = [np.std(data[J2][n])  for n in ns]
        ax.errorbar(ns, means, yerr=stds, fmt="o-", color=color,
                    label=f"J₂={J2}", capsize=3, markersize=4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("system size  N")
    ax.set_ylabel("relative error  |⟨E⟩ − E₀| / |E₀|")
    ax.set_title("Energy error scaling with system size (metropolis, tail-mean ± std over seeds)")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.legend(frameon=False, ncol=2, fontsize=7.5)

    fig.tight_layout()
    path = out / "fig_error_vs_N.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")


# ── Per-solver convergence grids (j1j2_1d, includes Optuna runs) ─────────────

def _solver_grid(records: list[dict], method: str,
                 row_vals: list, row_label: str, row_key: str,
                 col_vals: list, col_label: str,
                 col_fn,          # callable(config) → col_bin label or None
                 color_vals: list, color_label: str,
                 color_fn,        # callable(config) → color_bin label or None
                 color_map: dict,
                 out: Path, dpi: int,
                 title: str, fname: str,
                 snap_tol: float = 0.04):
    """
    Generic per-solver grid helper.

    Rows = row_vals (J2 values, snapped within snap_tol).
    Cols = col_vals (discrete bins produced by col_fn).
    Color = color_vals (bins produced by color_fn), drawn as line colour.

    Each cell shows all matching runs as thin α lines + bold mean.
    Y axis: E/N.  Dashed line: exact E₀/N.
    """
    # Group records
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        c = r["config"]
        if c.get("model") != "j1j2_1d":
            continue
        if c.get("sampling_method") != method:
            continue
        J2_raw = float(c.get("J2", -1))
        snap = min(row_vals, key=lambda x: abs(x - J2_raw))
        if abs(snap - J2_raw) > snap_tol:
            continue
        col_bin   = col_fn(c)
        color_bin = color_fn(c)
        if col_bin is None or color_bin is None:
            continue
        groups[(snap, col_bin, color_bin)].append(r)

    nrows, ncols = len(row_vals), len(col_vals)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.6, nrows * 2.0),
                              sharex=False, sharey=False)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=11, y=1.01)

    for ri, J2 in enumerate(row_vals):
        for ci, col_bin in enumerate(col_vals):
            ax = axes[ri, ci]
            ax.set_title(f"{row_key}={J2},  {col_label}={col_bin}",
                         fontsize=7.5, pad=3)

            # Find exact energy and N for this cell
            exact = None
            cell_N = None
            for cb in color_vals:
                for r in groups.get((J2, col_bin, cb), []):
                    if exact is None and r.get("exact_energy") is not None:
                        exact = r["exact_energy"]
                    if cell_N is None:
                        cell_N = int(r["config"].get("size", 8))
                    if exact is not None and cell_N is not None:
                        break
                if exact is not None and cell_N is not None:
                    break

            has_data = False
            best_re, best_label = float("inf"), ""

            for cb in color_vals:
                runs = groups.get((J2, col_bin, cb), [])
                if not runs:
                    continue
                color = color_map.get(cb, "#888888")
                all_sm = []
                for run in runs:
                    eng = np.array([e for e in run["history"]["energy"] if e is not None],
                                   dtype=float)
                    if eng.size == 0 or not np.all(np.isfinite(eng)):
                        continue
                    all_sm.append(_smooth(eng / run["config"]["size"]))

                if not all_sm:
                    continue
                has_data = True
                for sm in all_sm:
                    ax.plot(sm, color=color, alpha=0.25, linewidth=0.6)

                min_len = min(len(s) for s in all_sm)
                mean_ = np.array([s[:min_len] for s in all_sm]).mean(0)
                ax.plot(mean_, color=color, linewidth=1.5, label=f"{color_label}={cb}")

                if exact is not None and cell_N is not None:
                    # mean_ is already E/N; _converged_energy finds the stable tail
                    # (W=20 iters, rel change < 0.1 % between consecutive windows)
                    conv_per_spin = _converged_energy(mean_, window=20, rel_tol=1e-3)
                    re = _rel_err(conv_per_spin, exact / cell_N)
                    if re is not None and re < best_re:
                        best_re, best_label = re, cb

            if not has_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#bbbbbb", fontsize=7)

            if exact is not None:
                if cell_N is not None:
                    ax.axhline(exact / cell_N, color="#333333", linestyle="--",
                               linewidth=0.8, zorder=0)

                if has_data and math.isfinite(best_re):
                    col = "#2ca02c" if best_re < 0.01 else ("#ff7f0e" if best_re < 0.05 else "#d62728")
                    ax.annotate(f"best ε={best_re:.1%}",
                                xy=(0.97, 0.06), xycoords="axes fraction",
                                ha="right", fontsize=6, color=col,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          alpha=0.75, ec="none"))

            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if ri == nrows - 1:
                ax.set_xlabel("iteration", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"J₂/J₁ = {J2}\n⟨E⟩/N", fontsize=7.5)

    # Shared legend (colour axis only)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=color_map.get(cb, "#888"), linewidth=1.8,
                      label=f"{color_label}={cb}") for cb in color_vals]
    handles.append(Line2D([0], [0], color="#333", linestyle="--",
                          linewidth=0.9, label="exact E₀/N"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 6), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    path = out / f"{fname}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")


def _bin(val: float, edges: list[float], labels: list[str]) -> str | None:
    """Bin a continuous value. edges has len(labels)+1 entries."""
    for i, label in enumerate(labels):
        if edges[i] <= val < edges[i + 1]:
            return label
    return None


def fig_solver_grids(records: list[dict], out: Path, dpi: int):
    """One convergence-grid figure per solver, j1j2_1d, all Optuna runs included."""

    # ── Metropolis ────────────────────────────────────────────────────────────
    # Cols = learning rate (3 discrete values).  Color = N.
    _lr_vals    = [0.01, 0.05, 0.1]
    _lr_labels  = ["0.01", "0.05", "0.1"]
    _N_vals     = [8, 16, 32]
    _N_colors   = {8: _PALETTE[0], 16: _PALETTE[1], 32: _PALETTE[2]}

    def _metro_col_fn(c):
        lr = round(float(c.get("learning_rate", -1)), 4)
        snp = min(_lr_vals, key=lambda x: abs(x - lr))
        return str(snp) if abs(snp - lr) < 0.005 else None

    def _metro_color_fn(c):
        N = int(c.get("size", -1))
        return str(N) if N in _N_vals else None

    _solver_grid(
        records, method="metropolis",
        row_vals=[0.0, 0.3, 0.5, 0.7], row_label="J₂", row_key="J₂",
        col_vals=_lr_labels, col_label="lr",
        col_fn=_metro_col_fn,
        color_vals=[str(n) for n in _N_vals], color_label="N",
        color_fn=_metro_color_fn,
        color_map={str(n): _N_colors[n] for n in _N_vals},
        out=out, dpi=dpi,
        title="Metropolis sampler — J1–J2 Ising chain\nRows: frustration J₂/J₁ · Cols: learning rate · Colour: system size N",
        fname="fig_solver_metropolis",
    )

    # ── Gibbs ─────────────────────────────────────────────────────────────────
    # Cols = n_samples (binned: small / medium / large).  Color = lr bin.
    _ns_edges  = [0, 700, 1300, 1e9]
    _ns_labels = ["n_s ≤ 600", "n_s 800–1200", "n_s ≥ 1400"]
    _lr_edges  = [0, 0.008, 0.06, 1e9]
    _lr_g_labs = ["lr < 0.008", "lr 0.008–0.06", "lr > 0.06"]
    _lr_g_cols = {"lr < 0.008": _PALETTE[0], "lr 0.008–0.06": _PALETTE[1], "lr > 0.06": _PALETTE[2]}

    def _gibbs_col_fn(c):
        return _bin(float(c.get("n_samples", 0)), _ns_edges, _ns_labels)

    def _gibbs_color_fn(c):
        return _bin(float(c.get("learning_rate", 0)), _lr_edges, _lr_g_labs)

    _solver_grid(
        records, method="gibbs",
        row_vals=[0.0, 0.1, 0.45, 0.55], row_label="J₂", row_key="J₂",
        col_vals=_ns_labels, col_label="n_samples",
        col_fn=_gibbs_col_fn,
        color_vals=_lr_g_labs, color_label="lr",
        color_fn=_gibbs_color_fn,
        color_map=_lr_g_cols,
        out=out, dpi=dpi,
        title="Gibbs sampler — J1–J2 Ising chain  (Optuna-searched hyperparams)\nRows: frustration J₂/J₁ · Cols: batch size n_samples · Colour: learning rate range",
        fname="fig_solver_gibbs",
    )

    # ── LSB ───────────────────────────────────────────────────────────────────
    # Cols = lr bin (most varied axis for LSB).  Color = n_samples bin.
    _lr_lsb_edges = [0, 0.01, 0.05, 1e9]
    _lr_lsb_labs  = ["lr < 0.01", "lr 0.01–0.05", "lr > 0.05"]
    _ns_lsb_edges = [0, 800, 1400, 1e9]
    _ns_lsb_labs  = ["n_s ≤ 600", "n_s 800–1200", "n_s ≥ 1400"]
    _ns_lsb_cols  = {"n_s ≤ 600": _PALETTE[0], "n_s 800–1200": _PALETTE[1], "n_s ≥ 1400": _PALETTE[2]}

    def _lsb_col_fn(c):
        return _bin(float(c.get("learning_rate", 0)), _lr_lsb_edges, _lr_lsb_labs)

    def _lsb_color_fn(c):
        return _bin(float(c.get("n_samples", 0)), _ns_lsb_edges, _ns_lsb_labs)

    _solver_grid(
        records, method="lsb",
        row_vals=[0.0, 0.1, 0.3, 0.7], row_label="J₂", row_key="J₂",
        col_vals=_lr_lsb_labs, col_label="lr",
        col_fn=_lsb_col_fn,
        color_vals=_ns_lsb_labs, color_label="n_s",
        color_fn=_lsb_color_fn,
        color_map=_ns_lsb_cols,
        out=out, dpi=dpi,
        title="Langevin SB sampler (+ CEM) — J1–J2 Ising chain  (Optuna-searched hyperparams)\nRows: frustration J₂/J₁ · Cols: learning rate range · Colour: batch size",
        fname="fig_solver_lsb",
    )

    # ── Zephyr (QPU) ──────────────────────────────────────────────────────────
    # Rows = N.  Cols = J2.  Color = lr.
    _zep_lr_vals = [0.01, 0.05, 0.1, 0.2]
    _zep_lr_cols = {str(lr): _PALETTE[i] for i, lr in enumerate(_zep_lr_vals)}

    def _zep_col_fn(c):
        J2_raw = float(c.get("J2", -1))
        # col_fn here is called on J2 row snapping; repurpose for N row below
        return None   # overridden per-call below

    # Zephyr has N=8 and N=32; repurpose row_key=N
    _zep_groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        c = r["config"]
        if c.get("model") != "j1j2_1d":
            continue
        if c.get("sampling_method") not in ("zephyr", "zephyr_ra"):
            continue
        N  = int(c.get("size", -1))
        J2 = float(c.get("J2", -1))
        lr_raw = float(c.get("learning_rate", -1))
        snp_J2 = min([0.0, 0.1, 0.45, 0.5, 0.55, 0.7, 1.0],
                     key=lambda x: abs(x - J2))
        if abs(snp_J2 - J2) > 0.04:
            continue
        snp_lr = min(_zep_lr_vals, key=lambda x: abs(x - lr_raw))
        if abs(snp_lr - lr_raw) > 0.02:
            continue
        _zep_groups[(N, snp_J2, snp_lr)].append(r)

    _zep_N_vals = sorted({k[0] for k in _zep_groups})
    _zep_J2_vals = sorted({k[1] for k in _zep_groups})

    if _zep_N_vals and _zep_J2_vals:
        nrows_z, ncols_z = len(_zep_N_vals), len(_zep_J2_vals)
        fig_z, axs_z = plt.subplots(nrows_z, ncols_z,
                                     figsize=(ncols_z * 2.4, nrows_z * 2.2),
                                     sharex=False, sharey=False)
        if nrows_z == 1: axs_z = axs_z[np.newaxis, :]
        if ncols_z == 1: axs_z = axs_z[:, np.newaxis]

        fig_z.suptitle(
            "D-Wave Zephyr QPU — J1–J2 Ising chain\n"
            "Rows: system size N · Cols: frustration J₂/J₁ · Colour: learning rate",
            fontsize=10, y=1.01)

        for ri, N in enumerate(_zep_N_vals):
            for ci, J2 in enumerate(_zep_J2_vals):
                ax = axs_z[ri, ci]
                ax.set_title(f"N={N},  J₂={J2}", fontsize=7.5, pad=3)

                exact = None
                has_data = False
                for lr in _zep_lr_vals:
                    runs = _zep_groups.get((N, J2, lr), [])
                    if not runs:
                        continue
                    if exact is None and runs[0].get("exact_energy") is not None:
                        exact = runs[0]["exact_energy"]
                    col = _zep_lr_cols.get(str(lr), "#888")
                    for run in runs:
                        eng = np.array([e for e in run["history"]["energy"] if e is not None],
                                       dtype=float)
                        if eng.size == 0 or not np.all(np.isfinite(eng)):
                            continue
                        sm = _smooth(eng / N)
                        ax.plot(sm, color=col, linewidth=1.3, alpha=0.85,
                                label=f"lr={lr}")
                        has_data = True

                if not has_data:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="#bbb", fontsize=7)
                if exact is not None:
                    ax.axhline(exact / N, color="#333", linestyle="--",
                               linewidth=0.8, zorder=0)

                ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
                if ri == nrows_z - 1:
                    ax.set_xlabel("iteration", fontsize=7)
                if ci == 0:
                    ax.set_ylabel(f"N={N}\n⟨E⟩/N", fontsize=7.5)

        from matplotlib.lines import Line2D
        _seen = {}
        for ax in axs_z.flat:
            for h in ax.get_lines():
                _seen[h.get_label()] = h
        handles_z = [Line2D([0], [0], color=_zep_lr_cols.get(str(lr), "#888"),
                            linewidth=1.8, label=f"lr={lr}") for lr in _zep_lr_vals]
        handles_z.append(Line2D([0], [0], color="#333", linestyle="--",
                                linewidth=0.9, label="exact E₀/N"))
        fig_z.legend(handles=handles_z, loc="lower center",
                     ncol=len(handles_z), fontsize=7.5,
                     frameon=False, bbox_to_anchor=(0.5, -0.03))

        fig_z.tight_layout()
        path_z = out / "fig_solver_zephyr.pdf"
        fig_z.savefig(path_z, bbox_inches="tight", dpi=dpi)
        fig_z.savefig(path_z.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
        plt.close(fig_z)
        print(f"  → {path_z}")
    else:
        print("  ⚠  No Zephyr j1j2_1d data found — skipping fig_solver_zephyr")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    setup_style()

    _repo = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Generate summary plots from VMC results")
    parser.add_argument("--results-dir", default=str(_repo / "results"),
                        help="Root results directory (default: results/)")
    parser.add_argument("--output-dir", default=str(_repo / "plots"),
                        help="Where to save figures (default: plots/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results (including Optuna runs) from {results_dir} …")
    records = load_results(results_dir, include_hparam=True)
    print(f"  {len(records)} result files loaded\n")

    print("Generating summary figures ...")
    fig_convergence_grid(records, out_dir, args.dpi)
    fig_error_vs_J2(records, out_dir, args.dpi)
    fig_method_comparison(records, out_dir, args.dpi)
    fig_sample_quality(records, out_dir, args.dpi)
    fig_error_vs_N(records, out_dir, args.dpi)
    print("\nGenerating per-solver hyperparam grids ...")
    fig_solver_grids(records, out_dir, args.dpi)
    print(f"\nAll plots saved to  {out_dir}/")


if __name__ == "__main__":
    main()
