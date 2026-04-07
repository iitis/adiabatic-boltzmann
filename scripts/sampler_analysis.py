#!/usr/bin/env python3
"""
Comprehensive sampler analysis: convergence, sample quality, beta dynamics,
training health, efficiency, and correlations.

All plots are saved to --output (default: plots/sampler_analysis/).
The script handles old JSON files (missing ess/kl/sampling_time) gracefully.

Usage:
    python scripts/sampler_analysis.py
    python scripts/sampler_analysis.py --results results/ --model 1d --h 1.0
    python scripts/sampler_analysis.py --size 16 --output plots/analysis_n16/
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Colour + style
# ---------------------------------------------------------------------------

SAMPLER_COLORS = {
    "custom/metropolis":          "#2196F3",
    "custom/simulated_annealing": "#03A9F4",
    "custom/gibbs":               "#00BCD4",
    "custom/sbm":                 "#9C27B0",
    "dimod/simulated_annealing":  "#FF9800",
    "dimod/pegasus":              "#F44336",
    "dimod/zephyr":               "#E91E63",
    "velox/velox":                "#4CAF50",
}
DEFAULT_COLOR = "#888888"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
})


def color(key: str) -> str:
    return SAMPLER_COLORS.get(key, DEFAULT_COLOR)


def save(fig, path: Path, tight=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _safe_mean(lst):
    vals = [v for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def load_results(results_dir: Path, model_filter=None, h_filter=None, size_filter=None) -> pd.DataFrame:
    rows = []
    for path in results_dir.rglob("result_*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict) or "config" not in d:
            continue
        cfg  = d["config"]
        hist = d.get("history", {})

        if model_filter and cfg.get("model") != model_filter:
            continue
        if h_filter is not None and cfg.get("h") != h_filter:
            continue
        if size_filter is not None and cfg.get("size") != size_filter:
            continue

        sk   = f"{cfg.get('sampler','?')}/{cfg.get('sampling_method','?')}"
        n_it = len(hist.get("energy", []))

        # Relative-error trajectory  |E(t) - E_exact| / |E_exact|
        E_ex  = d.get("exact_energy")
        h_rel = []
        if E_ex and E_ex != 0:
            h_rel = [abs(e - E_ex) / abs(E_ex) for e in hist.get("energy", [])]

        # beta_eff_cem: drop None entries, keep (iter, value) pairs
        beff_raw = hist.get("beta_eff_cem", [])
        beff_iters  = [i for i, v in enumerate(beff_raw) if v is not None]
        beff_values = [v for v in beff_raw if v is not None]

        rows.append({
            # identity
            "file":        str(path),
            "sampler_key": sk,
            "sampler":     cfg.get("sampler",          "?"),
            "method":      cfg.get("sampling_method",  "?"),
            "model":       cfg.get("model",            "?"),
            "size":        cfg.get("size",             0),
            "n_hidden":    cfg.get("n_hidden",         0),
            "h":           cfg.get("h",                0.0),
            "lr":          cfg.get("learning_rate",    cfg.get("lr", 0)),
            "seed":        cfg.get("seed",             0),
            "n_samples":   cfg.get("n_samples",        0),
            "rbm":         cfg.get("rbm",              "full"),
            "n_iterations": n_it,
            # scalars
            "final_energy":    d.get("final_energy"),
            "exact_energy":    E_ex,
            "final_error":     d.get("error"),
            "rel_error":       abs(d["error"] / E_ex) if (d.get("error") and E_ex) else None,
            "sampling_time_s": d.get("sampling_time_s"),
            "sparsity":        d.get("sparsity"),
            "final_ess":             d.get("final_ess"),
            "mean_ess":              d.get("mean_ess"),
            "final_kl_exact":        d.get("final_kl_exact"),
            "mean_n_unique_ratio":   _safe_mean(hist.get("n_unique_ratio", [])),
            "final_n_unique_ratio":  hist["n_unique_ratio"][-1] if hist.get("n_unique_ratio") else None,
            # derived scalars from history
            "mean_beta_x":     _safe_mean(hist.get("beta_x", [])),
            "final_beta_x":    hist["beta_x"][-1] if hist.get("beta_x") else None,
            "mean_grad_norm":  _safe_mean(hist.get("grad_norm", [])),
            "final_grad_norm": hist["grad_norm"][-1] if hist.get("grad_norm") else None,
            "mean_cg_iter":    _safe_mean(hist.get("cg_iterations", [])),
            # history arrays
            "h_energy":        hist.get("energy",          []),
            "h_rel_error":     h_rel,
            "h_energy_error":  hist.get("energy_error",    []),
            "h_grad_norm":     hist.get("grad_norm",       []),
            "h_weight_norm":   hist.get("weight_norm",     []),
            "h_beta_x":        hist.get("beta_x",          []),
            "h_cg_iterations": hist.get("cg_iterations",   []),
            "h_ess":              hist.get("ess",              []),
            "h_kl_exact":         hist.get("kl_exact",         []),
            "h_n_unique_ratio":   hist.get("n_unique_ratio",   []),
            "h_sampling_time_s":  hist.get("sampling_time_s",  []),
            "beff_iters":      beff_iters,
            "beff_values":     beff_values,
            # hardware KL/ESS scalars written by eval_kl_hardware.py
            "kl_hardware":        d.get("kl_hardware"),
            "ess_hardware":       d.get("ess_hardware"),
            "kl_metro_baseline":  d.get("kl_metro_baseline"),
            "ess_metro":          d.get("ess_metro"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_band(arrays, lo=25, hi=75):
    """
    Compute median + percentile band from a list of 1-D arrays of possibly different lengths.
    Shorter arrays are NaN-padded so longer runs keep contributing after short ones end.
    """
    if not arrays:
        return None, None, None
    max_len = max(len(a) for a in arrays)
    mat = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        mat[i, : len(a)] = a
    med  = np.nanmedian(mat, axis=0)
    p_lo = np.nanpercentile(mat, lo, axis=0)
    p_hi = np.nanpercentile(mat, hi, axis=0)
    return med, p_lo, p_hi


def legend_handles(keys):
    return [Line2D([0], [0], color=color(k), lw=2, label=k) for k in keys]


# ---------------------------------------------------------------------------
# Figure 1 – Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(df: pd.DataFrame, output_dir: Path):
    """Energy convergence (relative error) per sampler, faceted by h."""
    h_vals = sorted(df["h"].unique())
    n_cols = len(h_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    all_keys = sorted(df["sampler_key"].unique())

    for ax, h in zip(axes, h_vals):
        sub = df[df["h"] == h]
        for sk in all_keys:
            grp = sub[sub["sampler_key"] == sk]
            arrays = [r for r in grp["h_rel_error"] if len(r) > 0]
            if not arrays:
                continue
            med, lo, hi = percentile_band(arrays)
            xs = np.arange(len(med))
            ax.plot(xs, med, color=color(sk), label=sk)
            ax.fill_between(xs, lo, hi, color=color(sk), alpha=0.15)

        ax.set_yscale("log")
        ax.set_title(f"h = {h}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|E − E_exact| / |E_exact|" if h == h_vals[0] else "")
        ax.grid(True, alpha=0.3, which="both")

    fig.legend(handles=legend_handles(all_keys), loc="upper right",
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    fig.suptitle("Convergence to Exact Ground State  (median ± IQR)", fontsize=12)
    save(fig, output_dir / "01_convergence.png")


# ---------------------------------------------------------------------------
# Figure 2 – Final error comparison
# ---------------------------------------------------------------------------

def plot_final_error(df: pd.DataFrame, output_dir: Path):
    """Strip + box: final relative error grouped by sampler, split by h."""
    h_vals = sorted(df["h"].unique())
    n_cols = len(h_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    all_keys = sorted(df["sampler_key"].unique())
    key_to_pos = {k: i for i, k in enumerate(all_keys)}

    for ax, h in zip(axes, h_vals):
        sub = df[(df["h"] == h) & df["rel_error"].notna()]
        data_by_key = {k: sub[sub["sampler_key"] == k]["rel_error"].dropna().tolist()
                       for k in all_keys}

        for k, vals in data_by_key.items():
            if not vals:
                continue
            x = key_to_pos[k]
            ax.boxplot(vals, positions=[x], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=color(k), alpha=0.5),
                       medianprops=dict(color="black", lw=2),
                       whiskerprops=dict(color=color(k)),
                       capprops=dict(color=color(k)),
                       flierprops=dict(markerfacecolor=color(k), markersize=3))

        ax.set_yscale("log")
        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels([k.split("/")[1] for k in all_keys], rotation=30, ha="right")
        ax.set_title(f"h = {h}")
        ax.set_xlabel("")
        ax.set_ylabel("Relative error" if h == h_vals[0] else "")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Final Relative Error  |E − E_exact| / |E_exact|  (all runs)", fontsize=12)
    save(fig, output_dir / "02_final_error.png")


# ---------------------------------------------------------------------------
# Figure 3 – Beta dynamics  (always important per user)
# ---------------------------------------------------------------------------

def plot_beta_dynamics(df: pd.DataFrame, output_dir: Path):
    """
    beta_x history per sampler (heuristic adaptation).
    Overlay beta_eff_cem scatter where available.

    For D-Wave samplers beta_x directly scales QPU couplings — deviations
    from 1.0 indicate effective temperature mismatch.
    For classical samplers (metropolis / SA) beta_x is passed to config
    but ignored by the sampler; it still shows the heuristic's behaviour.
    """
    all_keys = sorted(df["sampler_key"].unique())
    n = len(all_keys)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, sk in zip(axes, all_keys):
        grp = df[df["sampler_key"] == sk]
        # beta_x curves
        bx_arrays = [r for r in grp["h_beta_x"] if len(r) > 0]
        if bx_arrays:
            med, lo, hi = percentile_band(bx_arrays)
            xs = np.arange(len(med))
            ax.plot(xs, med, color=color(sk), label="beta_x (median)")
            ax.fill_between(xs, lo, hi, color=color(sk), alpha=0.15)

        # beta_eff_cem scatter (only non-None values)
        cem_iters = [i for row in grp.itertuples() for i in row.beff_iters]
        cem_vals  = [v for row in grp.itertuples() for v in row.beff_values]
        if cem_iters:
            ax.scatter(cem_iters, cem_vals, color="black", s=12,
                       zorder=5, label="β_eff CEM", alpha=0.7)

        ax.axhline(1.0, color="grey", lw=1, ls="--", alpha=0.6, label="β=1 (ideal)")
        ax.set_title(sk)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("β")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Beta Dynamics  (beta_x adaptation + CEM estimates)", fontsize=12)
    save(fig, output_dir / "03_beta_dynamics.png")


# ---------------------------------------------------------------------------
# Figure 4 – Sample quality  (ESS + KL, subset with new data)
# ---------------------------------------------------------------------------

def plot_sample_quality(df: pd.DataFrame, output_dir: Path):
    """
    Two-section figure:
      Top row  : per-iteration ESS and KL curves (classical samplers, new-format files)
      Bottom row: hardware post-hoc KL and ESS bar charts (all samplers with eval data)
    """
    has_ess = df[df["h_ess"].map(len) > 0]
    has_kl  = df[df["h_kl_exact"].map(lambda x: any(v is not None for v in x))]
    has_hw  = df[df["kl_hardware"].notna()]

    # Build figure: 2 rows × 2 cols; bottom row only shown if hardware data exists
    show_hw = not has_hw.empty
    nrows = 2 if show_hw else 1
    fig = plt.figure(figsize=(13, 4 * nrows))
    gs  = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── top-left: per-iteration ESS curves ───────────────────────────────────
    ax_ess = fig.add_subplot(gs[0, 0])
    if not has_ess.empty:
        for sk in sorted(has_ess["sampler_key"].unique()):
            grp    = has_ess[has_ess["sampler_key"] == sk]
            arrays = [r for r in grp["h_ess"] if len(r) > 0]
            if not arrays:
                continue
            med, lo, hi = percentile_band(arrays)
            xs = np.arange(len(med))
            ax_ess.plot(xs, med, color=color(sk), label=sk)
            ax_ess.fill_between(xs, lo, hi, color=color(sk), alpha=0.15)
    ax_ess.set_xlabel("Iteration")
    ax_ess.set_ylabel("ESS / n  [0, 1]")
    ax_ess.set_title("ESS per Iteration  (classical, new-format only)")
    ax_ess.set_ylim(0, 1.05)
    ax_ess.legend(fontsize=7)
    ax_ess.grid(True, alpha=0.3)

    # ── top-right: per-iteration KL curves ───────────────────────────────────
    ax_kl = fig.add_subplot(gs[0, 1])
    if not has_kl.empty:
        for sk in sorted(has_kl["sampler_key"].unique()):
            grp = has_kl[has_kl["sampler_key"] == sk]
            kl_arrays = []
            for row in grp.itertuples():
                clean = [v for v in row.h_kl_exact if v is not None]
                if clean:
                    kl_arrays.append(clean)
            if not kl_arrays:
                continue
            med, lo, hi = percentile_band(kl_arrays)
            xs = np.arange(len(med))
            ax_kl.plot(xs, med, color=color(sk), label=sk)
            ax_kl.fill_between(xs, lo, hi, color=color(sk), alpha=0.15)
    ax_kl.set_xlabel("Iteration")
    ax_kl.set_ylabel("KL(q ‖ p_exact)")
    ax_kl.set_title("KL per Iteration  (N ≤ 16, classical only)")
    ax_kl.legend(fontsize=7)
    ax_kl.grid(True, alpha=0.3)

    if not show_hw:
        fig.suptitle("Sample Quality Metrics  (median ± IQR)", fontsize=12)
        save(fig, output_dir / "04_sample_quality.png")
        return

    # ── bottom-left: hardware KL bar chart ───────────────────────────────────
    ax_hwkl = fig.add_subplot(gs[1, 0])
    # Aggregate per sampler_key: median kl_hardware and kl_metro_baseline
    hw_keys = sorted(has_hw["sampler_key"].unique())
    x_pos   = np.arange(len(hw_keys))
    bar_w   = 0.35

    kl_hw_meds    = []
    kl_metro_meds = []
    for sk in hw_keys:
        grp = has_hw[has_hw["sampler_key"] == sk]
        kl_hw_meds.append(float(np.nanmedian(grp["kl_hardware"].dropna())))
        mb = grp["kl_metro_baseline"].dropna()
        kl_metro_meds.append(float(np.nanmedian(mb)) if not mb.empty else np.nan)

    bars_hw = ax_hwkl.bar(x_pos - bar_w / 2, kl_hw_meds, bar_w,
                          color=[color(sk) for sk in hw_keys], alpha=0.85,
                          label="hardware")
    bars_mt = ax_hwkl.bar(x_pos + bar_w / 2, kl_metro_meds, bar_w,
                          color=[color(sk) for sk in hw_keys], alpha=0.35,
                          hatch="///", label="metropolis baseline")
    ax_hwkl.set_xticks(x_pos)
    ax_hwkl.set_xticklabels([sk.split("/")[1] for sk in hw_keys], rotation=20, ha="right")
    ax_hwkl.set_ylabel("KL(q ‖ p_exact)  [post-hoc]")
    ax_hwkl.set_title("Hardware KL vs Metropolis Baseline  (post-hoc eval)")
    ax_hwkl.legend(fontsize=8)
    ax_hwkl.grid(True, axis="y", alpha=0.3)

    # annotate bar tops
    for bar, val in zip(bars_hw, kl_hw_meds):
        if not np.isnan(val):
            ax_hwkl.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # ── bottom-right: hardware ESS bar chart ─────────────────────────────────
    ax_hwess = fig.add_subplot(gs[1, 1])
    ess_hw_meds    = []
    ess_metro_meds = []
    for sk in hw_keys:
        grp = has_hw[has_hw["sampler_key"] == sk]
        ess_hw_meds.append(float(np.nanmedian(grp["ess_hardware"].dropna())))
        em = grp["ess_metro"].dropna()
        ess_metro_meds.append(float(np.nanmedian(em)) if not em.empty else np.nan)

    ax_hwess.bar(x_pos - bar_w / 2, ess_hw_meds, bar_w,
                 color=[color(sk) for sk in hw_keys], alpha=0.85, label="hardware")
    ax_hwess.bar(x_pos + bar_w / 2, ess_metro_meds, bar_w,
                 color=[color(sk) for sk in hw_keys], alpha=0.35,
                 hatch="///", label="metropolis baseline")
    ax_hwess.set_xticks(x_pos)
    ax_hwess.set_xticklabels([sk.split("/")[1] for sk in hw_keys], rotation=20, ha="right")
    ax_hwess.set_ylabel("ESS / n  [0, 1]  (post-hoc)")
    ax_hwess.set_title("Hardware ESS vs Metropolis Baseline  (post-hoc eval)")
    ax_hwess.set_ylim(0, 1.05)
    ax_hwess.legend(fontsize=8)
    ax_hwess.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Sample Quality Metrics", fontsize=12)
    save(fig, output_dir / "04_sample_quality.png")


# ---------------------------------------------------------------------------
# Figure 5 – Convergence vs sample quality
# ---------------------------------------------------------------------------

def plot_convergence_vs_quality(df: pd.DataFrame, output_dir: Path):
    """
    Scatter plots that link sample-quality metrics to convergence speed:
      - mean_ess vs final_rel_error
      - mean_beta_x vs final_rel_error
      - sampling_time_s vs final_rel_error
    """
    all_keys = sorted(df["sampler_key"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        ("mean_ess",        "Mean ESS",              "x"),
        ("mean_beta_x",     "Mean beta_x",           "x"),
        ("sampling_time_s", "Total sampling time (s)", "x"),
    ]

    for ax, (xcol, xlabel, _) in zip(axes, panels):
        sub = df[df["rel_error"].notna() & df[xcol].notna()]
        for sk in all_keys:
            grp = sub[sub["sampler_key"] == sk]
            if grp.empty:
                continue
            ax.scatter(grp[xcol], grp["rel_error"],
                       color=color(sk), label=sk, s=25, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final relative error")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{xlabel} vs Convergence")

    axes[0].set_xscale("linear")
    # Add legend to last panel
    axes[2].legend(fontsize=7, loc="upper left")
    fig.suptitle("Sample Quality & Cost vs Final Convergence", fontsize=12)
    save(fig, output_dir / "05_quality_vs_convergence.png")


# ---------------------------------------------------------------------------
# Figure 6 – Training health
# ---------------------------------------------------------------------------

def plot_training_health(df: pd.DataFrame, output_dir: Path):
    """Gradient norm decay, weight norm growth, CG iterations per sampler."""
    all_keys = sorted(df["sampler_key"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("h_grad_norm",     "Gradient norm",   "log"),
        ("h_weight_norm",   "Weight norm",     "linear"),
        ("h_cg_iterations", "CG iterations",   "linear"),
    ]

    for ax, (col, ylabel, yscale) in zip(axes, metrics):
        for sk in all_keys:
            grp = df[df["sampler_key"] == sk]
            arrays = [r for r in grp[col] if len(r) > 0]
            if not arrays:
                continue
            med, lo, hi = percentile_band(arrays)
            xs = np.arange(len(med))
            ax.plot(xs, med, color=color(sk), label=sk)
            ax.fill_between(xs, lo, hi, color=color(sk), alpha=0.12)
        ax.set_yscale(yscale)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3, which="both")

    axes[2].legend(fontsize=7)
    fig.suptitle("Training Health Diagnostics  (median ± IQR)", fontsize=12)
    save(fig, output_dir / "06_training_health.png")


# ---------------------------------------------------------------------------
# Figure 7 – Efficiency frontier
# ---------------------------------------------------------------------------

def plot_efficiency(df: pd.DataFrame, output_dir: Path):
    """Error vs sampling time — efficiency Pareto view."""
    sub = df[df["sampling_time_s"].notna() & df["rel_error"].notna()]
    if sub.empty:
        print("  [skip] No sampling_time_s data found.")
        return

    all_keys = sorted(sub["sampler_key"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for sk in all_keys:
        grp = sub[sub["sampler_key"] == sk]
        ax.scatter(grp["sampling_time_s"], grp["rel_error"],
                   color=color(sk), label=sk, s=30, alpha=0.75)

    ax.set_xlabel("Total sampling time (s)")
    ax.set_ylabel("Final relative error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Efficiency Frontier: Sampling Cost vs Accuracy")
    save(fig, output_dir / "07_efficiency.png")


# ---------------------------------------------------------------------------
# Figure 8 – Scale analysis (error vs N)
# ---------------------------------------------------------------------------

def plot_scale(df: pd.DataFrame, output_dir: Path):
    """Final relative error vs system size N, one line per sampler."""
    all_keys = sorted(df["sampler_key"].unique())
    h_vals   = sorted(df["h"].unique())
    n_cols   = len(h_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, h in zip(axes, h_vals):
        sub = df[(df["h"] == h) & df["rel_error"].notna()]
        for sk in all_keys:
            grp = sub[sub["sampler_key"] == sk].groupby("size")["rel_error"]
            if grp.ngroups == 0:
                continue
            sizes  = sorted(grp.groups.keys())
            medians = [grp.get_group(s).median() for s in sizes]
            ax.plot(sizes, medians, "o-", color=color(sk), label=sk)
        ax.set_xlabel("System size N")
        ax.set_ylabel("Final relative error" if h == h_vals[0] else "")
        ax.set_yscale("log")
        ax.set_title(f"h = {h}")
        ax.grid(True, alpha=0.3, which="both")

    fig.legend(handles=legend_handles(all_keys), loc="upper right",
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Scaling: Convergence vs System Size", fontsize=12)
    save(fig, output_dir / "08_scale.png")


# ---------------------------------------------------------------------------
# Figure 10 – beta_x deep-dive
# ---------------------------------------------------------------------------

def plot_beta_x_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Three-panel deep-dive into beta_x behaviour across samplers:
      Left   : Box plot — distribution of final_beta_x per sampler
      Middle : Scatter — mean_beta_x vs final relative error
      Right  : Line — median beta_x trajectory per sampler (same as 03 but
               overlaid on one axis for direct comparison)
    """
    all_keys = sorted(df["sampler_key"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── panel 1: distribution of final_beta_x per sampler ────────────────
    ax = axes[0]
    sub_bx = df[df["final_beta_x"].notna()]
    keys_with_bx = [k for k in all_keys
                    if not sub_bx[sub_bx["sampler_key"] == k]["final_beta_x"].empty]
    box_data   = [sub_bx[sub_bx["sampler_key"] == k]["final_beta_x"].tolist()
                  for k in keys_with_bx]
    if box_data:
        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        for patch, sk in zip(bp["boxes"], keys_with_bx):
            patch.set_facecolor(color(sk))
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps", "fliers"]:
            for item, sk in zip(
                bp[element][::2 if element != "fliers" else 1],
                keys_with_bx * (2 if element != "fliers" else 1),
            ):
                item.set_color(color(sk if element != "fliers" else keys_with_bx[0]))
        ax.set_xticks(range(1, len(keys_with_bx) + 1))
        ax.set_xticklabels([k.split("/")[1] for k in keys_with_bx],
                           rotation=25, ha="right")
    ax.axhline(1.0, color="grey", lw=1, ls="--", alpha=0.6, label="β=1")
    ax.set_ylabel("final beta_x")
    ax.set_title("Final beta_x Distribution per Sampler")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # ── panel 2: mean_beta_x vs convergence scatter ───────────────────────
    ax = axes[1]
    sub_sc = df[df["mean_beta_x"].notna() & df["rel_error"].notna()]
    for sk in all_keys:
        grp = sub_sc[sub_sc["sampler_key"] == sk]
        if grp.empty:
            continue
        ax.scatter(grp["mean_beta_x"], grp["rel_error"],
                   color=color(sk), label=sk, s=28, alpha=0.75)
    ax.set_xlabel("mean beta_x (over training)")
    ax.set_ylabel("Final relative error")
    ax.set_yscale("log")
    ax.set_title("mean beta_x vs Final Error")
    ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ── panel 3: median trajectory per sampler (overlaid) ─────────────────
    ax = axes[2]
    for sk in all_keys:
        grp    = df[df["sampler_key"] == sk]
        arrays = [r for r in grp["h_beta_x"] if len(r) > 0]
        if not arrays:
            continue
        med, lo, hi = percentile_band(arrays)
        xs = np.arange(len(med))
        ax.plot(xs, med, color=color(sk), label=sk)
        ax.fill_between(xs, lo, hi, color=color(sk), alpha=0.12)
    ax.axhline(1.0, color="grey", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("beta_x")
    ax.set_title("beta_x Trajectory  (median ± IQR, all samplers)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("beta_x Analysis: Distribution, Correlation with Convergence, and Dynamics",
                 fontsize=12)
    save(fig, output_dir / "10_beta_x_analysis.png")


# ---------------------------------------------------------------------------
# Figure 9 – Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlations(df: pd.DataFrame, output_dir: Path):
    """Pearson correlations between all scalar metrics and final relative error."""
    numeric_cols = [
        "final_error", "rel_error", "mean_beta_x", "final_beta_x",
        "mean_grad_norm", "final_grad_norm", "mean_cg_iter",
        "mean_ess", "final_ess", "final_kl_exact",
        "sampling_time_s", "sparsity", "n_iterations",
    ]
    sub = df[[c for c in numeric_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(how="all")
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    # annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.7 else "white")
    ax.set_title("Pearson Correlations Between All Recorded Metrics", fontsize=11)
    save(fig, output_dir / "09_correlations.png")


# ---------------------------------------------------------------------------
# Figure 10 – Per-sampler summary card
# ---------------------------------------------------------------------------

def plot_summary_cards(df: pd.DataFrame, output_dir: Path):
    """
    One row per sampler: n_runs, median final error, median sampling time,
    median mean_ess, median final_beta_x.  Visual table.
    """
    all_keys = sorted(df["sampler_key"].unique())
    stats = []
    for sk in all_keys:
        grp = df[df["sampler_key"] == sk]
        stats.append({
            "sampler":          sk,
            "n_runs":           len(grp),
            "med_rel_err":      grp["rel_error"].median(),
            "med_samp_time":    grp["sampling_time_s"].median(),
            "med_mean_ess":     grp["mean_ess"].median(),
            "med_final_bx":     grp["final_beta_x"].median(),
            "med_mean_cg":      grp["mean_cg_iter"].median(),
        })
    stat_df = pd.DataFrame(stats).set_index("sampler")

    col_labels = ["n runs", "median\nrel.error", "median\nsampling_time (s)",
                  "median\nmean ESS", "median\nbeta_x (final)", "median\nCG iters"]
    cell_vals = []
    for _, row in stat_df.iterrows():
        cell_vals.append([
            f"{row['n_runs']:.0f}",
            f"{row['med_rel_err']:.3e}" if not np.isnan(row['med_rel_err']) else "—",
            f"{row['med_samp_time']:.1f}" if not np.isnan(row['med_samp_time']) else "—",
            f"{row['med_mean_ess']:.3f}" if not np.isnan(row['med_mean_ess']) else "—",
            f"{row['med_final_bx']:.3f}" if not np.isnan(row['med_final_bx']) else "—",
            f"{row['med_mean_cg']:.1f}" if not np.isnan(row['med_mean_cg']) else "—",
        ])

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(all_keys) + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_vals,
        rowLabels=all_keys,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    # Colour row headers by sampler
    for i, sk in enumerate(all_keys):
        tbl[(i + 1, -1)].set_facecolor(color(sk))
        tbl[(i + 1, -1)].get_text().set_color("white")
    ax.set_title("Sampler Summary Statistics", fontsize=12, pad=12)
    save(fig, output_dir / "00_summary.png", tight=False)


# ---------------------------------------------------------------------------
# Figure 11 – Sample diversity (unique configs)
# ---------------------------------------------------------------------------

def plot_sample_diversity(df: pd.DataFrame, output_dir: Path):
    """
    Two panels:
      Left : unique-config ratio trajectory per sampler (median ± IQR)
      Right: scatter — mean unique ratio vs final relative error
    """
    has_uniq = df[df["h_n_unique_ratio"].map(len) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── left: trajectory ──────────────────────────────────────────────────
    ax = axes[0]
    all_keys = sorted(df["sampler_key"].unique())
    for sk in all_keys:
        grp    = has_uniq[has_uniq["sampler_key"] == sk] if not has_uniq.empty else df.iloc[:0]
        arrays = [r for r in grp["h_n_unique_ratio"] if len(r) > 0]
        if not arrays:
            continue
        med, lo, hi = percentile_band(arrays)
        xs = np.arange(len(med))
        ax.plot(xs, med, color=color(sk), label=sk)
        ax.fill_between(xs, lo, hi, color=color(sk), alpha=0.15)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Unique configs / n_samples  [0, 1]")
    ax.set_title("Sample Diversity per Iteration  (median ± IQR)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── right: scatter vs convergence ────────────────────────────────────
    ax = axes[1]
    sub = df[df["mean_n_unique_ratio"].notna() & df["rel_error"].notna()]
    for sk in all_keys:
        grp = sub[sub["sampler_key"] == sk]
        if grp.empty:
            continue
        ax.scatter(grp["mean_n_unique_ratio"], grp["rel_error"],
                   color=color(sk), label=sk, s=28, alpha=0.75)
    ax.set_xlabel("Mean unique-config ratio (over training)")
    ax.set_ylabel("Final relative error")
    ax.set_yscale("log")
    ax.set_title("Sample Diversity vs Convergence")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Sample Diversity: Unique Configurations per Batch", fontsize=12)
    save(fig, output_dir / "11_sample_diversity.png")


# ---------------------------------------------------------------------------
# Figure 12 – Error diagnostics (grad norm + KL)
# ---------------------------------------------------------------------------

def plot_error_diagnostics(df: pd.DataFrame, output_dir: Path):
    """
    Two scatter panels probing what correlates with the final relative error:
      Left : final_grad_norm vs rel_error  — large grad norm → did SR converge?
      Right: final_kl_exact   vs rel_error — better samples → better energy?
    Each point is one run, coloured by sampler.
    Includes a log-log regression line per panel to make the trend explicit.
    """
    all_keys = sorted(df["sampler_key"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panels = [
        ("final_grad_norm", "Final gradient norm ‖∇‖",   "log"),
        ("final_kl_exact",  "Final KL(q ‖ p_exact)",     "log"),
    ]

    for ax, (xcol, xlabel, xscale) in zip(axes, panels):
        sub = df[df["rel_error"].notna() & df[xcol].notna()].copy()
        if sub.empty:
            ax.set_title(f"{xlabel} — no data")
            continue

        for sk in all_keys:
            grp = sub[sub["sampler_key"] == sk]
            if grp.empty:
                continue
            ax.scatter(grp[xcol], grp["rel_error"],
                       color=color(sk), label=sk, s=28, alpha=0.75)

        # log-log regression over all points (both axes positive)
        valid = sub[(sub[xcol] > 0) & (sub["rel_error"] > 0)]
        if len(valid) >= 5:
            lx = np.log10(valid[xcol].astype(float).values)
            ly = np.log10(valid["rel_error"].astype(float).values)
            slope, intercept = np.polyfit(lx, ly, 1)
            x_range = np.logspace(np.log10(valid[xcol].min()), np.log10(valid[xcol].max()), 60)
            ax.plot(x_range, 10 ** (slope * np.log10(x_range) + intercept),
                    "k--", lw=1.5, alpha=0.6,
                    label=f"log-log slope={slope:.2f}")

        ax.set_xscale(xscale)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final relative error")
        ax.set_title(f"{xlabel} vs Error")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Error Diagnostics: Gradient Norm and KL vs Convergence", fontsize=12)
    save(fig, output_dir / "12_error_diagnostics.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sampler analysis plots")
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="Path to results directory")
    parser.add_argument("--output",  type=Path, default=Path("plots/sampler_analysis"),
                        help="Output directory for plots")
    parser.add_argument("--model",   default=None, help="Filter by model (1d/2d)")
    parser.add_argument("--h",       type=float, default=None, help="Filter by h value")
    parser.add_argument("--size",    type=int,   default=None, help="Filter by system size")
    args = parser.parse_args()

    print(f"Loading results from {args.results} ...")
    df = load_results(args.results, args.model, args.h, args.size)
    if df.empty:
        print("No results found — check --results path and filters.")
        return
    print(f"Loaded {len(df)} experiments across samplers: {sorted(df['sampler_key'].unique())}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots:")
    plot_summary_cards(df, out)           # 00 — overview table
    plot_convergence(df, out)             # 01 — energy convergence
    plot_final_error(df, out)             # 02 — final error box
    plot_beta_dynamics(df, out)           # 03 — beta_x + beta_eff_cem per sampler
    plot_sample_quality(df, out)          # 04 — ESS + KL (training + hardware post-hoc)
    plot_convergence_vs_quality(df, out)  # 05 — scatter: quality vs error
    plot_training_health(df, out)         # 06 — grad/weight/CG curves
    plot_efficiency(df, out)              # 07 — error vs sampling time
    plot_scale(df, out)                   # 08 — error vs system size N
    plot_correlations(df, out)            # 09 — correlation heatmap
    plot_beta_x_analysis(df, out)         # 10 — beta_x distribution, scatter, trajectory
    plot_sample_diversity(df, out)        # 11 — unique configs / batch + correlation with error
    plot_error_diagnostics(df, out)       # 12 — grad norm + KL vs rel_error with regression

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
