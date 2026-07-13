#!/usr/bin/env python3
"""
CEM validation figures from cem_validation_sweep.py's JSON output.

Produces two SEPARATE standalone figures (matching the convention used by
cem_matching_demo.py / plots/cem/):

  cem_validation_calibration : CEM's beta_eff estimate vs. the exact
      ground-truth beta_eff (KL-argmin against the true visible marginal),
      pooled over all (h, beta_x) draws, colored by system size N. A
      perfect estimator lies on the dashed y=x line. Draws where CEM's
      optimizer saturated its search bound are marked with an "x" rather
      than folded into the main scatter.
  cem_validation_bias : mean(beta_cem - beta_ground_truth) +/- 95% CI at
      each training checkpoint (early/mid/late), one line per N, averaged
      over h and beta_x -- shows whether CEM's reliability holds up across
      a full optimization run.

Usage:
    python scripts/viz/plot_cem_validation.py
    python scripts/viz/plot_cem_validation.py --input plots/cem/cem_validation_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import matplotlib.pyplot as plt
from plot_style import setup_style

_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = _ROOT / "plots" / "cem"
_DEFAULT_INPUT = PLOTS_DIR / "cem_validation_results.json"

_N_COLORS = {8: "#2563eb", 12: "#16a34a", 16: "#d97706"}
_CKPT_ORDER = ["early", "mid", "late"]
_CEM_BOUNDS = (0.01, 50.0)  # must match encoder.estimate_beta_eff_cem's minimize_scalar bounds


def _ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _save(fig, fname):
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_calibration(records):
    setup_style(fontsize=9)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    ns = sorted({r["N"] for r in records})
    n_saturated = 0
    for N in ns:
        rows = [r for r in records if r["N"] == N]
        gt = np.array([r["beta_ground_truth"] for r in rows])
        cem = np.array([r["beta_cem"] for r in rows])
        saturated = (cem > 0.98 * _CEM_BOUNDS[1]) | (cem < 1.02 * _CEM_BOUNDS[0])
        n_saturated += int(saturated.sum())
        color = _N_COLORS.get(N, "#888")

        ax.scatter(gt[~saturated], cem[~saturated], s=10, alpha=0.55, color=color,
                   label=f"$N={N}$", edgecolors="none")
        if saturated.any():
            ax.scatter(gt[saturated], cem[saturated], s=32, alpha=0.9, color=color,
                       marker="x", linewidths=1.3,
                       label=f"$N={N}$ (hit search bound)" if N == ns[-1] else None)

        if len(gt) > 1:
            r = np.corrcoef(gt, cem)[0, 1]
            rmse = float(np.sqrt(np.mean((cem - gt) ** 2)))
            rmse_clean = (float(np.sqrt(np.mean((cem[~saturated] - gt[~saturated]) ** 2)))
                          if (~saturated).any() else float("nan"))
            print(f"  N={N}: r={r:.3f}  RMSE={rmse:.3f}  "
                  f"RMSE(excl. bound-saturated)={rmse_clean:.3f}  "
                  f"n={len(gt)}  saturated={int(saturated.sum())}")

    if n_saturated:
        print(f"  Total bound-saturated draws: {n_saturated}/{len(records)}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    lo = min(min(r["beta_ground_truth"], r["beta_cem"]) for r in records) * 0.8
    hi = max(max(r["beta_ground_truth"], r["beta_cem"]) for r in records) * 1.25
    ax.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=1.1, label="$y=x$ (perfect)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\beta_\mathrm{eff}$ (ground truth, exact KL)")
    ax.set_ylabel(r"$\beta_\mathrm{eff}$ (CEM)")
    ax.legend(fontsize=6, loc="lower right", frameon=True, edgecolor="black",
              handlelength=1.6, borderpad=0.4)

    _save(fig, "cem_validation_calibration")


def plot_bias_vs_checkpoint(records):
    setup_style(fontsize=9)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    ns = sorted({r["N"] for r in records})
    x = np.arange(len(_CKPT_ORDER))
    for i, N in enumerate(ns):
        means, cis = [], []
        for ckpt in _CKPT_ORDER:
            cell = [r for r in records if r["N"] == N and r["checkpoint"] == ckpt]
            cem_vals = np.array([r["beta_cem"] for r in cell])
            saturated = (cem_vals > 0.98 * _CEM_BOUNDS[1]) | (cem_vals < 1.02 * _CEM_BOUNDS[0])
            bias = [c["beta_cem"] - c["beta_ground_truth"]
                    for c, sat in zip(cell, saturated) if not sat]
            means.append(np.mean(bias) if bias else np.nan)
            cis.append(_ci95(bias) if bias else 0.0)
            if saturated.any():
                print(f"  N={N} checkpoint={ckpt}: excluded {int(saturated.sum())} "
                      f"bound-saturated draw(s) from bias/CI")
        color = _N_COLORS.get(N, "#888")
        offset = (i - (len(ns) - 1) / 2) * 0.08
        ax.errorbar(x + offset, means, yerr=cis, fmt="o-", color=color,
                    label=f"$N={N}$", capsize=3, linewidth=1.5, markersize=4)

    ax.axhline(0.0, color="#333", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in _CKPT_ORDER])
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel(r"Bias: $\beta_\mathrm{eff}^\mathrm{CEM} - \beta_\mathrm{eff}^\mathrm{exact}$")
    ax.legend(fontsize=7, loc="best", frameon=True, edgecolor="black")

    _save(fig, "cem_validation_bias")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_DEFAULT_INPUT))
    args = p.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"No records found in {args.input}")
    print(f"Loaded {len(records)} records from {args.input}")

    plot_calibration(records)
    plot_bias_vs_checkpoint(records)


if __name__ == "__main__":
    main()
