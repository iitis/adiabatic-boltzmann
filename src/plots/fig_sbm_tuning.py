#!/usr/bin/env python3
"""
Analyze results/sbm_tune.json — produced by scripts/run_sbm_hyperparameter_sweep.py.
Run from src/:  python plots/fig_sbm_tuning.py
Plots saved to: <repo>/plots/sbm_tune/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent.parent
IN_JSON = ROOT / "results" / "sbm_tune.json"
OUT_DIR = ROOT / "figures" / "fig_sbm_tuning"

# ── load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    raw = json.loads(IN_JSON.read_text())
    df  = pd.DataFrame([{k: v for k, v in r.items() if k != "energy_curve"} for r in raw])
    ok  = df[df.rel_error <= 100].copy()
    print(f"Loaded {len(df)} runs — "
          f"{(df.rel_error > 100).sum()} diverged, "
          f"{len(ok)} converged.")
    return df, ok, raw


# ── style constants ───────────────────────────────────────────────────────────

STEPS = [100, 300, 500, 1000, 2000, 5000]

COMBOS = [
    ("ballistic", False, "#1f77b4", "-",  "ballistic / cold"),
    ("ballistic", True,  "#1f77b4", "--", "ballistic / heated"),
    ("discrete",  False, "#d62728", "-",  "discrete / cold"),
    ("discrete",  True,  "#d62728", "--", "discrete / heated"),
]

COMBOS_CEM = [
    ("ballistic", False, True,  "#1f77b4", "ball/cold/CEM"),
    ("ballistic", False, False, "#aec7e8", "ball/cold"),
    ("ballistic", True,  False, "#ffbb78", "ball/hot"),
    ("ballistic", True,  True,  "#ff7f0e", "ball/hot/CEM"),
    ("discrete",  False, True,  "#d62728", "disc/cold/CEM"),
    ("discrete",  False, False, "#f7b6b6", "disc/cold"),
    ("discrete",  True,  False, "#c5b0d5", "disc/hot"),
    ("discrete",  True,  True,  "#9467bd", "disc/hot/CEM"),
]


def sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Filter df by keyword conditions, handling bool columns correctly."""
    mask = pd.Series(True, index=df.index)
    for col, val in kw.items():
        if isinstance(val, bool):
            mask &= df[col].astype(bool) == val
        else:
            mask &= df[col] == val
    return df[mask]


def savefig(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── Figure 1: 2×3 overview ────────────────────────────────────────────────────

def fig_overview(df: pd.DataFrame, ok: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"SBM hyperparameter sweep — {len(df)} runs, 50 iterations each",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # 1a — leaderboard box plot (mode × heated × CEM)
    ax = fig.add_subplot(gs[0, 0])
    data = [
        sel(ok, mode=m, heated=h, cem=c)["rel_error"].values
        for m, h, c, _, _ in COMBOS_CEM
    ]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, (*_, color, _) in zip(bp["boxes"], COMBOS_CEM):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(COMBOS_CEM) + 1))
    ax.set_xticklabels([lbl for *_, lbl in COMBOS_CEM],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("rel. error (%)")
    ax.set_title("Mode × heated × CEM")
    ax.grid(True, alpha=0.3, axis="y")

    # 1b — error vs max_steps
    ax = fig.add_subplot(gs[0, 1])
    for mode, heated, color, ls, label in COMBOS:
        medians = [
            sel(ok, mode=mode, heated=heated, max_steps=s)["rel_error"].median()
            for s in STEPS
        ]
        ax.plot(STEPS, medians, color=color, linestyle=ls,
                marker="o", linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("max_steps")
    ax.set_ylabel("median rel. error (%)")
    ax.set_title("Error vs max_steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # 1c — divergence rate vs max_steps
    ax = fig.add_subplot(gs[0, 2])
    for mode, heated, color, ls, label in COMBOS:
        rates = [
            (sel(df, mode=mode, heated=heated, max_steps=s)["rel_error"] > 100).mean()
            for s in STEPS
        ]
        ax.plot(STEPS, rates, color=color, linestyle=ls,
                marker="o", linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("max_steps")
    ax.set_ylabel("divergence rate")
    ax.set_title("Divergence rate vs max_steps")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, which="both")

    # 1d — error vs h
    ax = fig.add_subplot(gs[1, 0])
    for mode, heated, color, ls, label in COMBOS:
        sub = sel(ok, mode=mode, heated=heated)
        medians = [sub[sub.h == h]["rel_error"].median() for h in [0.5, 1.0, 2.0]]
        ax.plot([0.5, 1.0, 2.0], medians, color=color, linestyle=ls,
                marker="o", linewidth=2, label=label)
    ax.set_xlabel("h (transverse field)")
    ax.set_ylabel("median rel. error (%)")
    ax.set_title("Error vs h")
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1e — error vs system size
    ax = fig.add_subplot(gs[1, 1])
    for mode, heated, color, ls, label in COMBOS:
        sub = sel(ok, mode=mode, heated=heated)
        medians = [sub[sub["size"] == s]["rel_error"].median() for s in [4, 6, 8, 16]]
        ax.plot([4, 6, 8, 16], medians, color=color, linestyle=ls,
                marker="o", linewidth=2, label=label)
    ax.set_xlabel("system size N")
    ax.set_ylabel("median rel. error (%)")
    ax.set_title("Error vs system size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1f — sample diversity vs max_steps
    ax = fig.add_subplot(gs[1, 2])
    for mode, heated, color, ls, label in COMBOS:
        vals = [
            sel(ok, mode=mode, heated=heated, max_steps=s)["mean_unique"].median()
            for s in STEPS
        ]
        ax.plot(STEPS, vals, color=color, linestyle=ls,
                marker="o", linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("max_steps")
    ax.set_ylabel("median unique sample ratio")
    ax.set_title("Sample diversity vs max_steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    savefig(fig, "sbm_tune_overview.png")


# ── Figure 2: heatmaps (heated × max_steps) ───────────────────────────────────

def fig_heatmaps(ok: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SBM — median rel. error (%) heatmaps", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "ballistic", False, "Ballistic — no CEM"),
        (axes[0, 1], "ballistic", True,  "Ballistic — with CEM"),
        (axes[1, 0], "discrete",  False, "Discrete — no CEM"),
        (axes[1, 1], "discrete",  True,  "Discrete — with CEM"),
    ]
    for ax, mode, cem, title in panels:
        sub   = sel(ok, mode=mode, cem=cem)
        pivot = sub.groupby(["heated", "max_steps"])["rel_error"].median().unstack()
        im    = ax.imshow(pivot.values, aspect="auto",
                          cmap="RdYlGn_r", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist())
        ytick_labels = ["heated" if v else "cold" for v in pivot.index.tolist()]
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel("max_steps")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=9, color="white" if v > 60 else "black")

    fig.tight_layout()
    savefig(fig, "sbm_tune_heatmaps.png")


# ── Figure 3: convergence curves for best runs ────────────────────────────────

def fig_convergence(raw: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergence curves — best runs (rel. error < 20%)",
                 fontsize=13, fontweight="bold")

    for ax, model_filter, title in [(axes[0], "1d", "1D"), (axes[1], "2d", "2D")]:
        candidates = sorted(
            [r for r in raw if r["model"] == model_filter and r["rel_error"] <= 20],
            key=lambda x: x["rel_error"],
        )[:16]
        plotted = 0
        for r in candidates:
            curve = r["energy_curve"]
            if not curve:
                continue
            exact     = r["exact_energy"]
            rel_curve = [abs(e - exact) / abs(exact) * 100 for e in curve]
            label     = (f"{r['mode'][:4]}/"
                         f"{'hot' if r['heated'] else 'cold'}/"
                         f"s={r['max_steps']}")
            color = "#1f77b4" if r["mode"] == "ballistic" else "#d62728"
            ls    = "--" if r["heated"] else "-"
            ax.plot(rel_curve, color=color, linestyle=ls,
                    linewidth=1.2, alpha=0.7, label=label)
            plotted += 1
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("rel. error (%)")
        ax.set_title(f"{title} — runs with error < 20%  (n={plotted})")
        ax.grid(True, alpha=0.3, which="both")
        if plotted:
            ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    savefig(fig, "sbm_tune_convergence.png")


# ── Figure 4: CEM deep-dive ───────────────────────────────────────────────────

def fig_cem(ok: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CEM effect across all configurations",
                 fontsize=13, fontweight="bold")

    # 4a — CEM vs no-CEM overall
    ax = axes[0]
    for i, (cem, label) in enumerate([(False, "no CEM"), (True, "CEM")]):
        vals = sel(ok, cem=cem)["rel_error"]
        ax.boxplot(vals, positions=[i], patch_artist=True,
                   medianprops=dict(color="black", linewidth=2),
                   boxprops=dict(facecolor="#aec7e8" if not cem else "#ff7f0e",
                                 alpha=0.8))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no CEM", "CEM"])
    ax.set_ylabel("rel. error (%)")
    ax.set_title("Overall CEM effect")
    ax.grid(True, alpha=0.3, axis="y")

    # 4b — CEM vs no-CEM per mode
    ax = axes[1]
    x = np.arange(2)  # ballistic, discrete
    w = 0.35
    for i, (cem, label, color) in enumerate(
        [(False, "no CEM", "#aec7e8"), (True, "CEM", "#ff7f0e")]
    ):
        means = [
            sel(ok, cem=cem, mode=m)["rel_error"].median()
            for m in ["ballistic", "discrete"]
        ]
        ax.bar(x + i * w, means, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["ballistic", "discrete"])
    ax.set_ylabel("median rel. error (%)")
    ax.set_title("CEM effect per mode")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4c — CEM effect on diversity
    ax = axes[2]
    for i, (cem, label, color) in enumerate(
        [(False, "no CEM", "#aec7e8"), (True, "CEM", "#ff7f0e")]
    ):
        vals = sel(ok, cem=cem)["mean_unique"]
        ax.boxplot(vals, positions=[i], patch_artist=True,
                   medianprops=dict(color="black", linewidth=2),
                   boxprops=dict(facecolor=color, alpha=0.8))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no CEM", "CEM"])
    ax.set_ylabel("mean unique sample ratio")
    ax.set_title("CEM effect on sample diversity")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, "sbm_tune_cem.png")


# ── Figure 5: 1D vs 2D breakdown ──────────────────────────────────────────────

def fig_1d_vs_2d(ok: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("1D vs 2D model — error per SBM config",
                 fontsize=13, fontweight="bold")

    for ax, dim in zip(axes, ["1d", "2d"]):
        sub = ok[ok.model == dim]
        data = [
            sel(sub, mode=m, heated=h, cem=c)["rel_error"].values
            for m, h, c, _, _ in COMBOS_CEM
        ]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, (*_, color, _) in zip(bp["boxes"], COMBOS_CEM):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(COMBOS_CEM) + 1))
        ax.set_xticklabels([lbl for *_, lbl in COMBOS_CEM],
                           rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel("rel. error (%)")
        ax.set_title(f"{dim.upper()} model")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, "sbm_tune_1d_vs_2d.png")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, ok: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print("LEADERBOARD — median rel. error (%), sorted best→worst")
    print("=" * 90)
    grp = (
        ok.groupby(["mode", "heated", "max_steps", "cem"])["rel_error"]
        .agg(median="median", mean="mean", min="min", n="count")
        .reset_index()
        .sort_values("median")
    )
    print(f"{'mode':<12} {'heated':<8} {'steps':<7} {'cem':<6} "
          f"{'median%':>9} {'mean%':>9} {'min%':>7} {'n':>4}")
    print("-" * 90)
    for _, r in grp.head(20).iterrows():
        print(f"{r['mode']:<12} {str(r['heated']):<8} {int(r['max_steps']):<7} "
              f"{str(r['cem']):<6} {r['median']:>9.2f} {r['mean']:>9.2f} "
              f"{r['min']:>7.3f} {int(r['n']):>4}")
    print("=" * 90)

    print()
    print("DIVERGENCE RATE by (mode, heated, max_steps)")
    print("-" * 60)
    div = (
        df.groupby(["mode", "heated", "max_steps"])
        .apply(lambda x: (x["rel_error"] > 100).mean(), include_groups=False)
        .reset_index(name="div_rate")
    )
    print(div.to_string(index=False))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not IN_JSON.exists():
        print(f"ERROR: {IN_JSON} not found. Run scripts/sbm_tune.py first.",
              file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, ok, raw = load()

    print("Generating figures...")
    fig_overview(df, ok)
    fig_heatmaps(ok)
    fig_convergence(raw)
    fig_cem(ok)
    fig_1d_vs_2d(ok)

    print_summary(df, ok)
    print(f"All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
