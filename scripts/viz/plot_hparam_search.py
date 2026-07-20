#!/usr/bin/env python3
"""
plot_hparam_search.py — analyze a single Optuna hyperparameter-search run
(results/hparam_search/{hamiltonian}/{study}/{N}_{phys}/index.jsonl).

No existing script in scripts/viz/ analyzes a hparam search on its own terms:
plot_ttc.py / plot_ite.py can fold hparam_search runs into an ITE/TTC time
series via --include-hparam, but nothing plots optimization progress or
parameter sensitivity for the search itself. This fills that gap.

Two panels:
  (A) Optimization history — objective per trial in completion order, with
      the running incumbent (best-so-far) highlighted, so you can see whether
      the search had converged or was still improving when it stopped.
  (B) Parameter sensitivity — objective vs. each swept hyperparameter,
      colored by objective (single-hue sequential: lighter = better), to see
      which parameters the objective is actually sensitive to.

Usage:
    python scripts/viz/plot_hparam_search.py \\
        results/hparam_search/tfim_1d/veloxq_tfim/N128_h0.5
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
GOOD = "#0ca30c"

# single-hue sequential blue ramp (palette.md steps 150->650), light=low/good, dark=high/bad
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]
)

PARAM_SPECS = [
    ("n_hidden_alpha", "n_hidden / N (alpha)", False),
    ("learning_rate", "Learning rate", True),
    ("regularization", "Regularization", True),
    ("n_samples", "n_samples", False),
    ("T_initial", "SA initial temperature", True),
    ("num_sweeps", "SA sweeps", True),
]


def load_trials(search_dir):
    path = os.path.join(search_dir, "index.jsonl")
    trials = []
    with open(path) as f:
        for line in f:
            trials.append(json.loads(line))
    trials.sort(key=lambda t: t["datetime"])
    return trials


def style_axes(ax):
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot(search_dir, out_dir, title_tag):
    trials = load_trials(search_dir)
    n = len(trials)
    objectives = [t["objective"] for t in trials]
    order = list(range(1, n + 1))
    best_idx = int(np.argmin(objectives))
    incumbent = np.minimum.accumulate(objectives)

    norm_lo, norm_hi = np.log10(min(objectives)), np.log10(max(objectives))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = BLUE_RAMP((norm_hi - np.log10(objectives)) / (norm_hi - norm_lo))
    ax.scatter(order, objectives, c=colors, s=45, zorder=3, edgecolor="white", linewidth=0.5)
    ax.step(order, incumbent, where="post", color=INK, linewidth=1.3, zorder=2, label="incumbent (best so far)")
    ax.scatter([best_idx + 1], [objectives[best_idx]], marker="*", s=260, color="#eb6834",
               edgecolor=INK, linewidth=0.6, zorder=4, label=f"winning trial (#{trials[best_idx]['trial']})")
    ax.annotate(
        f"nh={trials[best_idx]['n_hidden']}, lr={trials[best_idx]['params']['learning_rate']:.4f},\n"
        f"reg={trials[best_idx]['params']['regularization']:.2e}, ns={trials[best_idx]['params']['n_samples']}",
        xy=(best_idx + 1, objectives[best_idx]), xytext=(0.6, 0.75), textcoords="axes fraction",
        fontsize=8, color=MUTED, ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    ax.set_yscale("log")
    ax.set_xlabel("Trial (completion order)")
    ax.set_ylabel("Objective (relative error)")
    ax.set_title(f"Optimization history — {n} completed trials, {title_tag}")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hparam_search_overview.{ext}")
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    # Second figure: full parameter-sensitivity panel (6 params, 2x3)
    fig2, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax2, (key, label, logx) in zip(axes.flat, PARAM_SPECS):
        vals = [t["params"][key] for t in trials]
        colors = BLUE_RAMP((norm_hi - np.log10(objectives)) / (norm_hi - norm_lo))
        ax2.scatter(vals, objectives, c=colors, s=40, edgecolor="white", linewidth=0.4, zorder=3)
        ax2.scatter([trials[best_idx]["params"][key]], [objectives[best_idx]], marker="*", s=220,
                    color="#eb6834", edgecolor=INK, linewidth=0.6, zorder=4)
        if logx:
            ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(label)
        ax2.set_ylabel("Objective" if key in ("n_hidden_alpha", "n_samples") else "")
        style_axes(ax2)
    fig2.suptitle(f"Parameter sensitivity — {title_tag}\n(lighter = lower/better objective; star = winning trial)", y=1.02)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hparam_search_param_sensitivity.{ext}")
        fig2.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("search_dir", help="Path to a results/hparam_search/.../N.../ directory containing index.jsonl")
    parser.add_argument("--out-dir", default="plots/hparam_search")
    parser.add_argument("--title-tag", default=None, help="Label for plot titles (default: derived from path)")
    args = parser.parse_args()

    title_tag = args.title_tag or os.path.basename(os.path.normpath(args.search_dir))
    plot(args.search_dir, args.out_dir, title_tag)


if __name__ == "__main__":
    main()
