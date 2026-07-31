#!/usr/bin/env python3
"""
plot_sparsity_ablation_heatmap.py

Regenerates plots/sparsity/sparsity_ablation_heatmap.{pdf,png} — Fig. 12a in
report.tex, the left subfigure of fig:sparsity-simple. No script generating
this figure was committed anywhere in the repo; this one reproduces it from
the committed cache alone.

Data: plots/sparsity/cache_sparsity_ablation.json — the classical-MCMC
sparsity ablation (N=16, alpha=1, zephyr, 4 pruned masks x 5 h values x 5
seeds, ClassicalSampler(method="metropolis"), 300 SR iterations, no
truncation). This is the ONLY sampler in this figure: the QPU cache has no
h != 1.0 entries and the exact-floor cache has no h != 1.0 entries either,
so panel (a) cannot and does not contain QPU or floor data.

Usage (from repo root):
    python scripts/viz/plot_sparsity_ablation_heatmap.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import setup_style

_REPO = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _REPO / "plots" / "sparsity"
CACHE_PATH = CACHE_DIR / "cache_sparsity_ablation.json"
OUT_STEM = CACHE_DIR / "sparsity_ablation_heatmap"

N = 16
TOPOLOGY = "zephyr"
SEEDS = ["42", "123", "456", "789", "1234"]
SPARSITY_LABELS = ["0.557", "0.682", "0.809", "0.877"]  # bottom -> top
H_VALUES = ["0.3", "0.7", "1.0", "1.3", "2.0"]  # left -> right
H_CRITICAL = "1.0"


def load(path):
    with open(path) as f:
        return json.load(f)


def build_grid(cache):
    """(n_sparsity, n_h) grid of mean relative error."""
    grid = np.full((len(SPARSITY_LABELS), len(H_VALUES)), np.nan)
    for row, sp_label in enumerate(SPARSITY_LABELS):
        for col, h in enumerate(H_VALUES):
            errs = []
            for seed in SEEDS:
                key = f"{N}_{sp_label}_{h}_{TOPOLOGY}_{seed}"
                rec = cache.get(key)
                if rec is None:
                    continue
                errs.append(rec["rel_error"])
            if errs:
                grid[row, col] = np.mean(errs)
    return grid


def main():
    setup_style(fontsize=9, scale=1.0)

    cache = load(CACHE_PATH)
    grid = build_grid(cache)

    if np.isnan(grid).any():
        missing = np.argwhere(np.isnan(grid))
        raise RuntimeError(
            f"Missing cache entries for (sparsity, h) cells: "
            f"{[(SPARSITY_LABELS[r], H_VALUES[c]) for r, c in missing]}"
        )

    fig, ax = plt.subplots()

    norm = mcolors.LogNorm(vmin=1e-2, vmax=1e0)
    im = ax.imshow(
        grid, origin="lower", aspect="auto", cmap="RdYlGn_r", norm=norm,
        extent=[-0.5, len(H_VALUES) - 0.5, -0.5, len(SPARSITY_LABELS) - 0.5],
    )

    ax.set_xticks(np.arange(len(H_VALUES)))
    ax.set_xticklabels([f"{float(h):.2f}" for h in H_VALUES], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(SPARSITY_LABELS)))
    ax.set_yticklabels(SPARSITY_LABELS)
    ax.set_xlabel(r"Transverse field $h$")
    ax.set_ylabel("Sparsity")
    ax.grid(False)

    h_c_idx = H_VALUES.index(H_CRITICAL)
    ax.axvline(h_c_idx, color="white", linewidth=0.8, linestyle=":")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"Rel. error $|\varepsilon|$")

    fig.tight_layout()
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
