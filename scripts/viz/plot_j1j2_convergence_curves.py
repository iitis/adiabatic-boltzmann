#!/usr/bin/env python3
"""
Convergence curves for the best RBM run at J2/J1 = 0.3, 0.5, 0.7, 0.9
for N = 8, 12 (2×4 grid).

Picks the file with the lowest final relative error for each (N, J2) pair,
plots E/N vs iteration, and marks the exact ground-state energy E_exact/N.

Usage:
    python scripts/viz/plot_j1j2_convergence_curves.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
from plot_style import setup_style

_ROOT    = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results" / "heisenberg_j1j2_1d"
_OUT     = _ROOT / "plots" / "j1j2"

J2_COLS = [0.3, 0.5, 0.7, 0.9]
N_ROWS  = [8, 12]


def load_best():
    """Return dict (N, J2) -> (energy_history/N, exact/N, final_error)."""
    best = {}
    for f in _RESULTS.rglob("*.json.gz"):
        try:
            with gzip.open(f) as fp:
                d = json.load(fp)
            cfg   = d.get("config", {})
            N     = int(cfg.get("N") or cfg.get("size"))
            J2    = round(float(cfg.get("J2", -1)), 4)
            err   = d.get("error")
            exact = d.get("exact_energy")
            hist  = d.get("history", {}).get("energy")
            if err is None or exact is None or not hist or N not in N_ROWS:
                continue
            j2_rounded = round(J2, 1)
            if j2_rounded not in J2_COLS:
                continue
            key = (N, j2_rounded)
            if key not in best or err < best[key][2]:
                best[key] = (
                    np.array(hist, dtype=float) / N,
                    float(exact) / N,
                    float(err),
                )
        except Exception:
            pass
    return best


def _rolling_mean(x, w=10):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def main():
    setup_style(fontsize=16, scale=1.0)
    best = load_best()

    fig, axes = plt.subplots(
        len(N_ROWS), len(J2_COLS),
        figsize=(20, 6),
        sharex=False,
        gridspec_kw={"hspace": 0.52, "wspace": 0.42},
    )

    for col, j2 in enumerate(J2_COLS):
        for row, N in enumerate(N_ROWS):
            ax  = axes[row][col]
            key = (N, j2)

            if key not in best:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#888")
                continue

            e_hist, e_exact, final_err = best[key]
            iters = np.arange(len(e_hist))

            # Raw curve (faint) + rolling mean (bold)
            w = max(1, len(e_hist) // 25)
            ax.plot(iters, e_hist, color="#93c5fd", lw=0.6, alpha=0.7)
            if len(e_hist) >= w:
                smooth = _rolling_mean(e_hist, w)
                ax.plot(np.arange(len(smooth)) + w // 2, smooth,
                        color="#1d4ed8", lw=1.6)

            # Exact GS line
            ax.axhline(e_exact, color="#dc2626", lw=1.3, ls="--")

            # J2/J1 label in every panel (top-left); N label only on leftmost panel of each row
            if col == 0:
                ax.set_title(rf"$N={N}$", pad=4)
            ax.text(0.04, 0.96,
                    rf"$J_2/J_1={j2}$",
                    ha="left", va="top", transform=ax.transAxes,
                    fontsize="small", color="#111")

            e_final = float(e_hist[-1])

            # Y-axis: zoom in around convergence region
            e_min = min(e_exact, np.min(e_hist[-max(1, len(e_hist)//3):]))
            e_max = np.percentile(e_hist, 90)
            margin = abs(e_exact) * 0.08
            ax.set_ylim(e_min - margin, max(e_max, e_exact + margin))

            # Inline "E/N = ..." label next to the converged (blue) curve
            ylo, yhi = ax.get_ylim()
            mid_idx = int(len(e_hist) * 0.68)
            y_label = min(e_hist[mid_idx] + 0.06 * (yhi - ylo), yhi - 0.16 * (yhi - ylo))
            ax.text(iters[mid_idx], y_label,
                    rf"$E/N={e_final:.3f}$", color="#1d4ed8", fontsize="small",
                    ha="center", va="bottom")

            if row == len(N_ROWS) - 1:
                ax.set_xlabel("Iteration")
            if col == 0:
                ax.set_ylabel("$E/N$")

    _OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = _OUT / f"fig_convergence_best.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
