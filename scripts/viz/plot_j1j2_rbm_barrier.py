#!/usr/bin/env python3
"""
Relative energy error of the best RBM run vs J2/J1 for the J1-J2 Heisenberg chain.

Loads all results from results/heisenberg_j1j2_1d/, groups by (N, J2), and
plots the minimum and median relative error.  The sign-problem wall at J2/J1=0.5
appears as a 3–4 order-of-magnitude jump.

Usage:
    python scripts/viz/plot_j1j2_rbm_barrier.py
"""
import collections
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

_ROOT    = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results" / "heisenberg_j1j2_1d"
_OUT     = _ROOT / "plots" / "j1j2"


def load_results():
    data = collections.defaultdict(list)
    for f in _RESULTS.rglob("*.json.gz"):
        try:
            with gzip.open(f) as fp:
                d = json.load(fp)
            cfg = d.get("config", {})
            N   = cfg.get("N") or cfg.get("size")
            J2  = cfg.get("J2")
            err = d.get("error")
            if N and J2 is not None and err is not None and err >= 0:
                data[(int(N), round(float(J2), 4))].append(float(err))
        except Exception:
            pass
    return data


def main():
    setup_style(fontsize=13, scale=1.0)

    data = load_results()

    systems = [8, 12, 16]
    colors  = ["#2563eb", "#16a34a", "#dc2626"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for N, color, marker in zip(systems, colors, markers):
        j2s     = sorted(j2 for (n, j2) in data if n == N)
        medians = [np.median(data[(N, j2)]) for j2 in j2s]
        mins    = [min(data[(N, j2)])        for j2 in j2s]

        kw = dict(color=color, marker=marker, markersize=5,
                  markerfacecolor="white", markeredgewidth=1.4)
        ax.plot(j2s, medians, lw=1.8, ls="-",  label=f"$N={N}$ (median)", **kw)
        ax.plot(j2s, mins,    lw=1.2, ls="--", label=f"$N={N}$ (best)",   **kw)

    # Shade the sign-problem region
    ax.axvspan(0.5, 1.05, color="#fee2e2", alpha=0.55, zorder=0,
               label="Sign-problem region")
    ax.axvline(0.5, color="#111", lw=1.2, ls=":", zorder=2)
    ax.text(0.505, 1.5e3, r"$J_2/J_1 = 0.5$" + "\n(Majumdar--Ghosh)",
            fontsize=8.5, color="#333", va="top")

    ax.set_yscale("log")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(5e-7, 1e4)
    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_ylabel(r"Relative energy error $|E_\mathrm{RBM} - E_\mathrm{exact}| / |E_\mathrm{exact}|$")
    ax.legend(ncol=2, fontsize=8.5, loc="lower left",
              handlelength=2.0, columnspacing=0.8)

    fig.tight_layout()
    _OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = _OUT / f"fig_rbm_barrier.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
