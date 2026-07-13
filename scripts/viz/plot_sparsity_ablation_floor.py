#!/usr/bin/env python3
"""
plot_sparsity_ablation_floor.py

Regenerates plots/sparsity/sparsity_ablation_qpu_vs_classical.{pdf,png} with a
third reference curve: the exact-ansatz floor (scripts/exper/exact_ansatz_floor.py),
which trains the same sparse masks via exact enumeration instead of Monte Carlo
sampling. Isolates the sparse ansatz's representational limit from optimization
and hardware error (referee point 9's error-source decomposition).

Reuses the same three caches:
    plots/sparsity/cache_full.json               (dense, sparsity=0 reference)
    plots/sparsity/cache_sparsity_ablation.json   (classical MCMC)
    plots/sparsity/cache_sparsity_ablation_qpu.json (real QPU)
    plots/sparsity/cache_sparsity_ablation_exact.json (exact-enumeration floor, new)

Usage (from repo root):
    python scripts/viz/plot_sparsity_ablation_floor.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import setup_style

_REPO = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _REPO / "plots" / "sparsity"
OUT_STEM = CACHE_DIR / "sparsity_ablation_qpu_vs_classical"

N = 16
H = 1.0
TOPOLOGY = "zephyr"
SEEDS = [42, 123, 456, 789, 1234]
TARGET_SPARSITIES = [0.557, 0.682, 0.809, 0.877]
NATIVE_SPARSITY = 0.55859375  # zephyr alpha=1 mask's own sparsity (see cache "0.557" point)


def load(path):
    with open(path) as f:
        return json.load(f)


def _per_spin_err(entry):
    """Absolute energy error per spin: |E_final - E_exact| / N. Referee point 19:
    an absolute error per site is easier to interpret than a relative error
    against a negative E_exact (which produced percentages above 100%)."""
    return abs(entry["E_final"] - entry["E_exact"]) / N


def classical_qpu_series(cache, sparsities, seeds, h=H, topology=TOPOLOGY):
    means, stds = [], []
    for ts in sparsities:
        errs = [_per_spin_err(cache[f"{N}_{ts}_{h}_{topology}_{s}"]) for s in seeds]
        means.append(np.mean(errs))
        stds.append(np.std(errs))
    return np.array(means), np.array(stds)


def dense_point(cache_full):
    errs = [_per_spin_err(cache_full[f"{N}_1_1_dense_{s}"]) for s in SEEDS]
    return np.mean(errs), np.std(errs)


def floor_series(cache_exact, sparsities):
    means, stds, bests = [], [], []
    for ts in sparsities:
        entry = cache_exact[str(ts)]
        errs = [
            abs(r["E_final"] - entry["E_exact"]) / N for r in entry["per_seed"]
        ]
        means.append(np.mean(errs))
        stds.append(np.std(errs))
        bests.append(min(errs))
    return np.array(means), np.array(stds), np.array(bests)


def main():
    setup_style(fontsize=9)

    cache_full = load(CACHE_DIR / "cache_full.json")
    cache_classical = load(CACHE_DIR / "cache_sparsity_ablation.json")
    cache_qpu = load(CACHE_DIR / "cache_sparsity_ablation_qpu.json")
    cache_exact = load(CACHE_DIR / "cache_sparsity_ablation_exact.json")

    dense_mean, dense_std = dense_point(cache_full)
    cl_mean, cl_std = classical_qpu_series(cache_classical, TARGET_SPARSITIES, SEEDS)
    qpu_mean, qpu_std = classical_qpu_series(cache_qpu, TARGET_SPARSITIES, SEEDS)
    floor_mean, floor_std, floor_best = floor_series(cache_exact, TARGET_SPARSITIES)

    x_classical = [0.0] + TARGET_SPARSITIES
    y_classical = np.concatenate([[dense_mean], cl_mean])
    yerr_classical = np.concatenate([[dense_std], cl_std])

    def safe_yerr(mean, std, min_frac=0.3):
        """Asymmetric [lower, upper] error clipped so the lower whisker never
        drops below min_frac*mean — a std that exceeds the mean would
        otherwise send the lower bound non-positive (invalid on a log axis)
        or stretch several decades down, exaggerating the uncertainty."""
        mean = np.asarray(mean)
        std = np.asarray(std)
        lower = np.minimum(std, mean * (1.0 - min_frac))
        lower = np.maximum(lower, 0.0)
        return np.array([lower, std])

    fig, ax = plt.subplots()

    ax.errorbar(
        x_classical, y_classical, yerr=safe_yerr(y_classical, yerr_classical),
        marker="o", color="#2166ac", linestyle="-",
        label="Classical", capsize=3, markersize=5, linewidth=1.4,
    )
    ax.errorbar(
        TARGET_SPARSITIES, qpu_mean, yerr=safe_yerr(qpu_mean, qpu_std),
        marker="s", color="#d62728", linestyle="--",
        label="Real QPU", capsize=3, markersize=5, linewidth=1.4,
    )
    ax.errorbar(
        TARGET_SPARSITIES, floor_mean, yerr=safe_yerr(floor_mean, floor_std),
        marker="^", color="#2ca02c", linestyle=":",
        label="Exact floor", capsize=3, markersize=5, linewidth=1.4,
    )

    ax.set_yscale("log")
    ax.axvline(NATIVE_SPARSITY, color="gray", linestyle=":", linewidth=0.8)
    ax.text(
        NATIVE_SPARSITY, 0.03, "native hardware floor",
        transform=ax.get_xaxis_transform(),
        rotation=90, va="bottom", ha="right", fontsize=8, color="gray",
    )

    ax.set_xlabel("Sparsity")
    ax.set_ylabel(r"Energy error per spin $|\varepsilon|/N$")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
