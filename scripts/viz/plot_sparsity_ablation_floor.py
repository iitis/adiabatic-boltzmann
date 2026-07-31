#!/usr/bin/env python3
"""
plot_sparsity_ablation_floor.py

Regenerates plots/sparsity/sparsity_ablation_qpu_vs_classical.{pdf,png}: three
classical samplers (Metropolis, simulated annealing, persistent Gibbs) against
the exact-ansatz floor (scripts/exper/exact_ansatz_floor.py, which trains the
same sparse masks via exact enumeration instead of Monte Carlo sampling).
Isolates the sparse ansatz's representational limit from optimization error
(referee point 9's error-source decomposition). The real-QPU arm previously
plotted here was dropped: its iteration budget was never matched to the
classical arms' (see F3 in audyt_cld_bg.md) and the report's conclusion now
explicitly scopes this ablation to classical hardware only -- keeping an
unmatched QPU curve in the same figure would contradict that.

Reuses these caches:
    plots/sparsity/cache_full.json               (dense, sparsity=0 reference;
                                                    also the classical-Metropolis
                                                    native-mask point)
    plots/sparsity/cache_sparsity_ablation.json   (classical MCMC, Metropolis)
    plots/sparsity/cache_sparsity_ablation_exact.json (exact-enumeration floor)
    plots/sparsity/cache_sparsity_ablation_simulated_annealing.json (classical SA)
    plots/sparsity/cache_sparsity_ablation_gibbs.json (classical persistent Gibbs)

SA and Gibbs (scripts/exper/sparsity_ablation_classical_baselines.py) test whether
the large classical-vs-floor gap is specific to Metropolis or general to
non-exact classical sampling; both also include the unpruned native-mask point
(sparsity 0.42578, label "native" in their cache keys), which the original
Metropolis ablation never re-ran (it exists only via cache_full.json).

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
NATIVE_SPARSITY = 0.42578125  # unpruned zephyr alpha=1 mask's own sparsity (cache_full.json "16_1_1_zephyr_*")


def load(path):
    with open(path) as f:
        return json.load(f)


def _per_spin_err(entry):
    """Absolute energy error per spin: |E_final - E_exact| / N. Referee point 19:
    an absolute error per site is easier to interpret than a relative error
    against a negative E_exact (which produced percentages above 100%)."""
    return abs(entry["E_final"] - entry["E_exact"]) / N


def classical_sampler_series(cache, sparsities, seeds, h=H, topology=TOPOLOGY):
    means, stds = [], []
    for ts in sparsities:
        errs = [_per_spin_err(cache[f"{N}_{ts}_{h}_{topology}_{s}"]) for s in seeds]
        means.append(np.mean(errs))
        stds.append(np.std(errs))
    return np.array(means), np.array(stds)


def native_point(cache, seeds, key_fmt):
    """Unpruned-mask point: key_fmt.format(seed) locates each seed's record.
    Two conventions coexist -- cache_full.json's "{N}_1_1_zephyr_{seed}"
    (alpha=1, h=1 as an int) vs. the new SA/Gibbs caches' own
    "{N}_native_{H}_zephyr_{seed}" -- so the caller supplies the format."""
    errs = [_per_spin_err(cache[key_fmt.format(s)]) for s in seeds]
    return np.mean(errs), np.std(errs)


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
    cache_exact = load(CACHE_DIR / "cache_sparsity_ablation_exact.json")
    cache_sa = load(CACHE_DIR / "cache_sparsity_ablation_simulated_annealing.json")
    cache_gibbs = load(CACHE_DIR / "cache_sparsity_ablation_gibbs.json")

    dense_mean, dense_std = dense_point(cache_full)
    native_cl_mean, native_cl_std = native_point(cache_full, SEEDS, "16_1_1_zephyr_{}")
    native_sa_mean, native_sa_std = native_point(cache_sa, SEEDS, "16_native_1.0_zephyr_{}")
    native_gi_mean, native_gi_std = native_point(cache_gibbs, SEEDS, "16_native_1.0_zephyr_{}")

    cl_mean, cl_std = classical_sampler_series(cache_classical, TARGET_SPARSITIES, SEEDS)
    sa_mean, sa_std = classical_sampler_series(cache_sa, TARGET_SPARSITIES, SEEDS)
    gi_mean, gi_std = classical_sampler_series(cache_gibbs, TARGET_SPARSITIES, SEEDS)
    floor_mean, floor_std, floor_best = floor_series(cache_exact, TARGET_SPARSITIES)

    x_classical = [0.0, NATIVE_SPARSITY] + TARGET_SPARSITIES
    y_classical = np.concatenate([[dense_mean, native_cl_mean], cl_mean])
    yerr_classical = np.concatenate([[dense_std, native_cl_std], cl_std])

    x_native = [NATIVE_SPARSITY] + TARGET_SPARSITIES
    y_sa = np.concatenate([[native_sa_mean], sa_mean])
    yerr_sa = np.concatenate([[native_sa_std], sa_std])
    y_gibbs = np.concatenate([[native_gi_mean], gi_mean])
    yerr_gibbs = np.concatenate([[native_gi_std], gi_std])

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
        label="Classical (Metropolis)", capsize=3, markersize=5, linewidth=1.4,
    )
    ax.errorbar(
        x_native, y_sa, yerr=safe_yerr(y_sa, yerr_sa),
        marker="D", color="#9467bd", linestyle="-.",
        label="Classical (SA)", capsize=3, markersize=4.5, linewidth=1.4,
    )
    ax.errorbar(
        x_native, y_gibbs, yerr=safe_yerr(y_gibbs, yerr_gibbs),
        marker="v", color="#ff7f0e", linestyle="--",
        label="Classical (Gibbs)", capsize=3, markersize=4.5, linewidth=1.4,
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
    ax.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
