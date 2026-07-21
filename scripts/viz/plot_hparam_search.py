#!/usr/bin/env python3
"""
plot_hparam_search.py — analyze a single Optuna hyperparameter-search run
(results/hparam_search/{hamiltonian}/{study}/{N}_{phys}/).

No existing script in scripts/viz/ analyzes a hparam search on its own terms:
plot_ttc.py / plot_ite.py can fold hparam_search runs into an ITE/TTC time
series via --include-hparam, but nothing plots optimization progress,
parameter sensitivity, or per-config convergence for the search itself.
This fills that gap.

Metric: energy error per spin, |E - E_exact| / N — the same convention as
scripts/viz/dashboard.py's error_per_spin and plot_sparsity_ablation_floor.py.
NOT relative error — all Hamiltonians in the hparam registry are 1D chains,
so N (chain length) is the spin count directly (see index.jsonl's "N" field).

Three panels:
  (A) Optimization history — err/spin per trial in completion order, with
      the running incumbent (best-so-far) highlighted, so you can see whether
      the search had converged or was still improving when it stopped.
  (B) Parameter sensitivity — err/spin vs. each swept hyperparameter,
      colored by err/spin (single-hue sequential: lighter = better), to see
      which parameters the metric is actually sensitive to.
  (C) Convergence per config — energy-error-per-spin vs. SR iteration for the
      top-K trials (by tail-mean err/spin, same statistic Optuna optimized),
      read from each trial's full result_*.json.gz history. Shows *how* the
      best configs got there, not just where they ended up.

Also writes best_config.json (winning trial's full config + final metrics)
next to the plots — directly usable as --hparam-dir input elsewhere.

Usage:
    python scripts/viz/plot_hparam_search.py \\
        results/hparam_search/tfim_1d/veloxq_tfim/N128_h0.5
"""

import argparse
import gzip
import json
import math
import os
from pathlib import Path

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
    """Load index.jsonl and attach err_per_spin = abs_error / N per trial.

    Trials without a usable abs_error (N beyond exact_max_N, so Optuna had
    no exact reference — see hparam_optuna.py) are dropped; the count is
    printed rather than silently ignored.
    """
    path = os.path.join(search_dir, "index.jsonl")
    trials = []
    with open(path) as f:
        for line in f:
            trials.append(json.loads(line))
    trials.sort(key=lambda t: t["datetime"])

    n_total = len(trials)
    valid = [
        t for t in trials
        if t.get("abs_error") is not None and math.isfinite(t["abs_error"]) and t.get("N")
    ]
    for t in valid:
        t["err_per_spin"] = t["abs_error"] / t["N"]
    if len(valid) < n_total:
        print(f"dropped {n_total - len(valid)}/{n_total} trials with no exact-energy "
              f"reference (N beyond exact_max_N) — no err/spin available")
    return valid


def load_histories(search_dir):
    """Load every trial's full result_*.json.gz (has per-iteration history).

    Returns a list of dicts: {err_per_spin (tail-mean, matches Optuna's
    objective statistic), series (per-iteration err/spin), config}.
    """
    records = []
    skipped_no_exact = 0
    for path in Path(search_dir).rglob("result_*.json.gz"):
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        exact = d.get("exact_energy")
        if exact is None:
            skipped_no_exact += 1
            continue
        config = d["config"]
        N = config["size"]
        energies = np.array(d["history"]["energy"], dtype=float)
        series = np.abs(energies - exact) / N
        # Match hparam_optuna.py's objective exactly: mean energy over the last
        # 20% of iterations, THEN diff against exact — not mean(|diff|), which
        # is inflated by per-iteration noise that a converged run averages out.
        tail_start = max(0, int(0.8 * len(energies)))
        tail_mean_energy = float(np.mean(energies[tail_start:]))
        tail_err = abs(tail_mean_energy - exact) / N
        records.append({"err_per_spin": tail_err, "series": series, "config": config})
    if skipped_no_exact:
        print(f"skipped {skipped_no_exact} result files with no exact-energy reference")
    return records


def style_axes(ax):
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot(search_dir, out_dir, title_tag, top_k):
    trials = load_trials(search_dir)
    n = len(trials)
    errs = [t["err_per_spin"] for t in trials]
    order = list(range(1, n + 1))
    best_idx = int(np.argmin(errs))
    incumbent = np.minimum.accumulate(errs)

    norm_lo, norm_hi = np.log10(min(errs)), np.log10(max(errs))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = BLUE_RAMP((norm_hi - np.log10(errs)) / (norm_hi - norm_lo))
    ax.scatter(order, errs, c=colors, s=45, zorder=3, edgecolor="white", linewidth=0.5)
    ax.step(order, incumbent, where="post", color=INK, linewidth=1.3, zorder=2, label="incumbent (best so far)")
    ax.scatter([best_idx + 1], [errs[best_idx]], marker="*", s=260, color="#eb6834",
               edgecolor=INK, linewidth=0.6, zorder=4, label=f"winning trial (#{trials[best_idx]['trial']})")
    ax.annotate(
        f"nh={trials[best_idx]['n_hidden']}, lr={trials[best_idx]['params']['learning_rate']:.4f},\n"
        f"reg={trials[best_idx]['params']['regularization']:.2e}, ns={trials[best_idx]['params']['n_samples']},\n"
        f"err/spin={errs[best_idx]:.2e}",
        xy=(best_idx + 1, errs[best_idx]), xytext=(0.6, 0.75), textcoords="axes fraction",
        fontsize=8, color=MUTED, ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    ax.set_yscale("log")
    ax.set_xlabel("Trial (completion order)")
    ax.set_ylabel("Energy error per spin  |ΔE|/N")
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
        # Studies that mix sampling methods (metropolis/exchange/SA/LSB/velox_sa)
        # only record method-specific params (e.g. T_initial, num_sweeps) for the
        # trials that used that method — plot whichever subset has this key.
        subset = [t for t in trials if key in t["params"]]
        if not subset:
            ax2.text(0.5, 0.5, "no trials used this param", ha="center", va="center",
                      fontsize=9, color=MUTED, transform=ax2.transAxes)
            ax2.set_xlabel(label)
            style_axes(ax2)
            continue
        vals = [t["params"][key] for t in subset]
        sub_errs = [t["err_per_spin"] for t in subset]
        colors = BLUE_RAMP((norm_hi - np.log10(sub_errs)) / (norm_hi - norm_lo))
        ax2.scatter(vals, sub_errs, c=colors, s=40, edgecolor="white", linewidth=0.4, zorder=3)
        if key in trials[best_idx]["params"]:
            ax2.scatter([trials[best_idx]["params"][key]], [errs[best_idx]], marker="*", s=220,
                        color="#eb6834", edgecolor=INK, linewidth=0.6, zorder=4)
        if logx:
            ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(label)
        ax2.set_ylabel("Err/spin" if key in ("n_hidden_alpha", "n_samples") else "")
        style_axes(ax2)
    fig2.suptitle(f"Parameter sensitivity — {title_tag}\n(lighter = lower/better err/spin; star = winning trial)", y=1.02)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hparam_search_param_sensitivity.{ext}")
        fig2.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig2)

    # Third figure: convergence per config — top-K trials by tail-mean err/spin,
    # read from each trial's full training history.
    records = load_histories(search_dir)
    if not records:
        print("no result_*.json.gz histories found with an exact-energy reference "
              "— skipping convergence panel")
        return
    records.sort(key=lambda r: r["err_per_spin"])
    shown = records[:top_k]
    if len(records) > top_k:
        print(f"convergence panel: showing top {top_k} of {len(records)} trials "
              f"(rest omitted, ranked by tail-mean err/spin)")

    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    rank_norm = np.linspace(0, 1, max(len(shown), 2))[: len(shown)]
    for rank, rec in enumerate(shown):
        cfg = rec["config"]
        label = (f"nh={cfg['n_hidden']} lr={cfg['learning_rate']:.2g} "
                 f"reg={cfg['regularization']:.1e} ns={cfg['n_samples']} seed={cfg['seed']}")
        is_best = rank == 0
        color = "#eb6834" if is_best else BLUE_RAMP(rank_norm[rank])
        ax3.plot(
            np.arange(len(rec["series"])), rec["series"],
            color=color, linewidth=2.2 if is_best else 1.1,
            alpha=1.0 if is_best else 0.75, zorder=5 if is_best else 3,
            label=(f"★ {label}" if is_best else label),
        )
    ax3.set_yscale("log")
    ax3.set_xlabel("SR iteration")
    ax3.set_ylabel("Energy error per spin  |ΔE(t)|/N")
    ax3.set_title(f"Convergence per config — top {len(shown)} of {len(records)} trials, {title_tag}")
    style_axes(ax3)
    ax3.legend(frameon=False, loc="upper right", fontsize=7)
    fig3.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hparam_search_convergence.{ext}")
        fig3.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig3)

    # Best-config summary, dumped for reuse elsewhere (e.g. run_fpga_best.py).
    best = shown[0]
    best_config_path = os.path.join(out_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump({"err_per_spin": best["err_per_spin"], "config": best["config"]}, f, indent=2)
    print(f"wrote {best_config_path}")
    print(f"best config: err/spin={best['err_per_spin']:.3e}  {best['config']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("search_dir", help="Path to a results/hparam_search/.../N.../ directory containing index.jsonl")
    parser.add_argument("--out-dir", default="plots/hparam_search")
    parser.add_argument("--title-tag", default=None, help="Label for plot titles (default: derived from path)")
    parser.add_argument("--top-k", type=int, default=8,
                         help="Number of trials to overlay in the convergence panel (default: 8)")
    args = parser.parse_args()

    title_tag = args.title_tag or os.path.basename(os.path.normpath(args.search_dir))
    plot(args.search_dir, args.out_dir, title_tag, args.top_k)


if __name__ == "__main__":
    main()
