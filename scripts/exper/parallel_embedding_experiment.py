#!/usr/bin/env python3
"""
parallel_embedding_experiment.py

Assesses the impact of QPU parallel embedding (Trainer's n_parallel config,
wired through DimodSampler.sample_parallel / ParallelEmbeddingComposite,
busclique-based like the single-embedding default) on RBM-VMC training for
the 1D TFIM.

Config: N=8, h=0.5, pegasus, lr=0.1, reg=1e-5, n_samples=990 (divisible by
1, 3, and 5) -- the archived-healthy candidate (results/tfim_1d/8/dimod/
pegasus/result_1d_h0.5_rbmfull_nh8_lr0.1_reg1e-05_ns1000_seed42_iter300_
cem0.json.gz, 0.09% error). A companion experiment
(embedding_algo_comparison.py) found this config's training is fragile near
convergence *regardless* of embedding algorithm (4/6 short runs hit NaN
blowups, both busclique and minorminer equally) -- a real, pre-existing
numerical robustness gap in the SR/CG pipeline, unrelated to n_parallel.
To get a meaningful n_parallel comparison despite that per-run noise, this
runs 3 seeds per n_parallel value and reports both the best surviving run
and the mean over non-crashed runs, plus each n_parallel's crash rate.

n_parallel in {1, 3, 5}, 3 seeds each = 9 short runs at 150 iterations.

Runs in-process -- Trainer/DimodSampler/FullyConnectedRBM are constructed
directly here, matching plot_sparsity_impact.py's _run_one_qpu pattern,
rather than shelling out to scripts/main.py. No `args` namespace is passed
to Trainer, so no checkpoint/dwave_samples file dumps fire; this script's
own JSON cache is the only side artifact besides the figure.

Usage (from repo root):
    python scripts/exper/parallel_embedding_experiment.py
    python scripts/exper/parallel_embedding_experiment.py --plot-only
"""
import argparse
import sys
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "viz"))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import setup_style
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import DimodSampler
from encoder import Trainer
from helpers import read_qpu_time_ms

QPU_TIME_PATH = Path("time.json")
QPU_BUDGET_MS = 30 * 60 * 1000
CACHE_PATH = Path(__file__).resolve().parent / "cache_parallel_embedding_np_seeds.json"

N = 8
H = 0.5
LR = 0.1
REG = 1e-05
N_SAMPLES = 990  # divisible by 1, 3, and 5
N_ITERS = 150
METHOD = "pegasus"
N_PARALLEL_VALUES = [1, 3, 5]
SEEDS = [42, 1, 7]


def _check_budget():
    used_ms = read_qpu_time_ms(QPU_TIME_PATH)
    if used_ms >= QPU_BUDGET_MS:
        raise RuntimeError(
            f"QPU budget exceeded: {used_ms / 60_000:.2f} min used >= "
            f"{QPU_BUDGET_MS / 60_000:.0f} min cap. Aborting."
        )
    return used_ms


def _run_one(n_parallel, seed):
    used_before = _check_budget()
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    ising = TransverseFieldIsing1D(size=N, h=H)
    rbm = FullyConnectedRBM(N, N, model_key)
    sampler = DimodSampler(method=METHOD)
    config = {
        "n_samples": N_SAMPLES,
        "n_iterations": N_ITERS,
        "learning_rate": LR,
        "regularization": REG,
        "seed": seed,
        "n_parallel": n_parallel,
        "save_checkpoints": False,
    }
    trainer = Trainer(rbm=rbm, ising_model=ising, sampler=sampler, config=config)
    history = trainer.train()
    used_after = read_qpu_time_ms(QPU_TIME_PATH)

    e = np.array(history["energy"])
    nan_iter = int(np.argmax(np.isnan(e))) if np.isnan(e).any() else None

    return {
        "n_parallel": n_parallel,
        "seed": seed,
        "energy": [float(x) for x in e],
        "sampling_time_s": history["sampling_time_s"],
        "E_exact": float(ising.exact_ground_energy()),
        "nan_iter": nan_iter,
        "qpu_time_ms_used": used_after - used_before,
    }


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def run_experiments(n_values=N_PARALLEL_VALUES, seeds=SEEDS):
    cache = load_cache()
    for n_parallel in n_values:
        for seed in seeds:
            key = f"{n_parallel}_{seed}"
            if key in cache:
                print(f"  {key}: cached, skipping")
                continue
            used = _check_budget()
            print(f"  {key}: running ({used / 60_000:.2f} min QPU used so far)")
            rec = _run_one(n_parallel, seed)
            cache[key] = rec
            save_cache(cache)
            e = rec["energy"]
            final = e[rec["nan_iter"] - 1] if rec["nan_iter"] else e[-1]
            status = f"NaN@{rec['nan_iter']}" if rec["nan_iter"] else "completed"
            rel_err = abs((final - rec["E_exact"]) / rec["E_exact"])
            print(f"    -> {status}, final={final:.4f} (exact={rec['E_exact']:.4f}, "
                  f"rel_err={rel_err:.4f}), qpu_ms={rec['qpu_time_ms_used']:.0f}")
    return cache


def _final_and_relerr(rec):
    e = rec["energy"]
    final = e[rec["nan_iter"] - 1] if rec["nan_iter"] else e[-1]
    rel_err = abs((final - rec["E_exact"]) / rec["E_exact"])
    return final, rel_err


def print_summary(cache):
    print(f"\n{'n_parallel':>10} {'seed':>5} {'status':>10} {'final':>9} {'rel_err':>8}")
    for n_parallel in N_PARALLEL_VALUES:
        for seed in SEEDS:
            rec = cache.get(f"{n_parallel}_{seed}")
            if rec is None:
                continue
            final, rel_err = _final_and_relerr(rec)
            status = f"NaN@{rec['nan_iter']}" if rec["nan_iter"] else "completed"
            print(f"{n_parallel:>10} {seed:>5} {status:>10} {final:>9.4f} {rel_err:>8.4f}")

    print(f"\n{'n_parallel':>10} {'crash_rate':>10} {'best_rel_err':>12} {'mean_rel_err(survivors)':>24}")
    for n_parallel in N_PARALLEL_VALUES:
        recs = [cache[f"{n_parallel}_{seed}"] for seed in SEEDS if f"{n_parallel}_{seed}" in cache]
        if not recs:
            continue
        n_crashed = sum(1 for r in recs if r["nan_iter"] is not None)
        rel_errs = [_final_and_relerr(r)[1] for r in recs]
        survivor_errs = [_final_and_relerr(r)[1] for r in recs if r["nan_iter"] is None]
        best = min(rel_errs)
        mean_survivors = float(np.mean(survivor_errs)) if survivor_errs else float("nan")
        print(f"{n_parallel:>10} {n_crashed}/{len(recs):>8} {best:>12.4f} {mean_survivors:>24.4f}")


def _best_seed_per_n_parallel(cache):
    """For each n_parallel, the seed with the lowest final relative error."""
    best = {}
    for n_parallel in N_PARALLEL_VALUES:
        recs = [(seed, cache[f"{n_parallel}_{seed}"]) for seed in SEEDS if f"{n_parallel}_{seed}" in cache]
        if not recs:
            continue
        seed, rec = min(recs, key=lambda sr: _final_and_relerr(sr[1])[1])
        best[n_parallel] = seed
    return best


def _plot_one_panel(cache, ax, x_key, best_only, colors, ls):
    """x_key: 'iteration' or 'qpu_time'."""
    best_seed = _best_seed_per_n_parallel(cache) if best_only else None
    exact = None
    for n_parallel in N_PARALLEL_VALUES:
        seeds_to_plot = [best_seed[n_parallel]] if best_only else SEEDS
        for seed in seeds_to_plot:
            rec = cache.get(f"{n_parallel}_{seed}")
            if rec is None:
                continue
            exact = rec["E_exact"]
            e = np.array(rec["energy"])
            blown = np.abs(e - exact) > 50
            cutoff = int(np.argmax(blown)) if blown.any() else len(e)

            label = (f"$n_\\mathrm{{parallel}}={n_parallel}$" if best_only
                     else f"$n_\\mathrm{{parallel}}={n_parallel}$, seed={seed}")
            linestyle = "-" if best_only else ls[seed]

            if x_key == "iteration":
                x = np.arange(cutoff)
                x_end = cutoff
            else:
                x = np.cumsum(rec["sampling_time_s"])[:cutoff]
                x_end = x[-1] if len(x) else 0

            ax.plot(x, e[:cutoff], color=colors[n_parallel], linestyle=linestyle,
                     alpha=0.85, label=label)
            if cutoff < len(e):
                ax.plot(x_end, e[cutoff - 1], marker="x", color=colors[n_parallel], ms=6)

    if exact is not None:
        ax.axhline(exact, color="black", ls="--", lw=0.8, label="exact")
        ax.set_ylim(exact - 5, exact + 15)
    ax.set_ylabel("energy")
    return exact


def make_figures(cache, out_dir, tag, best_only=False):
    """Two separate single-panel figures: energy-vs-iteration and
    energy-vs-cumulative-QPU-time. tag distinguishes the output filenames."""
    setup_style(fontsize=11, scale=2.2)
    colors = {1: "#2166ac", 3: "#d62728", 5: "#2ca02c"}
    ls = {42: "-", 1: "--", 7: ":"}
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    _plot_one_panel(cache, ax, "iteration", best_only, colors, ls)
    ax.set_xlabel("iteration")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = out_dir / f"{tag}_convergence"
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved figure to {p.with_suffix('.pdf')} / .png")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    _plot_one_panel(cache, ax, "qpu_time", best_only, colors, ls)
    ax.set_xlabel("cumulative QPU access time (s)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = out_dir / f"{tag}_qpu_time"
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved figure to {p.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-parallel", type=int, nargs="*", default=N_PARALLEL_VALUES)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    cli_args = parser.parse_args()

    cache = load_cache() if cli_args.plot_only else run_experiments(cli_args.n_parallel, cli_args.seeds)
    print_summary(cache)

    out_dir = _REPO / "plots" / "embedding" / "parallel_embedding"
    make_figures(cache, out_dir, tag="parallel_embedding_np_seeds", best_only=False)
    make_figures(cache, out_dir, tag="parallel_embedding_np_best", best_only=True)
