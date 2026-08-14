#!/usr/bin/env python3
"""
embedding_algo_comparison.py

Short experiment: does the embedding *algorithm* used by DimodSampler
(busclique.find_biclique_embedding, the current default, vs the older
minorminer.find_embedding it replaced in commit 03d0f69b4 on 2026-06-19)
change RBM-VMC training outcomes on real QPU hardware?

Motivation: every archived "healthy" real-QPU result we tried to reproduce
was committed on 2026-05-11 (f364e7bd6) -- over a month before the busclique
switch. Fresh re-runs with today's busclique-based embedding failed to
reproduce those archives 3/3 times. A one-off diagnostic with the old
minorminer algorithm reached -8.31 (vs exact -8.51) by iteration 10 -- far
faster than busclique ever got in 300 iterations -- but then hit a NaN abort
at iteration 34. This script runs both algorithms across multiple seeds to
see whether that pattern (minorminer reaches better energy but is less
numerically stable) replicates, or whether it was a one-off draw.

Config: N=8, h=0.5, pegasus, lr=0.1, reg=1e-5, n_samples=1000 -- the
archived-healthy config (results/tfim_1d/8/dimod/pegasus/result_1d_h0.5_
rbmfull_nh8_lr0.1_reg1e-05_ns1000_seed42_iter300_cem0.json.gz, 0.09% error).
150 iterations per run (archive's convergence "jump" happens around
iteration 140), 3 seeds x 2 algorithms = 6 runs.

DimodSampler._get_composite is monkeypatched for the minorminer trials only
(restored for busclique trials) -- src/sampler.py is never modified.

Usage (from repo root):
    python scripts/exper/embedding_algo_comparison.py
    python scripts/exper/embedding_algo_comparison.py --plot-only
"""
import argparse
import json
import sys
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
CACHE_PATH = Path(__file__).resolve().parent / "cache_embedding_algo_comparison.json"

N, H, LR, REG, N_SAMPLES = 8, 0.5, 0.1, 1e-05, 1000
N_ITERS = 150
SEEDS = [42, 1, 7]
ALGORITHMS = ["busclique", "minorminer"]

_ORIGINAL_GET_COMPOSITE = DimodSampler._get_composite


def _get_composite_minorminer(self, bqm, solver_name, rbm):
    """Old (pre-03d0f69b4) embedding path: generic minorminer.find_embedding
    over the whole hardware edgelist, instead of busclique's specialized
    biclique search."""
    from dwave.system import DWaveSampler, FixedEmbeddingComposite
    import minorminer

    cache_key = (getattr(self, "_n_cache", self.n_visible), solver_name, "minorminer")
    if cache_key not in self._embedding_cache:
        dwave_sampler = DWaveSampler(solver=solver_name)
        embedding = minorminer.find_embedding(
            list(bqm.quadratic.keys()), dwave_sampler.edgelist
        )
        if not embedding:
            raise RuntimeError(
                f"minorminer failed to find an embedding for "
                f"n_visible={self.n_visible} on solver '{solver_name}'."
            )
        composite = FixedEmbeddingComposite(dwave_sampler, embedding)
        chains = [len(v) for v in embedding.values()]
        self._last_chain_stats = {
            "max_chain": max(chains), "mean_chain": sum(chains) / len(chains),
            "qubits": sum(chains),
        }
        self._embedding_cache[cache_key] = composite
    else:
        composite = self._embedding_cache[cache_key]
    return composite, False, cache_key


def _check_budget():
    used_ms = read_qpu_time_ms(QPU_TIME_PATH)
    if used_ms >= QPU_BUDGET_MS:
        raise RuntimeError(
            f"QPU budget exceeded: {used_ms / 60_000:.2f} min used >= "
            f"{QPU_BUDGET_MS / 60_000:.0f} min cap. Aborting."
        )
    return used_ms


def _run_one(algorithm, seed):
    if algorithm == "minorminer":
        DimodSampler._get_composite = _get_composite_minorminer
    else:
        DimodSampler._get_composite = _ORIGINAL_GET_COMPOSITE

    used_before = _check_budget()
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    ising = TransverseFieldIsing1D(size=N, h=H)
    rbm = FullyConnectedRBM(N, N, model_key)
    sampler = DimodSampler(method="pegasus")
    sampler._last_chain_stats = None
    config = {
        "n_samples": N_SAMPLES,
        "n_iterations": N_ITERS,
        "learning_rate": LR,
        "regularization": REG,
        "seed": seed,
        "save_checkpoints": False,
    }
    trainer = Trainer(rbm=rbm, ising_model=ising, sampler=sampler, config=config)
    history = trainer.train()
    used_after = read_qpu_time_ms(QPU_TIME_PATH)

    e = np.array(history["energy"])
    nan_iter = int(np.argmax(np.isnan(e))) if np.isnan(e).any() else None
    exact = float(ising.exact_ground_energy())

    return {
        "algorithm": algorithm,
        "seed": seed,
        "energy": [float(x) for x in e],
        "exact": exact,
        "nan_iter": nan_iter,
        "chain_stats": sampler._last_chain_stats,
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


def run_experiments():
    cache = load_cache()
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            key = f"{algorithm}_{seed}"
            if key in cache:
                print(f"  {key}: cached, skipping")
                continue
            used = _check_budget()
            print(f"  {key}: running ({used / 60_000:.2f} min QPU used so far)")
            rec = _run_one(algorithm, seed)
            cache[key] = rec
            save_cache(cache)
            e = rec["energy"]
            final = e[rec["nan_iter"] - 1] if rec["nan_iter"] else e[-1]
            status = f"NaN at iter {rec['nan_iter']}" if rec["nan_iter"] else "completed"
            rel_err = abs((final - rec["exact"]) / rec["exact"])
            print(f"    -> {status}, final={final:.4f} (exact={rec['exact']:.4f}, "
                  f"rel_err={rel_err:.4f}), qpu_ms={rec['qpu_time_ms_used']:.0f}, "
                  f"chains={rec['chain_stats']}")
    return cache


def print_summary(cache):
    print(f"\n{'algorithm':>12} {'seed':>5} {'status':>14} {'final':>9} {'rel_err':>8} {'max_chain':>9} {'mean_chain':>10}")
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            rec = cache.get(f"{algorithm}_{seed}")
            if rec is None:
                continue
            e = rec["energy"]
            final = e[rec["nan_iter"] - 1] if rec["nan_iter"] else e[-1]
            status = f"NaN@{rec['nan_iter']}" if rec["nan_iter"] else "completed"
            rel_err = abs((final - rec["exact"]) / rec["exact"])
            cs = rec["chain_stats"] or {}
            print(f"{algorithm:>12} {seed:>5} {status:>14} {final:>9.4f} {rel_err:>8.4f} "
                  f"{cs.get('max_chain', '-'):>9} {cs.get('mean_chain', '-'):>10}")


def make_figure(cache, out_path):
    setup_style(fontsize=11, scale=2.2)
    colors = {"busclique": "#2166ac", "minorminer": "#d62728"}
    ls = {42: "-", 1: "--", 7: ":"}
    fig, ax = plt.subplots(figsize=(6, 4))

    exact = None
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            rec = cache.get(f"{algorithm}_{seed}")
            if rec is None:
                continue
            exact = rec["exact"]
            e = np.array(rec["energy"])
            # clip at divergence point to avoid swamping the y-axis
            blown = np.abs(e - exact) > 50
            cutoff = int(np.argmax(blown)) if blown.any() else len(e)
            ax.plot(np.arange(cutoff), e[:cutoff], color=colors[algorithm],
                     linestyle=ls[seed], label=f"{algorithm}, seed={seed}", alpha=0.85)
            if cutoff < len(e):
                ax.plot(cutoff, e[cutoff - 1], marker="x", color=colors[algorithm], ms=6)
    if exact is not None:
        ax.axhline(exact, color="black", ls="--", lw=0.8, label="exact")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_ylim(exact - 5, exact + 15)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    print(f"Saved figure to {out_path.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    cli_args = parser.parse_args()

    cache = load_cache() if cli_args.plot_only else run_experiments()
    print_summary(cache)
    out_path = _REPO / "plots" / "embedding" / "parallel_embedding" / "embedding_algo_comparison"
    make_figure(cache, out_path)
