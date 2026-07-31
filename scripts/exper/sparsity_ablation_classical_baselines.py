#!/usr/bin/env python3
"""
sparsity_ablation_classical_baselines.py

Extends the sparsity ablation (plot_sparsity_impact.py's
_run_one_sparsity_ablation, which only ever used ClassicalSampler("metropolis"))
with two more classical samplers on the *same* pruned Zephyr masks, same
hyperparameters, same seeds: simulated annealing and persistent block Gibbs.
Purpose: test whether the large gap between the exact-ansatz floor and
practical training (Metropolis, per the existing cache) is specific to one
sampler or general to non-exact classical sampling. Uses no QPU time at all
(ClassicalSampler is pure JAX/CPU) -- the hardware masks are loaded from the
disk-cached hardware graph (embeddings/_hwgraph_Advantage2_system1_live.json),
not fetched live.

Usage (from repo root):
    python scripts/exper/sparsity_ablation_classical_baselines.py --method simulated_annealing
    python scripts/exper/sparsity_ablation_classical_baselines.py --method gibbs
    python scripts/exper/sparsity_ablation_classical_baselines.py --method simulated_annealing --iters 20 --seeds 42
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))

import jax
jax.config.update("jax_enable_x64", True)

from ising import TransverseFieldIsing1D
from model import DWaveTopologyRBM
from sampler import ClassicalSampler
from encoder import Trainer
from plot_sparsity_impact import _make_pruned_rbm

N = 16
H = 1.0
TOPOLOGY = "zephyr"
LR = 0.05
REG = 1e-3
N_SAMPLES = 500
N_ITERS_DEFAULT = 300
ALL_SEEDS = [42, 123, 456, 789, 1234]
TARGET_SPARSITIES = [0.557, 0.682, 0.809, 0.877]

# Label for the unpruned native mask -- can't go through _make_pruned_rbm
# (which requires target_sparsity > native_sparsity); its true sparsity is
# 0.42578125, verified against cache_full.json's "16_1_1_zephyr_*" entries.
NATIVE_LABEL = "native"

_REPO = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _REPO / "plots" / "sparsity"


def run_one(method, target_sparsity, seed, n_iters):
    np.random.seed(seed)
    if target_sparsity == NATIVE_LABEL:
        # Same construction as _make_rbm's zephyr branch / _make_pruned_rbm:
        # subgraph-selection seed is always 42 (matches every other cache in
        # this study), only the weight-init key varies with the outer seed.
        key = jax.random.PRNGKey(seed)
        rbm = DWaveTopologyRBM(N, N, key, solver=TOPOLOGY, seed=42, live=True)
    else:
        rbm = _make_pruned_rbm(TOPOLOGY, N, target_sparsity, seed, live=True)
    ising = TransverseFieldIsing1D(size=N, h=H)
    sampler = ClassicalSampler(method=method)
    config = {
        "n_samples": N_SAMPLES,
        "n_iterations": n_iters,
        "learning_rate": LR,
        "regularization": REG,
        "stop_at_convergence": False,
        "save_checkpoints": False,
    }
    trainer = Trainer(rbm=rbm, ising_model=ising, sampler=sampler, config=config)
    t0 = time.perf_counter()
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    energies = np.array(history["energy"])
    E_exact = ising.exact_ground_energy()
    tail = max(1, len(energies) // 5)
    E_final = float(np.nanmean(energies[-tail:]))
    rel_err = abs(E_final - E_exact) / abs(E_exact)

    return {
        "energy_history": [float(e) for e in energies],
        "E_exact": float(E_exact),
        "E_final": E_final,
        "rel_error": rel_err,
        "n_params": rbm.n_parameters(),
        "sparsity": rbm.sparsity(),
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["simulated_annealing", "gibbs"],
                    choices=["simulated_annealing", "gibbs"])
    p.add_argument("--iters", type=int, default=N_ITERS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    p.add_argument("--sparsities", type=float, nargs="+", default=TARGET_SPARSITIES)
    p.add_argument("--no-native", action="store_true",
                    help="Skip the unpruned native-mask point (sparsity 0.426, included by default).")
    args = p.parse_args()

    sparsity_points = list(args.sparsities) + ([] if args.no_native else [NATIVE_LABEL])

    for method in args.methods:
        cache_path = CACHE_DIR / f"cache_sparsity_ablation_{method}.json"
        cache = {}
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)

        total = len(sparsity_points) * len(args.seeds)
        done = 0
        print(f"=== method={method} ===")
        for ts in sparsity_points:
            for seed in args.seeds:
                key = f"{N}_{ts}_{H}_{TOPOLOGY}_{seed}"
                done += 1
                if key in cache:
                    print(f"  [{done}/{total}] {key} -- cached, skipping")
                    continue
                print(f"  [{done}/{total}] {key} -- running ({args.iters} iters)...")
                rec = run_one(method, ts, seed, args.iters)
                cache[key] = rec
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                print(f"      rel_error={rec['rel_error']:.4%}  "
                      f"n_iters_logged={len(rec['energy_history'])}  "
                      f"elapsed={rec['elapsed_s']:.1f}s")

        print(f"Saved to {cache_path}\n")


if __name__ == "__main__":
    main()
