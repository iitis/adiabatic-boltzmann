#!/usr/bin/env python3
"""
exact_ansatz_floor.py

Pure ansatz-expressivity floor for the sparse-RBM masks used in the sparsity
ablation (plot_sparsity_impact.py's _make_pruned_rbm): trains via exact
enumeration of all 2^N visible configurations, weighted by the exact |Psi(v)|^2
probability, instead of Monte Carlo sampling. Removes finite-sample noise and
MCMC bias entirely, isolating the error caused by the sparse connectivity
pattern itself (referee point 9's error-source decomposition).

Only feasible because N=16 (2^16 = 65536 states) is small enough to enumerate
directly. Reuses the same masks, h, learning rate, and regularization as the
existing classical/QPU sparsity ablation for direct comparability.

Usage (from repo root):
    python scripts/exper/exact_ansatz_floor.py
"""
import sys
import json
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "viz"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ising import TransverseFieldIsing1D
from plot_sparsity_impact import _make_pruned_rbm

N = 16
H_FIELD = 1.0
TOPOLOGY = "zephyr"
SEEDS = [42, 123, 456, 789, 1234]  # same seeds as the classical/QPU ablation cache
TARGET_SPARSITIES = [0.557, 0.682, 0.809, 0.877]
LR = 0.05
REG = 1e-3
N_ITERATIONS = 3000  # exact (noise-free) training can hit long plateaus at high
# sparsity before escaping to the true optimum (observed: one seed at
# sparsity=0.877 sat flat for ~950 iterations before dropping) — enough budget
# for every run to actually converge instead of being cut off mid-plateau
CACHE_PATH = _REPO / "plots" / "sparsity" / "cache_sparsity_ablation_exact.json"


def enumerate_configs(n: int) -> jax.Array:
    """All 2^n spin configurations in {-1,+1}^n, bit-unpacking trick (same
    convention as encoder.py's _build_kl_cache)."""
    indices = np.arange(2**n, dtype=np.int64)
    all_v = ((indices[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(
        np.float64
    ) * 2 - 1
    return jnp.asarray(all_v)  # (2^n, n)


def exact_train(rbm, ising, all_v, n_iterations: int, lr: float, reg: float):
    """Exact SR training: replace the Monte-Carlo sample average with an
    exact-probability-weighted average over all 2^N visible configurations.

    Reuses rbm.log_psi / rbm.gradient_log_psi / rbm.set_weights directly (via
    vmap) rather than reimplementing them, so the sparse mask's zeroing of
    forbidden W entries and forbidden gradients is enforced exactly as in the
    rest of the codebase — no separate mask bookkeeping to get wrong here.
    """
    energies = []
    for it in range(n_iterations):
        log_psi_all = jax.vmap(rbm.log_psi)(all_v)  # (2^N,)
        log_p = 2.0 * log_psi_all
        log_p = log_p - jax.scipy.special.logsumexp(log_p)
        p = jnp.exp(log_p)  # (2^N,) exact |Psi|^2, sums to 1

        E_loc = ising.local_energy_batch(all_v, rbm)  # (2^N,) exact, no sampling noise
        E_mean = float(jnp.sum(p * E_loc))
        energies.append(E_mean)

        grads = jax.vmap(rbm.gradient_log_psi)(all_v)
        O_a = grads["a"]  # (2^N, N)
        O_b = grads["b"]  # (2^N, M)
        O_W = grads["W"].reshape(all_v.shape[0], -1)  # (2^N, N*M), matches get_weights() order
        O = jnp.concatenate([O_a, O_b, O_W], axis=1)  # (2^N, P)

        mu = p @ O  # (P,)
        Obar = O - mu[None, :]
        Ec = E_loc - E_mean

        F = Obar.T @ (p * Ec)  # (P,)
        S = (Obar * p[:, None]).T @ Obar + reg * jnp.eye(O.shape[1])

        delta = jnp.linalg.solve(S, F)

        w = rbm.get_weights()
        rbm.set_weights(w - lr * delta)

        if it % 20 == 0:
            print(f"    iter {it:3d}: E = {E_mean:.6f}")

    return energies


def main():
    ising = TransverseFieldIsing1D(size=N, h=H_FIELD)
    all_v = enumerate_configs(N)
    E_exact = ising.exact_ground_energy()
    print(f"Exact ground energy (N={N}, h={H_FIELD}): {E_exact:.6f}")

    results = {}
    for ts in TARGET_SPARSITIES:
        print(f"\n=== target_sparsity={ts} ===")
        per_seed = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            rbm = _make_pruned_rbm(TOPOLOGY, N, ts, seed, live=True)
            energies = exact_train(rbm, ising, all_v, N_ITERATIONS, LR, REG)
            elapsed = time.perf_counter() - t0

            tail = max(1, len(energies) // 5)
            E_final = float(np.nanmean(energies[-tail:]))
            rel_err = abs(E_final - E_exact) / abs(E_exact)
            print(
                f"  seed={seed:5d}  E_final={E_final:.6f}  "
                f"rel_err={rel_err:.4%}  ({elapsed:.1f}s)"
            )
            per_seed.append(
                {
                    "seed": seed,
                    "energy_history": energies,
                    "E_final": E_final,
                    "rel_error": rel_err,
                    "sparsity": rbm.sparsity(),
                    "n_params": rbm.n_parameters(),
                    "elapsed_s": elapsed,
                }
            )

        rel_errors = [r["rel_error"] for r in per_seed]
        best = min(per_seed, key=lambda r: r["rel_error"])
        print(
            f"  -> best rel_err={best['rel_error']:.4%} (seed={best['seed']}), "
            f"mean={np.mean(rel_errors):.4%}, std={np.std(rel_errors):.4%}"
        )

        results[str(ts)] = {
            "E_exact": E_exact,
            "per_seed": per_seed,
            "best_rel_error": best["rel_error"],
            "best_E_final": best["E_final"],
            "best_seed": best["seed"],
            "mean_rel_error": float(np.mean(rel_errors)),
            "std_rel_error": float(np.std(rel_errors)),
            "sparsity": best["sparsity"],
            "n_params": best["n_params"],
        }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
