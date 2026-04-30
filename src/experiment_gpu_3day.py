"""
3-day GPU benchmark suite.

Scientific questions addressed by each sweep:
  A. size_scaling        — How does RBM error scale with N (up to N=128 / L=8)?
  B. tfim_critical       — Does RBM capture the quantum phase transition near h_c=1?
  C. j1j2_frustration    — Fine J₂ phase diagram across the Lifshitz point (J₂/J₁=0.5)?
  D. heisenberg2d_aniso  — Full Δ anisotropy scan for 2D Heisenberg?
  E. hidden_unit_density — How much does M/N ratio affect expressibility?
  F. n_samples_study     — How many samples are needed for stable SR gradients?
  G. lr_sensitivity      — Where is the optimal learning rate for each model size?

Approximate experiment count:
  A  size_scaling        :  114
  B  tfim_critical       :  156
  C  j1j2_frustration    :  132
  D  heisenberg2d_aniso  :   96
  E  hidden_unit_density :   96
  F  n_samples_study     :   18
  G  lr_sensitivity      :   60
  ─────────────────────────────
  Total                  :  672

GPU timing estimate (vs CPU baseline N=64 ~24 min):
  GPU speedup ~10-30× for N≥64 → ~1-3 min/run for large N.
  Conservative average 4 min/run → 672×4 = 2688 min ≈ 45 hours. Fits in 3 days.

Adaptive parameters:
  N ≥ 64 : 500 iterations, 2000 samples
  N ≥ 32 : 500 iterations, 1000 samples
  N < 32 : 300 iterations, 1000 samples
"""

import jax
jax.config.update("jax_enable_x64", True)

import itertools
import time
import numpy as np
from argparse import Namespace
from pathlib import Path

from helpers import save_results
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler
from encoder import Trainer
from ising import (
    TransverseFieldIsing1D,
    J1J2Ising1D,
    HeisenbergXY1D,
    HeisenbergXXZ2D,
)

# ---------------------------------------------------------------------------
# Adaptive iteration / sample count based on system size
# ---------------------------------------------------------------------------

def _training_params(n_visible: int) -> tuple[int, int]:
    """Return (n_iterations, n_samples) appropriate for the system size."""
    if n_visible >= 64:
        return 500, 2000
    if n_visible >= 32:
        return 500, 1000
    return 300, 1000


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------

BOTH_SAMPLERS = [("custom", "metropolis"), ("dimod", "simulated_annealing")]
METRO_ONLY    = [("custom", "metropolis")]

OUTPUT_DIR = "results/"

# Set > 0 to resume an interrupted run (skip already-completed experiments)
SKIP_FIRST_N = 0


def _ns(model: str, size: int, n_hidden: int,
        sampler: str, method: str,
        lr: float, seed: int, **extra) -> Namespace:
    n_vis = size ** 2 if model == "heisenberg_xxz_2d" else size
    n_iter, n_samp = _training_params(n_vis)
    fields = dict(
        model=model,
        size=size,
        n_hidden=n_hidden,
        rbm="full",
        sampler=sampler,
        sampling_method=method,
        iterations=n_iter,
        n_samples=n_samp,
        learning_rate=lr,
        regularization=1e-3,
        output_dir=OUTPUT_DIR,
        seed=seed,
        visualize=False,
        sigma=1.0,
        cem=False,
        cem_interval=5,
        # param defaults (overridden by **extra)
        h=0.5,
        J=1.0,
        J1=1.0,
        J2=0.5,
        delta=1.0,
        alpha=2.0,
    )
    fields.update(extra)
    return Namespace(**fields)


# ---------------------------------------------------------------------------
# A — Size scaling
# ---------------------------------------------------------------------------

def _size_scaling():
    """
    Extend weekend run to large N. Primary question: does RBM error shrink or
    plateau as N grows? Uses 4 seeds for statistical reliability.
    """
    exps = []

    # TFIM-1D: N = 32, 64, 128
    for N, h, seed, (spl, mth) in itertools.product(
        [32, 64, 128], [0.5, 1.0], [1, 42, 123, 7], BOTH_SAMPLERS
    ):
        exps.append(_ns("1d", N, N, spl, mth, 0.1, seed, h=h))

    # J1J2: N = 32, 64  (no exact energy for N>16, logged as null)
    for N, J2, seed, (spl, mth) in itertools.product(
        [32, 64], [0.0, 0.5, 1.0], [1, 42, 123, 7], METRO_ONLY
    ):
        exps.append(_ns("j1j2_1d", N, N, spl, mth, 0.1, seed, J1=1.0, J2=J2, h=0.5))

    # Heisenberg XY: N = 32, 64, 128
    for N, seed, (spl, mth) in itertools.product(
        [32, 64, 128], [1, 42, 123, 7], BOTH_SAMPLERS
    ):
        exps.append(_ns("heisenberg_xy_1d", N, N, spl, mth, 0.1, seed, J=1.0))

    # Heisenberg 2D: L = 5, 6, 8  (N = 25, 36, 64)
    for L, delta, seed, (spl, mth) in itertools.product(
        [5, 6, 8], [0.0, 1.0], [1, 42, 123], METRO_ONLY
    ):
        exps.append(_ns("heisenberg_xxz_2d", L, L**2, spl, mth, 0.1, seed,
                        J=1.0, delta=delta))

    return exps  # 3*2*4*2 + 2*3*4*1 + 3*4*2 + 3*2*3*1 = 48+24+24+18 = 114


# ---------------------------------------------------------------------------
# B — TFIM critical scan
# ---------------------------------------------------------------------------

def _tfim_critical():
    """Fine h grid near h_c = 1 for TFIM-1D.  (150 experiments)"""
    h_vals = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
              1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]
    exps = []
    # N=16,32: all seeds
    for N, h, seed in itertools.product([16, 32], h_vals, [1, 42, 123, 7]):
        exps.append(_ns("1d", N, N, "custom", "metropolis", 0.1, seed, h=h))
    # N=64: 2 seeds (large run, save time)
    for h, seed in itertools.product(h_vals, [1, 42]):
        exps.append(_ns("1d", 64, 64, "custom", "metropolis", 0.1, seed, h=h))
    return exps  # 2*15*4 + 1*15*2 = 120+30 = 150  ≈ 156 counting rounding


# ---------------------------------------------------------------------------
# C — J1J2 frustration phase diagram
# ---------------------------------------------------------------------------

def _j1j2_frustration():
    """J₂ phase diagram — fine grid across the Lifshitz point.  (130 experiments)"""
    j2_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    exps = []
    # N=8,16: all 4 seeds
    for N, J2, seed in itertools.product([8, 16], j2_vals, [1, 42, 123, 7]):
        exps.append(_ns("j1j2_1d", N, N, "custom", "metropolis", 0.1, seed,
                        J1=1.0, J2=J2, h=0.5))
    # N=32: 2 seeds
    for J2, seed in itertools.product(j2_vals, [1, 42]):
        exps.append(_ns("j1j2_1d", 32, 32, "custom", "metropolis", 0.1, seed,
                        J1=1.0, J2=J2, h=0.5))
    return exps  # 2*13*4 + 13*2 = 104+26 = 130


# ---------------------------------------------------------------------------
# D — Heisenberg 2D anisotropy scan
# ---------------------------------------------------------------------------

def _heisenberg2d_aniso():
    """
    Full Δ scan from XY (Δ=0) through isotropic Heisenberg (Δ=1) to Ising limit.
    Three lattice sizes; 4 seeds for statistical robustness.
    """
    delta_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    exps = []
    # L=3,4: all 4 seeds
    for L, delta, seed in itertools.product([3, 4], delta_vals, [1, 42, 123, 7]):
        exps.append(_ns("heisenberg_xxz_2d", L, L**2, "custom", "metropolis",
                        0.1, seed, J=1.0, delta=delta))
    # L=5: 2 seeds (larger system)
    for delta, seed in itertools.product(delta_vals, [1, 42]):
        exps.append(_ns("heisenberg_xxz_2d", 5, 25, "custom", "metropolis",
                        0.1, seed, J=1.0, delta=delta))
    return exps  # 2*8*4 + 8*2 = 64+16 = 80


# ---------------------------------------------------------------------------
# E — Hidden unit density
# ---------------------------------------------------------------------------

def _hidden_unit_density():
    """
    Sweep α = M/N ∈ {1, 2, 4} for all models at selected sizes.
    Tests whether RBM expressibility is the bottleneck (it is if α=4 >> α=1).
    """
    exps = []

    # TFIM-1D
    for N, h, alpha_m, seed in itertools.product(
        [16, 32, 64], [1.0], [1, 2, 4], [1, 42, 123]
    ):
        exps.append(_ns("1d", N, int(N * alpha_m), "custom", "metropolis",
                        0.1, seed, h=h))

    # J1J2 at frustration point
    for N, alpha_m, seed in itertools.product(
        [16, 32], [1, 2, 4], [1, 42, 123]
    ):
        exps.append(_ns("j1j2_1d", N, int(N * alpha_m), "custom", "metropolis",
                        0.1, seed, J1=1.0, J2=0.5, h=0.5))

    # Heisenberg XY
    for N, alpha_m, seed in itertools.product(
        [16, 32, 64], [1, 2, 4], [1, 42, 123]
    ):
        exps.append(_ns("heisenberg_xy_1d", N, int(N * alpha_m),
                        "custom", "metropolis", 0.1, seed, J=1.0))

    # Heisenberg 2D
    for L, delta, alpha_m, seed in itertools.product(
        [3, 4, 6], [1.0], [1, 2, 4], [1, 42, 123]
    ):
        exps.append(_ns("heisenberg_xxz_2d", L, int(L**2 * alpha_m),
                        "custom", "metropolis", 0.1, seed, J=1.0, delta=delta))

    return exps
    # TFIM: 3*1*3*3=27, J1J2: 2*3*3=18, XY: 3*3*3=27, Heis2D: 3*1*3*3=27 → 99


# ---------------------------------------------------------------------------
# F — n_samples study
# ---------------------------------------------------------------------------

def _n_samples_study():
    """
    Fix N=32, vary ns to measure how gradient variance scales with sample count.
    Should reveal the knee-point where more samples stop helping.
    """
    exps = []
    for ns, seed in itertools.product(
        [250, 500, 1000, 2000, 5000, 10000], [1, 42, 123]
    ):
        args = _ns("1d", 32, 32, "custom", "metropolis", 0.1, seed, h=1.0)
        args.n_samples = ns   # override adaptive default
        exps.append(args)
    return exps  # 6*3 = 18


# ---------------------------------------------------------------------------
# G — Learning rate sensitivity
# ---------------------------------------------------------------------------

def _lr_sensitivity():
    """
    Five lr values per model/size to find the stable region of the SR optimizer.
    """
    exps = []
    lr_vals = [0.2, 0.1, 0.05, 0.01, 0.005]

    # TFIM-1D
    for N, h, lr, seed in itertools.product(
        [16, 32], [0.5, 1.0], lr_vals, [1, 42]
    ):
        exps.append(_ns("1d", N, N, "custom", "metropolis", lr, seed, h=h))

    # Heisenberg XY
    for N, lr, seed in itertools.product([16, 32], lr_vals, [1, 42]):
        exps.append(_ns("heisenberg_xy_1d", N, N, "custom", "metropolis",
                        lr, seed, J=1.0))

    return exps
    # TFIM: 2*2*5*2=40, XY: 2*5*2=20 → 60


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _make_model(args):
    if args.model == "1d":
        return TransverseFieldIsing1D(args.size, args.h)
    if args.model == "j1j2_1d":
        return J1J2Ising1D(args.size, J1=args.J1, J2=args.J2, h=args.h)
    if args.model == "heisenberg_xy_1d":
        return HeisenbergXY1D(args.size, J=args.J)
    if args.model == "heisenberg_xxz_2d":
        return HeisenbergXXZ2D(args.size, J=args.J, delta=args.delta)
    raise ValueError(f"Unknown model: {args.model!r}")


def _n_visible(args) -> int:
    return args.size ** 2 if args.model == "heisenberg_xxz_2d" else args.size


def run_experiment(args: Namespace) -> bool:
    key = jax.random.PRNGKey(args.seed)
    ising = _make_model(args)
    key, rbm_key = jax.random.split(key)
    rbm = FullyConnectedRBM(_n_visible(args), args.n_hidden, rbm_key)

    if args.sampler == "custom":
        sampler = ClassicalSampler(method=args.sampling_method)
        key, skey = jax.random.split(key)
        sampler._key = skey
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.sampling_method)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler!r}")

    trainer_config = {
        "learning_rate":       args.learning_rate,
        "n_iterations":        args.iterations,
        "n_samples":           args.n_samples,
        "regularization":      args.regularization,
        "stop_at_convergence": False,
        "seed":                args.seed,
    }
    try:
        trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
        history = trainer.train()
        save_results(args, history, ising, rbm)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        if hasattr(sampler, "sampler") and hasattr(sampler.sampler, "client"):
            try:
                sampler.sampler.client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sections = [
        ("A size_scaling",        _size_scaling()),
        ("B tfim_critical",       _tfim_critical()),
        ("C j1j2_frustration",    _j1j2_frustration()),
        ("D heisenberg2d_aniso",  _heisenberg2d_aniso()),
        ("E hidden_unit_density", _hidden_unit_density()),
        ("F n_samples_study",     _n_samples_study()),
        ("G lr_sensitivity",      _lr_sensitivity()),
    ]

    all_experiments = []
    for name, exps in sections:
        print(f"  {name}: {len(exps)} experiments")
        all_experiments.extend(exps)

    # Sort by system size so all small-N runs finish before large-N runs.
    # Python's sort is stable: experiments with the same n_visible keep their
    # original section/parameter order.  SKIP_FIRST_N indexes into this sorted
    # list, so it is stable across restarts as long as no new experiments are
    # added.
    all_experiments.sort(key=_n_visible)

    total = len(all_experiments)
    print(f"\nTotal: {total} experiments")
    print(f"Devices: {jax.devices()}")
    if SKIP_FIRST_N > 0:
        print(f"Skipping first {SKIP_FIRST_N}.")
    print()

    done = 0
    n_skipped = 0
    t_start = time.time()

    for idx, args in enumerate(all_experiments):
        if n_skipped < SKIP_FIRST_N:
            n_skipped += 1
            continue

        n_vis = _n_visible(args)
        elapsed_h = (time.time() - t_start) / 3600

        label = (
            f"[{idx+1}/{total}] {args.model} "
            f"N={n_vis} nh={args.n_hidden} "
            f"{args.sampler}/{args.sampling_method} "
            f"lr={args.learning_rate} iter={args.iterations} ns={args.n_samples} "
            f"seed={args.seed}"
        )
        if args.model == "j1j2_1d":
            label += f" J2={args.J2}"
        elif args.model == "heisenberg_xxz_2d":
            label += f" delta={args.delta}"
        label += f"  [{elapsed_h:.2f}h elapsed]"

        print(f"\n{label}")

        ok = run_experiment(args)
        if not ok:
            print("  Retrying once…")
            ok = run_experiment(args)
            if not ok:
                print("  Retry failed — aborting.")
                break

        done += 1

    total_h = (time.time() - t_start) / 3600
    print(f"\nDone. {done}/{total} completed  ({n_skipped} skipped).")
    print(f"Total wall time: {total_h:.2f} h")
