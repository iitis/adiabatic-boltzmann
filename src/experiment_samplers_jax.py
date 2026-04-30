"""
Weekend benchmarking sweep — new model variants.

Models:
  1. J₁-J₂ frustrated Ising chain   (j1j2_1d)
  2. Heisenberg XY chain             (heisenberg_xy_1d)
  3. Heisenberg XXZ 2D square lattice (heisenberg_xxz_2d)
  + TFIM 1D baseline for cross-model comparison

Samplers: custom/metropolis, dimod/simulated_annealing (no QPU).

Approximate experiment count and wall-time estimate:
  TFIM-1D       :  72 experiments  (baseline reference)
  J1J2-1D       :  60 experiments  (J₂ phase diagram)
  Heisenberg XY :  24 experiments
  Heisenberg 2D :  72 experiments
  ─────────────────────────────────
  Total         : 228 experiments

At ~3–5 min/run on CPU: 11–19 hours. Fits comfortably in a weekend.

Resumability: set SKIP_FIRST_N to skip already-completed experiments
(they run in deterministic order, so the skip index is stable across restarts).
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
# Fixed hyperparameters
# ---------------------------------------------------------------------------

N_ITERATIONS = 300
N_SAMPLES    = 1000
REGULARIZATION = 1e-3
OUTPUT_DIR   = "results/"

SAMPLER_METHODS = [
    ("custom",  "metropolis"),
    ("dimod",   "simulated_annealing"),
]

# Set > 0 to skip the first N experiments (for resuming interrupted runs)
SKIP_FIRST_N = 0


# ---------------------------------------------------------------------------
# Experiment factory helpers
# ---------------------------------------------------------------------------


def _ns(model, size, n_hidden, sampler, sampling_method, learning_rate, seed, **extra):
    """Build a Namespace with all fields required by Trainer and save_results."""
    fields = dict(
        model=model,
        size=size,
        n_hidden=n_hidden,
        rbm="full",
        sampler=sampler,
        sampling_method=sampling_method,
        iterations=N_ITERATIONS,
        learning_rate=learning_rate,
        regularization=REGULARIZATION,
        n_samples=N_SAMPLES,
        output_dir=OUTPUT_DIR,
        seed=seed,
        visualize=False,
        sigma=1.0,
        cem=False,
        cem_interval=5,
        # model param defaults — overridden by **extra below
        h=0.5,
        J=1.0,
        J1=1.0,
        J2=0.5,
        delta=1.0,
        alpha=2.0,
    )
    fields.update(extra)
    return Namespace(**fields)


def _tfim1d_sweep():
    """TFIM 1D: sweep h × size × lr × seed × sampler  (72 experiments)."""
    exps = []
    for h, size, lr, seed, (spl, mth) in itertools.product(
        [0.3, 0.5, 1.0], [8, 12, 16], [0.1, 0.01], [1, 42], SAMPLER_METHODS
    ):
        exps.append(_ns("1d", size, size, spl, mth, lr, seed, h=h))
    return exps


def _j1j2_sweep():
    """J1-J2 chain: sweep J₂ × size × seed × sampler  (60 experiments).

    J₁ = 1 (fixed), J₂ scans [0, 0.25, 0.5, 0.75, 1.0] to map the
    frustration-induced quantum phase transition (Lifshitz point at J₂/J₁ = 0.5).
    A single lr=0.1 is enough since the phase diagram is the primary interest.
    """
    exps = []
    for J2, size, seed, (spl, mth) in itertools.product(
        [0.0, 0.25, 0.5, 0.75, 1.0], [8, 12, 16], [1, 42], SAMPLER_METHODS
    ):
        exps.append(_ns(
            "j1j2_1d", size, size, spl, mth, 0.1, seed,
            J1=1.0, J2=J2, h=0.5,
        ))
    return exps


def _xy_sweep():
    """Heisenberg XY chain: sweep size × lr × seed × sampler  (24 experiments)."""
    exps = []
    for size, lr, seed, (spl, mth) in itertools.product(
        [8, 12, 16], [0.1, 0.01], [1, 42], SAMPLER_METHODS
    ):
        exps.append(_ns("heisenberg_xy_1d", size, size, spl, mth, lr, seed, J=1.0))
    return exps


def _heisenberg2d_sweep():
    """2D Heisenberg XXZ: sweep Δ × L × lr × seed × sampler  (72 experiments).

    Δ ∈ {0, 0.5, 1}: XY limit, partially anisotropic, isotropic Heisenberg.
    L ∈ {2, 3, 4}: N = 4, 9, 16 spins (exact reference available for all via ED).
    """
    exps = []
    for delta, L, lr, seed, (spl, mth) in itertools.product(
        [0.0, 0.5, 1.0], [2, 3, 4], [0.1, 0.01], [1, 42], SAMPLER_METHODS
    ):
        n_visible = L ** 2
        exps.append(_ns(
            "heisenberg_xxz_2d", L, n_visible, spl, mth, lr, seed,
            J=1.0, delta=delta,
        ))
    return exps


# ---------------------------------------------------------------------------
# Single experiment runner
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
    if args.model == "heisenberg_xxz_2d":
        return args.size ** 2
    return args.size


def run_experiment(args: Namespace) -> bool:
    """Run one experiment. Returns True on success, False on failure."""
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
        "learning_rate":   args.learning_rate,
        "n_iterations":    args.iterations,
        "n_samples":       args.n_samples,
        "regularization":  args.regularization,
        "stop_at_convergence": False,
        "seed":            args.seed,
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
# Main sweep
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    all_experiments = (
        _tfim1d_sweep()
        + _j1j2_sweep()
        + _xy_sweep()
        + _heisenberg2d_sweep()
    )

    total = len(all_experiments)
    print(f"Total experiments planned: {total}")
    print(f"Samplers: {[f'{s}/{m}' for s, m in SAMPLER_METHODS]}")
    if SKIP_FIRST_N > 0:
        print(f"Skipping first {SKIP_FIRST_N} experiments.")
    print()

    done = 0
    n_skipped = 0
    t_start = time.time()

    for idx, args in enumerate(all_experiments):
        label = (
            f"[{idx + 1}/{total}] "
            f"{args.model} N={args.size} "
            f"{args.sampler}/{args.sampling_method} "
            f"lr={args.learning_rate} seed={args.seed}"
        )
        # Model-specific param summary
        if args.model == "j1j2_1d":
            label += f" J2={args.J2}"
        elif args.model in ("heisenberg_xy_1d",):
            label += f" J={args.J}"
        elif args.model == "heisenberg_xxz_2d":
            label += f" delta={args.delta}"

        if n_skipped < SKIP_FIRST_N:
            n_skipped += 1
            print(f"  [skip {n_skipped}/{SKIP_FIRST_N}] {label}")
            continue

        elapsed_min = (time.time() - t_start) / 60
        print(f"\n{label}  (elapsed: {elapsed_min:.1f} min)")

        success = run_experiment(args)

        if not success:
            print("  Retrying once...")
            success = run_experiment(args)
            if not success:
                print("  Retry failed — aborting sweep.")
                break

        done += 1

    total_min = (time.time() - t_start) / 60
    print(f"\nDone. {done}/{total} experiments completed  ({n_skipped} skipped).")
    print(f"Total wall time: {total_min:.1f} min")
