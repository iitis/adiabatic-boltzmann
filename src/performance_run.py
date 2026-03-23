"""
Performance benchmark run.

Sweeps over:
  - sizes:           16, 32, 64
  - learning rates:  0.1, 0.01
  - samplers:        custom/metropolis, dimod/simulated_annealing, dimod/pegasus
  - seeds:           1, 42

Fixed:
  - model:           1D TFIM, h=0.5
  - rbm:             FullyConnectedRBM, n_hidden = N
  - n_samples:       1000
  - regularization:  1e-3
  - n_iterations:    300

D-Wave budget guard:
  QPU access time is logged to time.json (in milliseconds by DimodSampler).
  Time is cumulative across all sessions — never reset.
  Before each D-Wave experiment, we check the accumulated time and abort
  the entire D-Wave sweep if more than 20 minutes (1_200_000 ms) have been used.

Skip logic:
  Set SKIP_FIRST_N to skip the first N experiments in the sweep order.
  Useful for resuming after a partial run without re-running completed work.
"""

import json
import itertools
import numpy as np
from argparse import Namespace
from pathlib import Path

from helpers import save_results
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

SIZES = [16, 32, 64]
H = 0.5
LEARNING_RATES = [0.1, 0.01]
SEEDS = [1, 42]
N_SAMPLES = 1000
REGULARIZATION = 1e-3
N_ITERATIONS = 300
OUTPUT_DIR = "results/"

SAMPLER_METHODS = [
    ("custom", "metropolis"),
    ("dimod", "simulated_annealing"),
    ("dimod", "pegasus"),
]

# Set to N to skip the first N experiments in sweep order (0 = no skip)
SKIP_FIRST_N = 5

# D-Wave QPU budget — cumulative across all sessions, never reset
DWAVE_BUDGET_MS = 20 * 60 * 1000  # 20 minutes in milliseconds
DWAVE_TIME_FILE = Path("time.json")
DWAVE_SAMPLERS = {"pegasus", "zephyr"}


# ---------------------------------------------------------------------------
# QPU time helpers
# ---------------------------------------------------------------------------


def read_qpu_time_ms() -> float:
    """Return accumulated QPU access time in milliseconds from time.json."""
    if not DWAVE_TIME_FILE.exists():
        return 0.0
    try:
        with DWAVE_TIME_FILE.open("r") as f:
            return float(json.load(f).get("time_ms", 0.0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 0.0


def qpu_budget_exceeded() -> bool:
    used = read_qpu_time_ms()
    if used >= DWAVE_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] Accumulated QPU time {used / 60_000:.2f} min "
            f">= limit {DWAVE_BUDGET_MS / 60_000:.0f} min. "
            "Skipping remaining D-Wave experiments."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------


def run_experiment(args: Namespace) -> bool:
    """Run a single experiment. Returns True on success, False on failure."""
    np.random.seed(args.seed)

    ising = TransverseFieldIsing1D(args.size, args.h)
    rbm = FullyConnectedRBM(args.size, args.n_hidden)

    if args.sampler == "custom":
        sampler = ClassicalSampler(method=args.sampling_method)
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.sampling_method)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    trainer_config = {
        "learning_rate": args.learning_rate,
        "n_iterations": args.iterations,
        "n_samples": args.n_samples,
        "regularization": args.regularization,
    }

    try:
        trainer = Trainer(rbm, ising, sampler, trainer_config)
        history = trainer.train()
        save_results(args, history, ising)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        # Always close SDK connections to prevent file descriptor leaks
        if hasattr(sampler, "sampler") and hasattr(sampler.sampler, "client"):
            try:
                sampler.sampler.client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    used_min = read_qpu_time_ms() / 60_000
    print(
        f"QPU time budget : {DWAVE_BUDGET_MS / 60_000:.0f} min total  |  "
        f"already used: {used_min:.2f} min  |  "
        f"remaining: {max(0.0, DWAVE_BUDGET_MS / 60_000 - used_min):.2f} min"
    )

    if qpu_budget_exceeded():
        print(
            "Budget already exceeded before run started. No D-Wave experiments will run."
        )

    if SKIP_FIRST_N > 0:
        print(f"Skipping first {SKIP_FIRST_N} experiments in sweep order.")

    total = len(SIZES) * len(LEARNING_RATES) * len(SAMPLER_METHODS) * len(SEEDS)
    done = 0
    n_skipped = 0

    for size, lr, (sampler, sampling_method), seed in itertools.product(
        SIZES, LEARNING_RATES, SAMPLER_METHODS, SEEDS
    ):
        # Skip the first SKIP_FIRST_N without running them
        if n_skipped < SKIP_FIRST_N:
            n_skipped += 1
            print(
                f"  [skip {n_skipped}/{SKIP_FIRST_N}] "
                f"N={size} lr={lr} {sampler}/{sampling_method} seed={seed}"
            )
            continue

        # Check QPU budget before every D-Wave experiment
        is_dwave = sampling_method in DWAVE_SAMPLERS
        if is_dwave and qpu_budget_exceeded():
            remaining = total - done - n_skipped
            print(f"  Skipping {remaining} remaining experiments.")
            break

        args = Namespace(
            model="1d",
            size=size,
            h=H,
            rbm="full",
            n_hidden=size,
            sampler=sampler,
            sampling_method=sampling_method,
            iterations=N_ITERATIONS,
            learning_rate=lr,
            regularization=REGULARIZATION,
            n_samples=N_SAMPLES,
            output_dir=OUTPUT_DIR,
            seed=seed,
            visualize=False,
        )

        used_ms = read_qpu_time_ms()
        print(
            f"\n[{done + n_skipped + 1}/{total}] "
            f"N={size} lr={lr} {sampler}/{sampling_method} seed={seed}"
            + (f"  QPU used={used_ms / 60_000:.2f}min" if is_dwave else "")
        )

        success = run_experiment(args)

        if not success:
            print("  Retrying once...")
            success = run_experiment(args)

            if not success:
                print("  Retry failed — aborting sweep.")
                break

        done += 1

    print(f"\nDone. {done}/{total} experiments completed  ({n_skipped} skipped).")
    print(f"Total QPU time used: {read_qpu_time_ms() / 60_000:.2f} min")
