"""
Experiment: KL divergence as predictor of VMC convergence quality.

Scientific question
-------------------
Is the KL divergence D_KL(q_sampler ∥ |Ψ|²) the primary determinant of VMC
convergence quality, and does this relationship intensify at the quantum
critical point of the 1D TFIM?

Competing hypotheses
--------------------
H1 (KL hypothesis):   final KL is the primary predictor of final energy error.
H2 (SR hypothesis):   SR optimizer ill-conditioning (grad norm) is the primary
                       predictor, independently of sampler quality.

Experiment structure
--------------------
Part 1 — h-sweep (main result):
    1D, N=8, h ∈ {0.5, 1.0, 1.5, 2.0}, 5 samplers, 15 seeds → 300 runs.
    Primary output: Spearman ρ(KL, error) and ρ(grad_norm, error) per h.

Part 2 — size scaling (supplementary):
    1D, h=1.0 (critical point), N ∈ {4, 8, 16}, 5 samplers, 10 seeds → 150 runs.
    Primary output: does ρ(KL, error) grow with N?

Part 3 — 2D geometry (supplementary):
    2D, L=2 (N=4 spins), h ∈ {0.5, 1.0}, 5 samplers, 10 seeds → 100 runs.
    Primary output: does the 1D result hold in 2D?

Total: ~550 runs, all free classical samplers (no QPU).

Controlled hyperparameters (fixed across ALL runs)
---------------------------------------------------
  n_hidden    = n_visible  (α = 1 hidden-to-visible ratio)
  n_samples   = 1000
  lr          = 0.1
  reg         = 1e-5
  iterations  = 100
  rbm         = full
  CEM         = off
  visualize   = False

Usage
-----
    cd src
    python experiment_kl_convergence.py            # run everything
    python experiment_kl_convergence.py --dry-run  # print grid, no execution
    python experiment_kl_convergence.py --part 1   # run only Part 1
"""

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
from encoder import Trainer
from helpers import save_results
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler

# ---------------------------------------------------------------------------
# Fixed hyperparameters — do NOT change between parts or seeds
# ---------------------------------------------------------------------------

FIXED = dict(
    n_samples=1000,
    lr=0.1,
    reg=1e-5,
    iterations=100,
    rbm="full",
    use_cem=False,
    visualize=False,
    output_dir="results/",
)

# ---------------------------------------------------------------------------
# Sampler registry
# ---------------------------------------------------------------------------

SAMPLERS = {
    "metropolis": ("custom", "metropolis"),
    "gibbs": ("custom", "gibbs"),
    "sa_custom": ("custom", "simulated_annealing"),
    "sa_dimod": ("dimod", "simulated_annealing"),
    "tabu": ("dimod", "tabu"),
}

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------


@dataclass
class Run:
    model: str  # "1d" | "2d"
    size: int  # chain length or lattice L
    h: float
    sampler: str  # key in SAMPLERS
    seed: int
    part: int


def build_grid(parts: list[int]) -> list[Run]:
    grid = []
    sampler_keys = list(SAMPLERS.keys())

    if 1 in parts:
        # Part 1: h-sweep, 1D N=8
        for h in [0.5, 1.0, 1.5, 2.0]:
            for sampler in sampler_keys:
                for seed in range(1, 16):  # 15 seeds
                    grid.append(Run("1d", 8, h, sampler, seed, part=1))

    if 2 in parts:
        # Part 2: size scaling at the critical point
        for size in [4, 8, 16]:
            for sampler in sampler_keys:
                for seed in range(1, 11):  # 10 seeds
                    grid.append(Run("1d", size, 1.0, sampler, seed, part=2))

    if 3 in parts:
        # Part 3: 2D geometry
        for h in [0.5, 1.0]:
            for sampler in sampler_keys:
                for seed in range(1, 11):  # 10 seeds
                    grid.append(Run("2d", 2, h, sampler, seed, part=3))

    # Deduplicate (Part 1 and Part 2 both include 1D N=8 h=1.0)
    seen = set()
    unique = []
    for r in grid:
        key = (r.model, r.size, r.h, r.sampler, r.seed)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Result path — mirrors helpers.save_results naming convention
# ---------------------------------------------------------------------------


def result_path(run: Run) -> Path:
    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    n_visible = run.size if run.model == "1d" else run.size**2
    n_hidden = n_visible

    output_dir = Path(
        f"{FIXED['output_dir']}/{n_hidden}/{sampler_backend}/{sampling_method}"
    )
    fname = (
        f"result"
        f"_{run.model}"
        f"_h{run.h}"
        f"_rbm{FIXED['rbm']}"
        f"_nh{n_hidden}"
        f"_lr{FIXED['lr']}"
        f"_reg{FIXED['reg']}"
        f"_ns{FIXED['n_samples']}"
        f"_seed{run.seed}"
        f"_iter{FIXED['iterations']}"
        f"_cem0"
        f".json"
    )
    return output_dir / fname


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------


def build_args(run: Run) -> SimpleNamespace:
    """Construct the args namespace that helpers.save_results expects."""
    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    n_visible = run.size if run.model == "1d" else run.size**2
    n_hidden = n_visible

    return SimpleNamespace(
        model=run.model,
        size=run.size,
        h=run.h,
        rbm=FIXED["rbm"],
        n_hidden=n_hidden,
        sampler=sampler_backend,
        sampling_method=sampling_method,
        n_samples=FIXED["n_samples"],
        iterations=FIXED["iterations"],
        learning_rate=FIXED["lr"],
        regularization=FIXED["reg"],
        cem=FIXED["use_cem"],
        cem_interval=5,
        cem_n_samples=200,
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
    )


def make_sampler(run: Run):
    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    if sampler_backend == "custom":
        n_sweeps = 10 if sampling_method == "gibbs" else 1
        return ClassicalSampler(method=sampling_method, n_sweeps=n_sweeps)
    elif sampler_backend == "dimod":
        return DimodSampler(method=sampling_method)
    raise ValueError(f"Unknown sampler backend: {sampler_backend}")


def execute_run(run: Run) -> dict:
    """
    Execute one VMC training run. Returns a summary dict.
    Raises on any unrecoverable error — caller logs and continues.
    """
    np.random.seed(run.seed)

    args = build_args(run)
    n_visible = run.size if run.model == "1d" else run.size**2
    n_hidden = n_visible

    if run.model == "1d":
        ising = TransverseFieldIsing1D(run.size, run.h)
    else:
        ising = TransverseFieldIsing2D(run.size, run.h)

    rbm = FullyConnectedRBM(n_visible, n_hidden)
    sampler = make_sampler(run)

    trainer_config = dict(
        learning_rate=FIXED["lr"],
        n_iterations=FIXED["iterations"],
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=FIXED["use_cem"],
        cem_interval=5,
        cem_n_samples=200,
    )

    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    save_results(args, history, ising, rbm)

    exact = ising.exact_ground_energy()
    final = history["energy"][-1]
    rel_err = abs(final - exact) / abs(exact)
    kl = history["kl_exact"][-1]
    gn = history["grad_norm"][-1]

    return dict(
        rel_error=rel_err,
        final_kl=kl,
        grad_norm=gn,
    )


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------


def append_failure_log(log_path: Path, run: Run, exc: Exception):
    entry = dict(
        timestamp=datetime.now().isoformat(),
        model=run.model,
        size=run.size,
        h=run.h,
        sampler=run.sampler,
        seed=run.seed,
        part=run.part,
        error=type(exc).__name__,
        message=str(exc),
    )
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only this part (default: all three)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the run grid without executing"
    )
    parser.add_argument(
        "--sampler",
        choices=list(SAMPLERS),
        default=None,
        help="Restrict to a single sampler (for parallelism)",
    )
    args = parser.parse_args()

    parts = [args.part] if args.part else [1, 2, 3]
    grid = build_grid(parts)

    if args.sampler:
        grid = [r for r in grid if r.sampler == args.sampler]

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(
            f"{'Part':>4}  {'Model':>4}  {'N':>3}  {'h':>4}  {'Sampler':>12}  {'Seed':>4}  {'Exists':>6}"
        )
        print("-" * 65)
        n_todo = 0
        for r in grid:
            exists = result_path(r).exists()
            n_todo += 0 if exists else 1
            print(
                f"{r.part:>4}  {r.model:>4}  {r.size:>3}  {r.h:>4}  "
                f"{r.sampler:>12}  {r.seed:>4}  {'yes' if exists else 'no':>6}"
            )
        print(
            f"\nTotal: {len(grid)} runs, {n_todo} pending, {len(grid) - n_todo} already done."
        )
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    log_path = Path("experiment_failures.jsonl")
    n_total = len(grid)
    n_done = 0
    n_skip = 0
    n_fail = 0

    print(
        f"[{datetime.now():%H:%M:%S}] Starting experiment — {n_total} runs scheduled."
    )
    print(f"  Parts: {parts}  |  Samplers: {args.sampler or 'all'}")
    print(
        f"  Fixed: lr={FIXED['lr']}  reg={FIXED['reg']}  "
        f"ns={FIXED['n_samples']}  iter={FIXED['iterations']}\n"
    )

    for i, run in enumerate(grid, 1):
        path = result_path(run)
        tag = (
            f"[{i}/{n_total}] Part{run.part} {run.model.upper()} N={run.size} "
            f"h={run.h} {run.sampler} seed={run.seed}"
        )

        if path.exists():
            print(f"  SKIP  {tag}")
            n_skip += 1
            continue

        print(f"  RUN   {tag}")
        try:
            summary = execute_run(run)
            n_done += 1
            print(
                f"         rel_err={summary['rel_error']:.4f}  "
                f"kl={summary['final_kl']:.4f}  "
                f"grad_norm={summary['grad_norm']:.4f}"
            )
        except Exception as exc:
            n_fail += 1
            print(f"  FAIL  {tag}")
            print(f"         {type(exc).__name__}: {exc}")
            traceback.print_exc()
            append_failure_log(log_path, run, exc)
            # Do NOT re-raise — continue with next run

    print(f"\n[{datetime.now():%H:%M:%S}] Done.")
    print(f"  Completed : {n_done}")
    print(f"  Skipped   : {n_skip}  (results already exist)")
    print(f"  Failed    : {n_fail}  (see {log_path})")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
