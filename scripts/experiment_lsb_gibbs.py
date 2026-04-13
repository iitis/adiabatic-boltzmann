"""
Experiment: KL divergence as predictor of VMC convergence quality.
Includes extensive LR sweep for LSB and Gibbs samplers.

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
    1D, N=8, h ∈ {0.5, 1.0, 1.5, 2.0}, 6 samplers, 15 seeds → ~1440 runs.
    LSB and Gibbs: 5 LRs × 2 CEM variants × 15 seeds per h.
    Others (metro/SA/tabu): lr=0.1, CEM=off × 15 seeds per h.
    Primary output: Spearman ρ(KL, error) and ρ(grad_norm, error) per h.
Part 3 — 2D geometry (supplementary):
    2D, L=2 (N=4 spins), h ∈ {0.5, 1.0}, 6 samplers, 10 seeds → ~480 runs.
    Primary output: does the 1D result hold in 2D?

Total: ~2400 runs (after dedup), all free classical samplers (no QPU).

Controlled hyperparameters
--------------------------
  n_hidden    = n_visible  (α = 1 hidden-to-visible ratio)
  n_samples   = 1000
  lr          = 0.1 for metropolis/SA/tabu; swept over LEARNING_RATES for lsb/gibbs
  reg         = 1e-5
  iterations  = 100
  rbm         = full
  CEM         = off
  sigma       = 1.0  (LSB noise precision σ⁻²)
  lsb_steps   = 100  (LSB integration steps M)
  lsb_delta   = 1.0  (LSB time step Δ)
  visualize   = False

Usage
-----
    cd <repo-root>
    python scripts/experiment_lsb_gibbs.py                  # run everything
    python scripts/experiment_lsb_gibbs.py --dry-run        # print grid, no execution
    python scripts/experiment_lsb_gibbs.py --part 1         # run only Part 1
    python scripts/experiment_lsb_gibbs.py --sampler lsb    # only LSB runs
    python scripts/experiment_lsb_gibbs.py --sampler gibbs  # only Gibbs runs
"""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
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
    lr=0.1,  # default lr for samplers that are NOT in _LR_SWEEP_SAMPLERS
    reg=1e-5,
    iterations=300,
    rbm="full",
    use_cem=False,
    visualize=False,
    output_dir=str(_SRC / "results"),
    sigma=1.0,  # LSB noise precision σ⁻² (σ = 1/√(σ⁻²))
    lsb_steps=100,  # LSB integration steps M
    lsb_delta=1.0,  # LSB time step Δ
)

# Learning rates swept for LSB and Gibbs (log-spaced from 1e-4 to 1e-2)
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# ---------------------------------------------------------------------------
# Sampler registry
# ---------------------------------------------------------------------------

SAMPLERS = {
    "metropolis": ("custom", "metropolis"),
    "gibbs": ("custom", "gibbs"),
    "lsb": ("custom", "lsb"),
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
    lr: float = 0.1  # learning rate (swept for lsb/gibbs, fixed for others)
    use_cem: bool = False  # CEM β scheduling (swept for lsb/gibbs, off for others)


def build_grid() -> list[Run]:
    grid = []
    sampler_keys = list(SAMPLERS.keys())
    for size in [8, 16, 32, 64, 128]:
        for h in [0.5, 1.0, 2.0]:
            for sampler in sampler_keys:
                for lr in LEARNING_RATES:
                    for use_cem in [False, True] if sampler == "lsb" else [False]:
                        grid.append(
                            Run(
                                "1d",
                                size,
                                h,
                                sampler,
                                seed=1,
                                lr=lr,
                                use_cem=use_cem,
                            )
                        )

    # 2D geometry
    for size in [4, 8, 10, 12]:
        for h in [0.5, 1.0]:
            for sampler in sampler_keys:
                for lr in LEARNING_RATES:
                    for use_cem in [False, True] if sampler == "lsb" else [False]:
                        grid.append(
                            Run(
                                "2d",
                                size,
                                h,
                                sampler,
                                1,
                                lr=lr,
                                use_cem=use_cem,
                            )
                        )

    return grid


# ---------------------------------------------------------------------------
# Result path — mirrors helpers.save_results naming convention
# ---------------------------------------------------------------------------


def result_path(run: Run) -> Path:
    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    n_visible = run.size if run.model == "1d" else run.size**2
    n_hidden = n_visible

    # Directory mirrors save_results: output_dir/{size}/{sampler}/{method}/
    output_dir = Path(
        f"{FIXED['output_dir']}/{run.size}/{sampler_backend}/{sampling_method}"
    )
    fname = (
        f"result"
        f"_{run.model}"
        f"_h{run.h}"
        f"_rbm{FIXED['rbm']}"
        f"_nh{n_hidden}"
        f"_lr{run.lr}"
        f"_reg{FIXED['reg']}"
        f"_ns{FIXED['n_samples']}"
        f"_seed{run.seed}"
        f"_iter{FIXED['iterations']}"
        f"_cem{int(run.use_cem)}"
        f"_sigma{float(FIXED['sigma'])}"
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
        learning_rate=run.lr,
        regularization=FIXED["reg"],
        cem=run.use_cem,
        cem_interval=5,
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
        sigma=FIXED["sigma"],
        lsb_steps=FIXED["lsb_steps"],
        lsb_delta=FIXED["lsb_delta"],
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
        learning_rate=run.lr,
        n_iterations=FIXED["iterations"],
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=run.use_cem,
        cem_interval=5,
        lsb_sigma=FIXED["sigma"],
        lsb_steps=FIXED["lsb_steps"],
        lsb_delta=FIXED["lsb_delta"],
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
# Samplers that use batched GPU kernels (CuPy) vs. CPU-only
# ---------------------------------------------------------------------------

# ClassicalSampler methods dispatch to a GPU-batched path when CuPy is
# available (_DEVICE == "gpu" in sampler.py).  Multiple GPU processes compete
# for VRAM and SM time, so we limit their concurrency separately.
# DimodSampler (neal, TabuSampler) is pure CPU — use all available cores.
_GPU_SAMPLERS = {"metropolis", "gibbs", "lsb"}
_CPU_SAMPLERS: set[str] = set()


# ---------------------------------------------------------------------------
# Worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------


def _worker(run: Run) -> tuple:
    """
    Execute one run in a subprocess.
    Returns (run, summary_dict, exception_or_None).
    Never raises — exceptions are returned so the main process can log them.
    """
    try:
        summary = execute_run(run)
        return run, summary, None
    except Exception as exc:
        return run, None, exc


# ---------------------------------------------------------------------------
# Failure log (written only by the main process — no locking needed)
# ---------------------------------------------------------------------------


def _write_failure(log_path: Path, run: Run, exc: Exception):
    entry = dict(
        timestamp=datetime.now().isoformat(),
        model=run.model,
        size=run.size,
        h=run.h,
        sampler=run.sampler,
        lr=run.lr,
        seed=run.seed,
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
        "--dry-run", action="store_true", help="Print the run grid without executing"
    )
    parser.add_argument(
        "--sampler",
        choices=list(SAMPLERS),
        default=None,
        help="Restrict to a single sampler",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Max parallel workers for CPU-only samplers (default: 21)",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help="Max parallel workers for GPU-batched samplers "
        "(metropolis/gibbs/sa_custom). Lower = less VRAM contention. "
        "(default: 4)",
    )
    cli = parser.parse_args()

    grid = build_grid()
    if cli.sampler:
        grid = [r for r in grid if r.sampler == cli.sampler]

    # ── Dry run ──────────────────────────────────────────────────────────────
    if cli.dry_run:
        pending = sum(1 for r in grid if not result_path(r).exists())
        print(
            f"{'Model':>4}  {'N':>3}  {'h':>4}  "
            f"{'Sampler':>12}  {'LR':>8}  {'CEM':>3}  {'Seed':>4}  {'Pool':>3}  {'Done':>4}"
        )
        print("-" * 80)
        for r in grid:
            pool = "GPU" if r.sampler in _GPU_SAMPLERS else "CPU"
            done = 'no'
            print(
                f"{r.model:>4}  {r.size:>3}  {r.h:>4}  "
                f"{r.sampler:>12}  {r.lr:>8.4g}  {'Y' if r.use_cem else 'N':>3}  "
                f"{r.seed:>4}  {pool:>3}  {done:>4}"
            )
        print(
            f"\nTotal: {len(grid)} runs | pending: {pending} | "
            f"done: {len(grid) - pending}"
        )
        print(
            f"CPU pool: {cli.workers} workers  |  GPU pool: {cli.gpu_workers} workers"
        )
        return

    # ── Build pending lists per pool ─────────────────────────────────────────
    pending_gpu = [
        r for r in grid if r.sampler in _GPU_SAMPLERS and not result_path(r).exists()
    ]
    pending_cpu = [
        r for r in grid if r.sampler in _CPU_SAMPLERS and not result_path(r).exists()
    ]
    n_skip = sum(1 for r in grid if result_path(r).exists())
    n_total = len(grid)

    print(f"[{datetime.now():%H:%M:%S}] Experiment start — {n_total} total runs")
    print(
        f"  Pending : {len(pending_gpu)} GPU-pool  +  {len(pending_cpu)} CPU-pool  "
        f"({n_skip} already done)"
    )
    print(f"  Workers : GPU={cli.gpu_workers}  CPU={cli.workers}")
    print(
        f"  Fixed   : reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']}  sigma={FIXED['sigma']}\n"
        f"  LR sweep: {LEARNING_RATES}  CEM sweep: lsb only\n"
    )

    log_path = Path(__file__).resolve().parent / "experiment_lsb_gibbs_failures.jsonl"
    n_done = n_fail = 0

    # Use spawn to avoid inheriting any CUDA context from the parent process.
    # Fork + CUDA is undefined behaviour and causes random hangs/crashes.
    mp_ctx = multiprocessing.get_context("spawn")

    futures: dict = {}  # future → Run

    with (
        ProcessPoolExecutor(max_workers=cli.gpu_workers, mp_context=mp_ctx) as gpu_pool,
        ProcessPoolExecutor(max_workers=cli.workers, mp_context=mp_ctx) as cpu_pool,
    ):
        for run in pending_gpu:
            futures[gpu_pool.submit(_worker, run)] = run
        for run in pending_cpu:
            futures[cpu_pool.submit(_worker, run)] = run

        completed = 0
        total_pending = len(pending_gpu) + len(pending_cpu)

        for future in as_completed(futures):
            completed += 1
            run, summary, exc = future.result()
            tag = (
                f"[{completed}/{total_pending}] "
                f"{run.model.upper()} N={run.size} "
                f"h={run.h} {run.sampler} lr={run.lr:.4g} "
                f"cem={'Y' if run.use_cem else 'N'} seed={run.seed}"
            )

            if exc is not None:
                n_fail += 1
                print(f"  FAIL  {tag}")
                print(f"         {type(exc).__name__}: {exc}")
                _write_failure(log_path, run, exc)
            else:
                n_done += 1
                kl_str = (
                    f"{summary['final_kl']:.4f}"
                    if summary["final_kl"] is not None
                    else "N/A"
                )
                print(f"  DONE  {tag}")
                print(
                    f"         rel_err={summary['rel_error']:.4f}  "
                    f"kl={kl_str}  "
                    f"grad_norm={summary['grad_norm']:.4f}"
                )

    print(f"\n[{datetime.now():%H:%M:%S}] Finished.")
    print(f"  Completed : {n_done}")
    print(f"  Skipped   : {n_skip}  (already existed)")
    print(f"  Failed    : {n_fail}" + (f"  → see {log_path}" if n_fail else ""))

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
