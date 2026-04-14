"""
FPGA sampler sweep.

Sweeps: N=24, h ∈ {0.5, 1.0, 2.0}, lr over LEARNING_RATES, 1 seeds.
Sampler: fpga / fpga  (FPGASampler via VeloxQFPGA JTAG).

Usage
-----
    cd <repo-root>
    python scripts/experiment_fpga.py             # run everything
    python scripts/experiment_fpga.py --dry-run   # print grid, no execution
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
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import FPGASampler

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

FIXED = dict(
    n_samples=1000,
    reg=1e-5,
    iterations=100,
    rbm="full",
    visualize=False,
    output_dir=str(_SRC / "results"),
    sigma=1.0,
)

SIZES = [24]
H_VALUES = [0.5, 1.0, 2.0]
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = [1]

SAMPLERS = {
    "fpga": ("fpga", "fpga"),
}

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------


@dataclass
class Run:
    size: int
    h: float
    lr: float
    seed: int


def build_grid() -> list[Run]:
    grid = []
    for size in SIZES:
        for h in H_VALUES:
            for lr in LEARNING_RATES:
                for seed in SEEDS:
                    grid.append(Run(size=size, h=h, lr=lr, seed=seed))
    return grid


# ---------------------------------------------------------------------------
# Result path — mirrors helpers.save_results naming convention
# ---------------------------------------------------------------------------


def result_path(run: Run) -> Path:
    n_hidden = run.size
    output_dir = Path(f"{FIXED['output_dir']}/{run.size}/fpga/fpga")
    fname = (
        f"result_1d"
        f"_h{run.h}"
        f"_rbm{FIXED['rbm']}"
        f"_nh{n_hidden}"
        f"_lr{run.lr}"
        f"_reg{FIXED['reg']}"
        f"_ns{FIXED['n_samples']}"
        f"_seed{run.seed}"
        f"_iter{FIXED['iterations']}"
        f"_cem1"
        f"_sigma{float(FIXED['sigma'])}"
        f".json"
    )
    return output_dir / fname


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------


def build_args(run: Run) -> SimpleNamespace:
    return SimpleNamespace(
        model="1d",
        size=run.size,
        h=run.h,
        rbm=FIXED["rbm"],
        n_hidden=run.size,
        sampler="fpga",
        sampling_method="fpga",
        n_samples=FIXED["n_samples"],
        iterations=FIXED["iterations"],
        learning_rate=run.lr,
        regularization=FIXED["reg"],
        cem=True,
        cem_interval=5,
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
        sigma=FIXED["sigma"],
    )


def make_sampler(run: Run):
    return FPGASampler(transport="jtag")


def execute_run(run: Run) -> dict:
    np.random.seed(run.seed)

    args = build_args(run)
    ising = TransverseFieldIsing1D(run.size, run.h)
    rbm = FullyConnectedRBM(run.size, run.size)
    sampler = make_sampler(run)

    trainer_config = dict(
        learning_rate=run.lr,
        n_iterations=FIXED["iterations"],
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=True,
        cem_interval=5,
    )

    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    save_results(args, history, ising, rbm)

    exact = ising.exact_ground_energy()
    final = history["energy"][-1]
    rel_err = abs(final - exact) / abs(exact)
    kl = history["kl_exact"][-1]
    gn = history["grad_norm"][-1]

    return dict(rel_error=rel_err, final_kl=kl, grad_norm=gn)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker(run: Run) -> tuple:
    try:
        summary = execute_run(run)
        return run, summary, None
    except Exception as exc:
        return run, None, exc


def _write_failure(log_path: Path, run: Run, exc: Exception):
    entry = dict(
        timestamp=datetime.now().isoformat(),
        size=run.size,
        h=run.h,
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
        "--workers", type=int, default=1, help="Parallel workers (default: 1)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=FIXED["iterations"],
        help="Training iterations per run (default: 100)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run in-process (no multiprocessing). Recommended for FPGA/JTAG debugging.",
    )
    cli = parser.parse_args()
    FIXED["iterations"] = cli.iterations

    grid = build_grid()

    if cli.dry_run:
        pending = sum(1 for r in grid if not result_path(r).exists())
        print(f"{'N':>4}  {'h':>4}  {'LR':>8}  {'Seed':>4}  {'Done':>4}")
        print("-" * 36)
        for r in grid:
            done = "yes" if result_path(r).exists() else "no"
            print(f"{r.size:>4}  {r.h:>4}  {r.lr:>8.4g}  {r.seed:>4}  {done:>4}")
        print(
            f"\nTotal: {len(grid)} runs | pending: {pending} | done: {len(grid) - pending}"
        )
        return

    pending = [r for r in grid if not result_path(r).exists()]
    n_skip = len(grid) - len(pending)

    print(f"[{datetime.now():%H:%M:%S}] FPGA sweep — {len(grid)} total runs")
    print(f"  N={SIZES}  h={H_VALUES}  lr={LEARNING_RATES}")
    print(f"  Pending: {len(pending)}  skipped: {n_skip}  workers: {cli.workers}\n")

    log_path = Path(__file__).resolve().parent / "experiment_fpga_failures.jsonl"
    n_done = n_fail = 0

    if cli.serial:
        completed = 0
        for run in pending:
            completed += 1
            run, summary, exc = _worker(run)
            tag = (
                f"[{completed}/{len(pending)}] "
                f"N={run.size} h={run.h} lr={run.lr:.4g} seed={run.seed}"
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
                    f"         rel_err={summary['rel_error']:.4f}  kl={kl_str}  grad_norm={summary['grad_norm']:.4f}"
                )
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cli.workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_worker, run): run for run in pending}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                run, summary, exc = future.result()
                tag = (
                    f"[{completed}/{len(pending)}] "
                    f"N={run.size} h={run.h} lr={run.lr:.4g} seed={run.seed}"
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
                        f"         rel_err={summary['rel_error']:.4f}  kl={kl_str}  grad_norm={summary['grad_norm']:.4f}"
                    )

    print(f"\n[{datetime.now():%H:%M:%S}] Finished.")
    print(f"  Completed : {n_done}")
    print(f"  Skipped   : {n_skip}  (already existed)")
    print(f"  Failed    : {n_fail}" + (f"  → see {log_path}" if n_fail else ""))

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
