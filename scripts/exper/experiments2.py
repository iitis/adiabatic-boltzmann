"""
Experiment: KL divergence as predictor of VMC convergence quality.
Includes extensive LR sweep for LSB and Gibbs samplers.
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
    python scripts/exper/experiment_lsb_gibbs.py                  # run everything
    python scripts/exper/experiment_lsb_gibbs.py --dry-run        # print grid, no execution
    python scripts/exper/experiment_lsb_gibbs.py --part 1         # run only Part 1
    python scripts/exper/experiment_lsb_gibbs.py --sampler lsb    # only LSB runs
    python scripts/exper/experiment_lsb_gibbs.py --sampler gibbs  # only Gibbs runs
"""

import argparse
import json
import sys
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
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, _DEVICE

# ---------------------------------------------------------------------------
# Fixed hyperparameters — do NOT change between parts or seeds
# ---------------------------------------------------------------------------

FIXED = dict(
    n_samples=1000,
    lr=0.1,  # default lr for samplers that are NOT in _LR_SWEEP_SAMPLERS
    reg=1e-5,
    iterations=300,
    rbm="zephyr",
    use_cem=False,
    visualize=False,
    output_dir=str(_SRC.parent / "results"),
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
# dd

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
    for size in [8, 16, 32]:
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
    for size in [4, 6, 8]:
        for h in [0.5, 1.0, 2.0]:
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

    if FIXED["rbm"] == "full":
        rbm = FullyConnectedRBM(n_visible, n_hidden)
    else:
        rbm = DWaveTopologyRBM(n_visible, n_hidden, solver=FIXED["rbm"])

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
# Failure log
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
        "--gpu",
        action="store_true",
        help="Require GPU (CuPy) — exit with error if unavailable",
    )
    cli = parser.parse_args()

    if cli.gpu and _DEVICE != "gpu":
        sys.exit("ERROR: --gpu specified but CuPy is not available (device=cpu).")

    grid = build_grid()
    if cli.sampler:
        grid = [r for r in grid if r.sampler == cli.sampler]

    # ── Dry run ──────────────────────────────────────────────────────────────
    if cli.dry_run:
        pending = sum(1 for r in grid if not result_path(r).exists())
        print(
            f"{'Model':>4}  {'N':>3}  {'h':>4}  "
            f"{'Sampler':>12}  {'LR':>8}  {'CEM':>3}  {'Seed':>4}  {'Done':>4}"
        )
        print("-" * 72)
        for r in grid:
            done = "yes" if result_path(r).exists() else "no"
            print(
                f"{r.model:>4}  {r.size:>3}  {r.h:>4}  "
                f"{r.sampler:>12}  {r.lr:>8.4g}  {'Y' if r.use_cem else 'N':>3}  "
                f"{r.seed:>4}  {done:>4}"
            )
        print(
            f"\nTotal: {len(grid)} runs | pending: {pending} | "
            f"done: {len(grid) - pending}"
        )
        return

    pending = [r for r in grid if not result_path(r).exists()]
    n_skip = len(grid) - len(pending)
    n_total = len(grid)

    print(f"[{datetime.now():%H:%M:%S}] Experiment start — {n_total} total runs")
    print(f"  Pending : {len(pending)}  ({n_skip} already done)")
    print(
        f"  Fixed   : reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']}  sigma={FIXED['sigma']}\n"
        f"  LR sweep: {LEARNING_RATES}  CEM sweep: lsb only\n"
    )

    log_path = Path(__file__).resolve().parent / "experiment_lsb_gibbs_failures.jsonl"
    n_done = n_fail = 0

    for i, run in enumerate(pending, 1):
        tag = (
            f"[{i}/{len(pending)}] "
            f"{run.model.upper()} N={run.size} "
            f"h={run.h} {run.sampler} lr={run.lr:.4g} "
            f"cem={'Y' if run.use_cem else 'N'} seed={run.seed}"
        )
        try:
            summary = execute_run(run)
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
        except KeyboardInterrupt:
            print("\n[interrupted]")
            raise
        except Exception as exc:
            n_fail += 1
            print(f"  FAIL  {tag}")
            print(f"         {type(exc).__name__}: {exc}")
            _write_failure(log_path, run, exc)

    print(f"\n[{datetime.now():%H:%M:%S}] Finished.")
    print(f"  Completed : {n_done}")
    print(f"  Skipped   : {n_skip}  (already existed)")
    print(f"  Failed    : {n_fail}" + (f"  → see {log_path}" if n_fail else ""))

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
