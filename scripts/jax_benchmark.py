"""
JAX backend benchmark — Gibbs and LSB (with/without CEM).

Sweeps:
  1D  sizes  [12, 16, 24, 32, 48, 64]   h = [0.5, 1.0, 2.0]
  2D  sizes  [6, 8, 12]  (N = L²)        h = [0.5, 1.0, 2.0]
  LR         [3e-4, 1e-2]
  seeds      [1, 2]
  samplers   gibbs (cem=False)
             lsb   (cem=False, cem=True)

Fixed:
  rbm        FullyConnectedRBM, n_hidden = n_visible
  n_samples  1000
  reg        1e-5
  iterations 300
  sigma      1.0   (LSB noise precision σ⁻²)
  lsb_steps  100   (LSB integration steps M)
  lsb_delta  1.0   (LSB time step Δ)

Designed to run for ~8 h on a Titan GPU.
Results are written to jax_results/ (skips experiments that already exist).

Usage
-----
    cd <repo-root>
    python scripts/jax_benchmark.py              # run everything
    python scripts/jax_benchmark.py --dry-run    # print grid, no execution
    python scripts/jax_benchmark.py --sampler lsb
    python scripts/jax_benchmark.py --sampler gibbs
"""

import jax
jax.config.update("jax_enable_x64", True)

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from encoder import Trainer
from helpers import save_results
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import ClassicalSampler


# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

FIXED = dict(
    n_samples=1000,
    reg=1e-5,
    iterations=300,
    rbm="full",
    visualize=False,
    output_dir="jax_results",
    sigma=1.0,        # LSB noise precision σ⁻²
    lsb_steps=100,    # LSB integration steps M
    lsb_delta=1.0,    # LSB time step Δ
)

LEARNING_RATES = [3e-4, 1e-2]

SAMPLERS = {
    "gibbs": ("custom", "gibbs"),
    "lsb":   ("custom", "lsb"),
}


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

@dataclass
class Run:
    model:   str    # "1d" | "2d"
    size:    int    # chain length or lattice linear dim L
    h:       float
    sampler: str    # key in SAMPLERS
    lr:      float
    seed:    int
    use_cem: bool = False


def build_grid() -> list[Run]:
    grid: list[Run] = []

    # 1D
    for size in [12, 16, 24, 32, 48, 64]:
        for h in [0.5, 1.0, 2.0]:
            for lr in LEARNING_RATES:
                for seed in [1, 2]:
                    grid.append(Run("1d", size, h, "gibbs", lr, seed, use_cem=False))
                    grid.append(Run("1d", size, h, "lsb",   lr, seed, use_cem=False))
                    grid.append(Run("1d", size, h, "lsb",   lr, seed, use_cem=True))

    # 2D  (h values must be in TransverseFieldIsing2D.reference_energies_per_spin)
    for size in [6, 8, 12]:
        for h in [0.5, 1.0, 2.0]:
            for lr in LEARNING_RATES:
                for seed in [1, 2]:
                    grid.append(Run("2d", size, h, "gibbs", lr, seed, use_cem=False))
                    grid.append(Run("2d", size, h, "lsb",   lr, seed, use_cem=False))
                    grid.append(Run("2d", size, h, "lsb",   lr, seed, use_cem=True))

    return grid


# ---------------------------------------------------------------------------
# Result path — mirrors helpers.save_results naming exactly
# ---------------------------------------------------------------------------

def result_path(run: Run) -> Path:
    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    n_visible = run.size if run.model == "1d" else run.size ** 2
    n_hidden  = n_visible
    output_dir = Path(FIXED["output_dir"]) / str(run.size) / sampler_backend / sampling_method
    fname = (
        f"result_{run.model}"
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
    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    n_visible = run.size if run.model == "1d" else run.size ** 2
    return SimpleNamespace(
        model=run.model,
        size=run.size,
        h=run.h,
        rbm=FIXED["rbm"],
        n_hidden=n_visible,
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


def execute_run(run: Run) -> dict:
    """
    Execute one VMC training run.  Returns a summary dict.
    Raises on unrecoverable error — caller logs and continues.
    """
    key = jax.random.PRNGKey(run.seed)
    key, rbm_key = jax.random.split(key)

    args     = build_args(run)
    n_visible = run.size if run.model == "1d" else run.size ** 2
    n_hidden  = n_visible

    if run.model == "1d":
        ising = TransverseFieldIsing1D(run.size, run.h)
    else:
        ising = TransverseFieldIsing2D(run.size, run.h)

    rbm = FullyConnectedRBM(n_visible, n_hidden, rbm_key)

    sampler_backend, sampling_method = SAMPLERS[run.sampler]
    n_sweeps = 10 if sampling_method == "gibbs" else 1
    sampler = ClassicalSampler(method=sampling_method, n_sweeps=n_sweeps)
    key, sampler_key = jax.random.split(key)
    sampler._key = sampler_key

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
        seed=run.seed,
    )

    t0 = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history  = trainer.train()
    elapsed  = time.perf_counter() - t0

    save_results(args, history, ising, rbm)

    exact   = ising.exact_ground_energy()
    final   = history["energy"][-1]
    rel_err = abs(final - exact) / abs(exact)
    kl      = history.get("kl_exact", [None])[-1]
    gn      = history.get("grad_norm", [None])[-1]

    return dict(
        elapsed_s=elapsed,
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
        use_cem=run.use_cem,
        error=type(exc).__name__,
        message=str(exc),
    )
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run grid without executing")
    parser.add_argument("--sampler", choices=list(SAMPLERS), default=None,
                        help="Restrict to a single sampler")
    cli = parser.parse_args()

    print(f"JAX devices : {jax.devices()}")
    print(f"JAX version : {jax.__version__}")
    print(f"Output dir  : {FIXED['output_dir']}/")

    grid = build_grid()
    if cli.sampler:
        grid = [r for r in grid if r.sampler == cli.sampler]

    # ── Dry run ──────────────────────────────────────────────────────────────
    if cli.dry_run:
        pending = sum(1 for r in grid if not result_path(r).exists())
        print(
            f"\n{'Model':>4}  {'N':>4}  {'h':>4}  "
            f"{'Sampler':>6}  {'CEM':>3}  {'LR':>8}  {'Seed':>4}  {'Done':>4}"
        )
        print("-" * 60)
        for r in grid:
            done = "yes" if result_path(r).exists() else "no"
            print(
                f"{r.model:>4}  {r.size:>4}  {r.h:>4}  "
                f"{r.sampler:>6}  {'Y' if r.use_cem else 'N':>3}  "
                f"{r.lr:>8.4g}  {r.seed:>4}  {done:>4}"
            )
        print(f"\nTotal: {len(grid)}  pending: {pending}  done: {len(grid)-pending}")
        return

    pending = [r for r in grid if not result_path(r).exists()]
    n_skip  = len(grid) - len(pending)

    print(
        f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total runs  "
        f"({len(pending)} pending, {n_skip} already done)\n"
        f"  Fixed: reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']}  sigma={FIXED['sigma']}\n"
    )

    log_path = Path(__file__).resolve().parent / "jax_benchmark_failures.jsonl"
    n_done = n_fail = 0
    t_wall = time.perf_counter()

    for i, run in enumerate(pending, 1):
        tag = (
            f"[{i}/{len(pending)}] "
            f"{run.model.upper()} N={run.size:>3} "
            f"h={run.h}  {run.sampler}  "
            f"cem={'Y' if run.use_cem else 'N'}  "
            f"lr={run.lr:.4g}  seed={run.seed}"
        )

        # ETA
        if n_done > 0:
            avg_s  = (time.perf_counter() - t_wall) / n_done
            left_s = avg_s * (len(pending) - i + 1)
            eta    = f"  ETA ~{left_s/3600:.1f}h"
        else:
            eta = ""

        print(f"{tag}{eta}")

        try:
            summary = execute_run(run)
            n_done += 1
            kl_str = f"{summary['final_kl']:.4f}" if summary["final_kl"] is not None else "N/A"
            gn_str = f"{summary['grad_norm']:.4f}" if summary["grad_norm"] is not None else "N/A"
            print(
                f"  {summary['elapsed_s']:6.1f}s  "
                f"rel_err={summary['rel_error']:.4f}  "
                f"kl={kl_str}  grad_norm={gn_str}"
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            raise
        except Exception as exc:
            n_fail += 1
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            _write_failure(log_path, run, exc)

    total_h = (time.perf_counter() - t_wall) / 3600
    print(f"\n[{datetime.now():%H:%M:%S}]  Finished in {total_h:.2f}h")
    print(f"  Completed : {n_done}")
    print(f"  Skipped   : {n_skip}  (already existed)")
    print(f"  Failed    : {n_fail}" + (f"  → {log_path}" if n_fail else ""))

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
