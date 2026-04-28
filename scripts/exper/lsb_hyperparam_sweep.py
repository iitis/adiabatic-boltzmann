"""
LSB hyperparameter sweep — grid search over lsb_steps, lsb_delta, lsb_sigma, and
learning_rate for 1D TFIM, sizes up to 200.

Grid:
  sizes      [16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 200]
  h          [0.5, 1.0, 2.0, 3.044]
  lsb_steps  [50, 100, 200]
  lsb_delta  [0.5, 1.0, 2.0]
  lsb_sigma  [0.25, 1.0, 4.0]   (σ⁻², paper convention; actual σ = 1/√(lsb_sigma))
  lr         [5e-3, 1e-2, 5e-2]

Fixed:
  model      1d
  rbm        full, n_hidden = n_visible
  n_samples  1000
  reg        1e-5
  iterations 100   (reduced vs the main 300-iter runs for sweep efficiency)
  seed       1
  use_cem    True
  cem_interval 5

Results written to lsb_hyperparam_results/ (skips runs that already exist).

Usage
-----
    cd <repo-root>
    python scripts/exper/lsb_hyperparam_sweep.py                # full grid
    python scripts/exper/lsb_hyperparam_sweep.py --dry-run      # preview grid
    python scripts/exper/lsb_hyperparam_sweep.py --size 64      # single size
    python scripts/exper/lsb_hyperparam_sweep.py --h 1.0        # single h
    python scripts/exper/lsb_hyperparam_sweep.py --force        # re-run existing
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

from encoder import Trainer
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler


# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

SIZES_1D    = [16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 200]
H_VALUES    = [0.5, 1.0, 2.0, 3.044]
LSB_STEPS   = [50, 100, 200]
LSB_DELTA   = [0.5, 1.0, 2.0]
LSB_SIGMA   = [0.25, 1.0, 4.0]   # σ⁻²; σ = 1/√(lsb_sigma)
LEARNING_RATES = [5e-3, 1e-2, 5e-2]

FIXED = dict(
    n_samples=1000,
    reg=1e-5,
    iterations=100,
    seed=1,
    use_cem=True,
    cem_interval=5,
    output_dir="lsb_hyperparam_results",
)

OUTPUT_DIR = Path(str(FIXED["output_dir"]))


# ---------------------------------------------------------------------------
# Run dataclass and grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    size:      int
    h:         float
    lsb_steps: int
    lsb_delta: float
    lsb_sigma: float   # σ⁻²
    lr:        float


def build_grid(
    sizes: list[int] = SIZES_1D,
    h_values: list[float] = H_VALUES,
    lsb_steps: list[int] = LSB_STEPS,
    lsb_delta: list[float] = LSB_DELTA,
    lsb_sigma: list[float] = LSB_SIGMA,
    learning_rates: list[float] = LEARNING_RATES,
) -> list[Run]:
    grid = [
        Run(size, h, steps, delta, sigma, lr)
        for size, h, steps, delta, sigma, lr in product(
            sizes, h_values, lsb_steps, lsb_delta, lsb_sigma, learning_rates
        )
    ]
    grid.sort(key=lambda r: (r.size, r.h, r.lsb_steps, r.lsb_delta, r.lsb_sigma, r.lr))
    return grid


# ---------------------------------------------------------------------------
# Result path
# ---------------------------------------------------------------------------

def result_path(run: Run) -> Path:
    N = run.size
    out = OUTPUT_DIR / str(N) / "custom" / "lsb"
    fname = (
        f"result_1d"
        f"_h{run.h}"
        f"_rbmfull"
        f"_nh{N}"
        f"_lr{run.lr:.4g}"
        f"_reg{FIXED['reg']}"
        f"_ns{FIXED['n_samples']}"
        f"_seed{FIXED['seed']}"
        f"_iter{FIXED['iterations']}"
        f"_steps{run.lsb_steps}"
        f"_delta{run.lsb_delta}"
        f"_sigma{run.lsb_sigma}"
        f".json"
    )
    return out / fname


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_run(run: Run) -> dict:
    key = jax.random.PRNGKey(int(FIXED["seed"]))
    key, rbm_key = jax.random.split(key)

    ising  = TransverseFieldIsing1D(run.size, run.h)
    rbm    = FullyConnectedRBM(run.size, run.size, rbm_key)
    sampler = ClassicalSampler(method="lsb")

    key, sampler_key = jax.random.split(key)
    sampler._key = sampler_key

    trainer_config = dict(
        learning_rate=run.lr,
        n_iterations=FIXED["iterations"],
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=FIXED["use_cem"],
        cem_interval=FIXED["cem_interval"],
        lsb_sigma=run.lsb_sigma,
        lsb_steps=run.lsb_steps,
        lsb_delta=run.lsb_delta,
        seed=FIXED["seed"],
    )

    t0      = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    exact   = ising.exact_ground_energy()
    final   = history["energy"][-1]
    rel_err = abs(final - exact) / abs(exact)
    kl      = history.get("kl_exact", [None])[-1]
    gn      = history.get("grad_norm", [None])[-1]

    result = dict(
        # identity
        model="1d",
        size=run.size,
        h=run.h,
        lsb_steps=run.lsb_steps,
        lsb_delta=run.lsb_delta,
        lsb_sigma=run.lsb_sigma,
        lr=run.lr,
        # fixed
        n_samples=FIXED["n_samples"],
        reg=FIXED["reg"],
        iterations=FIXED["iterations"],
        seed=FIXED["seed"],
        use_cem=FIXED["use_cem"],
        cem_interval=FIXED["cem_interval"],
        # outcomes
        exact_energy=exact,
        final_energy=final,
        rel_error=rel_err,
        final_kl=kl,
        final_grad_norm=gn,
        elapsed_s=elapsed,
        # full history for post-hoc analysis
        history=history,
    )

    path = result_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    return dict(elapsed_s=elapsed, rel_error=rel_err, final_kl=kl, grad_norm=gn)


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------

def _write_failure(log_path: Path, run: Run, exc: Exception):
    entry = dict(
        timestamp=datetime.now().isoformat(),
        size=run.size,
        h=run.h,
        lsb_steps=run.lsb_steps,
        lsb_delta=run.lsb_delta,
        lsb_sigma=run.lsb_sigma,
        lr=run.lr,
        error=type(exc).__name__,
        message=str(exc),
    )
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", type=int, default=None,
                        help="Filter to this 1D chain length only")
    parser.add_argument("--h", type=float, default=None,
                        help="Filter to this transverse field value only")
    parser.add_argument("--lr", type=float, default=None,
                        help="Filter to this learning rate only")
    parser.add_argument("--lsb-steps", type=int, default=None,
                        help="Filter to this lsb_steps value only")
    parser.add_argument("--lsb-delta", type=float, default=None,
                        help="Filter to this lsb_delta value only")
    parser.add_argument("--lsb-sigma", type=float, default=None,
                        help="Filter to this lsb_sigma (σ⁻²) value only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run grid without executing")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists")
    cli = parser.parse_args()

    lr_list    = [cli.lr]         if cli.lr         is not None else LEARNING_RATES
    steps_list = [cli.lsb_steps] if cli.lsb_steps  is not None else LSB_STEPS
    delta_list = [cli.lsb_delta] if cli.lsb_delta  is not None else LSB_DELTA
    sigma_list = [cli.lsb_sigma] if cli.lsb_sigma  is not None else LSB_SIGMA
    sizes      = [cli.size]       if cli.size        is not None else SIZES_1D
    h_values   = [cli.h]          if cli.h           is not None else H_VALUES

    grid = build_grid(sizes, h_values, steps_list, delta_list, sigma_list, lr_list)

    print(f"JAX devices : {jax.devices()}")
    print(f"JAX version : {jax.__version__}")
    print(f"Output dir  : {OUTPUT_DIR}/")
    print(f"Fixed       : iter={FIXED['iterations']}  ns={FIXED['n_samples']}"
          f"  reg={FIXED['reg']}  seed={FIXED['seed']}  cem=on")

    if cli.dry_run:
        pending = sum(1 for r in grid if cli.force or not result_path(r).exists())
        print(f"\n{'N':>4}  {'h':>6}  {'steps':>5}  {'delta':>5}  {'sigma':>5}  "
              f"{'lr':>8}  {'done':>4}")
        print("-" * 55)
        for r in grid:
            exists = result_path(r).exists()
            done = "yes" if exists else "no"
            print(f"{r.size:>4}  {r.h:>6}  {r.lsb_steps:>5}  {r.lsb_delta:>5}  "
                  f"{r.lsb_sigma:>5}  {r.lr:>8.4g}  {done}")
        print(f"\nTotal: {len(grid)}  pending: {pending}  done: {len(grid)-pending}")
        return

    pending = [r for r in grid if cli.force or not result_path(r).exists()]
    n_skip  = len(grid) - len(pending)

    print(f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total runs  "
          f"({len(pending)} pending, {n_skip} already done)\n")

    log_path = Path(__file__).resolve().parent / "lsb_hyperparam_sweep_failures.jsonl"
    n_done = n_fail = 0
    t_wall = time.perf_counter()

    for i, run in enumerate(pending, 1):
        tag = (
            f"[{i}/{len(pending)}] "
            f"1D N={run.size:>3}  h={run.h}  "
            f"steps={run.lsb_steps}  delta={run.lsb_delta}  "
            f"sigma={run.lsb_sigma}  lr={run.lr:.4g}"
        )

        if n_done > 0:
            avg_s  = (time.perf_counter() - t_wall) / n_done
            left_s = avg_s * (len(pending) - i + 1)
            tag += f"  ETA ~{left_s/3600:.1f}h"

        print(tag)

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
