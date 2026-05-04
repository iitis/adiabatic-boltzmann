"""
D-Wave QPU experiment runner — pegasus and zephyr samplers, 1D TFIM only.

Grid:
  1D TFIM    sizes 16..200 spins    h = [0.5, 1.0, 2.0, 3.044]
  LR         [1e-2]
  seeds      [1, 2, 3, 4, 5]

Samplers: dimod/pegasus  dimod/zephyr

RBM strategy per run:
  1. FullyConnectedRBM — triggers minorminer to find a QPU embedding.
  2. If minorminer cannot embed → fall back to DWaveTopologyRBM (chain-free,
     trivial embedding on the matching QPU topology).
  The result filename records which RBM was actually used (rbmfull,
  rbmpegasus, or rbmzephyr) so skip-detection is unambiguous.

Budget (QPU access time via time.json):
  QPU_BUDGET_MS   = 15 * 60 * 1000  (15 min; override with --budget-ms)
  QPU_MS_PER_ITER = 200              (empirical QPU-access-time estimate per iteration)
  QPU time consumed since script start is read from time.json before each
  experiment.  If the file cannot be read the script aborts rather than
  silently exceeding budget.  Runs are capped to the iterations that fit;
  if fewer than MIN_ITERATIONS would fit the loop stops.

Results written to:
  jax_results/tfim_1d/{size}/dimod/{method}/
Skips runs whose result file already exists (checks both rbmfull and
rbm{method} variants, across any iteration count).

Usage
-----
    cd <repo-root>
    python scripts/exper/experiment_dwave_qpu.py                    # both QPU types
    python scripts/exper/experiment_dwave_qpu.py --sampler pegasus
    python scripts/exper/experiment_dwave_qpu.py --h 0.5 --size 16
    python scripts/exper/experiment_dwave_qpu.py --seeds 1 2 3
    python scripts/exper/experiment_dwave_qpu.py --dry-run
    python scripts/exper/experiment_dwave_qpu.py --force              # re-run existing
    python scripts/exper/experiment_dwave_qpu.py --budget-ms 600000  # 10-min budget
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

from encoder import Trainer
from helpers import save_results, read_qpu_time_ms
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import DimodSampler


# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

QPU_BUDGET_MS  = 15 * 60 * 1000   # default QPU-time budget (ms); override with --budget-ms
QPU_MS_PER_ITER = 200              # empirical QPU-access-time estimate per iteration (ms)
MIN_ITERATIONS  = 30               # abort loop when remaining budget fits fewer iters
TIME_PATH       = Path("time.json")


# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

FIXED = dict(
    n_samples=1000,
    reg=1e-5,
    iterations=300,
    visualize=False,
    output_dir="jax_results",
    sigma=1.0,
    cem_interval=5,      # CEM step every N iterations
    annealing_time=20,   # QPU annealing time in µs
)

LEARNING_RATES  = [1e-2]
SEEDS           = [1]
SAMPLER_BACKEND = "dimod"

H_VALUES  = [0.5, 1.0, 2.0]
SIZES_1D  = [16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 200]

# Error string raised by minorminer inside DimodSampler.dwave()
_EMBED_FAIL_MSG = "minorminer failed to find an embedding"


# ---------------------------------------------------------------------------
# Run dataclass
# ---------------------------------------------------------------------------

@dataclass
class Run:
    size:   int
    h:      float
    lr:     float
    seed:   int
    method: str   # "pegasus" | "zephyr"


def build_grid(
    methods: list[str],
    learning_rates: list[float] = LEARNING_RATES,
    seeds: list[int] = SEEDS,
) -> list[Run]:
    grid: list[Run] = []
    for method in methods:
        for size in SIZES_1D:
            for h in H_VALUES:
                for lr in learning_rates:
                    for seed in seeds:
                        grid.append(Run(size, h, lr, seed, method))
    # zephyr before pegasus, then small systems first
    _method_order = {"zephyr": 0, "pegasus": 1}
    grid.sort(key=lambda r: (_method_order.get(r.method, 99), r.size, r.h, r.seed))
    return grid


# ---------------------------------------------------------------------------
# Skip-detection
# ---------------------------------------------------------------------------

def _result_dir(run: Run) -> Path:
    return Path(FIXED["output_dir"]) / "tfim_1d" / str(run.size) / SAMPLER_BACKEND / run.method


def is_done(run: Run) -> bool:
    """True if any result file exists for this (size, h, lr, method) combination.

    Matches any seed, any iteration count, and both rbmfull / rbm{method} variants.
    """
    d = _result_dir(run)
    if not d.exists():
        return False
    for rbm_type in ("full", run.method):
        pattern = (
            f"result_1d_h{run.h}_rbm{rbm_type}_nh{run.size}"
            f"_lr{run.lr}_reg{FIXED['reg']}_ns{FIXED['n_samples']}"
            f"_seed*_iter*_cem1_sigma*.json"
        )
        if any(d.glob(pattern)):
            return True
    return False


# ---------------------------------------------------------------------------
# Build args namespace (consumed by Trainer and save_results)
# ---------------------------------------------------------------------------

def build_args(run: Run, n_iterations: int, rbm_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="1d",
        size=run.size,
        h=run.h,
        J=1.0,
        delta=1.0,
        alpha=2.0,
        rbm=rbm_type,
        n_hidden=run.size,
        sampler=SAMPLER_BACKEND,
        sampling_method=run.method,
        n_samples=FIXED["n_samples"],
        iterations=n_iterations,
        learning_rate=run.lr,
        regularization=FIXED["reg"],
        cem=True,
        cem_interval=FIXED["cem_interval"],
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
        sigma=FIXED["sigma"],
        lsb_steps=100,
        lsb_delta=1.0,
    )


# ---------------------------------------------------------------------------
# Single-attempt training helper
# ---------------------------------------------------------------------------

def _train_once(
    run: Run, n_iterations: int, rbm_type: str
) -> tuple:
    """
    Instantiate RBM + sampler + trainer and run.

    Returns (rbm, ising, args, history, elapsed_s).
    Raises RuntimeError with _EMBED_FAIL_MSG if minorminer cannot embed
    (only possible when rbm_type == "full").
    """
    key = jax.random.PRNGKey(run.seed)
    key, rbm_key = jax.random.split(key)

    ising = TransverseFieldIsing1D(run.size, run.h)

    if rbm_type == "full":
        rbm = FullyConnectedRBM(run.size, run.size, rbm_key)
    else:
        rbm = DWaveTopologyRBM(run.size, run.size, rbm_key, solver=rbm_type)

    sampler = DimodSampler(method=run.method)
    args = build_args(run, n_iterations, rbm_type)

    trainer_config = dict(
        learning_rate=run.lr,
        n_iterations=n_iterations,
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=True,
        cem_interval=FIXED["cem_interval"],
        lsb_sigma=FIXED["sigma"],
        seed=run.seed,
    )

    t0 = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    return rbm, ising, args, history, elapsed


# ---------------------------------------------------------------------------
# Run execution with full/topology fallback
# ---------------------------------------------------------------------------

def execute_run(run: Run, n_iterations: int) -> dict:
    """
    Run with FullyConnectedRBM first; fall back to DWaveTopologyRBM on
    embedding failure.  Saves results and returns a summary dict.
    """
    rbm_type = "full"
    try:
        rbm, ising, args, history, elapsed = _train_once(run, n_iterations, rbm_type)
    except RuntimeError as exc:
        if _EMBED_FAIL_MSG not in str(exc):
            raise
        print(
            f"  [embed] Full RBM: no embedding found for N={run.size} "
            f"on {run.method} → retrying with DWaveTopologyRBM"
        )
        rbm_type = run.method
        rbm, ising, args, history, elapsed = _train_once(run, n_iterations, rbm_type)

    save_results(args, history, ising, rbm)

    try:
        exact   = ising.exact_ground_energy()
        final   = history["energy"][-1]
        rel_err = abs(final - exact) / abs(exact)
    except NotImplementedError:
        rel_err = float("nan")

    return dict(
        elapsed_s=elapsed,
        rel_error=rel_err,
        rbm_type=rbm_type,
        n_iters=len(history["energy"]),
    )


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------

def _write_failure(log_path: Path, run: Run, exc: Exception) -> None:
    entry = dict(
        timestamp=datetime.now().isoformat(),
        size=run.size,
        h=run.h,
        lr=run.lr,
        seed=run.seed,
        method=run.method,
        error=type(exc).__name__,
        message=str(exc),
    )
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sampler", choices=["pegasus", "zephyr", "all"], default="all",
        help="QPU topology to use (default: both)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run grid without executing")
    parser.add_argument("--size",  type=int,   default=None,
                        help="Run only this system size")
    parser.add_argument("--h",     type=float, default=None,
                        help="Run only this transverse field value")
    parser.add_argument("--lr",    type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: 1 2 3 4 5)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists")
    parser.add_argument("--budget-ms", type=float, default=QPU_BUDGET_MS,
                        help=f"QPU-time budget in milliseconds (default: {QPU_BUDGET_MS:.0f})")
    cli = parser.parse_args()

    methods    = ["zephyr", "pegasus"] if cli.sampler == "all" else [cli.sampler]
    lr_list    = [cli.lr]   if cli.lr    is not None else LEARNING_RATES
    seed_list  = cli.seeds  if cli.seeds is not None else SEEDS
    budget_ms  = cli.budget_ms

    print(f"JAX devices  : {jax.devices()}")
    print(f"JAX version  : {jax.__version__}")
    print(f"Samplers     : {', '.join(methods)}")
    print(f"Seeds        : {seed_list}")
    print(f"QPU budget   : {budget_ms/60000:.1f} min  ({budget_ms:.0f} ms)")
    print(f"Iters/run    : {FIXED['iterations']}  @ ~{QPU_MS_PER_ITER} ms QPU/iter")
    print(f"Output dir   : {FIXED['output_dir']}/")

    grid = build_grid(methods, lr_list, seed_list)

    if cli.size is not None:
        grid = [r for r in grid if r.size == cli.size]
    if cli.h is not None:
        grid = [r for r in grid if r.h == cli.h]

    if cli.dry_run:
        pending  = sum(1 for r in grid if cli.force or not is_done(r))
        max_runs = int(budget_ms / (FIXED["iterations"] * QPU_MS_PER_ITER))
        print(
            f"\n{'Method':>10}  {'N':>4}  {'h':>6}  {'LR':>8}  {'Seed':>4}  {'Done':>4}"
        )
        print("-" * 55)
        for r in grid:
            done_flag = is_done(r)
            done_str  = "yes" if done_flag else "no"
            if done_flag and cli.force:
                done_str += " (force)"
            print(
                f"{r.method:>10}  {r.size:>4}  {r.h:>6}  "
                f"{r.lr:>8.4g}  {r.seed:>4}  {done_str}"
            )
        print(
            f"\nTotal: {len(grid)}  pending: {pending}  done: {len(grid)-pending}"
            f"\nBudget allows ~{max_runs} full runs at {QPU_MS_PER_ITER} ms QPU/iter"
        )
        return

    pending = [r for r in grid if cli.force or not is_done(r)]
    n_skip  = len(grid) - len(pending)

    qpu_start_ms = read_qpu_time_ms(TIME_PATH)

    print(
        f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total  "
        f"({len(pending)} pending, {n_skip} already done)\n"
        f"  Fixed: reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']}  cem=on  cem_interval={FIXED['cem_interval']}\n"
        f"  RBM: FullyConnectedRBM → DWaveTopologyRBM on embedding failure\n"
        f"  QPU time at start: {qpu_start_ms/60000:.2f} min\n"
    )

    log_path = Path(__file__).resolve().parent / "experiment_dwave_qpu_failures.jsonl"
    n_done = n_fail = 0
    t_wall = time.perf_counter()

    for i, run in enumerate(pending, 1):
        # ── Budget check before every experiment ─────────────────────
        try:
            qpu_used_ms = read_qpu_time_ms(TIME_PATH) - qpu_start_ms
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"\n[{datetime.now():%H:%M:%S}]  Cannot read QPU time: {exc} — aborting.")
            break

        remaining_ms = budget_ms - qpu_used_ms
        max_iters    = min(FIXED["iterations"], int(remaining_ms / QPU_MS_PER_ITER))

        if max_iters < MIN_ITERATIONS:
            print(
                f"\n[{datetime.now():%H:%M:%S}]  QPU budget exhausted  "
                f"({qpu_used_ms/60000:.1f}/{budget_ms/60000:.1f} min used, "
                f"only {max_iters} iters would fit — need {MIN_ITERATIONS}). "
                f"Stopping after {n_done} runs."
            )
            break

        # ── Progress line ─────────────────────────────────────────────
        elapsed_s = time.perf_counter() - t_wall
        if n_done > 0:
            avg_s  = elapsed_s / n_done
            left_s = avg_s * (len(pending) - i + 1)
            eta    = f"  ETA ~{left_s/3600:.1f}h"
        else:
            eta = ""

        budget_note = (
            f"  [{remaining_ms/60000:.1f} min QPU left"
            + (f", capped to {max_iters} iters" if max_iters < FIXED["iterations"] else "")
            + "]"
        )

        print(
            f"[{i}/{len(pending)}] "
            f"1d N={run.size:>3}  h={run.h}  {run.method}  "
            f"lr={run.lr:.4g}  seed={run.seed}"
            f"{budget_note}{eta}"
        )

        try:
            summary = execute_run(run, max_iters)
            n_done += 1
            print(
                f"  {summary['elapsed_s']:6.1f}s  "
                f"rbm={summary['rbm_type']}  "
                f"iters={summary['n_iters']}  "
                f"rel_err={summary['rel_error']:.4f}"
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            raise
        except Exception as exc:
            n_fail += 1
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            _write_failure(log_path, run, exc)

    total_s      = time.perf_counter() - t_wall
    qpu_total_ms = read_qpu_time_ms(TIME_PATH) - qpu_start_ms
    print(f"\n[{datetime.now():%H:%M:%S}]  Finished in {total_s/3600:.2f}h  "
          f"(QPU time this session: {qpu_total_ms/60000:.2f} min)")
    print(f"  Completed : {n_done}")
    print(f"  Skipped   : {n_skip}  (already existed)")
    print(f"  Failed    : {n_fail}" + (f"  → {log_path}" if n_fail else ""))

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
