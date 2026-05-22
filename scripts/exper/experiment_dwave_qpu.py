"""
D-Wave QPU experiment runner — lr1d size sweep on Zephyr (TTE probe).

Grid:
  LR-TFIM (α=2.0)  sizes [8, 24, 32, 64, 96, 200]  h=0.5
  LR         [0.01]
  seeds      [42]
  iterations 150 per run (zephyr) / 300 per run (gibbs companion)

Sampler: dimod/zephyr + Gibbs companion for each run.

RBM strategy per run:
  1. FullyConnectedRBM — triggers minorminer to find a QPU embedding.
  2. If minorminer cannot embed → fall back to DWaveTopologyRBM (chain-free,
     trivial embedding on the matching QPU topology).
  The result filename records which RBM was actually used (rbmfull or
  rbmzephyr) so skip-detection is unambiguous.

Budget (QPU access time via time.json):
  QPU_BUDGET_MS   = 20 * 60 * 1000  (20 min; override with --budget-ms)
  QPU_MS_PER_ITER = 200              (empirical QPU-access-time estimate per iteration)
  QPU time consumed since script start is read from time.json before each
  experiment.  If the file cannot be read the script aborts rather than
  silently exceeding budget.  Runs are capped to the iterations that fit;
  if fewer than MIN_ITERATIONS would fit the loop stops.

Results written to:
  results/lr_tfim_1d/{size}/dimod/zephyr/
Skips runs whose result file already exists (checks both rbmfull and
rbmzephyr variants, across any iteration count).

Usage
-----
    cd <repo-root>
    python scripts/exper/experiment_dwave_qpu.py
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
from ising import LongRangeTFIM1D
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler


# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

QPU_BUDGET_MS  = 20 * 60 * 1000   # default QPU-time budget (ms); override with --budget-ms
QPU_MS_PER_ITER = 200              # empirical QPU-access-time estimate per iteration (ms)
MIN_ITERATIONS  = 20               # abort loop when remaining budget fits fewer iters
TIME_PATH       = Path("time.json")

GIBBS_ITERATIONS = 300
GIBBS_N_SWEEPS   = 10


# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

FIXED = dict(
    n_samples=1000,
    reg=1e-5,
    iterations=150,
    visualize=False,
    output_dir="results",
    sigma=1.0,
    cem_interval=5,
    annealing_time=20,   # QPU annealing time in µs
)

LEARNING_RATES  = [0.01]
SEEDS           = [42]
SAMPLER_BACKEND = "dimod"

H_VALUES  = [0.5]
SIZES_1D  = [8, 24, 32, 64, 96, 200]
ALPHA     = 2.0   # LR-TFIM power-law exponent

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
    grid.sort(key=lambda r: (r.size, r.h, r.seed))
    return grid


# ---------------------------------------------------------------------------
# Skip-detection
# ---------------------------------------------------------------------------

def _result_dir(run: Run) -> Path:
    return Path(FIXED["output_dir"]) / "lr_tfim_1d" / str(run.size) / SAMPLER_BACKEND / run.method


def is_done(run: Run) -> bool:
    """True if any result file exists for this (size, h, lr, method) combination.

    Matches any rbm type and any iteration count.
    """
    d = _result_dir(run)
    if not d.exists():
        return False
    for rbm_type in ("full", run.method):
        pattern = (
            f"result_lr1d_h{run.h}_alpha{ALPHA}_rbm{rbm_type}_nh{run.size}"
            f"_lr{run.lr}_reg{FIXED['reg']}_ns{FIXED['n_samples']}"
            f"_seed{run.seed}_iter*_cem0_sigma*.json"
        )
        if any(d.glob(pattern)):
            return True
    return False


# ---------------------------------------------------------------------------
# Gibbs companion helpers
# ---------------------------------------------------------------------------

def _gibbs_result_dir(run: Run) -> Path:
    return Path(FIXED["output_dir"]) / "lr_tfim_1d" / str(run.size) / "custom" / "gibbs"


def is_gibbs_done(run: Run) -> bool:
    d = _gibbs_result_dir(run)
    if not d.exists():
        return False
    pattern = (
        f"result_lr1d_h{run.h}_alpha{ALPHA}_rbmfull_nh{run.size}"
        f"_lr{run.lr}_reg{FIXED['reg']}_ns{FIXED['n_samples']}"
        f"_seed{run.seed}_iter*_cem0_sigma*.json"
    )
    return any(d.glob(pattern))


def build_gibbs_args(run: Run) -> SimpleNamespace:
    return SimpleNamespace(
        model="lr1d",
        size=run.size,
        h=run.h,
        J=1.0,
        J1=1.0,
        J2=0.5,
        delta=1.0,
        alpha=ALPHA,
        ansatz="rbm",
        dbm_hidden="8",
        n_mf_steps=10,
        rbm="full",
        n_hidden=run.size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        patch_size=2,
        sampler="custom",
        sampling_method="gibbs",
        mh_warmup=0,
        mh_sweeps=1,
        ra_s_target=0.45,
        ra_pause_time=10,
        ra_anneal_time=10,
        n_samples=FIXED["n_samples"],
        iterations=GIBBS_ITERATIONS,
        learning_rate=run.lr,
        regularization=FIXED["reg"],
        cem=False,
        cem_interval=FIXED["cem_interval"],
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
        sigma=FIXED["sigma"],
    )


def execute_gibbs_run(run: Run) -> dict:
    """Run a Gibbs companion (FullyConnectedRBM, ClassicalSampler/gibbs) for this config."""
    _gibbs_result_dir(run).mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(run.seed)
    key, rbm_key = jax.random.split(key)

    ising = LongRangeTFIM1D(run.size, run.h, alpha=ALPHA)
    rbm = FullyConnectedRBM(run.size, run.size, rbm_key)
    sampler = ClassicalSampler(method="gibbs", n_sweeps=GIBBS_N_SWEEPS)
    args = build_gibbs_args(run)

    trainer_config = dict(
        learning_rate=run.lr,
        n_iterations=GIBBS_ITERATIONS,
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=False,
        cem_interval=FIXED["cem_interval"],
        lsb_sigma=FIXED["sigma"],
        seed=run.seed,
    )

    t0 = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    save_results(args, history, ising, rbm)

    try:
        exact = ising.exact_ground_energy()
        final = history["energy"][-1]
        rel_err = abs(final - exact) / abs(exact)
    except NotImplementedError:
        rel_err = float("nan")

    return dict(
        elapsed_s=elapsed,
        rel_error=rel_err,
        n_iters=len(history["energy"]),
    )


# ---------------------------------------------------------------------------
# Build args namespace (consumed by Trainer and save_results)
# ---------------------------------------------------------------------------

def build_args(run: Run, n_iterations: int, rbm_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="lr1d",
        size=run.size,
        h=run.h,
        J=1.0,
        J1=1.0,
        J2=0.5,
        delta=1.0,
        alpha=ALPHA,
        ansatz="rbm",
        dbm_hidden="8",
        n_mf_steps=10,
        rbm=rbm_type,
        n_hidden=run.size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        patch_size=2,
        sampler=SAMPLER_BACKEND,
        sampling_method=run.method,
        mh_warmup=0,
        mh_sweeps=1,
        ra_s_target=0.45,
        ra_pause_time=10,
        ra_anneal_time=10,
        n_samples=FIXED["n_samples"],
        iterations=n_iterations,
        learning_rate=run.lr,
        regularization=FIXED["reg"],
        cem=False,
        cem_interval=FIXED["cem_interval"],
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
        sigma=FIXED["sigma"],
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

    ising = LongRangeTFIM1D(run.size, run.h, alpha=ALPHA)

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
        use_cem=False,
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
    _result_dir(run).mkdir(parents=True, exist_ok=True)
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
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run grid without executing")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists")
    parser.add_argument("--budget-ms", type=float, default=QPU_BUDGET_MS,
                        help=f"QPU-time budget in milliseconds (default: {QPU_BUDGET_MS:.0f})")
    cli = parser.parse_args()

    budget_ms = cli.budget_ms

    print(f"JAX devices  : {jax.devices()}")
    print(f"JAX version  : {jax.__version__}")
    print(f"Model        : lr1d  h={H_VALUES[0]}  α={ALPHA}")
    print(f"Sizes        : {SIZES_1D}")
    print(f"Sampler      : dimod/zephyr + gibbs companion  CEM=off")
    print(f"QPU budget   : {budget_ms/60000:.1f} min  ({budget_ms:.0f} ms)")
    print(f"Iters/run    : {FIXED['iterations']} (zephyr)  {GIBBS_ITERATIONS} (gibbs)  @ ~{QPU_MS_PER_ITER} ms QPU/iter")
    print(f"Output dir   : {FIXED['output_dir']}/lr_tfim_1d/")

    grid = build_grid(["zephyr"])

    if cli.dry_run:
        max_runs = int(budget_ms / (FIXED["iterations"] * QPU_MS_PER_ITER))
        print(f"\n{'N':>6}  {'h':>6}  {'LR':>8}  {'Seed':>4}  {'Zephyr':>8}  {'Gibbs':>6}")
        print("-" * 52)
        for r in grid:
            dw_str  = ("yes" if is_done(r) else "no") + (" (force)" if is_done(r) and cli.force else "")
            gb_str  = ("yes" if is_gibbs_done(r) else "no") + (" (force)" if is_gibbs_done(r) and cli.force else "")
            print(f"{r.size:>6}  {r.h:>6}  {r.lr:>8.4g}  {r.seed:>4}  {dw_str:>8}  {gb_str:>6}")
        n_dw_pending  = sum(1 for r in grid if cli.force or not is_done(r))
        n_gb_pending  = sum(1 for r in grid if cli.force or not is_gibbs_done(r))
        print(
            f"\nTotal: {len(grid)}  zephyr pending: {n_dw_pending}  gibbs pending: {n_gb_pending}"
            f"\nBudget allows ~{max_runs} full zephyr runs at {QPU_MS_PER_ITER} ms QPU/iter"
        )
        return

    # Pre-compute done status for all runs
    run_status = [
        (r, is_done(r) and not cli.force, is_gibbs_done(r) and not cli.force)
        for r in grid
    ]
    # Items that need at least one of D-Wave or Gibbs
    pending_items = [(r, dw_done, gb_done) for r, dw_done, gb_done in run_status
                     if not dw_done or not gb_done]
    n_fully_done = len(grid) - len(pending_items)

    qpu_start_ms = read_qpu_time_ms(TIME_PATH)

    print(
        f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total  "
        f"({len(pending_items)} needing work, {n_fully_done} fully done)\n"
        f"  Fixed: reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']} (zephyr)  {GIBBS_ITERATIONS} (gibbs)  cem=off\n"
        f"  RBM: FullyConnectedRBM → DWaveTopologyRBM on embedding failure\n"
        f"  QPU time at start: {qpu_start_ms/60000:.2f} min\n"
    )

    n_dw_done   = 0
    n_gb_done   = 0
    t_wall      = time.perf_counter()
    budget_exhausted = False

    for i, (run, dwave_already_done, gibbs_already_done) in enumerate(pending_items, 1):
        elapsed_s = time.perf_counter() - t_wall
        eta = ""
        if i > 1:
            avg_s  = elapsed_s / (i - 1)
            left_s = avg_s * (len(pending_items) - i + 1)
            eta    = f"  ETA ~{left_s/3600:.1f}h"

        # ── D-Wave part ───────────────────────────────────────────────
        if not dwave_already_done:
            qpu_used_ms  = 0
            remaining_ms = 0
            max_iters    = 0
            if not budget_exhausted:
                try:
                    qpu_used_ms = read_qpu_time_ms(TIME_PATH) - qpu_start_ms
                except (OSError, json.JSONDecodeError, KeyError) as exc:
                    print(
                        f"\n[{datetime.now():%H:%M:%S}]  Cannot read QPU time: {exc}"
                        f" — stopping D-Wave runs."
                    )
                    budget_exhausted = True

            if not budget_exhausted:
                remaining_ms = budget_ms - qpu_used_ms
                max_iters    = min(FIXED["iterations"], int(remaining_ms / QPU_MS_PER_ITER))

                if max_iters < MIN_ITERATIONS:
                    print(
                        f"\n[{datetime.now():%H:%M:%S}]  QPU budget exhausted  "
                        f"({qpu_used_ms/60000:.1f}/{budget_ms/60000:.1f} min used, "
                        f"only {max_iters} iters would fit — need {MIN_ITERATIONS}). "
                        f"Stopping D-Wave runs (Gibbs companions will continue)."
                    )
                    budget_exhausted = True

            if not budget_exhausted:
                budget_note = (
                    f"  [{remaining_ms/60000:.1f} min QPU left"
                    + (f", capped to {max_iters} iters" if max_iters < FIXED["iterations"] else "")
                    + "]"
                )
                print(
                    f"[{i}/{len(pending_items)}] zephyr "
                    f"lr1d N={run.size:>4}  h={run.h}  α={ALPHA}  "
                    f"lr={run.lr:.4g}  seed={run.seed}"
                    f"{budget_note}{eta}"
                )
                try:
                    summary = execute_run(run, max_iters)
                    n_dw_done += 1
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
                    print(f"  ERROR  {type(exc).__name__}: {exc}")
                    sys.exit(1)

        # ── Gibbs companion ───────────────────────────────────────────
        if not gibbs_already_done:
            print(
                f"[{i}/{len(pending_items)}] gibbs  "
                f"lr1d N={run.size:>4}  h={run.h}  α={ALPHA}  "
                f"lr={run.lr:.4g}  seed={run.seed}  iter={GIBBS_ITERATIONS}{eta}"
            )
            try:
                g_summary = execute_gibbs_run(run)
                n_gb_done += 1
                print(
                    f"  {g_summary['elapsed_s']:6.1f}s  "
                    f"rbm=full  "
                    f"iters={g_summary['n_iters']}  "
                    f"rel_err={g_summary['rel_error']:.4f}"
                )
            except KeyboardInterrupt:
                print("\n[interrupted]")
                raise
            except Exception as exc:
                print(f"  ERROR  {type(exc).__name__}: {exc}")
                sys.exit(1)

    total_s      = time.perf_counter() - t_wall
    qpu_total_ms = read_qpu_time_ms(TIME_PATH) - qpu_start_ms
    print(f"\n[{datetime.now():%H:%M:%S}]  Finished in {total_s/3600:.2f}h  "
          f"(QPU time this session: {qpu_total_ms/60000:.2f} min)")
    print(f"  Zephyr completed : {n_dw_done}")
    print(f"  Gibbs  completed : {n_gb_done}")
    print(f"  Fully skipped    : {n_fully_done}  (both already existed)")


if __name__ == "__main__":
    main()
