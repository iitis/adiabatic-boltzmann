"""
D-Wave fast-anneal experiment runner — LR-TFIM 1D size sweep on Pegasus.

Grid:
  LR-TFIM (α=2.0)  10 sizes equally spaced from N=16 to N=200
  h ∈ {0.5, 1.0}   CEM ∈ {True, False}
  →  10 × 2 × 2 = 40 QPU runs + 40 Gibbs companion runs

  QPU  : lr=0.1, sampler=dimod/pegasus_fast, iter=150
  Gibbs: lr=0.01, sampler=custom/gibbs,      iter=150

Sampler: pegasus_fast (fast_anneal=True, 7 ns).
h biases (RBM a, b) are silently dropped before each QPU call;
only J (W couplings) are submitted.

RBM strategy per QPU run:
  1. FullyConnectedRBM — triggers minorminer embedding.
  2. On embedding failure → DWaveTopologyRBM (chain-free, trivial embedding).

Budget (QPU access time, *this session only*):
  QPU_BUDGET_MS = 40 × 60 × 1000  (default 40 min; override with --budget-ms)
  Read from time.json at startup; delta ≤ budget_ms is enforced.
  Runs are capped to the iterations that fit; when fewer than MIN_ITERATIONS
  would fit, D-Wave loop stops.  Gibbs companions always continue.

Results:
  results/lr_tfim_1d/{size}/dimod/pegasus_fast/
  results/lr_tfim_1d/{size}/custom/gibbs/

Usage
-----
    cd <repo-root>
    python scripts/exper/experiment_dwave_qpu.py
    python scripts/exper/experiment_dwave_qpu.py --dry-run
    python scripts/exper/experiment_dwave_qpu.py --force
    python scripts/exper/experiment_dwave_qpu.py --budget-ms 1200000
    python scripts/exper/experiment_dwave_qpu.py --method pegasus_fast
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

import numpy as np

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

QPU_BUDGET_MS   = 40 * 60 * 1000  # 40 min — delta since script start
# Conservative fast-anneal estimate: programming + readout overhead dominates
# at 7 ns anneal time; actual QPU access time is ~5–15 ms per VMC iteration.
QPU_MS_PER_ITER = 30
MIN_ITERATIONS  = 20
TIME_PATH       = Path("time.json")


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
)

QPU_LR          = 0.1
GIBBS_LR        = 0.01
GIBBS_ITERS     = 150
GIBBS_N_SWEEPS  = 10
SEED            = 42
SAMPLER_BACKEND = "dimod"
ALPHA           = 2.0   # LR-TFIM power-law exponent

H_VALUES  = [0.5, 1.0]
CEM_FLAGS = [True, False]

# 10 sizes equally spaced from N=16 to N=200 (rounded to nearest integer)
SIZES_1D: list[int] = sorted(
    set(int(round(x)) for x in np.linspace(16, 200, 10))
)

_EMBED_FAIL_MSG = "minorminer failed to find an embedding"


# ---------------------------------------------------------------------------
# Run dataclass
# ---------------------------------------------------------------------------

@dataclass
class Run:
    size:   int
    h:      float
    cem:    bool
    method: str   # "pegasus_fast" | "zephyr_fast"


def build_grid(method: str) -> list[Run]:
    grid = [
        Run(size, h, cem, method)
        for size in SIZES_1D
        for h in H_VALUES
        for cem in CEM_FLAGS
    ]
    grid.sort(key=lambda r: (r.size, r.h, r.cem))
    return grid


# ---------------------------------------------------------------------------
# Skip-detection helpers
# ---------------------------------------------------------------------------

def _cem_tag(cem: bool) -> str:
    return "cem1" if cem else "cem0"


def _result_dir(run: Run) -> Path:
    return (
        Path(FIXED["output_dir"])
        / "lr_tfim_1d"
        / str(run.size)
        / SAMPLER_BACKEND
        / run.method
    )


def is_done(run: Run) -> bool:
    """True if a QPU result exists for this exact (size, h, cem, method)."""
    d = _result_dir(run)
    if not d.exists():
        return False
    topology = run.method.replace("_fast", "")
    for rbm_type in ("full", topology):
        pattern = (
            f"result_lr1d_h{run.h}_alpha{ALPHA}_rbm{rbm_type}_nh{run.size}"
            f"_lr{QPU_LR}_reg{FIXED['reg']}_ns{FIXED['n_samples']}"
            f"_seed{SEED}_iter*_{_cem_tag(run.cem)}_sigma*.json"
        )
        if any(d.glob(pattern)):
            return True
    return False


def _gibbs_result_dir(run: Run) -> Path:
    return (
        Path(FIXED["output_dir"])
        / "lr_tfim_1d"
        / str(run.size)
        / "custom"
        / "gibbs"
    )


def is_gibbs_done(run: Run) -> bool:
    d = _gibbs_result_dir(run)
    if not d.exists():
        return False
    pattern = (
        f"result_lr1d_h{run.h}_alpha{ALPHA}_rbmfull_nh{run.size}"
        f"_lr{GIBBS_LR}_reg{FIXED['reg']}_ns{FIXED['n_samples']}"
        f"_seed{SEED}_iter*_cem0_sigma*.json"
    )
    return any(d.glob(pattern))


# ---------------------------------------------------------------------------
# Gibbs companion
# ---------------------------------------------------------------------------

def _gibbs_args(run: Run) -> SimpleNamespace:
    return SimpleNamespace(
        model="lr1d", size=run.size, h=run.h,
        J=1.0, J1=1.0, J2=0.5, delta=1.0, alpha=ALPHA,
        ansatz="rbm", dbm_hidden="8", n_mf_steps=10,
        rbm="full", n_hidden=run.size,
        d_model=32, n_layers=2, n_heads=4, patch_size=2,
        sampler="custom", sampling_method="gibbs",
        mh_warmup=0, mh_sweeps=1,
        ra_s_target=0.45, ra_pause_time=10, ra_anneal_time=10,
        n_samples=FIXED["n_samples"],
        iterations=GIBBS_ITERS,
        learning_rate=GIBBS_LR,
        regularization=FIXED["reg"],
        cem=False, cem_interval=FIXED["cem_interval"],
        seed=SEED, visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"], sigma=FIXED["sigma"],
    )


def execute_gibbs_run(run: Run) -> dict:
    _gibbs_result_dir(run).mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(SEED)
    key, rbm_key = jax.random.split(key)

    ising   = LongRangeTFIM1D(run.size, run.h, alpha=ALPHA)
    rbm     = FullyConnectedRBM(run.size, run.size, rbm_key)
    sampler = ClassicalSampler(method="gibbs", n_sweeps=GIBBS_N_SWEEPS)
    args    = _gibbs_args(run)

    trainer_config = dict(
        learning_rate=GIBBS_LR,
        n_iterations=GIBBS_ITERS,
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=False,
        cem_interval=FIXED["cem_interval"],
        lsb_sigma=FIXED["sigma"],
        seed=SEED,
    )

    t0 = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    save_results(args, history, ising, rbm)

    try:
        exact   = ising.exact_ground_energy()
        final   = history["energy"][-1]
        rel_err = abs(final - exact) / abs(exact)
    except NotImplementedError:
        rel_err = float("nan")

    return dict(elapsed_s=elapsed, rel_error=rel_err, n_iters=len(history["energy"]))


# ---------------------------------------------------------------------------
# QPU run helpers
# ---------------------------------------------------------------------------

def _qpu_args(run: Run, n_iterations: int, rbm_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="lr1d", size=run.size, h=run.h,
        J=1.0, J1=1.0, J2=0.5, delta=1.0, alpha=ALPHA,
        ansatz="rbm", dbm_hidden="8", n_mf_steps=10,
        rbm=rbm_type, n_hidden=run.size,
        d_model=32, n_layers=2, n_heads=4, patch_size=2,
        sampler=SAMPLER_BACKEND, sampling_method=run.method,
        mh_warmup=0, mh_sweeps=1,
        ra_s_target=0.45, ra_pause_time=10, ra_anneal_time=10,
        n_samples=FIXED["n_samples"],
        iterations=n_iterations,
        learning_rate=QPU_LR,
        regularization=FIXED["reg"],
        cem=run.cem, cem_interval=FIXED["cem_interval"],
        seed=SEED, visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"], sigma=FIXED["sigma"],
    )


def _train_once(run: Run, n_iterations: int, rbm_type: str) -> tuple:
    key = jax.random.PRNGKey(SEED)
    key, rbm_key = jax.random.split(key)

    ising = LongRangeTFIM1D(run.size, run.h, alpha=ALPHA)

    if rbm_type == "full":
        rbm = FullyConnectedRBM(run.size, run.size, rbm_key)
    else:
        rbm = DWaveTopologyRBM(run.size, run.size, rbm_key, solver=rbm_type)

    sampler = DimodSampler(method=run.method)
    args    = _qpu_args(run, n_iterations, rbm_type)

    trainer_config = dict(
        learning_rate=QPU_LR,
        n_iterations=n_iterations,
        n_samples=FIXED["n_samples"],
        regularization=FIXED["reg"],
        save_checkpoints=False,
        checkpoint_interval=10,
        use_cem=run.cem,
        cem_interval=FIXED["cem_interval"],
        lsb_sigma=FIXED["sigma"],
        seed=SEED,
    )

    t0 = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    return rbm, ising, args, history, elapsed


def execute_run(run: Run, n_iterations: int) -> dict:
    """Try FullyConnectedRBM; fall back to DWaveTopologyRBM on embedding failure."""
    _result_dir(run).mkdir(parents=True, exist_ok=True)
    rbm_type = "full"
    try:
        rbm, ising, args, history, elapsed = _train_once(run, n_iterations, rbm_type)
    except RuntimeError as exc:
        if _EMBED_FAIL_MSG not in str(exc):
            raise
        topology = run.method.replace("_fast", "")
        print(
            f"  [embed] no embedding for N={run.size} on {run.method}"
            f" → retrying with DWaveTopologyRBM ({topology})"
        )
        rbm_type = topology
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
    parser.add_argument(
        "--budget-ms", type=float, default=float(QPU_BUDGET_MS),
        help=f"Session QPU-time budget in ms (default: {QPU_BUDGET_MS} = 40 min)",
    )
    parser.add_argument(
        "--method", default="pegasus_fast",
        choices=["pegasus_fast", "zephyr_fast"],
        help="D-Wave fast-anneal solver (default: pegasus_fast)",
    )
    cli = parser.parse_args()

    budget_ms = cli.budget_ms
    grid      = build_grid(cli.method)

    print(f"JAX devices  : {jax.devices()}")
    print(f"JAX version  : {jax.__version__}")
    print(f"Model        : lr1d  α={ALPHA}")
    print(f"h values     : {H_VALUES}")
    print(f"Sizes ({len(SIZES_1D):2d}) : {SIZES_1D}")
    print(f"CEM variants : {CEM_FLAGS}  (interval={FIXED['cem_interval']})")
    print(f"Method       : {cli.method}  (fast_anneal=True, 7 ns, h-biases dropped)")
    print(f"QPU  LR      : {QPU_LR}      Gibbs LR: {GIBBS_LR}")
    print(f"Iters/run    : {FIXED['iterations']} (QPU)  {GIBBS_ITERS} (Gibbs)")
    print(f"Total runs   : {len(grid)} QPU + {len(grid)} Gibbs")
    print(f"QPU budget   : {budget_ms/60000:.1f} min  ({budget_ms:.0f} ms, this session)")
    print(f"Output dir   : {FIXED['output_dir']}/lr_tfim_1d/")

    if cli.dry_run:
        max_runs = int(budget_ms / (FIXED["iterations"] * QPU_MS_PER_ITER))
        print(f"\n{'N':>6}  {'h':>5}  {'CEM':>5}  {'QPU':>6}  {'Gibbs':>6}")
        print("-" * 38)
        for r in grid:
            print(
                f"{r.size:>6}  {r.h:>5}  {'on' if r.cem else 'off':>5}"
                f"  {'done' if is_done(r) else 'no':>6}"
                f"  {'done' if is_gibbs_done(r) else 'no':>6}"
            )
        n_dw = sum(1 for r in grid if cli.force or not is_done(r))
        n_gb = sum(1 for r in grid if cli.force or not is_gibbs_done(r))
        print(
            f"\nTotal: {len(grid)}  QPU pending: {n_dw}  Gibbs pending: {n_gb}"
            f"\nBudget allows ~{max_runs} full QPU runs"
            f" ({QPU_MS_PER_ITER} ms/iter × {FIXED['iterations']} iters)"
        )
        return

    run_status = [
        (r, is_done(r) and not cli.force, is_gibbs_done(r) and not cli.force)
        for r in grid
    ]
    pending_items = [
        (r, dw_done, gb_done)
        for r, dw_done, gb_done in run_status
        if not dw_done or not gb_done
    ]
    n_fully_done = len(grid) - len(pending_items)

    qpu_start_ms = read_qpu_time_ms(TIME_PATH)

    print(
        f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total  "
        f"({len(pending_items)} needing work, {n_fully_done} fully done)\n"
        f"  reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']} (QPU)  {GIBBS_ITERS} (Gibbs)\n"
        f"  QPU time at session start: {qpu_start_ms/60000:.2f} min\n"
    )

    n_dw_done        = 0
    n_gb_done        = 0
    t_wall           = time.perf_counter()
    budget_exhausted = False

    for i, (run, dwave_already_done, gibbs_already_done) in enumerate(pending_items, 1):
        elapsed_s = time.perf_counter() - t_wall
        eta = ""
        if i > 1:
            avg_s  = elapsed_s / (i - 1)
            left_s = avg_s * (len(pending_items) - i + 1)
            eta    = f"  ETA ~{left_s/3600:.1f}h"

        cem_str = "cem=ON" if run.cem else "cem=OFF"

        # ── D-Wave fast anneal ────────────────────────────────────────
        if not dwave_already_done:
            qpu_used_ms  = 0.0
            remaining_ms = budget_ms
            max_iters    = FIXED["iterations"]

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
                        f"\n[{datetime.now():%H:%M:%S}]  QPU budget exhausted"
                        f"  ({qpu_used_ms/60000:.1f}/{budget_ms/60000:.1f} min used,"
                        f" only {max_iters} iters would fit — need {MIN_ITERATIONS})."
                        f" Stopping D-Wave runs (Gibbs companions continue)."
                    )
                    budget_exhausted = True

            if not budget_exhausted:
                budget_note = (
                    f"  [{remaining_ms/60000:.1f} min left"
                    + (f", capped to {max_iters} iters" if max_iters < FIXED["iterations"] else "")
                    + "]"
                )
                print(
                    f"[{i}/{len(pending_items)}] {run.method}"
                    f"  lr1d N={run.size:>4}  h={run.h}  α={ALPHA}"
                    f"  lr={QPU_LR}  {cem_str}"
                    f"{budget_note}{eta}"
                )
                try:
                    summary = execute_run(run, max_iters)
                    n_dw_done += 1
                    print(
                        f"  {summary['elapsed_s']:6.1f}s"
                        f"  rbm={summary['rbm_type']}"
                        f"  iters={summary['n_iters']}"
                        f"  rel_err={summary['rel_error']:.4f}"
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
                f"[{i}/{len(pending_items)}] gibbs"
                f"  lr1d N={run.size:>4}  h={run.h}  α={ALPHA}"
                f"  lr={GIBBS_LR}  {cem_str}  iter={GIBBS_ITERS}{eta}"
            )
            try:
                g_summary = execute_gibbs_run(run)
                n_gb_done += 1
                print(
                    f"  {g_summary['elapsed_s']:6.1f}s"
                    f"  rbm=full"
                    f"  iters={g_summary['n_iters']}"
                    f"  rel_err={g_summary['rel_error']:.4f}"
                )
            except KeyboardInterrupt:
                print("\n[interrupted]")
                raise
            except Exception as exc:
                print(f"  ERROR  {type(exc).__name__}: {exc}")
                sys.exit(1)

    total_s      = time.perf_counter() - t_wall
    qpu_total_ms = read_qpu_time_ms(TIME_PATH) - qpu_start_ms
    print(
        f"\n[{datetime.now():%H:%M:%S}]  Finished in {total_s/3600:.2f}h"
        f"  (QPU time this session: {qpu_total_ms/60000:.2f} min)"
    )
    print(f"  QPU runs completed : {n_dw_done}")
    print(f"  Gibbs  completed   : {n_gb_done}")
    print(f"  Fully skipped      : {n_fully_done}  (both already existed)")


if __name__ == "__main__":
    main()
