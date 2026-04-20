"""
D-Wave benchmark — Pegasus and Zephyr QPU samplers with full/sparse RBMs.

Sweeps:
  1D sizes         [8, 12, 16]
  2D sizes         []              (empty by default — runs are slow)
  h                [0.5, 1.0, 2.0]
  LR               [0.1, 0.01]
  seeds            [1, 42]
  sampling methods pegasus, zephyr
  RBM types        full, pegasus, zephyr

Fixed:
  model            1D TFIM
  n_samples        1000
  reg              1e-5
  iterations       300
  n_hidden         n_visible

D-Wave QPU budget:
  Accumulated QPU access time is read from time.json (written by DimodSampler).
  The budget is cumulative across all sessions — never reset automatically.
  Before every D-Wave run the remaining budget is checked; the entire sweep
  aborts if the limit is exceeded.  A corrupt or unreadable time.json raises
  immediately rather than silently assuming 0 ms used.

Results are written to dwave_results/ (skips experiments that already exist).

Usage
-----
    cd <repo-root>
    python scripts/dwave_benchmark.py              # run everything
    python scripts/dwave_benchmark.py --dry-run    # print grid, no execution
    python scripts/dwave_benchmark.py --method pegasus
    python scripts/dwave_benchmark.py --method zephyr
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
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import DimodSampler


# ---------------------------------------------------------------------------
# Sweep parameters — edit these to change what gets run
# ---------------------------------------------------------------------------

SIZES_1D = [8, 12, 16]  # 1D chain lengths
SIZES_2D: list = []  # 2D lattice linear dims (L × L)
H_VALUES = [0.5, 1.0, 2.0]
LEARNING_RATES = [0.1, 0.01]
SEEDS = [1, 42]

# D-Wave sampling methods to include: "pegasus", "zephyr", or both
SAMPLING_METHODS = ["pegasus", "zephyr"]

# RBM architectures to include: "full", "pegasus", "zephyr", or any subset
RBM_TYPES = ["full", "pegasus", "zephyr"]

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

N_SAMPLES = 1000
REG = 1e-5
N_ITERATIONS = 300
OUTPUT_DIR = "dwave_results"
USE_CEM = True
CEM_INTERVAL = 5

# ---------------------------------------------------------------------------
# QPU budget
# ---------------------------------------------------------------------------

DWAVE_BUDGET_MS = 60 * 60 * 1000  # 20 minutes in milliseconds
DWAVE_TIME_FILE = Path("time.json")  # written by DimodSampler (path relative to cwd)


# ---------------------------------------------------------------------------
# QPU time helpers — no silent fallbacks
# ---------------------------------------------------------------------------


def read_qpu_time_ms() -> float:
    """Return accumulated QPU access time in ms.

    Raises on any read / parse failure so that a corrupt time.json never
    makes the budget check silently pass.
    """
    if not DWAVE_TIME_FILE.exists():
        return 0.0
    with DWAVE_TIME_FILE.open("r") as f:
        data = json.load(f)
    return float(data["time_ms"])


def qpu_budget_exceeded() -> bool:
    used = read_qpu_time_ms()
    if used >= DWAVE_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] Accumulated QPU time {used / 60_000:.2f} min "
            f">= limit {DWAVE_BUDGET_MS / 60_000:.0f} min. "
            "Aborting remaining D-Wave experiments."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------


@dataclass
class Run:
    model: str  # "1d" | "2d"
    size: int  # chain length or lattice linear dim L
    h: float
    sampling_method: str  # "pegasus" | "zephyr"
    rbm: str  # "full" | "pegasus" | "zephyr"
    lr: float
    seed: int


def build_grid() -> list[Run]:
    grid: list[Run] = []

    def _add(model: str, size: int):
        for h in H_VALUES:
            for method in SAMPLING_METHODS:
                for rbm in RBM_TYPES:
                    for lr in LEARNING_RATES:
                        for seed in SEEDS:
                            grid.append(Run(model, size, h, method, rbm, lr, seed))

    for size in SIZES_1D:
        _add("1d", size)
    for size in SIZES_2D:
        _add("2d", size)

    return grid


# ---------------------------------------------------------------------------
# Result path — mirrors helpers.save_results naming exactly
# ---------------------------------------------------------------------------


def result_path(run: Run) -> Path:
    n_visible = run.size if run.model == "1d" else run.size**2
    n_hidden = n_visible
    output_dir = Path(OUTPUT_DIR) / str(run.size) / "dimod" / run.sampling_method
    fname = (
        f"result_{run.model}"
        f"_h{run.h}"
        f"_rbm{run.rbm}"
        f"_nh{n_hidden}"
        f"_lr{run.lr}"
        f"_reg{REG}"
        f"_ns{N_SAMPLES}"
        f"_seed{run.seed}"
        f"_iter{N_ITERATIONS}"
        f"_cem{int(USE_CEM)}"
        f"_sigma1.0"
        f".json"
    )
    return output_dir / fname


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------


def build_args(run: Run) -> SimpleNamespace:
    n_visible = run.size if run.model == "1d" else run.size**2
    return SimpleNamespace(
        model=run.model,
        size=run.size,
        h=run.h,
        rbm=run.rbm,
        n_hidden=n_visible,
        sampler="dimod",
        sampling_method=run.sampling_method,
        n_samples=N_SAMPLES,
        iterations=N_ITERATIONS,
        learning_rate=run.lr,
        regularization=REG,
        cem=USE_CEM,
        cem_interval=CEM_INTERVAL,
        seed=run.seed,
        visualize=False,
        output_dir=OUTPUT_DIR,
        sigma=1.0,
    )


def execute_run(run: Run) -> dict:
    """Execute one VMC training run.  Returns a summary dict.  Raises on error."""
    key = jax.random.PRNGKey(run.seed)
    key, rbm_key = jax.random.split(key)

    args = build_args(run)
    n_visible = run.size if run.model == "1d" else run.size**2
    n_hidden = n_visible

    if run.model == "1d":
        ising = TransverseFieldIsing1D(run.size, run.h)
    else:
        ising = TransverseFieldIsing2D(run.size, run.h)

    if run.rbm == "full":
        rbm = FullyConnectedRBM(n_visible, n_hidden, rbm_key)
    else:
        rbm = DWaveTopologyRBM(n_visible, n_hidden, rbm_key, solver=run.rbm)

    sampler = DimodSampler(method=run.sampling_method)

    trainer_config = dict(
        learning_rate=run.lr,
        n_iterations=N_ITERATIONS,
        n_samples=N_SAMPLES,
        regularization=REG,
        save_checkpoints=True,
        checkpoint_interval=10,
        use_cem=USE_CEM,
        cem_interval=CEM_INTERVAL,
        lsb_sigma=1.0,
        seed=run.seed,
    )

    t0 = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    save_results(args, history, ising, rbm)

    exact = ising.exact_ground_energy()
    final = history["energy"][-1]
    rel_err = abs(final - exact) / abs(exact)
    kl = history.get("kl_exact", [None])[-1]
    gn = history.get("grad_norm", [None])[-1]

    if run.rbm != "full":
        sparsity = rbm.connectivity_summary()["sparsity"]
    else:
        sparsity = None

    return dict(
        elapsed_s=elapsed,
        rel_error=rel_err,
        final_kl=kl,
        grad_norm=gn,
        sparsity=sparsity,
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
        sampling_method=run.sampling_method,
        rbm=run.rbm,
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
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the run grid without executing"
    )
    parser.add_argument(
        "--method",
        choices=["pegasus", "zephyr"],
        default=None,
        help="Restrict to a single D-Wave sampling method",
    )
    cli = parser.parse_args()

    print(f"JAX devices : {jax.devices()}")
    print(f"JAX version : {jax.__version__}")
    print(f"Output dir  : {OUTPUT_DIR}/")

    try:
        used_ms = read_qpu_time_ms()
    except Exception as e:
        print(f"[QPU BUDGET ERROR] Cannot read {DWAVE_TIME_FILE}: {e} — aborting.")
        sys.exit(1)

    print(
        f"QPU budget  : {DWAVE_BUDGET_MS / 60_000:.0f} min total  |  "
        f"used: {used_ms / 60_000:.2f} min  |  "
        f"remaining: {max(0.0, DWAVE_BUDGET_MS / 60_000 - used_ms / 60_000):.2f} min"
    )

    grid = build_grid()
    if cli.method:
        grid = [r for r in grid if r.sampling_method == cli.method]

    # ── Dry run ──────────────────────────────────────────────────────────────
    if cli.dry_run:
        pending = sum(1 for r in grid if not result_path(r).exists())
        print(
            f"\n{'Model':>4}  {'N':>4}  {'h':>4}  "
            f"{'Method':>8}  {'RBM':>8}  {'LR':>8}  {'Seed':>4}  {'Done':>4}"
        )
        print("-" * 68)
        for r in grid:
            done = "yes" if result_path(r).exists() else "no"
            print(
                f"{r.model:>4}  {r.size:>4}  {r.h:>4}  "
                f"{r.sampling_method:>8}  {r.rbm:>8}  "
                f"{r.lr:>8.4g}  {r.seed:>4}  {done:>4}"
            )
        print(f"\nTotal: {len(grid)}  pending: {pending}  done: {len(grid) - pending}")
        return

    pending = [r for r in grid if not result_path(r).exists()]
    n_skip = len(grid) - len(pending)

    print(
        f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total runs  "
        f"({len(pending)} pending, {n_skip} already done)\n"
        f"  Fixed: reg={REG}  ns={N_SAMPLES}  iter={N_ITERATIONS}\n"
    )

    log_path = Path(__file__).resolve().parent / "dwave_benchmark_failures.jsonl"
    n_done = n_fail = 0
    t_wall = time.perf_counter()

    for i, run in enumerate(pending, 1):
        # Check QPU budget before every run
        try:
            if qpu_budget_exceeded():
                remaining = len(pending) - i + 1
                print(f"  Aborting {remaining} remaining experiments.")
                break
        except Exception as e:
            print(f"[QPU BUDGET ERROR] Cannot read {DWAVE_TIME_FILE}: {e} — aborting.")
            break

        if n_done > 0:
            avg_s = (time.perf_counter() - t_wall) / n_done
            left_s = avg_s * (len(pending) - i + 1)
            eta = f"  ETA ~{left_s / 3600:.1f}h"
        else:
            eta = ""

        try:
            used_ms = read_qpu_time_ms()
        except Exception as e:
            print(f"[QPU BUDGET ERROR] Cannot read {DWAVE_TIME_FILE}: {e} — aborting.")
            break

        print(
            f"[{i}/{len(pending)}] "
            f"{run.model.upper()} N={run.size:>3} h={run.h}  "
            f"{run.sampling_method}  rbm={run.rbm}  "
            f"lr={run.lr:.4g}  seed={run.seed}  "
            f"QPU used={used_ms / 60_000:.2f}min"
            f"{eta}"
        )

        try:
            summary = execute_run(run)
            n_done += 1
            kl_str = (
                f"{summary['final_kl']:.4f}"
                if summary["final_kl"] is not None
                else "N/A"
            )
            gn_str = (
                f"{summary['grad_norm']:.4f}"
                if summary["grad_norm"] is not None
                else "N/A"
            )
            sp_str = (
                f"sparsity={summary['sparsity']:.3f}"
                if summary["sparsity"] is not None
                else ""
            )
            print(
                f"  {summary['elapsed_s']:6.1f}s  "
                f"rel_err={summary['rel_error']:.4f}  "
                f"kl={kl_str}  grad_norm={gn_str}" + (f"  {sp_str}" if sp_str else "")
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

    try:
        print(f"  Total QPU : {read_qpu_time_ms() / 60_000:.2f} min used")
    except Exception:
        pass

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
