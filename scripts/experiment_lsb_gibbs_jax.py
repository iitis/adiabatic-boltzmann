"""
JAX experiment runner — Gibbs and/or LSB sampler, full RBM.

Grid:
  1D  sizes 16..200 spins           h = [0.5, 1.0, 2.0, 3.044]
  2D  L=4..14  (N=L²=16..196 spins) h = [0.5, 1.0, 2.0, 3.044]
  LR  [3e-4, 1e-2]
  seed 1
  Runs sorted by n_visible ascending (small systems first).

Sampler behaviour:
  gibbs  — CEM off, n_sweeps=10
  lsb    — CEM on,  cem_interval=5

Fixed (both methods):
  rbm        FullyConnectedRBM, n_hidden = n_visible
  n_samples  1000
  reg        1e-5
  iterations 300
  sigma      1.0   (LSB / beta-x scaling)
  lsb_steps  100
  lsb_delta  1.0

Results written to jax_results/ (skips runs that already exist).

Usage
-----
    cd <repo-root>
    python scripts/experiment_lsb_gibbs_jax.py                    # gibbs + lsb
    python scripts/experiment_lsb_gibbs_jax.py --sampler gibbs    # gibbs only
    python scripts/experiment_lsb_gibbs_jax.py --sampler lsb      # lsb only
    python scripts/experiment_lsb_gibbs_jax.py --dry-run          # preview grid
    python scripts/experiment_lsb_gibbs_jax.py --rerun-collapsed  # redo Gibbs runs
                                                                   # with collapse-
                                                                   # reinit bias
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
    sigma=1.0,
    lsb_steps=100,
    lsb_delta=1.0,
    n_sweeps=10,       # Gibbs only
    cem_interval=5,    # LSB only
)

LEARNING_RATES  = [3e-4, 1e-2]
SAMPLER_BACKEND = "custom"

# Per-method settings — CEM is only ever enabled for LSB
METHOD_CONFIG = {
    "gibbs": dict(use_cem=False),
    "lsb":   dict(use_cem=True),
}


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

@dataclass
class Run:
    model:  str    # "1d" | "2d"
    size:   int    # chain length or lattice linear dim L
    h:      float
    lr:     float
    seed:   int
    method: str    # "gibbs" | "lsb"


def build_grid(methods: list[str]) -> list[Run]:
    grid: list[Run] = []

    sizes_1d = [16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 200]
    sizes_2d = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for method in methods:
        for size in sizes_1d:
            for h in [0.5, 1.0, 2.0, 3.044]:
                for lr in LEARNING_RATES:
                    grid.append(Run("1d", size, h, lr, seed=1, method=method))
        for size in sizes_2d:
            for h in [0.5, 1.0, 2.0, 3.044]:
                for lr in LEARNING_RATES:
                    grid.append(Run("2d", size, h, lr, seed=1, method=method))

    # small systems first, then by method so same-size runs are adjacent
    grid.sort(key=lambda r: (r.size if r.model == "1d" else r.size ** 2, r.method))

    return grid


# ---------------------------------------------------------------------------
# Result path — mirrors helpers.save_results naming exactly
# ---------------------------------------------------------------------------

def result_path(run: Run) -> Path:
    n_visible = run.size if run.model == "1d" else run.size ** 2
    use_cem   = METHOD_CONFIG[run.method]["use_cem"]
    output_dir = (
        Path(str(FIXED["output_dir"])) / str(run.size) / SAMPLER_BACKEND / run.method
    )
    fname = (
        f"result_{run.model}"
        f"_h{run.h}"
        f"_rbm{FIXED['rbm']}"
        f"_nh{n_visible}"
        f"_lr{run.lr}"
        f"_reg{FIXED['reg']}"
        f"_ns{FIXED['n_samples']}"
        f"_seed{run.seed}"
        f"_iter{FIXED['iterations']}"
        f"_cem{int(use_cem)}"
        f"_sigma{float(FIXED['sigma'])}"
        f".json"
    )
    return output_dir / fname


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def build_args(run: Run) -> SimpleNamespace:
    n_visible = run.size if run.model == "1d" else run.size ** 2
    use_cem   = METHOD_CONFIG[run.method]["use_cem"]
    return SimpleNamespace(
        model=run.model,
        size=run.size,
        h=run.h,
        rbm=FIXED["rbm"],
        n_hidden=n_visible,
        sampler=SAMPLER_BACKEND,
        sampling_method=run.method,
        n_samples=FIXED["n_samples"],
        iterations=FIXED["iterations"],
        learning_rate=run.lr,
        regularization=FIXED["reg"],
        cem=use_cem,
        cem_interval=FIXED["cem_interval"],
        seed=run.seed,
        visualize=FIXED["visualize"],
        output_dir=FIXED["output_dir"],
        sigma=FIXED["sigma"],
        lsb_steps=FIXED["lsb_steps"],
        lsb_delta=FIXED["lsb_delta"],
    )


def execute_run(run: Run) -> dict:
    key = jax.random.PRNGKey(run.seed)
    key, rbm_key = jax.random.split(key)

    args      = build_args(run)
    n_visible = run.size if run.model == "1d" else run.size ** 2
    use_cem   = METHOD_CONFIG[run.method]["use_cem"]

    if run.model == "1d":
        ising = TransverseFieldIsing1D(run.size, run.h)
    else:
        ising = TransverseFieldIsing2D(run.size, run.h)

    rbm = FullyConnectedRBM(n_visible, n_visible, rbm_key)

    if run.method == "gibbs":
        sampler = ClassicalSampler(method="gibbs", n_sweeps=int(FIXED["n_sweeps"]))
    else:
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
        use_cem=use_cem,
        cem_interval=FIXED["cem_interval"],
        lsb_sigma=FIXED["sigma"],
        lsb_steps=FIXED["lsb_steps"],
        lsb_delta=FIXED["lsb_delta"],
        seed=run.seed,
    )

    t0      = time.perf_counter()
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
    kl = history.get("kl_exact", [None])[-1]
    gn = history.get("grad_norm", [None])[-1]

    return dict(elapsed_s=elapsed, rel_error=rel_err, final_kl=kl, grad_norm=gn)


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------

def _write_failure(log_path: Path, run: Run, exc: Exception):
    entry = dict(
        timestamp=datetime.now().isoformat(),
        model=run.model,
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
# Collapse detection (Gibbs only)
# ---------------------------------------------------------------------------

def _is_collapsed(path: Path) -> bool:
    """
    True if the saved Gibbs result shows collapse-reinit bias:
    any sampling_time_s after iter 0 exceeds 0.1 s.
    """
    try:
        with open(path) as f:
            d = json.load(f)
        times = d.get("history", {}).get("sampling_time_s", [])
        return any(t > 0.1 for t in times[1:])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sampler", choices=["gibbs", "lsb", "both"], default="both",
        help="Which sampler(s) to run (default: both)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run grid without executing")
    parser.add_argument("--rerun-collapsed", action="store_true",
                        help="Delete and re-run Gibbs results with collapse-reinit bias")
    cli = parser.parse_args()

    methods = ["gibbs", "lsb"] if cli.sampler == "both" else [cli.sampler]

    print(f"JAX devices : {jax.devices()}")
    print(f"JAX version : {jax.__version__}")
    print(f"Samplers    : {', '.join(methods)}")
    print(f"Output dir  : {FIXED['output_dir']}/")

    grid = build_grid(methods)

    if cli.dry_run:
        pending = sum(1 for r in grid if not result_path(r).exists())
        print(
            f"\n{'Method':>5}  {'Model':>4}  {'N':>4}  {'h':>6}  {'LR':>8}  {'Done':>4}"
        )
        print("-" * 52)
        for r in grid:
            done = "yes" if result_path(r).exists() else "no"
            print(
                f"{r.method:>5}  {r.model:>4}  {r.size:>4}  {r.h:>6}  "
                f"{r.lr:>8.4g}  {done:>4}"
            )
        print(f"\nTotal: {len(grid)}  pending: {pending}  done: {len(grid)-pending}")
        return

    if cli.rerun_collapsed:
        gibbs_runs = [r for r in grid if r.method == "gibbs"]
        collapsed  = [r for r in gibbs_runs if _is_collapsed(result_path(r))]
        if collapsed:
            print(f"Deleting {len(collapsed)} Gibbs result(s) with collapse-reinit bias...")
            for r in collapsed:
                result_path(r).unlink()
                print(f"  deleted {result_path(r).name}")
        else:
            print("No collapsed Gibbs results found.")

    pending = [r for r in grid if not result_path(r).exists()]
    n_skip  = len(grid) - len(pending)

    print(
        f"\n[{datetime.now():%H:%M:%S}]  {len(grid)} total runs  "
        f"({len(pending)} pending, {n_skip} already done)\n"
        f"  Fixed: reg={FIXED['reg']}  ns={FIXED['n_samples']}  "
        f"iter={FIXED['iterations']}\n"
        f"  gibbs: n_sweeps={FIXED['n_sweeps']}  cem=off\n"
        f"  lsb:   lsb_steps={FIXED['lsb_steps']}  cem=on  "
        f"cem_interval={FIXED['cem_interval']}\n"
    )

    log_path = Path(__file__).resolve().parent / "experiment_lsb_gibbs_jax_failures.jsonl"
    n_done = n_fail = 0
    t_wall = time.perf_counter()

    for i, run in enumerate(pending, 1):
        tag = (
            f"[{i}/{len(pending)}] "
            f"{run.model.upper()} N={run.size:>3} "
            f"h={run.h}  {run.method}  "
            f"lr={run.lr:.4g}  seed={run.seed}"
        )

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
