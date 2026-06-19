"""
Hardware sampler sweep — FPGA, Langevin SB, and VeloxQ SA backends.

Sweeps: N, h ∈ {0.5, 1.0, 2.0}, learning rates, seeds on TFIM 1D.
Select backend with --sampler; each has its own sensible defaults for sizes
and learning rates which can be overridden via CLI.

Usage (from project root):
    python scripts/exper/experiment_hardware.py --sampler fpga
    python scripts/exper/experiment_hardware.py --sampler langevin
    python scripts/exper/experiment_hardware.py --sampler veloxq
    python scripts/exper/experiment_hardware.py --sampler fpga --dry-run
    python scripts/exper/experiment_hardware.py --sampler fpga --sizes 24 48 --lrs 0.01 0.1
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

import jax
jax.config.update("jax_enable_x64", True)

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

from encoder import Trainer
from helpers import find_latest_checkpoint, restore_rbm_from_checkpoint, save_results
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM

_REPO = _SRC.parent

# ── Per-sampler defaults ──────────────────────────────────────────────────────

_SAMPLER_DEFAULTS = {
    "fpga":    dict(sizes=[24, 48, 64, 100, 128, 200], lrs=[0.01, 0.1],  use_cem=False,
                    sampler_tag="fpga",    method_tag="fpga"),
    "langevin": dict(sizes=[24],           lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], use_cem=True,
                    sampler_tag="langevin", method_tag="langevin"),
    "veloxq":  dict(sizes=[24],            lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], use_cem=True,
                    sampler_tag="veloxq",  method_tag="sa"),
}

FIXED = dict(n_samples=1000, reg=1e-5, iterations=300, rbm="full", sigma=1.0)
H_VALUES = [0.5, 1.0, 2.0]
SEEDS    = [1]

# Langevin SB integration knobs
LANGEVIN_NUM_STEPS   = 1000
LANGEVIN_DT          = 0.25
LANGEVIN_NOISE_SIGMA = 1.0
LANGEVIN_DETUNING    = 1.0
LANGEVIN_SCALE       = 1.0


# ── Grid ──────────────────────────────────────────────────────────────────────


@dataclass
class Run:
    size: int
    h: float
    lr: float
    seed: int


def build_grid(sizes, lrs) -> list[Run]:
    return [Run(size=s, h=h, lr=lr, seed=seed)
            for s in sizes for h in H_VALUES for lr in lrs for seed in SEEDS]


# ── Result path ───────────────────────────────────────────────────────────────


def result_path(run: Run, sampler_tag: str, method_tag: str, use_cem: bool) -> Path:
    output_dir = _REPO / "results" / "tfim_1d" / str(run.size) / sampler_tag / method_tag
    fname = (
        f"result_1d"
        f"_h{run.h}"
        f"_rbm{FIXED['rbm']}"
        f"_nh{run.size}"
        f"_lr{run.lr}"
        f"_reg{FIXED['reg']}"
        f"_ns{FIXED['n_samples']}"
        f"_seed{run.seed}"
        f"_iter{FIXED['iterations']}"
        f"_cem{int(use_cem)}"
        f"_sigma{FIXED['sigma']}"
        f".json"
    )
    return output_dir / fname


# ── Sampler factory ───────────────────────────────────────────────────────────


def make_sampler(sampler_name: str, n_samples: int):
    if sampler_name == "fpga":
        from sampler import FPGASampler
        return FPGASampler(transport="auto")
    if sampler_name == "langevin":
        from sampler import LangevinSampler
        return LangevinSampler(
            num_rep=max(1024, n_samples),
            num_steps=LANGEVIN_NUM_STEPS,
            dt=LANGEVIN_DT,
            sigma=LANGEVIN_NOISE_SIGMA,
            detuning=LANGEVIN_DETUNING,
            scale=LANGEVIN_SCALE,
        )
    if sampler_name == "veloxq":
        from sampler import VeloxQStandardSASampler
        return VeloxQStandardSASampler()
    raise ValueError(f"Unknown sampler: {sampler_name!r}")


# ── Single run ────────────────────────────────────────────────────────────────


def build_args(run: Run, sampler_tag: str, method_tag: str, use_cem: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model="1d", size=run.size, h=run.h, rbm=FIXED["rbm"], n_hidden=run.size,
        sampler=sampler_tag, sampling_method=method_tag,
        n_samples=FIXED["n_samples"], iterations=FIXED["iterations"],
        learning_rate=run.lr, regularization=FIXED["reg"],
        cem=use_cem, cem_interval=5, seed=run.seed,
        visualize=False, output_dir=str(_REPO / "results"), sigma=FIXED["sigma"],
    )


def execute_run(run: Run, sampler_name: str, sampler_tag: str, method_tag: str,
                use_cem: bool) -> dict:
    key = jax.random.PRNGKey(run.seed)
    key, rbm_key = jax.random.split(key)

    args   = build_args(run, sampler_tag, method_tag, use_cem)
    ising  = TransverseFieldIsing1D(run.size, run.h)
    rbm    = FullyConnectedRBM(run.size, run.size, rbm_key)
    sampler = make_sampler(sampler_name, FIXED["n_samples"])

    start_iteration = 0
    latest_ckpt = find_latest_checkpoint(args)
    if latest_ckpt is not None:
        ckpt_iter = restore_rbm_from_checkpoint(rbm, latest_ckpt)
        start_iteration = ckpt_iter + 1
        print(f"  [Resume] checkpoint iter {ckpt_iter} → resuming from {start_iteration}")

    trainer_config = dict(
        learning_rate=run.lr, n_iterations=FIXED["iterations"],
        n_samples=FIXED["n_samples"], regularization=FIXED["reg"],
        save_checkpoints=True, checkpoint_interval=10,
        use_cem=use_cem, cem_interval=5, seed=run.seed,
    )

    try:
        trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
        history = trainer.train(start_iteration=start_iteration)
        save_results(args, history, ising, rbm)

        exact   = ising.exact_ground_energy()
        final   = history["energy"][-1]
        rel_err = abs(final - exact) / abs(exact)
        kl      = history["kl_exact"][-1]
        gn      = history["grad_norm"][-1]
        return dict(rel_error=rel_err, final_kl=kl, grad_norm=gn)
    finally:
        close = getattr(sampler, "close", None)
        if callable(close):
            close()


# ── Worker / failure log ──────────────────────────────────────────────────────


def _worker(args_tuple) -> tuple:
    run, sampler_name, sampler_tag, method_tag, use_cem = args_tuple
    try:
        summary = execute_run(run, sampler_name, sampler_tag, method_tag, use_cem)
        return run, summary, None
    except Exception as exc:
        return run, None, exc


def _write_failure(log_path: Path, run: Run, exc: Exception):
    entry = dict(timestamp=datetime.now().isoformat(), size=run.size, h=run.h,
                 lr=run.lr, seed=run.seed, error=type(exc).__name__, message=str(exc))
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sampler", choices=["fpga", "langevin", "veloxq"], required=True,
                   help="Hardware backend to run")
    p.add_argument("--sizes",      type=int,   nargs="+", default=None,
                   help="System sizes (default: sampler-specific)")
    p.add_argument("--lrs",        type=float, nargs="+", default=None,
                   help="Learning rates (default: sampler-specific)")
    p.add_argument("--iterations", type=int,   default=FIXED["iterations"],
                   help="Training iterations per run (default: 300)")
    p.add_argument("--workers",    type=int,   default=1)
    p.add_argument("--serial",     action="store_true",
                   help="Run in-process (no multiprocessing)")
    p.add_argument("--dry-run",    action="store_true")
    cli = p.parse_args()

    defaults     = _SAMPLER_DEFAULTS[cli.sampler]
    sizes        = cli.sizes or defaults["sizes"]
    lrs          = cli.lrs   or defaults["lrs"]
    sampler_tag  = defaults["sampler_tag"]
    method_tag   = defaults["method_tag"]
    use_cem      = defaults["use_cem"]
    FIXED["iterations"] = cli.iterations

    grid    = build_grid(sizes, lrs)
    rp      = lambda r: result_path(r, sampler_tag, method_tag, use_cem)
    pending = [r for r in grid if not rp(r).exists()]
    n_skip  = len(grid) - len(pending)

    if cli.dry_run:
        print(f"{'N':>4}  {'h':>4}  {'LR':>10}  {'Seed':>4}  {'Done':>4}")
        print("-" * 40)
        for r in grid:
            done = "yes" if rp(r).exists() else "no"
            print(f"{r.size:>4}  {r.h:>4}  {r.lr:>10.4g}  {r.seed:>4}  {done:>4}")
        print(f"\nTotal: {len(grid)} | pending: {len(pending)} | done: {n_skip}")
        return

    print(f"[{datetime.now():%H:%M:%S}] {cli.sampler} sweep — {len(grid)} total runs")
    print(f"  N={sizes}  h={H_VALUES}  lr={lrs}")
    print(f"  Pending: {len(pending)}  skipped: {n_skip}  workers: {cli.workers}\n")

    log_path = Path(__file__).resolve().parent / f"experiment_{cli.sampler}_failures.jsonl"
    n_done = n_fail = 0
    worker_args = [(r, cli.sampler, sampler_tag, method_tag, use_cem) for r in pending]

    def _print_result(idx, run, summary, exc):
        nonlocal n_done, n_fail
        tag = (f"[{idx}/{len(pending)}] "
               f"N={run.size} h={run.h} lr={run.lr:.4g} seed={run.seed}")
        if exc is not None:
            n_fail += 1
            print(f"  FAIL  {tag}\n         {type(exc).__name__}: {exc}")
            _write_failure(log_path, run, exc)
        else:
            n_done += 1
            kl_str = f"{summary['final_kl']:.4f}" if summary.get("final_kl") is not None else "N/A"
            print(f"  DONE  {tag}")
            print(f"         rel_err={summary['rel_error']:.4f}  kl={kl_str}"
                  f"  grad_norm={summary['grad_norm']:.4f}")

    if cli.serial:
        for idx, wa in enumerate(worker_args, 1):
            run, summary, exc = _worker(wa)
            _print_result(idx, run, summary, exc)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cli.workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_worker, wa): i for i, wa in enumerate(worker_args, 1)}
            for future in as_completed(futures):
                idx = futures[future]
                run, summary, exc = future.result()
                _print_result(idx, run, summary, exc)

    print(f"\n[{datetime.now():%H:%M:%S}] Finished.")
    print(f"  Completed: {n_done}  Skipped: {n_skip}  Failed: {n_fail}"
          + (f"  → see {log_path}" if n_fail else ""))
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
