#!/usr/bin/env python3
"""
Parallel experiment runner — drop-in replacement for experiment_runner.sh.
Each experiment is an independent Python subprocess; a thread pool runs N of
them concurrently.  Shared state (QPU budget in time.json) is protected by a
lock so parallel D-Wave jobs don't race on the budget counter.

Usage:
    python parallel_runner.py              # uses all CPU cores
    python parallel_runner.py --workers 4  # cap at 4 parallel jobs
    python parallel_runner.py --dry-run    # print all commands, don't execute
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — mirrors experiment_runner.sh
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results/"
LOG_FILE = "parallel_benchmark.log"
SCRIPT = "src/single_experiment.py"

# ── sweep axes ────────────────────────────────────────────────────────────────
SIZES_1D = [8, 16, 32, 64]
SIZES_2D = [4, 6, 8]
H_VALUES = [0.5, 1.0, 2.0]
LEARNING_RATES = [0.1]
SEEDS = [1, 42]

# ── fixed hyperparameters ─────────────────────────────────────────────────────
SAMPLER  = "custom"
METHOD   = "sbm"
RBM      = "full"
N_NH_STEPS  = 1      # n_hidden = n_visible (α = 1)
ITERATIONS  = 300
_REG        = 1e-5   # best reg from sbm_tune results
_NS         = 1000

# ── optimal SBM config (from sbm_tune analysis) ───────────────────────────────
SB_MODE      = "discrete"   # beats ballistic at equal step budget
SB_HEATED    = False        # heated consistently hurts
SB_MAX_STEPS = 500          # sweet spot: best reach-rate, zero divergence

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Thread-safe shared counters + locks
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()
_count_lock = threading.Lock()
_procs_lock = threading.Lock()
_done = 0
_failed = 0
_active_procs: list[subprocess.Popen] = []  # all live child processes


def _shutdown(signum, frame):
    """Kill all tracked child processes on SIGTERM/SIGINT, then exit."""
    log(
        f"  [signal {signum}] Terminating {len(_active_procs)} active subprocess(es)..."
    )
    with _procs_lock:
        for proc in _active_procs:
            try:
                proc.terminate()
            except OSError:
                pass
    sys.exit(1)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )


def log(msg: str):
    with _log_lock:
        logging.info(msg)


# ---------------------------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------------------------


def _n_hidden_steps(n_visible: int) -> list[int]:
    """Generate N_NH_STEPS evenly-spaced n_hidden values ending at n_visible."""
    return [n_visible * k // N_NH_STEPS for k in range(1, N_NH_STEPS + 1)]


def build_experiments() -> list[dict]:
    experiments = []
    for model, sizes in [("1d", SIZES_1D), ("2d", SIZES_2D)]:
        for size in sizes:
            n_visible = size if model == "1d" else size * size
            for n_hidden in _n_hidden_steps(n_visible):
                for h in H_VALUES:
                    for lr in LEARNING_RATES:
                        for seed in SEEDS:
                            experiments.append({
                                "model":     model,
                                "size":      size,
                                "n_visible": n_visible,
                                "n_hidden":  n_hidden,
                                "alpha":     n_hidden / n_visible,
                                "h":         h,
                                "lr":        lr,
                                "seed":      seed,
                            })
    return experiments


# ---------------------------------------------------------------------------
# QPU budget (only relevant for D-Wave sampling methods)
# ---------------------------------------------------------------------------


def result_exists(exp: dict) -> bool:
    """Return True if the result file for this experiment already exists."""
    path = (
        Path(OUTPUT_DIR)
        / str(exp["n_hidden"])
        / SAMPLER
        / METHOD
        / (
            f"result_{exp['model']}"
            f"_h{exp['h']}"
            f"_rbm{RBM}"
            f"_nh{exp['n_hidden']}"
            f"_lr{exp['lr']}"
            f"_reg{_REG}"
            f"_ns{_NS}"
            f"_seed{exp['seed']}"
            f"_iter{ITERATIONS}"
            f".json"
        )
    )
    return path.exists()


# ---------------------------------------------------------------------------
# Single-experiment worker
# ---------------------------------------------------------------------------


def run_experiment(exp: dict, idx: int, total: int, dry_run: bool) -> bool:
    global _done, _failed

    label = (
        f"[{idx:>4}/{total}] "
        f"{exp['model']} N={exp['size']} nh={exp['n_hidden']} "
        f"h={exp['h']} lr={exp['lr']} seed={exp['seed']}"
    )

    if result_exists(exp):
        log(f"  [skip] result already exists — {label}")
        with _count_lock:
            _done += 1
        return True

    log(label)

    cmd = [
        "python3", SCRIPT,
        "--model",       exp["model"],
        "--size",        str(exp["size"]),
        "--n-hidden",    str(exp["n_hidden"]),
        "--h",           str(exp["h"]),
        "--lr",          str(exp["lr"]),
        "--sampler",     SAMPLER,
        "--method",      METHOD,
        "--rbm",         RBM,
        "--iterations",  str(ITERATIONS),
        "--seed",        str(exp["seed"]),
        "--output-dir",  OUTPUT_DIR,
        "--sb-mode",     SB_MODE,
        "--sb-max-steps", str(SB_MAX_STEPS),
    ]
    if SB_HEATED:
        cmd += ["--sb-heated"]

    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        with _count_lock:
            _done += 1
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            log(f"  Retrying (attempt {attempt}/{MAX_RETRIES}) — {label}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        with _procs_lock:
            _active_procs.append(proc)

        stdout, _ = proc.communicate()  # blocks until child exits

        with _procs_lock:
            _active_procs.remove(proc)

        # Flush subprocess output into the shared log (thread-safe)
        if stdout.strip():
            with _log_lock:
                for line in stdout.strip().splitlines():
                    logging.info("    | %s", line)

        if proc.returncode == 0:
            with _count_lock:
                _done += 1
            return True

        log(f"  Attempt {attempt}/{MAX_RETRIES} failed (exit {proc.returncode})")

    log(f"  All {MAX_RETRIES} attempts failed — skipping.")
    with _count_lock:
        _failed += 1
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Parallel VMC experiment runner")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4; SBM is GPU-bound, more workers fragment the GPU).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing them.",
    )
    args = parser.parse_args()

    _setup_logging()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    experiments = build_experiments()
    total = len(experiments)

    log("=" * 60)
    log(f"Parallel runner started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Workers                 : {args.workers}")
    log(f"Total experiments       : {total}")
    log(f"1D sizes                : {SIZES_1D}")
    log(f"2D sizes                : {SIZES_2D}")
    log(f"H values                : {H_VALUES}")
    log(f"Sampler                 : {SAMPLER}/{METHOD}")
    log(f"SBM config              : mode={SB_MODE}  heated={SB_HEATED}  max_steps={SB_MAX_STEPS}")
    log(f"Iterations              : {ITERATIONS}  seeds={SEEDS}")
    if args.dry_run:
        log("DRY RUN — no experiments will be executed")
    log("=" * 60)

    global _done, _failed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_experiment, exp, i + 1, total, args.dry_run): exp
            for i, exp in enumerate(experiments)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                exp = futures[future]
                log(f"  Unhandled exception for {exp}: {exc}")
                with _count_lock:
                    _failed += 1

    log("")
    log("=" * 60)
    log(f"Benchmark finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Completed          : {_done} / {total}")
    log(f"Failed (skipped)   : {_failed}")
    log("=" * 60)


if __name__ == "__main__":
    main()
