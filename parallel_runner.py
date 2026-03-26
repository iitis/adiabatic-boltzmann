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
import json
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
MODEL = "2d"
SIZES = [6, 8]  # N=4 is complete for all samplers

N_NH_STEPS = 1  # generates N evenly-spaced steps ending at n_visible
LEARNING_RATES = [0.1]  # only lr=0.1 is used in plots
SAMPLERS = [
    ("custom", "metropolis"),
    ("dimod", "simulated_annealing"),
    ("velox", "velox"),
]  # list of (sampler, method) pairs
RBMS = ["full"]
H_VALUES = [0.5, 1.0, 2.0]
SEEDS = [1, 42]
ITERATIONS = 300

DWAVE_BUDGET_MS = 1_800_000  # 30 minutes in ms
TIME_FILE = Path("time.json")

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Thread-safe shared counters + locks
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()
_qpu_lock = threading.Lock()
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
    for size in SIZES:
        n_visible = size * size if MODEL == "2d" else size
        for n_hidden in _n_hidden_steps(n_visible):
            alpha = n_hidden / n_visible
            for rbm in RBMS:
                for h in H_VALUES:
                    for lr in LEARNING_RATES:
                        for sampler, method in SAMPLERS:
                            for seed in SEEDS:
                                experiments.append(
                                    {
                                        "size": size,
                                        "n_visible": n_visible,
                                        "n_hidden": n_hidden,
                                        "alpha": alpha,
                                        "rbm": rbm,
                                        "h": h,
                                        "lr": lr,
                                        "sampler": sampler,
                                        "method": method,
                                        "seed": seed,
                                    }
                                )
    return experiments


# ---------------------------------------------------------------------------
# QPU budget (only relevant for D-Wave sampling methods)
# ---------------------------------------------------------------------------


def _is_dwave(method: str) -> bool:
    return method in ("pegasus", "zephyr")


def _read_qpu_time_ms() -> int:
    return int(json.loads(TIME_FILE.read_text()).get("time_ms"))


def _check_qpu_budget() -> tuple[bool, int]:
    """Returns (budget_ok, used_ms). Must be called with _qpu_lock held."""
    used = _read_qpu_time_ms()
    return used < DWAVE_BUDGET_MS, used


# ---------------------------------------------------------------------------
# Single-experiment worker
# ---------------------------------------------------------------------------


def run_experiment(exp: dict, idx: int, total: int, dry_run: bool) -> bool:
    global _done, _failed

    label = (
        f"[{idx:>4}/{total}] "
        f"size={exp['size']} nh={exp['n_hidden']} (α={exp['alpha']:.2f}) "
        f"rbm={exp['rbm']} h={exp['h']} lr={exp['lr']} "
        f"{exp['sampler']}/{exp['method']} seed={exp['seed']}"
    )

    # QPU budget check — serialised so parallel D-Wave jobs don't race
    if _is_dwave(exp["method"]):
        with _qpu_lock:
            ok, used = _check_qpu_budget()
            if not ok:
                log(f"  [QPU BUDGET] {used / 60000:.2f} min used — skipping {label}")
                with _count_lock:
                    _failed += 1
                return False
        qpu_info = f"  QPU used={used / 60000:.2f}min"
    else:
        qpu_info = ""

    log(f"{label}{qpu_info}")

    cmd = [
        "python3",
        SCRIPT,
        "--size",
        str(exp["size"]),
        "--lr",
        str(exp["lr"]),
        "--sampler",
        exp["sampler"],
        "--method",
        exp["method"],
        "--seed",
        str(exp["seed"]),
        "--output-dir",
        OUTPUT_DIR,
        "--model",
        MODEL,
        "--n-hidden",
        str(exp["n_hidden"]),
        "--h",
        str(exp["h"]),
        "--iterations",
        str(ITERATIONS),
        "--rbm",
        exp["rbm"],
    ]

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
        default=os.cpu_count(),
        help=f"Parallel workers (default: {os.cpu_count()} = cpu count). "
        "Rule of thumb: cpu_count for CPU-bound runs, "
        "more for I/O-bound or QPU-queued runs.",
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
    used_ms_start = _read_qpu_time_ms()

    log("=" * 60)
    log(f"Parallel runner started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Workers                 : {args.workers}")
    log(f"Total experiments       : {total}")
    log(f"Model                   : {MODEL}  sizes={SIZES}")
    log(f"N_NH_STEPS              : {N_NH_STEPS}")
    log(f"RBMs                    : {RBMS}")
    log(f"H values                : {H_VALUES}")
    log(f"Samplers                : {SAMPLERS}")
    log(
        f"QPU budget              : {DWAVE_BUDGET_MS / 60000:.1f} min  "
        f"(used so far: {used_ms_start / 60000:.2f} min)"
    )
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

    used_ms_end = _read_qpu_time_ms()
    log("")
    log("=" * 60)
    log(f"Benchmark finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Completed          : {_done} / {total}")
    log(f"Failed (skipped)   : {_failed}")
    log(f"QPU time this run  : {(used_ms_end - used_ms_start) / 60000:.2f} min")
    log(f"QPU time total     : {used_ms_end / 60000:.2f} min")
    log("=" * 60)


if __name__ == "__main__":
    main()
