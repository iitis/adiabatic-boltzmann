#!/usr/bin/env python3
"""
Comparison runner: our VMC vs NetKet VMC, small-size representative grid.

For every experiment in the grid this script launches TWO jobs in parallel:
  1. src/single_experiment.py   → results/         (our own RBM + SR)
  2. scripts/netket_lsb_vmc.py  → results_netket/  (NetKet VMC + SR)

Both result sets share the same JSON schema so sampler_analysis.py can load
and compare them.  Run sampler_analysis.py against each folder separately, or
pass --results results/ and --results results_netket/ twice.

Usage:
    python compare_runner.py              # run everything
    python compare_runner.py --dry-run    # print commands only
    python compare_runner.py --workers 2
    python compare_runner.py --skip-netket  # only run our RBM
    python compare_runner.py --skip-ours    # only run NetKet
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
# Experiment grid — keep small: representative samplers, small sizes
# ---------------------------------------------------------------------------

OURS_OUTPUT_DIR = "results/"
NETKET_OUTPUT_DIR = "results_netket/"
NETKET_PLOTS_DIR = "plots/netket_vmc/"

OURS_SCRIPT = "src/single_experiment.py"
NETKET_SCRIPT = "scripts/netket_lsb_vmc.py"

LOG_FILE = "compare_runner.log"

# ── sweep axes ──────────────────────────────────────────────────────────────
SIZES = [6, 8]  # small enough to run fast on CPU
H_VALUES = [0.5, 1.0]
SEEDS = [42]

# ── (sampler, method, rbm) triples to compare ───────────────────────────────
# Each entry runs both our RBM and NetKet with the same sampler.
# Add ("dimod","pegasus","full") etc. if QPU access is available.
COMBOS = [
    ("custom", "metropolis", "full"),
    ("custom", "lsb", "full"),
    ("dimod", "simulated_annealing", "full"),
    ("dimod", "zephyr", "full"),
]

# ── fixed hyperparameters (match parallel_runner defaults) ──────────────────
ITERATIONS = 100  # enough to see convergence on small sizes
N_SAMPLES = 512
LR = 0.1
REG = 1e-5
DIAG_SHIFT = 1e-4  # NetKet QGT regularisation

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Thread-safe state
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()
_count_lock = threading.Lock()
_procs_lock = threading.Lock()
_done = 0
_failed = 0
_active_procs: list[subprocess.Popen] = []


def _shutdown(signum, frame):
    with _procs_lock:
        for p in _active_procs:
            try:
                p.terminate()
            except OSError:
                pass
    sys.exit(1)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


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


def build_experiments() -> list[dict]:
    exps = []
    for sampler, method, rbm in COMBOS:
        for size in SIZES:
            n_vis = size  # 1d only
            n_hid = n_vis
            for h in H_VALUES:
                for seed in SEEDS:
                    exps.append(
                        {
                            "model": "1d",
                            "size": size,
                            "n_vis": n_vis,
                            "n_hid": n_hid,
                            "h": h,
                            "seed": seed,
                            "sampler": sampler,
                            "method": method,
                            "rbm": rbm,
                        }
                    )
    return exps


# ---------------------------------------------------------------------------
# Result-exists checks (skip re-running finished experiments)
# ---------------------------------------------------------------------------

_COMMON_SUFFIX = f"_lr{LR}_reg{REG}_ns{N_SAMPLES}"


def _ours_result_path(exp: dict) -> Path:
    """Mirrors the path logic in helpers.save_results() / single_experiment.py."""
    n_hid = exp["n_hid"]
    return (
        Path(OURS_OUTPUT_DIR)
        / str(n_hid)
        / exp["sampler"]
        / exp["method"]
        / (
            f"result_{exp['model']}"
            f"_h{exp['h']}"
            f"_rbm{exp['rbm']}"
            f"_nh{n_hid}"
            f"_lr{LR}"
            f"_reg{REG}"
            f"_ns{N_SAMPLES}"
            f"_seed{exp['seed']}"
            f"_iter{ITERATIONS}"
            f"_cem0.json"
        )
    )


def _netket_result_path(exp: dict) -> Path:
    """Mirrors the path logic in netket_lsb_vmc.py."""
    n_hid = exp["n_hid"]
    method_key = f"{exp['sampler']}_{exp['method']}"
    return (
        Path(NETKET_OUTPUT_DIR)
        / str(n_hid)
        / "netket"
        / method_key
        / (
            f"result_{exp['model']}"
            f"_h{exp['h']}"
            f"_rbm{exp['rbm']}"
            f"_nh{n_hid}"
            f"_lr{LR}"
            f"_ns{N_SAMPLES}"
            f"_seed{exp['seed']}"
            f"_iter{ITERATIONS}"
            f".json"
        )
    )


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def _ours_cmd(exp: dict) -> list[str]:
    return [
        "python3",
        OURS_SCRIPT,
        "--model",
        exp["model"],
        "--size",
        str(exp["size"]),
        "--n-hidden",
        str(exp["n_hid"]),
        "--h",
        str(exp["h"]),
        "--lr",
        str(LR),
        "--sampler",
        exp["sampler"],
        "--method",
        exp["method"],
        "--rbm",
        exp["rbm"],
        "--iterations",
        str(ITERATIONS),
        "--n-samples",
        str(N_SAMPLES),
        "--seed",
        str(exp["seed"]),
        "--output-dir",
        OURS_OUTPUT_DIR,
    ]


def _netket_cmd(exp: dict) -> list[str]:
    # netket_lsb_vmc.py is run from the repo root; src/ is added to sys.path inside it
    return [
        "python3",
        NETKET_SCRIPT,
        "--model",
        exp["model"],
        "--size",
        str(exp["size"]),
        "--n-hidden",
        str(exp["n_hid"]),
        "--h",
        str(exp["h"]),
        "--lr",
        str(LR),
        "--sampler",
        exp["sampler"],
        "--sampling-method",
        exp["method"],
        "--rbm",
        exp["rbm"],
        "--iterations",
        str(ITERATIONS),
        "--n-samples",
        str(N_SAMPLES),
        "--seed",
        str(exp["seed"]),
        "--diag-shift",
        str(DIAG_SHIFT),
        "--results-dir",
        NETKET_OUTPUT_DIR,
        "--output",
        NETKET_PLOTS_DIR,
    ]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], label: str) -> bool:
    global _done, _failed

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            log(f"  Retrying (attempt {attempt}/{MAX_RETRIES}) — {label}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        with _procs_lock:
            _active_procs.append(proc)

        stdout, _ = proc.communicate()

        with _procs_lock:
            _active_procs.remove(proc)

        if stdout.strip():
            with _log_lock:
                for line in stdout.strip().splitlines():
                    logging.info("    | %s", line)

        if proc.returncode == 0:
            return True

        log(
            f"  Attempt {attempt}/{MAX_RETRIES} failed (exit {proc.returncode}) — {label}"
        )

    return False


def run_job(job: dict, idx: int, total: int, dry_run: bool) -> bool:
    """A job is one (experiment, framework) pair."""
    global _done, _failed

    exp = job["exp"]
    framework = job["framework"]  # "ours" or "netket"
    result_p = job["result_path"]
    cmd = job["cmd"]
    label = (
        f"[{idx:>4}/{total}] [{framework:>6}] "
        f"N={exp['size']} h={exp['h']} "
        f"{exp['sampler']}/{exp['method']}"
    )

    if result_p.exists():
        log(f"  [skip] {label}")
        with _count_lock:
            _done += 1
        return True

    log(label)

    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        with _count_lock:
            _done += 1
        return True

    ok = _run_cmd(cmd, label)
    with _count_lock:
        if ok:
            _done += 1
        else:
            _failed += 1
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Our VMC vs NetKet comparison runner")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ours", action="store_true", help="Skip our RBM jobs")
    parser.add_argument("--skip-netket", action="store_true", help="Skip NetKet jobs")
    args = parser.parse_args()

    _setup_logging()
    Path(OURS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(NETKET_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(NETKET_PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    experiments = build_experiments()

    # Build job list: each experiment → up to 2 jobs (ours + netket)
    jobs = []
    for exp in experiments:
        if not args.skip_ours:
            jobs.append(
                {
                    "exp": exp,
                    "framework": "ours",
                    "result_path": _ours_result_path(exp),
                    "cmd": _ours_cmd(exp),
                }
            )
        if not args.skip_netket:
            jobs.append(
                {
                    "exp": exp,
                    "framework": "netket",
                    "result_path": _netket_result_path(exp),
                    "cmd": _netket_cmd(exp),
                }
            )

    total = len(jobs)

    log("=" * 65)
    log(f"Compare runner started  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Workers                 : {args.workers}")
    log(
        f"Total jobs              : {total}  "
        f"({'ours+netket' if not args.skip_ours and not args.skip_netket else 'partial'})"
    )
    log(f"Sizes                   : {SIZES}")
    log(f"H values                : {H_VALUES}")
    for s, m, r in COMBOS:
        log(f"Combo                   : {s}/{m}  rbm={r}")
    log(f"Iterations              : {ITERATIONS}  n_samples={N_SAMPLES}")
    log(f"Our results  →  {OURS_OUTPUT_DIR}")
    log(f"NetKet results → {NETKET_OUTPUT_DIR}")
    if args.dry_run:
        log("DRY RUN — no experiments will be executed")
    log("=" * 65)

    global _done, _failed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_job, job, i + 1, total, args.dry_run): job
            for i, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                job = futures[future]
                log(f"  Unhandled exception ({job['framework']}): {exc}")
                with _count_lock:
                    _failed += 1

    log("")
    log("=" * 65)
    log(f"Finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Done     : {_done} / {total}")
    log(f"Failed   : {_failed}")
    log("")
    log("To analyse results:")
    log(
        f"  python scripts/sampler_analysis.py --results {OURS_OUTPUT_DIR}   --output plots/analysis_ours/"
    )
    log(
        f"  python scripts/sampler_analysis.py --results {NETKET_OUTPUT_DIR} --output plots/analysis_netket/"
    )
    log("=" * 65)


if __name__ == "__main__":
    main()
