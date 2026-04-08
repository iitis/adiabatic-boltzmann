#!/usr/bin/env python3
"""
D-Wave vs classical scaling benchmark.

Scientific question
-------------------
Does D-Wave quantum annealing produce better VMC samples than classical MCMC
at large system sizes and near the quantum critical point of the 1D/2D TFIM?

Experiment design
-----------------
All three samplers use the SAME DWaveTopologyRBM (zephyr topology), so
architecture effects are held fixed and only the sampling strategy varies:

  A) dimod / zephyr          — D-Wave QPU (trivial embedding, no chains)
  B) custom / metropolis      — Metropolis-Hastings, same RBM connectivity
  C) custom / simulated_annealing — SA, same RBM connectivity

Sweep axes:
  * sizes_1d / L_values_2d   — loaded automatically from probe_results/zephyr_capacity.csv
                                (run probe_capacity.py first; the script will error if the
                                CSV is missing)
  * h ∈ {0.5, 1.0, 2.0}     — ordered / critical / disordered
  * seeds × 5                — for error bars
  * 300 iterations each

Metrics saved per run (via helpers.save_results + encoder history):
  - energy convergence curve
  - final_error (|E_VMC − E_exact|)
  - final_kl_exact
  - final_n_unique_ratio  ← sample diversity
  - mean_n_unique_ratio
  - mean_ess

QPU budget: 60 minutes total (DWAVE_BUDGET_MS).  Classical jobs are
unlimited and run in parallel up to --workers threads.  D-Wave jobs are
gated behind a serialised QPU lock (same as parallel_runner.py).

Usage (from project root, venv active):
    cd src && python ../dwave_scaling_benchmark.py
    cd src && python ../dwave_scaling_benchmark.py --workers 8
    cd src && python ../dwave_scaling_benchmark.py --dry-run
"""

import argparse
import csv
import json
import logging
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# Sizes are loaded at runtime from the probe CSV (see load_probe_sizes below).
# Run probe_capacity.py first — the benchmark will raise an error if the CSV
# is missing.
PROBE_CSV = Path("probe_results/zephyr_capacity.csv")

H_VALUES = [0.5, 1.0, 2.0]
LEARNING_RATES = [0.1]
SEEDS = [42]
ITERATIONS = 300
_REG = 1e-3
_NS = 1000

# All three samplers use the same DWaveTopologyRBM (rbm="zephyr")
# so only the sampler changes.  Classical combos consume no QPU budget.
COMBOS = [
    # (sampler,  method,                rbm)
    ("dimod", "zephyr", "zephyr"),  # D-Wave QPU
    ("custom", "metropolis", "zephyr"),  # classical MH
    ("custom", "simulated_annealing", "zephyr"),  # classical SA
]

DWAVE_BUDGET_MS = 3_600_000  # 60 minutes

# ---------------------------------------------------------------------------
# Fixed paths / limits
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results/"
LOG_FILE = "dwave_scaling_benchmark.log"
SCRIPT = "src/single_experiment.py"
TIME_FILE = Path("time.json")
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Thread-safe counters and locks
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()
_qpu_lock = threading.Lock()
_count_lock = threading.Lock()
_procs_lock = threading.Lock()

_done = 0
_failed = 0
_active_procs: list[subprocess.Popen] = []


def _shutdown(_signum, _frame):
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
# Probe CSV loader
# ---------------------------------------------------------------------------


def load_probe_sizes(csv_path: Path) -> tuple[list[int], list[int]]:
    """
    Read probe_capacity.py output and return (sizes_1d, l_values_2d).

    Selects all "well_connected" rows; falls back to all "viable" rows
    (with a warning) if no well-connected rows exist for a model.

    Raises FileNotFoundError if the CSV is absent — run probe_capacity.py first.
    Raises RuntimeError if the CSV contains no usable rows at all.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Probe results not found: {csv_path}\n"
            "Run probe_capacity.py first:\n"
            "    cd src && python ../probe_capacity.py"
        )

    rows_1d: list[dict] = []
    rows_2d: list[dict] = []

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error"):  # skip failed probe entries
                continue
            if row["model"] == "1d":
                rows_1d.append(row)
            elif row["model"] == "2d":
                rows_2d.append(row)

    def _pick(rows: list[dict], model_label: str) -> list[int]:
        wc = [r for r in rows if r["well_connected"] == "True"]
        if wc:
            return sorted({int(r["size"]) for r in wc})
        viable = [r for r in rows if r["viable"] == "True"]
        if viable:
            print(
                f"  [probe] WARNING: no well-connected {model_label} sizes found; "
                f"falling back to viable sizes (deg_vis_min > 0)."
            )
            return sorted({int(r["size"]) for r in viable})
        raise RuntimeError(
            f"No viable {model_label} sizes found in {csv_path}. "
            "The Zephyr topology may not support the requested model sizes."
        )

    sizes_1d = _pick(rows_1d, "1D")
    l_values_2d = _pick(rows_2d, "2D")
    return sizes_1d, l_values_2d


# ---------------------------------------------------------------------------
# Experiment list
# ---------------------------------------------------------------------------


def build_experiments(sizes_1d: list[int], l_values_2d: list[int]) -> list[dict]:
    experiments = []
    for sampler, method, rbm in COMBOS:
        for model in ("1d", "2d"):
            raw_sizes = sizes_1d if model == "1d" else l_values_2d
            for raw_size in raw_sizes:
                n_visible = raw_size if model == "1d" else raw_size * raw_size
                n_hidden = n_visible  # α = 1
                for h in H_VALUES:
                    for lr in LEARNING_RATES:
                        for seed in SEEDS:
                            experiments.append(
                                {
                                    "model": model,
                                    "size": raw_size,
                                    "n_visible": n_visible,
                                    "n_hidden": n_hidden,
                                    "h": h,
                                    "lr": lr,
                                    "seed": seed,
                                    "sampler": sampler,
                                    "method": method,
                                    "rbm": rbm,
                                }
                            )
    return experiments


# ---------------------------------------------------------------------------
# QPU budget
# ---------------------------------------------------------------------------


def _is_dwave(method: str) -> bool:
    return method in ("pegasus", "zephyr")


def _read_qpu_time_ms() -> int:
    try:
        return int(json.loads(TIME_FILE.read_text()).get("time_ms", 0))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"QPU time file {TIME_FILE} not found — cannot enforce budget."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read QPU time from {TIME_FILE}: {exc}") from exc


def _check_qpu_budget() -> tuple[bool, int]:
    """Returns (budget_ok, used_ms). Must be called with _qpu_lock held."""
    used = _read_qpu_time_ms()
    return used < DWAVE_BUDGET_MS, used


# ---------------------------------------------------------------------------
# Skip-if-exists check
# ---------------------------------------------------------------------------


def result_exists(exp: dict) -> bool:
    """
    Matches the file that helpers.save_results() would write:
        results/{size}/{sampler}/{method}/result_{model}_h{h}_rbm{rbm}
        _nh{nh}_lr{lr}_reg{reg}_ns{ns}_seed{seed}_iter{iter}_cem0.json
    """
    path = (
        Path(OUTPUT_DIR)
        / str(exp["size"])
        / exp["sampler"]
        / exp["method"]
        / (
            f"result_{exp['model']}"
            f"_h{exp['h']}"
            f"_rbm{exp['rbm']}"
            f"_nh{exp['n_hidden']}"
            f"_lr{exp['lr']}"
            f"_reg{_REG}"
            f"_ns{_NS}"
            f"_seed{exp['seed']}"
            f"_iter{ITERATIONS}"
            f"_cem0"
            f".json"
        )
    )
    return path.exists()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def run_experiment(exp: dict, idx: int, total: int, dry_run: bool) -> bool:
    global _done, _failed

    label = (
        f"[{idx:>4}/{total}] "
        f"{exp['model']} N={exp['size']:>4} nh={exp['n_hidden']:>4} "
        f"h={exp['h']} lr={exp['lr']} seed={exp['seed']} "
        f"[{exp['sampler']}/{exp['method']}]"
    )

    if result_exists(exp):
        log(f"  [skip] {label}")
        with _count_lock:
            _done += 1
        return True

    # QPU budget gate — serialised across all threads
    if _is_dwave(exp["method"]):
        with _qpu_lock:
            ok, used = _check_qpu_budget()
            if not ok:
                log(
                    f"  [QPU BUDGET EXHAUSTED] {used / 60000:.2f} min used — skipping {label}"
                )
                with _count_lock:
                    _failed += 1
                return False
        qpu_info = f"  QPU used={used / 60000:.2f}min"
    else:
        qpu_info = ""

    log(f"  {label}{qpu_info}")

    cmd = [
        "python3",
        SCRIPT,
        "--model",
        exp["model"],
        "--size",
        str(exp["size"]),
        "--n-hidden",
        str(exp["n_hidden"]),
        "--h",
        str(exp["h"]),
        "--lr",
        str(exp["lr"]),
        "--sampler",
        exp["sampler"],
        "--method",
        exp["method"],
        "--rbm",
        exp["rbm"],
        "--iterations",
        str(ITERATIONS),
        "--seed",
        str(exp["seed"]),
        "--output-dir",
        OUTPUT_DIR,
    ]

    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        with _count_lock:
            _done += 1
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            log(f"  Retry {attempt}/{MAX_RETRIES} — {label}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    parser = argparse.ArgumentParser(
        description="D-Wave vs classical scaling benchmark"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4). D-Wave jobs are further serialised "
        "by the QPU budget lock, so extra workers mainly speed up classical runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing.",
    )
    args = parser.parse_args()

    _setup_logging()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    sizes_1d, l_values_2d = load_probe_sizes(PROBE_CSV)
    experiments = build_experiments(sizes_1d, l_values_2d)
    total = len(experiments)

    n_dwave = sum(1 for e in experiments if _is_dwave(e["method"]))
    n_classical = total - n_dwave
    used_ms_start = _read_qpu_time_ms()

    log("=" * 70)
    log(f"D-Wave scaling benchmark  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Workers                   : {args.workers}")
    log(
        f"Total experiments         : {total}  ({n_dwave} QPU + {n_classical} classical)"
    )
    log(f"1D sizes (from probe)     : {sizes_1d}")
    log(f"2D L-values (from probe)  : {l_values_2d}")
    log(f"H values                  : {H_VALUES}")
    log(f"Seeds                     : {SEEDS}")
    log(f"Iterations                : {ITERATIONS}")
    log(f"α (n_hidden/n_visible)    : 1")
    log(f"RBM topology              : DWaveTopologyRBM (zephyr) — same for all")
    for sampler, method, rbm in COMBOS:
        log(f"  Combo                   : {sampler}/{method}  rbm={rbm}")
    log(
        f"QPU budget                : {DWAVE_BUDGET_MS / 60000:.0f} min  "
        f"(used so far: {used_ms_start / 60000:.2f} min)"
    )
    if args.dry_run:
        log("DRY RUN — no experiments will be executed")
    log("=" * 70)

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
    log("=" * 70)
    log(f"Benchmark finished        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Completed                 : {_done} / {total}")
    log(f"Failed / skipped          : {_failed}")
    log(f"QPU time this run         : {(used_ms_end - used_ms_start) / 60000:.2f} min")
    log(f"QPU time total            : {used_ms_end / 60000:.2f} min")
    log("=" * 70)


if __name__ == "__main__":
    main()
