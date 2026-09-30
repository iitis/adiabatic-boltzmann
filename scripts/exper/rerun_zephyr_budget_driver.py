"""Usage (from repo root): python scripts/exper/rerun_zephyr_budget_driver.py [sizes...]

Budget-guarded batch driver around scripts/exper/rerun_cem_fixed_headline.py.

Runs Zephyr cem1 reruns one at a time. Before EVERY run it re-reads the root
time.json (authoritative QPU counter). Stops the batch when the batch's device
time reaches BATCH_CAP_S, and refuses to start any run that could push total
usage past HARD_CAP_MIN. Any failure to read time.json aborts (no fallback).
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TIME_JSON = REPO / "time.json"
BASELINE_MS = 2152152.199360213
HARD_CAP_MIN = 57.0        # 60 min allowance minus safety margin
BATCH_CAP_S = 3.5 * 60     # per-invocation device-time cap
RESERVE_S = 15.0           # worst-case device time of one run (observed max ~6.7s)

assert Path.cwd().resolve() == REPO, f"must run from repo root, cwd={Path.cwd()}"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_cem_fixed_headline as R  # noqa: E402  (sets x64, imports src/)


def used_s() -> float:
    d = json.loads(TIME_JSON.read_text())  # raises on any read/parse problem
    return (float(d["time_ms"]) - BASELINE_MS) / 1000.0


def main(order):
    start_used = used_s()
    print(f"[budget] start: {start_used/60:.2f} min used, {60-start_used/60:.2f} min left", flush=True)
    samplers = {}
    for N, seed in order:
        out = R.expected_output_path(N, "zephyr", seed)
        if out.exists():
            continue
        u = used_s()
        batch = u - start_used
        if batch + RESERVE_S > BATCH_CAP_S:
            print(f"[budget] batch cap reached ({batch:.1f}s this batch); stopping", flush=True)
            break
        if (u + RESERVE_S) / 60 > HARD_CAP_MIN:
            print(f"[budget] HARD CAP: {u/60:.2f} min used; refusing further runs", flush=True)
            break
        if N not in samplers:
            s = R.DimodSampler(method="zephyr")
            assert s.time_path.resolve() == TIME_JSON, s.time_path.resolve()
            samplers[N] = s
        t0 = time.time()
        try:
            R.run_one(N, "zephyr", seed, samplers[N])
        except Exception:
            print(f"[error] N={N} seed={seed} failed, skipping:", flush=True)
            traceback.print_exc()
            continue
        u2 = used_s()
        print(f"[budget] N={N} seed={seed}: qpu {u2-u:.2f}s, wall {time.time()-t0:.0f}s, "
              f"total {u2/60:.2f} min used ({60-u2/60:.2f} left)", flush=True)
    end = used_s()
    print(f"[budget] batch done: {end-start_used:.1f}s device; total {end/60:.2f} min used", flush=True)


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] or [8, 16, 32, 64]
    main([(N, s) for N in sizes for s in range(20)])
