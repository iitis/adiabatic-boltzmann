#!/usr/bin/env python3
"""
Measure representative QPU access times for 1D TFIM D-Wave experiments.

Scans jax_results/tfim_1d/ for result files produced by pegasus/zephyr,
runs one QPU sample call per (solver, size) pair using the same RBM
configuration that was used in the actual experiments, and writes the
measured QPU access times to scripts/dwave_sampling_times.json.

That JSON is the authoritative source for D-Wave per-iteration timing used
by scripts/viz/aggregate_results.py.

Run from the repo root:
    python scripts/measure_dwave_times.py [--results-dir jax_results/tfim_1d]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import DWaveTopologyRBM, FullyConnectedRBM  # noqa: E402
from sampler import DimodSampler  # noqa: E402

DWAVE_METHODS = {"pegasus", "zephyr"}
OUT_PATH = Path(__file__).resolve().parent / "dwave_sampling_times.json"


def collect_configs(results_dir: Path) -> dict:
    """
    Return dict mapping (solver, size) -> representative config.

    Scans every .json under results_dir that has a D-Wave sampling_method.
    For each (solver, size) pair, prefer h=0.5 and rbm=full to keep the
    measurement simple (full RBM → minorminer embedding, same as most runs).
    """
    seen = defaultdict(list)
    for path in sorted(results_dir.rglob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue
        cfg = data.get("config", {})
        if cfg.get("sampling_method") not in DWAVE_METHODS:
            continue
        seen[(cfg["sampling_method"], int(cfg["size"]))].append(cfg)

    representative = {}
    for (solver, size), cfgs in sorted(seen.items()):
        preferred = [c for c in cfgs if c.get("h") == 0.5] or cfgs
        full_rbm = [c for c in preferred if c.get("rbm") == "full"]
        representative[(solver, size)] = (full_rbm or preferred)[0]
    return representative


def build_rbm(cfg: dict, key: jax.Array):
    n_visible = int(cfg["size"])
    n_hidden = int(cfg.get("n_hidden", n_visible))
    rbm_type = cfg.get("rbm", "full")
    if rbm_type == "full":
        return FullyConnectedRBM(n_visible, n_hidden, key)
    return DWaveTopologyRBM(n_visible, n_hidden, key, solver=rbm_type)


def measure_one(solver: str, size: int, cfg: dict) -> float:
    """Run one QPU sample call and return access time in seconds."""
    rbm = build_rbm(cfg, jax.random.PRNGKey(0))
    n_samples = int(cfg.get("n_samples", 1000))
    print(
        f"  [{solver}  N={size:4d}]  rbm={cfg.get('rbm','full')}  "
        f"n_hidden={rbm.n_hidden}  n_samples={n_samples} ...",
        flush=True,
    )
    sampler = DimodSampler(method=solver)
    sampler.sample(rbm, n_samples, config={})

    time_s = sampler.last_sampling_time_s
    if time_s is None:
        raise RuntimeError(
            f"last_sampling_time_s not set after QPU call for {solver} N={size}."
        )
    print(f"    → {time_s * 1000:.1f} ms")
    return time_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=REPO_ROOT / "jax_results" / "tfim_1d",
        type=Path,
    )
    args = parser.parse_args()

    configs = collect_configs(args.results_dir)
    if not configs:
        raise RuntimeError(f"No D-Wave result files found under {args.results_dir}")

    print(f"Found {len(configs)} (solver, size) pairs:")
    for (solver, size), cfg in sorted(configs.items()):
        print(f"  {solver:10s}  N={size:4d}  rbm={cfg.get('rbm', 'full')}")
    print()

    times: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for (solver, size), cfg in sorted(configs.items()):
        try:
            t = measure_one(solver, size, cfg)
            times.setdefault(solver, {})[str(size)] = t
        except Exception as e:
            msg = f"{solver} N={size}: {e}"
            print(f"  ERROR {msg}")
            failures.append(msg)

    if times:
        OUT_PATH.write_text(json.dumps(times, indent=2) + "\n")
        print(f"\nSaved → {OUT_PATH}")

    if failures:
        print(f"\n{len(failures)} measurement(s) failed:")
        for msg in failures:
            print(f"  {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
