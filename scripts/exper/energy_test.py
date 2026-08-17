#!/usr/bin/env python3
"""
energy_test.py — measure solver-only GPU energy (src/energy.py's
GPUEnergyMeter, scoped to sampler.sample() calls) for the classical
samplers at the same system sizes as Figure 10c's classical-sampler panels
(scripts/viz/paper_figures.py:fig10c_tte_vs_n_self_convergence, `_sizes`).

Reuses mcmc_matched_sweep.run_one() unchanged (same Trainer/ClassicalSampler
construction as scripts/main.py --sampler custom), just pointed at
results/energy_test/ so it can never collide with or overwrite the final,
already-published result files under results/.

QPU (pegasus/zephyr) and FPGA are not included: their sample() call doesn't
run on the GPU, so GPUEnergyMeter has nothing to measure there (D-Wave
QPU cost is tracked separately as access-time budget in time.json, not
energy — see CLAUDE.md).

Usage:
    python scripts/exper/energy_test.py
    python scripts/exper/energy_test.py --methods metropolis gibbs --sizes 8 16
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
jax.config.update("jax_enable_x64", True)

from mcmc_matched_sweep import run_one  # noqa: E402

# Same sizes as fig10c_tte_vs_n_self_convergence's classical-sampler `_sizes`.
FIG10C_SIZES = [8, 12, 16, 24, 32, 64]
DEFAULT_METHODS = ["metropolis", "gibbs", "simulated_annealing", "lsb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=FIG10C_SIZES)
    p.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                   choices=["metropolis", "gibbs", "lsb", "simulated_annealing"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--cem", action="store_true", default=False)
    p.add_argument("--output-dir", type=str, default=str(_ROOT / "results" / "energy_test"))
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def result_path(args, method, size):
    return (
        Path(args.output_dir) / "tfim_1d" / str(size) / "custom" / method /
        f"result_1d_h{args.h}_rbmfull_nh{size}_lr{args.lr}_reg{args.reg}"
        f"_ns{args.n_samples}_seed{args.seed}_iter{args.iterations}_cem{int(args.cem)}_sigma1.0.json.gz"
    )


def main():
    args = parse_args()
    # run_one() (from mcmc_matched_sweep) also reads these; fixed here since
    # this script's job is the energy number, not tuning sampler internals.
    args.gibbs_sweeps = 10
    args.sa_sweeps = None
    args.n_warmup = None
    args.variant = ""

    rows = []
    for method in args.methods:
        for size in args.sizes:
            print(f"=== {method} N={size} ===")
            run_one(size, method, args.seed, args)
            with gzip.open(result_path(args, method, size)) as f:
                wh = json.load(f).get("gpu_energy_wh")
            rows.append((method, size, wh))

    print(f"\n{'method':<20}{'N':>5}  gpu_energy_wh")
    for method, size, wh in rows:
        print(f"{method:<20}{size:>5}  {wh}")


if __name__ == "__main__":
    main()
