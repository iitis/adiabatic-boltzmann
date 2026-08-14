#!/usr/bin/env python3
"""
hparam_veloxq_tfim.py — VeloxQ SA hyperparameter search for TFIM N=32/64.

Zero-argument preset for the investigation into why the generalization sweep
(scripts/fpga/run_fpga_best.py --generalize) breaks down at N=32 and N=64:
those runs held learning_rate/regularization/n_samples fixed at the N=16/24
Optuna optimum while n_hidden grew, and VeloxQ SA degraded in lockstep with
the FPGA backend under those fixed hyperparameters — evidence this is a
hyperparameter-transfer problem, not a hardware/sampling artifact. This
script re-runs a proper Optuna search per size instead of extrapolating.

Requires Julia + VeloxQstandard installed and reachable from --julia-project
(see scripts/fpga/julia_local/Project.toml). Everything below is a default;
pass any hparam_optuna.py flag to override it — later flags win.

Uses a fixed --study-name (not timestamped), so if it crashes, hits Ctrl-C,
or the Julia server dies partway through: just rerun the exact same command.
Each combo resumes from its existing study.db and only runs however many
trials are still needed to reach --n-trials completed — combos that already
have enough are skipped outright. No flags to remember, no timestamps to copy.

Usage:
    python scripts/hparam/hparam_veloxq_tfim.py
    python scripts/hparam/hparam_veloxq_tfim.py --N 64          # just one size
    python scripts/hparam/hparam_veloxq_tfim.py --n-trials 100  # longer search
    python scripts/hparam/hparam_veloxq_tfim.py --dry-run       # sanity-check the grid first
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import hparam_optuna

_DEFAULT_ARGV = [
    "--hamiltonian", "tfim_1d",
    "--study-name", "veloxq_tfim",
    "--N", "32", "64",
    "--J2", "0.5",
    "--sampling-methods", "velox_sa",
    "--ansatz-types", "rbm",
    "--n-trials", "60",
    "--iterations", "150",
    "--n-samples-max", "4000",
    "--julia-project", str(Path(__file__).parent.parent / "fpga" / "julia_local"),
    "--server-timeout", "600",
]


def main():
    hparam_optuna.main(_DEFAULT_ARGV + sys.argv[1:])


if __name__ == "__main__":
    main()
