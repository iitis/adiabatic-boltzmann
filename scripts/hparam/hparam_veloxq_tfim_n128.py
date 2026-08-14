#!/usr/bin/env python3
"""
hparam_veloxq_tfim_n128.py — VeloxQ SA hyperparameter search for TFIM N=128.

Companion to hparam_veloxq_tfim.py (which covers N=32/64). N=128 has no
existing Optuna search — this runs one from scratch. Same search space and
budget as the N=32/64 study, EXCEPT --n-samples-max is raised 4000 -> 6000:
a 2x larger system plausibly needs more samples for a stable SR gradient
estimate, and this is a cheap, CLI-exposed, isolated widening (doesn't touch
hparam_optuna.py's shared search-space code, so it can't affect other
studies). If N=128's best trials also cluster at a search-space boundary
(same signal that flagged T_initial's floor as too tight for N=32/64 — see
hparam_optuna.py's widened T_initial range), that's a sign to widen further;
check with `optuna.importance`/percentile position of the best trial before
assuming this space is wide enough.

Requires Julia + VeloxQstandard installed and reachable from --julia-project
(see scripts/fpga/julia_local/Project.toml). Everything below is a default;
pass any hparam_optuna.py flag to override it — later flags win.

Uses a fixed --study-name (not timestamped), so if it crashes, hits Ctrl-C,
or the Julia server dies partway through: just rerun the exact same command.
Resumes from its existing study.db and only runs however many trials are
still needed to reach --n-trials completed.

Usage:
    python scripts/hparam/hparam_veloxq_tfim_n128.py
    python scripts/hparam/hparam_veloxq_tfim_n128.py --n-trials 100  # longer search
    python scripts/hparam/hparam_veloxq_tfim_n128.py --dry-run       # sanity-check the grid first
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import hparam_optuna

_DEFAULT_ARGV = [
    "--hamiltonian", "tfim_1d",
    "--study-name", "veloxq_tfim",
    "--N", "128",
    "--J2", "0.5",
    "--sampling-methods", "velox_sa",
    "--ansatz-types", "rbm",
    "--n-trials", "60",
    "--iterations", "150",
    "--n-samples-max", "6000",
    "--julia-project", str(Path(__file__).parent.parent / "fpga" / "julia_local"),
    "--server-timeout", "600",
]


def main():
    hparam_optuna.main(_DEFAULT_ARGV + sys.argv[1:])


if __name__ == "__main__":
    main()
