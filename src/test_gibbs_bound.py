#!/usr/bin/env python3
"""
Variational bound check for the Gibbs sampler with n_sweeps=10.

Tests 1D TFIM with h=1 and h=2 on small systems using the Gibbs sampler.
Reports for each case:
  - Final relative energy error after 300 iterations
  - Whether the energy ever dropped below the exact ground state

Usage:
    cd src && python test_gibbs_bound.py
"""

import contextlib
import io
import sys
import numpy as np

from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer


CASES = [
    dict(size=8,  h=1.0),
    dict(size=8,  h=2.0),
    dict(size=12, h=1.0),
    dict(size=12, h=2.0),
]

N_ITER    = 300
N_SAMPLES = 1000
N_SWEEPS  = 10
LR        = 0.01
REG       = 1e-5
SEED      = 42


def run_case(size: int, h: float) -> dict:
    np.random.seed(SEED)

    ising   = TransverseFieldIsing1D(size, h)
    E_exact = ising.exact_ground_energy()

    rbm     = FullyConnectedRBM(n_visible=size, n_hidden=size)
    sampler = ClassicalSampler(method="gibbs")
    config  = {
        "learning_rate":  LR,
        "n_iterations":   N_ITER,
        "n_samples":      N_SAMPLES,
        "regularization": REG,
        "n_sweeps":       N_SWEEPS,
    }

    trainer = Trainer(rbm, ising, sampler, config, args=None)

    with contextlib.redirect_stdout(io.StringIO()):
        history = trainer.train()

    energies     = history["energy"]
    final_energy = energies[-1]
    min_energy   = min(energies)
    violations   = [e for e in energies if e < E_exact]

    return {
        "E_exact":      E_exact,
        "final_energy": final_energy,
        "rel_error":    abs(final_energy - E_exact) / abs(E_exact),
        "min_energy":   min_energy,
        "n_violations": len(violations),
        "worst":        min_energy - E_exact,  # negative = below ground state
    }


def main() -> int:
    print(f"Gibbs sampler variational bound test  (n_sweeps={N_SWEEPS}, iter={N_ITER})\n")

    failed = False
    for c in CASES:
        label = f"1D  N={c['size']}  h={c['h']}"
        print(f"  running {label} ...", end="", flush=True)
        r = run_case(**c)
        print("\r", end="")

        ok = r["n_violations"] == 0
        if not ok:
            failed = True

        tag = "[PASS]" if ok else "[FAIL]"
        print(
            f"{tag} {label:<18}"
            f"  E_exact={r['E_exact']:9.5f}"
            f"  E_final={r['final_energy']:9.5f}"
            f"  err={r['rel_error']*100:5.2f}%"
            f"  violations={r['n_violations']:3d}"
            f"  worst={r['worst']:+.6f}"
        )

    print()
    if not failed:
        print("All cases passed — energy never dropped below the ground state.")
        return 0
    else:
        print("FAILED — energy violated the variational bound in one or more cases.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
