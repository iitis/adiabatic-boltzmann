#!/usr/bin/env python3
"""DMRG reference energies for the general-graph TFIMs of src/graph_tfim.py (e.g. the antiferromagnetic
triangular lattice on a torus or a cylinder), H = -sum_{i<j} J_ij sz_i sz_j - h sum_i sx_i, built from the same
coupling matrix as the VMC runs.

Implementation (TeNPy, no changes to the library):
- The lattice is laid out as a 1D MPS in the model's site order (x * L2 + y, i.e. column by column around the
  periodic width); every J_ij becomes a (possibly long-range) MPO coupling.
- Z2 symmetry: H commutes with P = prod_i sx_i. In the rotated basis (sz <-> sx), H' = -sum J_ij sx sx - h sum sz,
  and P = prod sz is a conserved charge (SpinHalfSite(conserve='parity')), so every tensor is block-diagonal in
  parity. Energies are basis independent.
- chi ladder: each chi is warm-started from the previous chi's state. Only for chi <= --multistart-chi are several
  starts compared (paramagnet and random basis states, in both parity sectors); the ground-state sector found there
  is kept for the larger chi.
- Recorded energy: <H> of the returned MPS (a variational upper bound), plus the truncation error of the last
  sweep, so E(chi -> inf) can be extrapolated in the analysis. For N <= 20 the exact (Lanczos) energy is recorded.

  python scripts/validation/dmrg_graph.py --model tricyl --size 6x6 --h 1.0 --chis 32,64,128,256,512,1024
Output: one line per (model, size, h, chi) appended to results/dmrg_baseline/graph.jsonl
(lines written by the earlier, non-symmetric version lack the "version" field).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # graph_tfim imports JAX; keep this CPU job off the GPU

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from graph_tfim import make

from tenpy.algorithms import dmrg
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

logging.getLogger("tenpy").setLevel(logging.WARNING)
OUT = REPO / "results" / "dmrg_baseline" / "graph.jsonl"
VERSION = "parity_v2"


class RotatedGraphModel(CouplingMPOModel):
    """H' = -sum_{i<j} J_ij Sx_i Sx_j - h sum_i Sz_i (Pauli matrices), parity of the number of down spins conserved."""

    def init_sites(self, p):
        return SpinHalfSite(conserve="parity")

    def init_lattice(self, p):
        return Chain(p["L"], self.init_sites(p), bc="open", bc_MPS="finite")

    def init_terms(self, p):
        J, h = p["J"], p["h"]
        self.add_onsite(-h, 0, "Sigmaz")
        for i, j in zip(*np.nonzero(np.triu(J))):
            self.add_coupling_term(-float(J[i, j]), int(i), int(j), "Sigmax", "Sigmax")


def basis_state(N, parity, rng=None):
    """'up'/'down' product state with the given parity of the number of 'down' spins (paramagnet if rng is None)."""
    s = np.zeros(N, int) if rng is None else rng.integers(0, 2, N)
    if s.sum() % 2 != parity:
        s[0] ^= 1
    return ["down" if b else "up" for b in s]


def run(M, psi, chi, sweeps):
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dict(
        trunc_params=dict(chi_max=chi, svd_min=1e-10), mixer=True, max_sweeps=sweeps, min_sweeps=3,
        max_E_err=1e-10, max_trunc_err=1.0))
    _, psi = eng.run()
    psi.canonical_form()
    E = float(np.real(M.H_MPO.expectation_value(psi)))
    trunc = eng.sweep_stats["max_trunc_err"]
    return E, psi, float(trunc[-1]) if len(trunc) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tricyl", choices=("tri", "tricyl", "chain"))
    ap.add_argument("--size", required=True)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--chis", default="32,64,128,256,512,1024")
    ap.add_argument("--multistart-chi", type=int, default=32)
    ap.add_argument("--sweeps", type=int, default=20)
    args = ap.parse_args()
    graph = make(args.model, args.size, args.h)
    N = graph.size
    M = RotatedGraphModel(dict(L=N, J=graph.J_np, h=graph.h))
    sites = M.lat.mps_sites()
    e_exact = None
    if N <= 20:
        import scipy.sparse.linalg as spla
        e_exact = float(spla.eigsh(graph.sparse_hamiltonian(), k=1, which="SA", tol=1e-12)[0][0])
    emf = graph.mean_field(n_starts=64)[0]
    rng = np.random.default_rng(0)
    psi_best = None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for chi in map(int, args.chis.split(",")):
        t0 = time.time()
        if chi <= args.multistart_chi:
            cands = [(f"para_p{p}", MPS.from_product_state(sites, basis_state(N, p), bc="finite")) for p in (0, 1)]
            cands += [(f"rand_p{p}", MPS.from_product_state(sites, basis_state(N, p, rng), bc="finite")) for p in (0, 1)]
            if psi_best is not None:
                cands.append(("warm", psi_best))
        else:
            cands = [("warm", psi_best)]
        results = {}
        for name, psi0 in cands:
            E, psi, trunc = run(M, psi0.copy(), chi, args.sweeps)
            results[name] = (E, psi, trunc)
        start = min(results, key=lambda k: results[k][0])
        E, psi_best, trunc = results[start]
        rec = dict(version=VERSION, model=args.model, size=args.size, N=N, h=args.h, chi=chi, E=E, E_per_spin=E / N,
                   start=start, start_energies={k: v[0] for k, v in results.items()}, max_trunc_err=trunc,
                   chi_reached=int(max(psi_best.chi)), t_dmrg=time.time() - t0,
                   threads=int(os.environ.get("OMP_NUM_THREADS", "0") or 0), E_mf=emf, E_exact=e_exact)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{args.model} {args.size} h={args.h:g} chi={chi}: E/N={E / N:.10f} "
              + (f"(exact {e_exact / N:.10f}, err/N {(E - e_exact) / N:.2e}) " if e_exact is not None else "")
              + f"trunc={trunc:.1e} start={start} t={rec['t_dmrg']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
