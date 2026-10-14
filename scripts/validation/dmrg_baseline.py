#!/usr/bin/env python3
"""DMRG (MPS) baseline for the 1D TFIM, H = -sum sz_i sz_{i+1} - h sum sx_i, periodic BC.

For every (N, h, chi) the ground state is found with two-site DMRG (TeNPy, open MPS carrying the
periodic bond in the MPO), best of three starts (all up, all +x, previous chi's state; small chi otherwise
lands in start-dependent local minima), its energy taken as <H> of the returned MPS. Then n_samples
i.i.d. z-basis configurations are drawn exactly from |psi|^2 by sequential conditional sampling of the
right-canonical MPS. Records energy error per spin against the exact energy (reference only), wall time
of DMRG and of sampling, and a sample check (<sz_i sz_i+1> over all bonds, samples vs MPS).
chi = 1 is not run: truncated two-site DMRG does not find the best product state (N=16, h=1: -19.988 vs
-20); use the analytic product state of src/mean_field.py instead.

  python scripts/validation/dmrg_baseline.py --N 16,32,64,128 --chis 2,4,8,16,32
Output: one line per (N, h, chi) appended to results/dmrg_baseline/tfim1d.jsonl
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ising import TransverseFieldIsing1D
from mean_field import mean_field_energy

from tenpy.algorithms import dmrg
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

logging.getLogger("tenpy").setLevel(logging.WARNING)
OUT = REPO / "results" / "dmrg_baseline" / "tfim1d.jsonl"
H_GRID = (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0)


class TFIM(CouplingMPOModel):
    def init_sites(self, p):
        return SpinHalfSite(conserve=None)  # local basis = sz eigenbasis, index 0 = up

    def init_terms(self, p):
        self.add_onsite(-p.get("h", 1.0), 0, "Sigmax")
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-1.0, u1, "Sigmaz", u2, "Sigmaz", dx)


def tfim(N, h):
    return TFIM(dict(L=N, h=h, bc_MPS="finite", bc_x="periodic", lattice=Chain))


def starts(M, warm):
    """Initial states: all up (z-ordered), all along +x (paramagnet), and the previous chi's ground state."""
    sites = M.lat.mps_sites()
    yield "up", MPS.from_product_state(sites, ["up"] * M.lat.N_sites, bc="finite")
    yield "x", MPS.from_product_state(sites, [np.array([1.0, 1.0]) / np.sqrt(2)] * M.lat.N_sites, bc="finite")
    if warm is not None:
        yield "warm", warm.copy()


def ground_state(M, chi, warm=None):
    """Best DMRG state over several starts (small chi gets stuck in start-dependent local minima).

    Returns (<H> of the returned, canonicalised MPS, MPS, name of the winning start, per-start energies);
    the engine's own energy is taken before the final truncation and can lie below <H> (even below the
    best product state at chi = 1)."""
    best, energies = None, {}
    for name, psi in starts(M, warm):
        eng = dmrg.TwoSiteDMRGEngine(psi, M, dict(
            trunc_params=dict(chi_max=chi, svd_min=1e-12), mixer=True, max_sweeps=20, min_sweeps=4,
            max_E_err=1e-12, max_trunc_err=1.0))  # truncation at small chi is intended
        _, psi = eng.run()
        psi.canonical_form()  # sample() relies on exact right-canonical B tensors
        E = float(np.real(M.H_MPO.expectation_value(psi)))
        energies[name] = E
        if best is None or E < best[0]:
            best = (E, psi, name)
    return (*best, energies)


def sample(psi, n, rng, z2_symmetrize=True):
    """Exact i.i.d. samples from |psi|^2 in the sz basis, returned as +-1 (shape n x N).

    In the ordered phase DMRG returns a symmetry-broken state (h=0.5: all samples have m > 0), while the finite-N
    ground state is the Z2-symmetric cat with 50/50 sectors. z2_symmetrize flips each sample globally with
    probability 1/2, i.e. samples (|psi(v)|^2 + |psi(-v)|^2)/2, the Born distribution of the cat (psi + P psi)/sqrt(2)
    up to the overlap <psi|P psi>, which is exponentially small in N in the ordered phase and where it is not,
    psi is already (nearly) symmetric. Z2-even statistics (zz, |m|) are unchanged by it."""
    out = np.empty((n, psi.L))
    env = np.ones((n, 1))
    for i in range(psi.L):
        B = psi.get_B(i, "B").to_ndarray()  # (vL, p, vR), right-canonical
        theta = np.einsum("sl,lpr->spr", env, B)
        p = np.einsum("spr,spr->sp", theta, theta.conj()).real
        p /= p.sum(1, keepdims=True)
        k = (rng.random(n) > p[:, 0]).astype(int)  # 0 = up, 1 = down
        out[:, i] = 1 - 2 * k
        env = theta[np.arange(n), k] / np.sqrt(p[np.arange(n), k])[:, None]
    if z2_symmetrize:
        out *= rng.choice([-1.0, 1.0], size=(n, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="16,32,64,128")
    ap.add_argument("--hs", default=",".join(map(str, H_GRID)))
    ap.add_argument("--chis", default="2,4,8,16,32")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for N in map(int, a.N.split(",")):
        for h in map(float, a.hs.split(",")):
            ising = TransverseFieldIsing1D(N, h)
            e0, emf = ising.exact_ground_energy(), mean_field_energy(ising)
            M, warm = tfim(N, h), None
            for chi in sorted(map(int, a.chis.split(","))):
                t = time.perf_counter()
                E, psi, start, energies = ground_state(M, chi, warm)
                t_dmrg = time.perf_counter() - t  # all starts: the cost of a reliable answer
                warm = psi
                t = time.perf_counter()
                V = sample(psi, a.samples, rng)
                t_samp = time.perf_counter() - t
                C = psi.correlation_function("Sigmaz", "Sigmaz")
                zz_mps = float(np.mean([C[i, (i + 1) % N] for i in range(N)]))  # all N bonds
                zz_each = np.mean(V * np.roll(V, -1, axis=1), axis=1)  # bond average per sample (i.i.d.)
                zz_smp, zz_se = float(zz_each.mean()), float(zz_each.std(ddof=1) / np.sqrt(len(zz_each)))
                rec = dict(N=N, h=h, chi=chi, E=E, E0=e0, E_mf=emf, err=(E - e0) / N,
                           mf_err=(emf - e0) / N, chi_used=int(max(psi.chi)), start=start,
                           err_by_start={k: (v - e0) / N for k, v in energies.items()}, t_dmrg=t_dmrg,
                           n_samples=a.samples, t_sample=t_samp, zz_mps=zz_mps, zz_samples=zz_smp,
                           zz_samples_se=zz_se,
                           mz_abs_samples=float(np.mean(np.abs(V.mean(1)))), m_samples=float(V.mean()))
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"N={N:4d} h={h:<5} chi={chi:3d} err/N={rec['err']:.2e} (MF {rec['mf_err']:.2e}) "
                      f"[{start}] dmrg {t_dmrg:6.2f}s sample {t_samp:5.2f}s zz {zz_mps:.4f}/{zz_smp:.4f}", flush=True)


if __name__ == "__main__":
    main()
