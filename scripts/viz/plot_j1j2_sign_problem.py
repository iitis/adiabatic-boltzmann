#!/usr/bin/env python3
"""
Visualise the sign problem in the J1-J2 Heisenberg chain for real-valued RBMs.

For each J2/J1 ratio we exact-diagonalise the chain (small N, no sampler needed),
apply the Marshall sign gauge phi(v) = s(v)*psi(v), and measure two quantities:

  1. Sign impurity — probability weight on negative components of phi.
     A real RBM represents A(v) >= 0, so in the Marshall gauge it represents
     phi_RBM(v) = s(v)*A(v).  If any phi(v) < 0 after optimally choosing the
     sign of psi, the ground state cannot be represented by any real positive
     function and sign impurity > 0 is a FUNDAMENTAL limitation, not a training
     issue.

  2. Energy penalty / site — variational energy of |phi|/||phi|| minus the exact
     ground-state energy, per site.  |phi| is one specific positive approximation;
     its energy is an upper bound on the irreducible representational error.

Marshall sign: s(v) = prod_{i odd} sigma_i  (product of sigma_z on sublattice B).

--- Why J2/J1 = 0.5 is the threshold ---

At J2=0 the Marshall theorem guarantees a positive ground state.  For 0 < J2/J1 < 0.5
the chain evolves continuously (through a KT phase transition at J2/J1=0.241 where
the energy gap opens, but without changing sign structure) and the ground state
remains positive in the Marshall basis.

The Majumdar-Ghosh (MG) point J2/J1=0.5 is a degenerate ground state of two dimer
coverings |D1> and |D2> (period-2 singlet products), both of which are all-positive
in the Marshall basis.  At J2/J1=0.5+eps the degeneracy is broken and the new ground
state is the ANTISYMMETRIC combination (|D1>-|D2>)/sqrt(2) in the k=pi sector, which
has equal positive and negative Marshall weight (sign_imp = 0.5).  This is a genuine
level crossing, not a crossover: sign_imp jumps discontinuously from 0 to 0.5.

--- Oscillations above J2/J1 = 0.5 ---

For J2/J1 > 0.5 the chain develops incommensurate spiral correlations, but finite
systems can only host discrete momenta.  As J2/J1 increases, further level crossings
occur between states in different momentum sectors, each with its own sign structure.
The oscillations in sign_imp are these finite-size level crossings.  They are real
physics but not representative of the thermodynamic limit, where sign_imp converges
to a smooth monotone increase from 0 at J2/J1=0.5.  Using N divisible by 4 (so the
MG dimer state tiles perfectly) gives the cleanest behaviour at the onset.

Note: the dimerization-gap transition at J2/J1=0.241 (KT point) has NO feature in
either panel — sign_imp=0 throughout [0, 0.5) — so it is not marked here.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from plot_style import setup_style

# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def _coupling_matrix(N: int, bond_len: int) -> sp.csr_matrix:
    """Sparse J=1 Heisenberg coupling matrix for bonds of length bond_len (PBC)."""
    dim = 2 ** N
    k = np.arange(dim)
    rows, cols, vals = [], [], []
    diag = np.zeros(dim)

    for i in range(N):
        j = (i + bond_len) % N
        si = 1 - 2 * ((k >> i) & 1)   # sigma_i = ±1 for every basis state
        sj = 1 - 2 * ((k >> j) & 1)

        diag += 0.25 * si * sj         # Sz_i Sz_j contribution

        # S+_i S-_j + h.c. acts on opposite-spin pairs
        opp = si != sj
        k_src = k[opp]
        k_dst = k_src ^ (1 << i) ^ (1 << j)
        rows.append(k_dst)
        cols.append(k_src)
        vals.append(np.full(opp.sum(), 0.5))

    rows = np.concatenate(rows + [k])
    cols = np.concatenate(cols + [k])
    vals = np.concatenate(vals + [diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def build_h1_h2(N: int):
    """Return (H_NN, H_NNN) so that H(J1,J2) = J1*H_NN + J2*H_NNN."""
    return _coupling_matrix(N, 1), _coupling_matrix(N, 2)


# ---------------------------------------------------------------------------
# Marshall sign
# ---------------------------------------------------------------------------

def marshall_signs(N: int) -> np.ndarray:
    """s(v) = product of sigma_i on odd-indexed sites = ±1 per basis state."""
    dim = 2 ** N
    k = np.arange(dim)
    s = np.ones(dim)
    for i in range(1, N, 2):
        s *= 1 - 2 * ((k >> i) & 1)
    return s


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _sign_impurity(phi: np.ndarray) -> float:
    """Probability weight on negative components; already accounts for global sign."""
    si = float(phi[phi < 0] @ phi[phi < 0])
    return min(si, 1.0 - si)   # flip global sign if majority is negative


def _min_sign_impurity_2d(phi1: np.ndarray, phi2: np.ndarray) -> tuple:
    """Sweep the 2D degenerate subspace for the combination with minimum sign impurity."""
    best_si, best_phi = 1.0, phi1
    for theta in np.linspace(0, np.pi, 600):
        phi = np.cos(theta) * phi1 + np.sin(theta) * phi2
        phi /= np.linalg.norm(phi)
        si = _sign_impurity(phi)
        if si < best_si:
            best_si, best_phi = si, phi.copy()
    return best_si, best_phi


def sweep(N: int, ratios: np.ndarray, J1: float = 1.0):
    """Return (sign_impurities, energy_penalties_per_site, exact_energies_per_site)."""
    H_nn, H_nnn = build_h1_h2(N)
    s = marshall_signs(N)

    sign_imps  = np.empty(len(ratios))
    e_penalties = np.empty(len(ratios))
    e_exact_ps  = np.empty(len(ratios))

    for idx, r in enumerate(ratios):
        H = J1 * H_nn + (r * J1) * H_nnn

        # Ground state via sparse Lanczos; k=2 needed for degenerate points
        evals, evecs = eigsh(H, k=2, which="SA", tol=1e-12, maxiter=100_000)
        E_exact = float(evals[0])
        e_exact_ps[idx] = E_exact / N

        phi0 = s * evecs[:, 0]
        phi1 = s * evecs[:, 1]

        # Near degeneracy (e.g. Majumdar-Ghosh point) eigsh can return any linear
        # combination of the degenerate ground state.  Rotate to find the combination
        # with minimum sign impurity — this is the representability floor.
        degenerate = abs(evals[1] - evals[0]) < 1e-6 * abs(evals[0]) + 1e-10
        if degenerate:
            sign_imp, phi = _min_sign_impurity_2d(phi0, phi1)
        else:
            phi = phi0
            phi /= np.linalg.norm(phi)
            sign_imp = _sign_impurity(phi)
            # Ensure majority-positive convention for the energy calculation
            if float(phi[phi < 0] @ phi[phi < 0]) > 0.5:
                phi = -phi

        sign_imps[idx] = sign_imp

        # Energy of the positive projection |phi|/||phi||
        # E_pos = <phi_pos | H_M | phi_pos> = <s*phi_pos | H | s*phi_pos>
        phi_pos = np.abs(phi)
        E_pos = float((s * phi_pos) @ (H @ (s * phi_pos)))
        e_penalties[idx] = (E_pos - E_exact) / N

    return sign_imps, e_penalties, e_exact_ps


# ---------------------------------------------------------------------------
# Plot + JSON export
# ---------------------------------------------------------------------------

def main():
    setup_style()

    ratios  = np.linspace(0.0, 1.0, 120)
    systems = [8, 12, 16]
    colors  = ["#2563eb", "#16a34a", "#dc2626"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    results = {}
    for N, color, marker in zip(systems, colors, markers):
        print(f"  diagonalising N={N} ...", flush=True)
        sign_imps, e_pen, e_exact = sweep(N, ratios)

        results[f"N{N}"] = {
            "J2_J1":       ratios.tolist(),
            "sign_imp":    sign_imps.tolist(),
            "e_penalty_per_site": e_pen.tolist(),
            "e_exact_per_site":   e_exact.tolist(),
        }

        kw = dict(color=color, lw=1.6, label=f"$N={N}$",
                  markevery=15, markersize=4, marker=marker,
                  markerfacecolor="white", markeredgewidth=1.2)
        axes[0].plot(ratios, sign_imps, **kw)
        axes[1].plot(ratios, e_pen,     **kw)

    for ax in axes:
        ax.axvline(0.5, color="#222", ls=":", lw=1.2, zorder=0)
        ax.set_xlim(0, 1.0)

    # Annotation: MG point / sign problem onset
    axes[0].annotate(
        "Majumdar--Ghosh\n" + r"($J_2/J_1=0.5$)",
        xy=(0.500, 0.28), xytext=(0.58, 0.38),
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.9),
        fontsize=9, color="#333",
    )

    axes[0].set_ylabel(
        r"Sign impurity $\displaystyle\sum_{\phi<0}\phi(v)^2$")
    axes[0].set_ylim(-0.01, 0.55)
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].set_ylabel(
        r"Energy penalty / site" + "\n"
        r"$(E_{\mathrm{pos}}-E_{\mathrm{exact}})/N$")
    axes[1].set_xlabel(r"$J_2/J_1$")
    axes[1].set_ylim(-0.005, None)
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "j1j2")
    os.makedirs(out, exist_ok=True)

    for ext in ("pdf", "png"):
        path = os.path.join(out, f"j1j2_sign_problem.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")

    json_path = os.path.join(out, "j1j2_sign_problem.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved {json_path}")

    plt.show()


if __name__ == "__main__":
    main()
