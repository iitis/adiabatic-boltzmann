#!/usr/bin/env python3
"""
Direct proof that the RBM energy barrier at J2/J1 > 0.5 is caused by the
sign problem, not insufficient expressivity or training.

For each of 4 J2/J1 values (0.3, 0.5, 0.7, 0.9) and two system sizes
(N=8, N=12), we compute by exact enumeration:

  1. Sign impurity ε = Σ_{φ<0} φ(v)²  (0 iff ground state is Marshall-positive)
  2. Theoretical energy floor = (E_{|φ|} − E_exact) / |E_exact|
        where |φ|/‖|φ|‖ is the positive projection of the true GS in
        the Marshall basis.  No real positive ansatz can do better.

We then load the best RBM run from results/ and compare its error to the floor.
If RBM_error ≥ floor everywhere, the barrier is entirely explained by the sign
problem.

Usage:
    python scripts/viz/plot_j1j2_sign_barrier_proof.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plot_style import setup_style

_ROOT    = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results" / "heisenberg_j1j2_1d"
_OUT     = _ROOT / "plots" / "j1j2"

J2_VALS = [0.3, 0.5, 0.7, 0.9]
N_VALS  = [8, 12]


# ---------------------------------------------------------------------------
# Exact diagonalisation helpers
# ---------------------------------------------------------------------------

def _coupling_matrix(N: int, bond_len: int) -> sp.csr_matrix:
    dim = 2 ** N
    k   = np.arange(dim)
    rows, cols, vals = [], [], []
    diag = np.zeros(dim)
    for i in range(N):
        j  = (i + bond_len) % N
        si = 1 - 2 * ((k >> i) & 1)
        sj = 1 - 2 * ((k >> j) & 1)
        diag += 0.25 * si * sj
        opp   = si != sj
        k_src = k[opp]
        k_dst = k_src ^ (1 << i) ^ (1 << j)
        rows.append(k_dst); cols.append(k_src)
        vals.append(np.full(opp.sum(), 0.5))
    rows = np.concatenate(rows + [k])
    cols = np.concatenate(cols + [k])
    vals = np.concatenate(vals + [diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def _marshall_signs(N: int) -> np.ndarray:
    k = np.arange(2 ** N)
    s = np.ones(2 ** N)
    for i in range(1, N, 2):
        s *= 1 - 2 * ((k >> i) & 1)
    return s


def _sign_impurity(phi: np.ndarray) -> float:
    si = float(phi[phi < 0] @ phi[phi < 0])
    return min(si, 1.0 - si)


def _best_phi(H_nn, H_nnn, marshall, J2: float, J1: float = 1.0) -> tuple:
    """Return (E_exact, phi_best, sign_imp) minimising sign impurity."""
    H = J1 * H_nn + J2 * H_nnn
    evals, evecs = eigsh(H, k=2, which="SA", tol=1e-12, maxiter=200_000)
    E_exact = float(evals[0])

    phi0 = marshall * evecs[:, 0]
    phi1 = marshall * evecs[:, 1]

    degenerate = abs(evals[1] - evals[0]) < 1e-6 * abs(evals[0]) + 1e-10
    if degenerate:
        best_si, best_phi = 1.0, phi0 / np.linalg.norm(phi0)
        for theta in np.linspace(0, np.pi, 600):
            phi = np.cos(theta) * phi0 + np.sin(theta) * phi1
            phi /= np.linalg.norm(phi)
            si = _sign_impurity(phi)
            if si < best_si:
                best_si, best_phi = si, phi.copy()
        return E_exact, best_phi, best_si
    else:
        phi = phi0 / np.linalg.norm(phi0)
        if float(phi[phi < 0] @ phi[phi < 0]) > 0.5:
            phi = -phi
        return E_exact, phi, _sign_impurity(phi)


def compute_floor(N: int, J2: float) -> dict:
    """Return sign_imp and relative energy floor for real-positive ansatz."""
    H_nn, H_nnn = _coupling_matrix(N, 1), _coupling_matrix(N, 2)
    marshall     = _marshall_signs(N)
    H            = H_nn + J2 * H_nnn

    E_exact, phi, sign_imp = _best_phi(H_nn, H_nnn, marshall, J2)

    # Energy of positive projection |φ|/‖|φ|‖ — theoretical floor
    phi_pos  = np.abs(phi)
    phi_pos /= np.linalg.norm(phi_pos)
    psi_pos  = marshall * phi_pos          # back to original basis
    E_floor  = float(psi_pos @ (H @ psi_pos))

    rel_floor = (E_floor - E_exact) / abs(E_exact)
    return {"sign_imp": sign_imp, "rel_floor": rel_floor, "E_exact": E_exact}


# ---------------------------------------------------------------------------
# Load best RBM errors from results/
# ---------------------------------------------------------------------------

def load_best_rbm_errors() -> dict:
    """Return dict (N, J2_rounded) -> min relative error."""
    best = {}
    for f in _RESULTS.rglob("*.json.gz"):
        try:
            with gzip.open(f) as fp:
                d = json.load(fp)
            cfg   = d.get("config", {})
            N     = int(cfg.get("N") or cfg.get("size"))
            J2    = round(float(cfg.get("J2", -1)), 1)
            err   = d.get("error")
            if err is None or N not in N_VALS or J2 not in J2_VALS:
                continue
            key = (N, J2)
            if key not in best or float(err) < best[key]:
                best[key] = float(err)
        except Exception:
            pass
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style(fontsize=16, scale=1.0)

    print("Computing exact floors …")
    ed = {}
    for N in N_VALS:
        H_nn  = _coupling_matrix(N, 1)
        H_nnn = _coupling_matrix(N, 2)
        marshall = _marshall_signs(N)
        for J2 in J2_VALS:
            print(f"  N={N}, J2/J1={J2}", flush=True)
            ed[(N, J2)] = compute_floor(N, J2)

    rbm = load_best_rbm_errors()

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8),
                             gridspec_kw={"hspace": 0.12})

    x      = np.arange(len(J2_VALS))
    width  = 0.18
    colors = {"8": "#2563eb", "12": "#16a34a"}
    offsets = {"8": -0.20, "12": 0.20}

    # ── Panel 1: sign impurity ───────────────────────────────────────────────
    ax = axes[0]
    for N in N_VALS:
        si_vals = [ed[(N, j2)]["sign_imp"] for j2 in J2_VALS]
        ax.bar(x + offsets[str(N)], si_vals, width,
               color=colors[str(N)], alpha=0.85, label=f"$N={N}$",
               edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="#aaa", lw=0.7, ls="--")
    ax.set_ylabel(r"Sign impurity $\epsilon$")
    ax.set_ylim(-0.02, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.legend(loc="upper left", handlelength=1.4)
    ax.text(0.98, 0.95, r"$\epsilon = 0$: no sign problem",
            ha="right", va="top", transform=ax.transAxes,
            fontsize="small", color="#555")

    # ── Panel 2: floor vs RBM error ─────────────────────────────────────────
    ax = axes[1]
    for N in N_VALS:
        floor_vals = [ed[(N, j2)]["rel_floor"] for j2 in J2_VALS]
        rbm_vals   = [rbm.get((N, j2), np.nan)  for j2 in J2_VALS]

        # Theoretical floor — filled bar
        ax.bar(x + offsets[str(N)], floor_vals, width,
               color=colors[str(N)], alpha=0.55, label=f"$N={N}$ floor",
               edgecolor="white", linewidth=0.5)

        # Actual best RBM — marker on top
        for xi, (fl, rv) in enumerate(zip(floor_vals, rbm_vals)):
            if not np.isnan(rv):
                ax.plot(xi + offsets[str(N)], rv, marker="D",
                        color=colors[str(N)], markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=5,
                        label=f"$N={N}$ RBM best" if xi == 0 else None)

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)
    ax.set_ylabel(r"Relative energy error")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$J_2/J_1={j}$" for j in J2_VALS])

    # Legend: distinguish floor (bar) from RBM (marker)
    floor_patch = mpatches.Patch(color="#888", alpha=0.55, label="Sign-problem floor (exact)")
    rbm_marker  = plt.Line2D([0], [0], marker="D", color="#888",
                              markersize=5, linestyle="None",
                              label="Best RBM run")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[floor_patch, rbm_marker] + handles[:2],
              loc="lower right", handlelength=1.4, fontsize="small")

    _OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = _OUT / f"fig_sign_barrier_proof.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
