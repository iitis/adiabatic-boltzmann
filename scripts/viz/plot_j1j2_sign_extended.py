#!/usr/bin/env python3
"""
Extended sign-problem visualisations for the J1-J2 Heisenberg chain.

Figure 1 — Average sign and QMC cost  (j1j2_avg_sign.pdf/png)
  ⟨s⟩ = 1 − 2·(sign impurity) ranges from 1 (no problem) to 0 (maximally
  hard).  The QMC variance overhead scales as 1/⟨s⟩², which diverges at the
  Majumdar–Ghosh point and shows the *exponential* cost of the sign problem
  more starkly than the impurity weight alone.

Figure 2 — Sign-structure heatmap  (j1j2_sign_heatmap.pdf/png)
  For N=8 (256 states), restrict to the Sz=0 sector (70 states) which
  carries almost all ground-state weight.  Sort states once by their
  Marshall-gauged amplitude φ(v) = s(v)·Ψ(v) at J2/J1 = 0.51 (just above
  the MG point), so that |D1⟩ dimer states (positive φ) sit on the left and
  |D2⟩ dimer states (negative φ) on the right.

  The heatmap shows φ(v) normalised per row, so both sign and relative weight
  are visible:
    · All-blue  (J2/J1 < 0.5):  ground state is Marshall-positive.
    · Clean blue/red split at MG: antisymmetric dimer (|D1⟩−|D2⟩)/√2 has
      exactly half positive, half negative weight.
    · Evolving pattern above MG: sign structure changes with each finite-size
      level crossing.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_style import setup_style

# ---------------------------------------------------------------------------
# Hamiltonian + Marshall sign  (mirrors plot_j1j2_sign_problem.py)
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
    dim = 2 ** N
    k = np.arange(dim)
    s = np.ones(dim)
    for i in range(1, N, 2):
        s *= 1 - 2 * ((k >> i) & 1)
    return s


def _sign_impurity(phi: np.ndarray) -> float:
    si = float(phi[phi < 0] @ phi[phi < 0])
    return min(si, 1.0 - si)


def _ground_phi(H_nn, H_nnn, marshall, J1: float, ratio: float) -> np.ndarray:
    """Return majority-positive Marshall-gauged GS (degenerate-subspace sweep)."""
    H = J1 * H_nn + ratio * J1 * H_nnn
    evals, evecs = eigsh(H, k=2, which="SA", tol=1e-12, maxiter=200_000)
    phi0 = marshall * evecs[:, 0]
    phi1 = marshall * evecs[:, 1]

    degenerate = abs(evals[1] - evals[0]) < 1e-6 * abs(evals[0]) + 1e-10
    if degenerate:
        best_si, best_phi = 1.0, phi0
        for theta in np.linspace(0, np.pi, 600):
            phi = np.cos(theta) * phi0 + np.sin(theta) * phi1
            phi /= np.linalg.norm(phi)
            si = _sign_impurity(phi)
            if si < best_si:
                best_si, best_phi = si, phi.copy()
        return best_phi
    else:
        phi = phi0 / np.linalg.norm(phi0)
        if float(phi[phi < 0] @ phi[phi < 0]) > 0.5:
            phi = -phi
        return phi


# ---------------------------------------------------------------------------
# Figure 1: average sign ⟨s⟩ and QMC cost 1/⟨s⟩²
# ---------------------------------------------------------------------------

def fig_avg_sign(ratios: np.ndarray, systems=(8, 12, 16), J1=1.0, out_dir="plots"):
    setup_style(fontsize=12, scale=1.0)

    colors  = ["#2563eb", "#16a34a", "#dc2626"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True,
                             gridspec_kw={"hspace": 0.06})

    for N, color, marker in zip(systems, colors, markers):
        H_nn, H_nnn = _coupling_matrix(N, 1), _coupling_matrix(N, 2)
        marshall    = _marshall_signs(N)

        avg_signs = []
        for r in ratios:
            phi = _ground_phi(H_nn, H_nnn, marshall, J1, r)
            si  = _sign_impurity(phi)
            avg_signs.append(1.0 - 2.0 * si)
        avg_signs = np.array(avg_signs)

        kw = dict(color=color, lw=1.5, label=f"$N={N}$",
                  markevery=12, markersize=4.5, marker=marker,
                  markerfacecolor="white", markeredgewidth=1.3)

        axes[0].plot(ratios, avg_signs, **kw)

        # 1/⟨s⟩²: use NaN where |⟨s⟩| is numerically zero to avoid spurious
        # lines to ±∞; the gap in the curve makes the divergence visible.
        safe = np.where(np.abs(avg_signs) < 5e-3, np.nan, avg_signs)
        axes[1].plot(ratios, 1.0 / safe**2, **kw)

    for ax in axes:
        ax.axvline(0.5, color="#333", ls=":", lw=1.1, zorder=0)
        ax.set_xlim(0, 1.0)

    axes[0].annotate(
        r"Majumdar--Ghosh ($J_2/J_1=0.5$)",
        xy=(0.500, 0.02), xytext=(0.56, 0.20),
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
        fontsize=8.5, color="#333",
    )
    axes[0].axhline(0, color="#aaa", lw=0.7, ls="--", zorder=0)
    axes[0].set_ylabel(r"Average sign $\langle s \rangle = 1 - 2\,\epsilon$")
    axes[0].set_ylim(-0.08, 1.06)
    axes[0].legend(loc="upper right", fontsize=9, handlelength=1.8)

    # Annotate the divergence
    axes[1].annotate(r"$\to\infty$ at MG", xy=(0.5, 5e4),
                     xytext=(0.6, 2e4),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
                     fontsize=8.5, color="#555")
    axes[1].set_ylabel(r"QMC cost $1/\langle s\rangle^2$")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.7, 2e5)
    axes[1].set_xlabel(r"$J_2/J_1$")
    axes[1].legend(loc="upper left", fontsize=9, handlelength=1.8)
    axes[1].yaxis.set_major_formatter(mticker.LogFormatterMathtext())

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"j1j2_avg_sign.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: sign-structure heatmap (N=8, Sz=0 sector)
# ---------------------------------------------------------------------------

def fig_sign_heatmap(ratios: np.ndarray, N: int = 8, J1: float = 1.0,
                     out_dir: str = "plots"):
    """2D heatmap: x = Sz=0 basis states (sorted), y = J2/J1, colour = φ(v)."""
    H_nn, H_nnn = _coupling_matrix(N, 1), _coupling_matrix(N, 2)
    marshall    = _marshall_signs(N)

    # ── Sz=0 sector ─────────────────────────────────────────────────────────
    k          = np.arange(2 ** N)
    popcount   = np.array([bin(x).count("1") for x in k])
    sz0_idx    = np.where(popcount == N // 2)[0]   # 70 states for N=8

    # ── Sort states once using the reference GS just above MG ───────────────
    ref_phi         = _ground_phi(H_nn, H_nnn, marshall, J1, 0.51)
    ref_phi_sz0     = ref_phi[sz0_idx]
    sort_order      = np.argsort(-ref_phi_sz0)      # most-positive first
    sorted_sz0_idx  = sz0_idx[sort_order]
    n_states        = len(sorted_sz0_idx)

    # Find the sign boundary in the reference (where φ crosses zero)
    ref_sorted = ref_phi_sz0[sort_order]
    n_positive = int(np.sum(ref_sorted > 0))        # number of D1-dominant states

    # ── Build the (n_ratios × n_states) amplitude matrix ────────────────────
    print(f"  Building heatmap: {len(ratios)} J2/J1 values × {n_states} states …")
    amp_matrix = np.zeros((len(ratios), n_states))
    for row_idx, r in enumerate(ratios):
        phi = _ground_phi(H_nn, H_nnn, marshall, J1, r)
        phi_sz0 = phi[sorted_sz0_idx]
        # Normalise per row so sign and relative weight are both visible
        mx = np.max(np.abs(phi_sz0))
        amp_matrix[row_idx] = phi_sz0 / mx if mx > 0 else phi_sz0

    # ── Plot ─────────────────────────────────────────────────────────────────
    setup_style(fontsize=12, scale=1.0)

    fig, ax = plt.subplots(figsize=(7, 5))

    # pcolormesh: x = state index, y = J2/J1 (increasing upward)
    x_edges = np.arange(n_states + 1) - 0.5
    y_edges = np.concatenate([
        [ratios[0] - (ratios[1] - ratios[0]) / 2],
        (ratios[:-1] + ratios[1:]) / 2,
        [ratios[-1] + (ratios[-1] - ratios[-2]) / 2],
    ])

    cmap = plt.get_cmap("RdBu")
    mesh = ax.pcolormesh(x_edges, y_edges, amp_matrix,
                         cmap=cmap, vmin=-1, vmax=1,
                         shading="flat", rasterized=True)

    # MG line
    ax.axhline(0.5, color="#222", lw=1.3, ls="--", zorder=3,
               label=r"$J_2/J_1 = 0.5$ (MG)")

    # Vertical line at sign boundary
    ax.axvline(n_positive - 0.5, color="#555", lw=0.9, ls=":",
               zorder=3)

    # Region labels on x-axis
    mid_d1  = n_positive / 2
    mid_d2  = n_positive + (n_states - n_positive) / 2
    ax.text(mid_d1,  ratios[-1] + 0.04,  r"$|D_1\rangle$ support",
            ha="center", va="bottom", fontsize=9, color="#1a4fa0",
            transform=ax.get_xaxis_transform())
    ax.text(mid_d2,  ratios[-1] + 0.04,  r"$|D_2\rangle$ support",
            ha="center", va="bottom", fontsize=9, color="#9b1c1c",
            transform=ax.get_xaxis_transform())

    # Colourbar
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(r"$\phi(v)/\max|\phi|$", fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax.set_xlabel(r"Basis state (Sz=0 sector, sorted by $\phi$ at $J_2/J_1=0.51$)",
                  fontsize=10)
    ax.set_ylabel(r"$J_2/J_1$")
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(ratios[0], ratios[-1])

    # Remove dense x ticks; keep a few landmarks
    ax.set_xticks([0, n_positive - 1, n_positive, n_states - 1])
    ax.set_xticklabels(["0", f"{n_positive-1}", f"{n_positive}", f"{n_states-1}"],
                       fontsize=9)

    title_str = (
        rf"Sign structure heatmap — $N={N}$, $S_z=0$ sector ({n_states} states)"
    )
    ax.set_title(title_str, fontsize=10, pad=18)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"j1j2_sign_heatmap.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "j1j2")

    print("Figure 1: average sign and QMC cost …")
    ratios_coarse = np.linspace(0.0, 1.0, 120)
    fig_avg_sign(ratios_coarse, out_dir=out_dir)

    print("Figure 2: sign-structure heatmap (N=8) …")
    ratios_fine = np.linspace(0.0, 1.0, 80)
    fig_sign_heatmap(ratios_fine, N=8, out_dir=out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
