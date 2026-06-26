#!/usr/bin/env python3
"""
plot_phase_transition_j1j2_heisenberg.py

Trains a small RBM-VMC on the 1D J₁-J₂ Heisenberg chain at three values of
J₂/J₁ and visualises how antiferromagnetic / dimerised ordering changes
across the phase diagram:

  Row 0: heatmap of spin configurations sampled from the trained RBM
          (rows sorted by staggered magnetisation M_s = Σᵢ (-1)^i σᶻᵢ)
  Row 1: spin-spin correlation C(r) = <σ⁰z σᵣz> vs distance r,
          VMC samples (solid) vs exact ground state (dashed)

Hamiltonian:
    H = J₁ Σᵢ [σˣᵢσˣᵢ₊₁ + σʸᵢσʸᵢ₊₁ + σᶻᵢσᶻᵢ₊₁]
      + J₂ Σᵢ [σˣᵢσˣᵢ₊₂ + σʸᵢσʸᵢ₊₂ + σᶻᵢσᶻᵢ₊₂]    (J₁ = 1, Δ = 1)

Phase diagram (J₁ = 1, Δ = 1):
    J₂/J₁ < 0.241   gapless Luttinger liquid (Néel correlations)
    J₂/J₁ ≈ 0.241   KT transition → spin-Peierls dimerisation gap opens
    0.241 < J₂ < 0.5 gapped dimerised phase (still Marshall sign-free)
    J₂/J₁ = 0.5     Majumdar-Ghosh point: exact singlet-pair ground state
    J₂/J₁ > 0.5     real RBM fails — sign problem (not shown here)

Usage (from repo root):
    python scripts/viz/plot_phase_transition_j1j2_heisenberg.py
    python scripts/viz/plot_phase_transition_j1j2_heisenberg.py --size 12 --iters 400
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax

from plot_style import setup_style
from ising import J1J2HeisenbergXXZ1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer


# ── Phase points ──────────────────────────────────────────────────────────────

J2_POINTS = [
    (0.1,  r"LL / N\'{e}el  ($J_2/J_1 = 0.1$)"),
    (0.35, r"Dimerised  ($J_2/J_1 = 0.35$)"),
    (0.5,  r"Majumdar-Ghosh  ($J_2/J_1 = 0.5$)"),
]

_COLORS = ["#2166ac", "#756bb1", "#d7191c"]


# ── Exact reference ───────────────────────────────────────────────────────────

def _build_j1j2_sparse(N: int, J1: float, J2: float, delta: float) -> sp.csr_matrix:
    """
    Vectorised J₁-J₂ Hamiltonian with ±1 spin convention.
    Bit (N-1-i) of index s encodes spin at site i (same as ising.py).
    Off-diagonal: 2*J for each bond between opposite-spin neighbours.
    """
    dim = 2 ** N
    k = np.arange(dim, dtype=np.int64)

    diag = np.zeros(dim)
    rows_list, cols_list, vals_list = [k], [k], [None]

    for bond_len, J_bond in ((1, J1), (2, J2)):
        for i in range(N):
            j = (i + bond_len) % N
            si = (1 - 2 * ((k >> (N - 1 - i)) & 1)).astype(float)
            sj = (1 - 2 * ((k >> (N - 1 - j)) & 1)).astype(float)
            diag += J_bond * delta * si * sj

            opp = si * sj < 0
            k_src = k[opp]
            flip = (1 << (N - 1 - i)) | (1 << (N - 1 - j))
            k_dst = k_src ^ flip
            rows_list.append(k_dst)
            cols_list.append(k_src)
            vals_list.append(np.full(opp.sum(), 2.0 * J_bond))

    vals_list[0] = diag
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def exact_spin_corr(N: int, J2: float, max_r: int) -> np.ndarray:
    """C_exact(r) = <ψ₀| σ₀z σᵣz |ψ₀>  (J₁ = 1, Δ = 1)."""
    H = _build_j1j2_sparse(N, J1=1.0, J2=J2, delta=1.0)
    _, vecs = eigsh(H, k=1, which="SA")
    psi2 = vecs[:, 0] ** 2

    k = np.arange(2 ** N, dtype=np.int64)
    s0 = (1 - 2 * ((k >> (N - 1)) & 1)).astype(float)

    corr = np.empty(max_r + 1)
    for r in range(max_r + 1):
        sr = (1 - 2 * ((k >> (N - 1 - r)) & 1)).astype(float)
        corr[r] = np.sum(psi2 * s0 * sr)
    return corr


def vmc_spin_corr(V: np.ndarray, max_r: int) -> np.ndarray:
    N = V.shape[1]
    return np.array([np.mean(V[:, 0] * V[:, r % N]) for r in range(max_r + 1)])


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_sample(
    size: int,
    n_hidden: int,
    J2: float,
    n_train_samples: int,
    n_iters: int,
    seed: int,
    n_viz_samples: int,
):
    import warnings
    key = jax.random.PRNGKey(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress frustrated-regime warning for J2=0.35/0.5
        model = J1J2HeisenbergXXZ1D(size=size, J1=1.0, J2=J2, delta=1.0)
    rbm = FullyConnectedRBM(n_visible=size, n_hidden=n_hidden, key=key)
    sampler = ClassicalSampler(method="metropolis")
    config = {
        "n_samples": n_train_samples,
        "n_iterations": n_iters,
        "learning_rate": 0.03,
        "regularization": 1e-3,
    }
    trainer = Trainer(rbm=rbm, ising_model=model, sampler=sampler, config=config)
    history = trainer.train()

    V = np.asarray(sampler.sample(rbm, n_viz_samples, config={"beta_x": 1.0}))
    E_vmc   = float(np.mean(history["energy"][-20:]))
    E_exact = float(model.exact_ground_energy())
    return V, E_vmc, E_exact


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(args):
    N = args.size
    max_r = N // 2

    out = _REPO / "plots" / "phase_transitions"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"j1j2_cache_N{N}_nh{args.n_hidden}_it{args.iters}_ns{args.samples}_s{args.seed}.npz"

    # ── Data: train or load from cache ────────────────────────────────────────
    if args.plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache at {cache_path} — run without --plot-only first."
            )
        raw = np.load(cache_path, allow_pickle=False)
        phase_data = [
            (raw[f"V_{i}"], float(raw["E_vmc"][i]), float(raw["E_exact"][i]))
            for i in range(len(J2_POINTS))
        ]
        print(f"  loaded cache  {cache_path}")
    else:
        phase_data = []
        for J2, _ in J2_POINTS:
            print(f"\n── J₂/J₁ = {J2} ──────────────────────────────")
            V, E_vmc, E_exact = train_and_sample(
                size=N, n_hidden=args.n_hidden, J2=J2,
                n_train_samples=args.samples, n_iters=args.iters,
                seed=args.seed, n_viz_samples=args.viz_samples,
            )
            rel_err = abs(E_vmc - E_exact) / abs(E_exact) * 100
            print(f"   E_VMC = {E_vmc:.4f}  E_exact = {E_exact:.4f}  ε = {rel_err:.2f}%")
            phase_data.append((V, E_vmc, E_exact))
        np.savez(
            cache_path,
            **{f"V_{i}": d[0] for i, d in enumerate(phase_data)},
            E_vmc   = np.array([d[1] for d in phase_data]),
            E_exact = np.array([d[2] for d in phase_data]),
        )
        print(f"\n  cached  {cache_path}")

    setup_style(fontsize=10, scale=1.0)

    cmap_spins = mcolors.ListedColormap(["#2166ac", "#d7191c"])
    spin_norm  = mcolors.BoundaryNorm([-1.5, 0, 1.5], cmap_spins.N)

    fig = plt.figure(figsize=(12, 5.5))
    gs = GridSpec(
        2, 4, figure=fig,
        height_ratios=[2.2, 1],
        width_ratios=[1, 1, 1, 0.12],
        hspace=0.55, wspace=0.32,
    )
    heat_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    corr_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    cbar_ax   = fig.add_subplot(gs[0, 3])

    for col, ((J2, label), (V, E_vmc, E_exact)) in enumerate(zip(J2_POINTS, phase_data)):
        color = _COLORS[col]

        # Sort by staggered magnetisation (highlights Néel vs. dimer structure)
        stag = np.array([((-1) ** np.arange(N)) @ V[s] for s in range(len(V))])
        order = np.argsort(stag)[::-1]
        V_plot = V[order]

        # ── heatmap ────────────────────────────────────────────────────────
        ax_h = heat_axes[col]
        ax_h.imshow(
            V_plot, aspect="auto", cmap=cmap_spins, norm=spin_norm,
            interpolation="none",
        )
        ax_h.grid(False)
        ax_h.set_title(label, fontsize=10, pad=5)
        ax_h.set_xlabel("Site $i$", fontsize=9)
        if col == 0:
            ax_h.set_ylabel("Sample", fontsize=9)
        else:
            ax_h.set_yticks([])

        rel_err = abs(E_vmc - E_exact) / abs(E_exact) * 100
        ax_h.text(
            0.97, 0.03,
            rf"$\varepsilon = {rel_err:.2f}\%$",
            transform=ax_h.transAxes,
            ha="right", va="bottom", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#222", alpha=0.55, ec="none"),
        )

        # ── correlation ────────────────────────────────────────────────────
        ax_c = corr_axes[col]
        r_vals  = np.arange(max_r + 1)
        c_exact = exact_spin_corr(N, J2, max_r)
        c_vmc   = vmc_spin_corr(V, max_r)

        ax_c.plot(r_vals, c_exact, "k--", lw=1.3, label="Exact")
        ax_c.plot(r_vals, c_vmc,   color=color, lw=1.8, label="VMC")
        ax_c.axhline(0, color="#aaaaaa", lw=0.7, ls=":")
        ax_c.set_xlabel(r"Distance $r$", fontsize=9)
        if col == 0:
            ax_c.set_ylabel(r"$\langle \sigma^z_0 \sigma^z_r \rangle$", fontsize=9)
        ax_c.legend(fontsize=8, loc="upper right", handlelength=1.5)
        ax_c.set_xlim(0, max_r)
        ax_c.set_xticks(range(0, max_r + 1, 2))

        # Mark the KT point on the first correlation panel
        if col == 0:
            ax_c.annotate(r"$J_2/J_1=0.241$: KT", xy=(0, 0), xycoords="axes fraction",
                          xytext=(0.02, 0.06), textcoords="axes fraction",
                          fontsize=6.5, color="#555")

    # ── spin legend: two discrete swatches (spins are binary, no gradient) ──
    cbar_ax.axis("off")
    for y0, fc, lbl in [(0.60, "#d7191c", r"$\uparrow$"), (0.10, "#2166ac", r"$\downarrow$")]:
        cbar_ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, y0), 0.9, 0.28, boxstyle="square,pad=0",
            facecolor=fc, edgecolor="none",
            transform=cbar_ax.transAxes, clip_on=False,
        ))
        cbar_ax.text(0.5, y0 + 0.14, lbl, transform=cbar_ax.transAxes,
                     ha="center", va="center", fontsize=11, color="white")

    for ext in ("pdf", "png"):
        path = out / f"spin_ordering_j1j2_heisenberg.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"\n  saved  {path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size",        type=int, default=12,   help="chain length N (divisible by 4 for clean MG tiling)")
    p.add_argument("--n-hidden",    type=int, default=None, help="RBM hidden units (default: 2N)")
    p.add_argument("--iters",       type=int, default=400,  help="VMC training iterations")
    p.add_argument("--samples",     type=int, default=500,  help="samples per training step")
    p.add_argument("--viz-samples", type=int, default=2000, help="samples shown in heatmap")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--plot-only",   action="store_true",
                   help="skip training; load samples from cache and regenerate plots only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.n_hidden is None:
        args.n_hidden = 2 * args.size
    make_figure(args)
