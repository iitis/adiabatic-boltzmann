#!/usr/bin/env python3
"""
plot_j1j2_structure_factor.py

Structure factor S(k) = sum_r C(r) exp(-i k r) for the frustrated 1D
J1-J2 transverse-field Ising chain (J1J2Ising1D):

    H = -J1 sum_i sigma^z_i sigma^z_{i+1} - J2 sum_i sigma^z_i sigma^z_{i+2}
        - h sum_i sigma^x_i

IMPORTANT sign-convention note (verified numerically, not assumed):
    With this class's convention both -J1 and -J2 terms favour ALIGNMENT
    when J1, J2 > 0 -- i.e. positive J1 and J2 together are a plain
    (unfrustrated) ferromagnet with S(k) peaked at k=0 for every ratio.
    The genuinely frustrated ANNNI-type regime -- ferromagnetic nearest
    neighbours competing against antiferromagnetic next-nearest neighbours,
    with a Lifshitz point at |J2|/J1 = 0.5 -- requires J1 > 0 and J2 < 0 in
    this sign convention. That is the regime swept below. (A naive
    `--J2 > 0` sweep, as the class's default kwargs might suggest, would
    show no physics at all: the peak stays pinned at k=0.)

As |J2|/J1 crosses the Lifshitz point at 0.5, a transverse field h > 0
turns the classical devil's staircase into a continuous "floating"
incommensurate phase, so the structure-factor peak slides smoothly from
k=0 (ferromagnetic) to k=pi/2 (antiphase <2,2> order) instead of jumping
between commensurate plateaus.

  Left panel:  S(k) vs k for a sequence of |J2|/J1 ratios (VMC), sequential
               blue ramp from light (weak frustration) to dark (strong).
  Right panel: peak momentum k*/pi vs |J2|/J1, VMC vs exact diagonalization,
               showing the shift through the Lifshitz point.

Usage (from repo root):
    python scripts/viz/plot_j1j2_structure_factor.py
    python scripts/viz/plot_j1j2_structure_factor.py --size 16 --h 0.6
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # plot_style

import jax

from plot_style import setup_style
from ising import J1J2Ising1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer

RATIOS = [0.0, 0.3, 0.5, 0.7, 1.0, 2.0]   # |J2| / J1, frustrated (J2 < 0) regime
LIFSHITZ = 0.5


# ── Exact reference ───────────────────────────────────────────────────────────

def _build_j1j2_ising_sparse(N: int, J1: float, J2: float, h: float) -> sp.csr_matrix:
    """H = -J1 sum sigma^z_i sigma^z_{i+1} - J2 sum sigma^z_i sigma^z_{i+2} - h sum sigma^x_i."""
    dim = 2 ** N
    k = np.arange(dim, dtype=np.int64)

    diag = np.zeros(dim)
    for bond_len, J in ((1, J1), (2, J2)):
        for i in range(N):
            j = (i + bond_len) % N
            si = (1 - 2 * ((k >> (N - 1 - i)) & 1)).astype(float)
            sj = (1 - 2 * ((k >> (N - 1 - j)) & 1)).astype(float)
            diag -= J * si * sj

    row_list, col_list, val_list = [k], [k], [diag]
    for i in range(N):
        flip = k ^ (1 << (N - 1 - i))
        row_list.append(k)
        col_list.append(flip)
        val_list.append(np.full(dim, -h))

    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    vals = np.concatenate(val_list)
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def exact_spin_corr(N: int, J1: float, J2: float, h: float) -> np.ndarray:
    """C_exact(r) = <psi_0| sigma^z_0 sigma^z_r |psi_0> for r = 0..N-1."""
    H = _build_j1j2_ising_sparse(N, J1, J2, h)
    _, vecs = eigsh(H, k=1, which="SA")
    psi2 = vecs[:, 0] ** 2

    k = np.arange(2 ** N, dtype=np.int64)
    s0 = (1 - 2 * ((k >> (N - 1)) & 1)).astype(float)

    corr = np.empty(N)
    for r in range(N):
        sr = (1 - 2 * ((k >> (N - 1 - r)) & 1)).astype(float)
        corr[r] = np.sum(psi2 * s0 * sr)
    return corr


def vmc_spin_corr(V: np.ndarray) -> np.ndarray:
    """C_VMC(r) = mean over samples of sigma_0 * sigma_r, for r = 0..N-1."""
    N = V.shape[1]
    return np.array([np.mean(V[:, 0] * V[:, r]) for r in range(N)])


def structure_factor(corr: np.ndarray):
    """S(k) = sum_r C(r) cos(k r) at the N allowed momenta k in [0, pi]."""
    N = len(corr)
    r = np.arange(N)
    ks_full = 2.0 * np.pi * np.arange(N) / N
    mask = ks_full <= np.pi + 1e-9
    ks = ks_full[mask]
    Sk = np.array([np.sum(corr * np.cos(kval * r)) for kval in ks])
    return ks, Sk


def peak_momentum(ks: np.ndarray, Sk: np.ndarray) -> float:
    return float(ks[np.argmax(Sk)])


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_sample(size, n_hidden, J1, J2, h, n_train_samples, n_iters, seed, n_viz_samples):
    key = jax.random.PRNGKey(seed)
    model = J1J2Ising1D(size=size, J1=J1, J2=J2, h=h)
    rbm = FullyConnectedRBM(n_visible=size, n_hidden=n_hidden, key=key)
    sampler = ClassicalSampler(method="metropolis")
    config = {
        "n_samples": n_train_samples,
        "n_iterations": n_iters,
        "learning_rate": 0.05,
        "regularization": 1e-3,
    }
    trainer = Trainer(rbm=rbm, ising_model=model, sampler=sampler, config=config)
    trainer.train()
    V = np.asarray(sampler.sample(rbm, n_viz_samples, config={"beta_x": 1.0}))
    return V


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(args):
    N = args.size
    J1 = 1.0

    out = _REPO / "plots" / "j1j2"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = (
        out / f"structure_factor_cache_N{N}_nh{args.n_hidden}_h{args.h}"
              f"_it{args.iters}_ns{args.samples}_s{args.seed}.npz"
    )

    if args.plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache at {cache_path} -- run without --plot-only first."
            )
        raw = np.load(cache_path)
        C_exact = raw["C_exact"]
        C_vmc   = raw["C_vmc"]
        print(f"  loaded cache  {cache_path}")
    else:
        C_exact = np.empty((len(RATIOS), N))
        C_vmc   = np.empty((len(RATIOS), N))
        for i, ratio in enumerate(RATIOS):
            J2 = -ratio * J1   # antiferromagnetic NNN: the frustrated regime
            print(f"\n── |J2|/J1 = {ratio:.2f}  (J2={J2:.2f})  ({i + 1}/{len(RATIOS)}) ──")
            V = train_and_sample(
                size=N, n_hidden=args.n_hidden, J1=J1, J2=J2, h=args.h,
                n_train_samples=args.samples, n_iters=args.iters,
                seed=args.seed, n_viz_samples=args.viz_samples,
            )
            C_exact[i] = exact_spin_corr(N, J1, J2, args.h)
            C_vmc[i]   = vmc_spin_corr(V)
            print(f"   C_vmc(1) = {C_vmc[i, 1]:.3f}  C_exact(1) = {C_exact[i, 1]:.3f}")
        np.savez(cache_path, C_exact=C_exact, C_vmc=C_vmc)
        print(f"\n  cached  {cache_path}")

    setup_style(fontsize=10, scale=1.0)
    fig, (ax_sk, ax_peak) = plt.subplots(1, 2, figsize=(11, 4.2))

    cmap = cm.get_cmap("Blues")
    shades = np.linspace(0.35, 0.95, len(RATIOS))

    peak_vmc = np.empty(len(RATIOS))
    peak_exact = np.empty(len(RATIOS))
    for i, (ratio, shade) in enumerate(zip(RATIOS, shades)):
        ks, Sk_vmc = structure_factor(C_vmc[i])
        _, Sk_exact = structure_factor(C_exact[i])
        peak_vmc[i] = peak_momentum(ks, Sk_vmc)
        peak_exact[i] = peak_momentum(ks, Sk_exact)
        ax_sk.plot(ks / np.pi, Sk_vmc, color=cmap(shade), lw=1.8,
                   label=rf"$|J_2|/J_1={ratio:.2f}$")

    ax_sk.set_xlabel(r"$k/\pi$")
    ax_sk.set_ylabel(r"$S(k)$")
    ax_sk.set_xlim(0, 1)
    ax_sk.legend(fontsize=8, loc="upper left", handlelength=1.5)

    ax_peak.plot(RATIOS, peak_exact / np.pi, "k--", lw=1.3, label="Exact")
    ax_peak.plot(RATIOS, peak_vmc / np.pi, "o-", color="#2166ac", lw=1.6, ms=5, label="VMC")
    ax_peak.axvline(LIFSHITZ, color="#888888", ls=":", lw=1.2, zorder=0)
    ax_peak.text(LIFSHITZ, 1.02, "Lifshitz\npoint", transform=ax_peak.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.5, color="#666666")
    ax_peak.axhline(0.5, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax_peak.set_xlabel(r"$|J_2|/J_1$")
    ax_peak.set_ylabel(r"Peak momentum $k^*/\pi$")
    ax_peak.set_ylim(-0.02, 0.55)
    ax_peak.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = out / f"j1j2_structure_factor.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"\n  saved  {path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size",        type=int, default=14,   help="chain length N (<=16 for exact diag)")
    p.add_argument("--n-hidden",    type=int, default=None, help="RBM hidden units (default: 2N)")
    p.add_argument("--h",           type=float, default=0.6, help="transverse field (floating-phase regime)")
    p.add_argument("--iters",       type=int, default=400,  help="VMC training iterations")
    p.add_argument("--samples",     type=int, default=600,  help="samples per training step")
    p.add_argument("--viz-samples", type=int, default=3000, help="samples used for C(r) estimate")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--plot-only",   action="store_true",
                   help="skip training; load correlation data from cache and regenerate plot only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.n_hidden is None:
        args.n_hidden = 2 * args.size
    if args.size > 16:
        raise ValueError("Exact diagonalization requires size <= 16 (2^16 states).")
    make_figure(args)
