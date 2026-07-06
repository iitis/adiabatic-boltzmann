#!/usr/bin/env python3
"""
plot_correlation_length.py

Extracts the correlation length xi(h) of the 1D TFIM from the spin-spin
correlation function C(r) = <sigma^z_0 sigma^z_r> and shows it diverge at
the quantum critical point h_c = 1.

C(r) is fit to  C(r) = C_inf + A * exp(-r/xi),  where the C_inf offset
absorbs the long-range order plateau in the ordered phase (h < h_c) so the
same 3-parameter model works on both sides of the transition.

  Left panel:  C(r) vs r at three representative h, VMC samples (markers)
               vs exact diagonalization (dashed) vs the fitted exponential
               (thin solid), on a log scale to show the fit quality.
  Right panel: xi(h) vs h, VMC vs exact, peaking at h_c = 1.

Usage (from repo root):
    python scripts/viz/plot_correlation_length.py
    python scripts/viz/plot_correlation_length.py --size 16 --n-h 17
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # plot_style

import jax

from plot_style import setup_style
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer

H_C = 1.0
_COLORS = ["#2166ac", "#756bb1", "#d7191c"]


# ── Exact reference ───────────────────────────────────────────────────────────

def _build_tfim_sparse(N: int, h: float) -> sp.csr_matrix:
    """TFIM sparse Hamiltonian; bit (N-1-i) of index s encodes spin i."""
    dim = 2 ** N
    k = np.arange(dim, dtype=np.int64)

    diag = np.zeros(dim)
    for i in range(N):
        j = (i + 1) % N
        si = (1 - 2 * ((k >> (N - 1 - i)) & 1)).astype(float)
        sj = (1 - 2 * ((k >> (N - 1 - j)) & 1)).astype(float)
        diag -= si * sj

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


def exact_spin_corr(N: int, h: float, max_r: int) -> np.ndarray:
    """C_exact(r) = <psi_0| sigma^z_0 sigma^z_r |psi_0> from full diagonalization."""
    H = _build_tfim_sparse(N, h)
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
    """C_VMC(r) = mean over samples of sigma_0 * sigma_r."""
    N = V.shape[1]
    return np.array([np.mean(V[:, 0] * V[:, r % N]) for r in range(max_r + 1)])


# ── Correlation-length fit ────────────────────────────────────────────────────

def _corr_model(r, xi, A, C_inf):
    return C_inf + A * np.exp(-r / xi)


def fit_correlation_length(c_vals: np.ndarray, N: int):
    """Fit C(r) (r=1..max_r) to C_inf + A*exp(-r/xi). Returns (xi, popt|None).

    When the decaying part is negligible (deep in the ordered phase, where
    C(r) is nearly flat at the long-range-order plateau), xi is unconstrained
    by the data and the optimizer runs away to the upper bound. That bound
    hit is not a measurement -- report it as NaN (unreliable) rather than a
    number that looks like a real correlation length.
    """
    r_fit = np.arange(1, len(c_vals))
    c_fit = c_vals[1:]
    c_tail = float(np.mean(c_fit[-2:]))
    p0 = [max(N / 8.0, 0.5), c_fit[0] - c_tail, c_tail]
    xi_upper = 10.0 * N
    bounds = ([1e-3, -2.0, -1.0], [xi_upper, 2.0, 1.0])
    try:
        popt, _ = curve_fit(_corr_model, r_fit, c_fit, p0=p0, bounds=bounds, maxfev=20000)
        xi = float(popt[0])
        if xi > 0.99 * xi_upper:
            print(f"    fit saturated upper bound (xi={xi:.1f}) -- decay too weak to constrain, marking unreliable")
            return float("nan"), None
        return xi, popt
    except RuntimeError as e:
        print(f"    fit failed: {e}")
        return float("nan"), None


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_sample(size, n_hidden, h, n_train_samples, n_iters, seed, n_viz_samples):
    key = jax.random.PRNGKey(seed)
    ising = TransverseFieldIsing1D(size=size, h=h)
    rbm = FullyConnectedRBM(n_visible=size, n_hidden=n_hidden, key=key)
    sampler = ClassicalSampler(method="metropolis")
    config = {
        "n_samples": n_train_samples,
        "n_iterations": n_iters,
        "learning_rate": 0.05,
        "regularization": 1e-3,
    }
    trainer = Trainer(rbm=rbm, ising_model=ising, sampler=sampler, config=config)
    trainer.train()
    V = np.asarray(sampler.sample(rbm, n_viz_samples, config={"beta_x": 1.0}))
    return V


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(args):
    N = args.size
    max_r = N // 2
    h_grid = np.linspace(args.h_min, args.h_max, args.n_h)

    out = _REPO / "plots" / "phase_transitions"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = (
        out / f"xi_cache_N{N}_nh{args.n_hidden}_it{args.iters}"
              f"_ns{args.samples}_s{args.seed}_pts{args.n_h}.npz"
    )

    if args.plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache at {cache_path} -- run without --plot-only first."
            )
        raw = np.load(cache_path)
        h_grid  = raw["h_grid"]
        C_exact = raw["C_exact"]
        C_vmc   = raw["C_vmc"]
        print(f"  loaded cache  {cache_path}")
    else:
        C_exact = np.empty((len(h_grid), max_r + 1))
        C_vmc   = np.empty((len(h_grid), max_r + 1))
        for i, h in enumerate(h_grid):
            print(f"\n── h = {h:.3f}  ({i + 1}/{len(h_grid)}) ──────────────────")
            V = train_and_sample(
                size=N, n_hidden=args.n_hidden, h=float(h),
                n_train_samples=args.samples, n_iters=args.iters,
                seed=args.seed, n_viz_samples=args.viz_samples,
            )
            C_exact[i] = exact_spin_corr(N, float(h), max_r)
            C_vmc[i]   = vmc_spin_corr(V, max_r)
            print(f"   C_vmc(1) = {C_vmc[i, 1]:.3f}  C_exact(1) = {C_exact[i, 1]:.3f}")
        np.savez(cache_path, h_grid=h_grid, C_exact=C_exact, C_vmc=C_vmc)
        print(f"\n  cached  {cache_path}")

    xi_exact = np.array([fit_correlation_length(C_exact[i], N)[0] for i in range(len(h_grid))])
    xi_vmc   = np.array([fit_correlation_length(C_vmc[i], N)[0] for i in range(len(h_grid))])

    setup_style(fontsize=10, scale=1.0)
    fig, (ax_c, ax_xi) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ── Left: representative C(r) + fits ──────────────────────────────────────
    r_vals = np.arange(max_r + 1)
    r_dense = np.linspace(0.5, max_r, 200)
    targets = [0.5, 1.0, 1.5]
    idxs = [int(np.argmin(np.abs(h_grid - t))) for t in targets]

    for color, i in zip(_COLORS, idxs):
        h = h_grid[i]
        xi_i, popt = fit_correlation_length(C_vmc[i], N)
        ax_c.plot(r_vals, np.abs(C_exact[i]), color=color, ls="--", lw=1.3)
        ax_c.plot(r_vals, np.abs(C_vmc[i]), "o", color=color, ms=4,
                   label=rf"$h={h:.2f}$  ($\xi_{{\rm VMC}}={xi_i:.2f}$)")
        if popt is not None:
            ax_c.plot(r_dense, np.abs(_corr_model(r_dense, *popt)), color=color, lw=1.0, alpha=0.7)

    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Distance $r$")
    ax_c.set_ylabel(r"$|\langle \sigma^z_0 \sigma^z_r \rangle|$")
    ax_c.set_xlim(0, max_r)
    ax_c.legend(fontsize=8, loc="upper right", handlelength=1.5)

    # ── Right: xi(h) ───────────────────────────────────────────────────────────
    ax_xi.plot(h_grid, xi_exact, "k--", lw=1.3, label="Exact")
    ax_xi.plot(h_grid, xi_vmc, "o-", color=_COLORS[0], lw=1.6, ms=4, label="VMC")
    ax_xi.axvline(H_C, color="#888888", ls=":", lw=1.2, zorder=0)
    ax_xi.text(H_C, 1.02, r"$h_c=1$", transform=ax_xi.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=8, color="#666666")
    ax_xi.set_xlabel(r"$h$")
    ax_xi.set_ylabel(r"Correlation length $\xi$")
    ax_xi.set_xlim(h_grid.min(), h_grid.max())
    ax_xi.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = out / f"tfim_correlation_length.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"\n  saved  {path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size",        type=int, default=16,   help="chain length N")
    p.add_argument("--n-hidden",    type=int, default=None, help="RBM hidden units (default: 2N)")
    p.add_argument("--h-min",       type=float, default=0.2)
    p.add_argument("--h-max",       type=float, default=1.8)
    p.add_argument("--n-h",         type=int, default=13,   help="number of h points in the sweep")
    p.add_argument("--iters",       type=int, default=300,  help="VMC training iterations")
    p.add_argument("--samples",     type=int, default=500,  help="samples per training step")
    p.add_argument("--viz-samples", type=int, default=3000, help="samples used for C(r) estimate")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--plot-only",   action="store_true",
                   help="skip training; load correlation data from cache and regenerate plot only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.n_hidden is None:
        args.n_hidden = 2 * args.size
    make_figure(args)
