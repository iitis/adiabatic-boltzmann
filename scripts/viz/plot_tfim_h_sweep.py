#!/usr/bin/env python3
"""
plot_tfim_h_sweep.py

Sweeps the 1D transverse-field Ising model across its quantum phase
transition at h_c = 1 and compares VMC (RBM) against exact references:

  Row 0: ground-state energy per site E/N -- VMC vs exact (free-fermion
         integral formula)
  Row 1: order parameter <|m^z|> -- VMC (sampled |mean_i sigma_i^z|, the
         standard finite-size estimator for a symmetry-broken order
         parameter) vs the exact Pfeuty (1970) thermodynamic-limit result
         <sigma^z> = (1-h^2)^(1/8) for h<1, else 0
  Row 2: relative energy error eps(h) = |E_vmc - E_exact| / |E_exact|,
         log scale -- shows where the ansatz starts struggling near h_c

Usage (from repo root):
    python scripts/viz/plot_tfim_h_sweep.py
    python scripts/viz/plot_tfim_h_sweep.py --size 12 --n-h 15
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # plot_style

import jax

from plot_style import setup_style
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer

_COLOR = "#2166ac"
H_C = 1.0


# ── Exact reference ───────────────────────────────────────────────────────────

def exact_order_parameter(h: float) -> float:
    """Pfeuty (1970): <sigma^z> = (1-h^2)^(1/8) for h<1, else 0 (L -> infinity)."""
    return (1.0 - h ** 2) ** 0.125 if h < 1.0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_measure(
    size: int,
    n_hidden: int,
    h: float,
    n_train_samples: int,
    n_iters: int,
    seed: int,
    n_viz_samples: int,
):
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
    history = trainer.train()

    V = np.asarray(sampler.sample(rbm, n_viz_samples, config={"beta_x": 1.0}))
    E_vmc = float(np.mean(history["energy"][-20:])) / size
    E_exact = float(ising.exact_ground_energy()) / size
    M_vmc = float(np.mean(np.abs(np.mean(V, axis=1))))
    return E_vmc, E_exact, M_vmc


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(args):
    N = args.size
    h_grid = np.linspace(args.h_min, args.h_max, args.n_h)

    out = _REPO / "plots" / "phase_transitions"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = (
        out / f"tfim_hsweep_cache_N{N}_nh{args.n_hidden}_it{args.iters}"
              f"_ns{args.samples}_s{args.seed}_pts{args.n_h}.npz"
    )

    # ── Data: train or load from cache ────────────────────────────────────────
    if args.plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache at {cache_path} -- run without --plot-only first."
            )
        raw = np.load(cache_path)
        h_grid  = raw["h_grid"]
        E_vmc   = raw["E_vmc"]
        E_exact = raw["E_exact"]
        M_vmc   = raw["M_vmc"]
        print(f"  loaded cache  {cache_path}")
    else:
        E_vmc   = np.empty(len(h_grid))
        E_exact = np.empty(len(h_grid))
        M_vmc   = np.empty(len(h_grid))
        for i, h in enumerate(h_grid):
            print(f"\n── h = {h:.3f}  ({i + 1}/{len(h_grid)}) ──────────────────")
            e_v, e_x, m_v = train_and_measure(
                size=N, n_hidden=args.n_hidden, h=float(h),
                n_train_samples=args.samples, n_iters=args.iters,
                seed=args.seed, n_viz_samples=args.viz_samples,
            )
            rel_err = abs(e_v - e_x) / abs(e_x) * 100
            print(f"   E_VMC/N = {e_v:.4f}  E_exact/N = {e_x:.4f}  "
                  f"ε = {rel_err:.2f}%  M_vmc = {m_v:.3f}")
            E_vmc[i], E_exact[i], M_vmc[i] = e_v, e_x, m_v
        np.savez(cache_path, h_grid=h_grid, E_vmc=E_vmc, E_exact=E_exact, M_vmc=M_vmc)
        print(f"\n  cached  {cache_path}")

    setup_style(fontsize=10, scale=1.0)

    M_exact = np.array([exact_order_parameter(h) for h in h_grid])
    rel_err = np.abs(E_vmc - E_exact) / np.abs(E_exact) * 100

    fig, (ax_e, ax_m, ax_eps) = plt.subplots(
        3, 1, figsize=(6, 8), sharex=True, gridspec_kw={"hspace": 0.12},
    )

    ax_e.plot(h_grid, E_exact, "k--", lw=1.3, label="Exact")
    ax_e.plot(h_grid, E_vmc, "o-", color=_COLOR, lw=1.6, ms=4, label="VMC")
    ax_e.set_ylabel(r"$E/N$")
    ax_e.legend(fontsize=9, loc="best")

    ax_m.plot(h_grid, M_exact, "k--", lw=1.3, label=r"Exact ($L\to\infty$)")
    ax_m.plot(h_grid, M_vmc, "o-", color=_COLOR, lw=1.6, ms=4, label="VMC")
    ax_m.set_ylabel(r"$\langle |m^z| \rangle$")
    ax_m.legend(fontsize=9, loc="best")

    ax_eps.plot(h_grid, rel_err, "o-", color=_COLOR, lw=1.6, ms=4)
    ax_eps.set_yscale("log")
    ax_eps.set_ylabel(r"$\varepsilon$ (\%)")
    ax_eps.set_xlabel(r"$h$")

    for ax in (ax_e, ax_m, ax_eps):
        ax.axvline(H_C, color="#888888", ls=":", lw=1.2, zorder=0)
    ax_e.text(H_C, 1.03, r"$h_c=1$", transform=ax_e.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=8, color="#666666")

    ax_e.set_xlim(h_grid.min(), h_grid.max())

    for ext in ("pdf", "png"):
        path = out / f"tfim_h_sweep.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"\n  saved  {path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size",        type=int, default=12,   help="chain length N")
    p.add_argument("--n-hidden",    type=int, default=None, help="RBM hidden units (default: 2N)")
    p.add_argument("--h-min",       type=float, default=0.0)
    p.add_argument("--h-max",       type=float, default=2.0)
    p.add_argument("--n-h",         type=int, default=11,   help="number of h points in the sweep")
    p.add_argument("--iters",       type=int, default=200,  help="VMC training iterations")
    p.add_argument("--samples",     type=int, default=400,  help="samples per training step")
    p.add_argument("--viz-samples", type=int, default=2000, help="samples used for magnetization estimate")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--plot-only",   action="store_true",
                   help="skip training; load results from cache and regenerate plot only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.n_hidden is None:
        args.n_hidden = 2 * args.size
    make_figure(args)
