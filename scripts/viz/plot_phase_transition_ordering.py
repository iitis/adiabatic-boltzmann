#!/usr/bin/env python3
"""
plot_phase_transition_ordering.py

Trains a small RBM-VMC on the 1D TFIM at three values of the transverse
field h (ordered / critical / disordered) and visualises how spin ordering
changes across the quantum phase transition at h_c = 1, via a heatmap of
spin configurations sampled from the trained RBM (rows = samples sorted by
magnetisation, columns = sites).

The relative energy error ε is shown in each panel to demonstrate that the
model has actually converged to the correct ground state.

Usage (from repo root):
    python scripts/viz/plot_phase_transition_ordering.py
    python scripts/viz/plot_phase_transition_ordering.py --size 12 --iters 200
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

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # plot_style

import jax
jax.config.update("jax_enable_x64", True)

from plot_style import setup_style
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer


# ── Phase points to visualise ─────────────────────────────────────────────────

H_POINTS = [
    (0.3, r"Ordered  ($h = 0.3 < h_c$)"),
    (1.0, r"Critical  ($h = 1.0 = h_c$)"),
    (1.7, r"Disordered  ($h = 1.7 > h_c$)"),
]


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_sample(
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
    E_vmc = float(np.mean(history["energy"][-20:]))
    E_exact = float(ising.exact_ground_energy())
    return V, E_vmc, E_exact


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(args):
    N = args.size

    out = _REPO / "plots" / "phase_transitions"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"tfim_cache_N{N}_nh{N}_it{args.iters}_ns{args.samples}_s{args.seed}.npz"

    # ── Data: train or load from cache ────────────────────────────────────────
    if args.plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache at {cache_path} — run without --plot-only first."
            )
        raw = np.load(cache_path, allow_pickle=False)
        phase_data = [
            (raw[f"V_{i}"], float(raw["E_vmc"][i]), float(raw["E_exact"][i]))
            for i in range(len(H_POINTS))
        ]
        print(f"  loaded cache  {cache_path}")
    else:
        phase_data = []
        for h, _ in H_POINTS:
            print(f"\n── h = {h} ──────────────────────────────")
            V, E_vmc, E_exact = train_and_sample(
                size=N, n_hidden=N, h=h,
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

    fig = plt.figure(figsize=(12, 3.3))
    gs = GridSpec(
        1, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.12],
        wspace=0.32,
    )
    heat_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    cbar_ax   = fig.add_subplot(gs[0, 3])

    for col, ((_, label), (V, E_vmc, E_exact)) in enumerate(zip(H_POINTS, phase_data)):
        # Sort rows by total magnetisation: ordered → two clear blocks (↑↑ / ↓↓)
        order = np.argsort(V.sum(axis=1))[::-1]
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
        path = out / f"spin_ordering_across_qpt.{ext}"
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
    p.add_argument("--iters",       type=int, default=300,  help="VMC training iterations")
    p.add_argument("--samples",     type=int, default=500,  help="samples per training step")
    p.add_argument("--viz-samples", type=int, default=2000, help="samples shown in heatmap")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--plot-only",   action="store_true",
                   help="skip training; load samples from cache and regenerate plots only")
    return p.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
