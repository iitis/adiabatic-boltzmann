"""
cem_mechanism.py -- what CEM actually fits, made visible.

At every CEM step (src/encoder.py:estimate_beta_eff_cem), CEM takes the
joint LSB samples (V, H) drawn that iteration, forms the per-hidden-unit
local field Theta = V @ W + b, and picks the scalar beta that makes
tanh(beta * Theta) match the sampled H in a least-squares sense. This
script makes that fit visible directly: it trains a small TFIM 1D RBM
with CEM, freezes the RBM twice (early vs converged), draws a large fresh
LSB sample batch at each freeze point, and plots

  (1) the empirical <h> vs Theta response curve (binned over all
      samples x hidden units), against the CEM-fitted tanh(beta*Theta)
      and the naive tanh(1*Theta), and
  (2) the marginal histogram of Theta itself, sharing the x-axis.

The histogram matters, not just the curve: when weights are small
(early training), Theta clusters near 0 where tanh is ~linear, so beta
is barely identifiable from the data -- any beta fits comparably well.
Once weights grow, Theta spreads into the saturating tails and beta
becomes well-constrained. This is why beta_x wobbles early and settles
late in scripts/viz/plot_cem_dynamics.py.

Trains via a single persistent Trainer object, advanced in two calls to
train(start_iteration=...), so the RBM/optimizer/beta_x state carries
over between snapshots without restarting from scratch (see
src/encoder.py:Trainer.train).

Usage (from project root):
    python scripts/exper/cem_mechanism.py
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "viz"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encoder import Trainer, estimate_beta_eff_cem
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from plot_style import setup_style

PLOTS_DIR = _REPO / "plots" / "cem_mechanism"

N = 16
H_FIELD = 1.0
SEED = 1
LEARNING_RATE = 0.01
N_SAMPLES_TRAIN = 1000
SNAPSHOT_ITERS = [20, 300]  # (early, converged)
SNAPSHOT_N_SAMPLES = 5000   # larger batch for a clean response curve
N_BINS = 40
MIN_BIN_COUNT = 30

COLOR_FIT = "#0072B2"     # CEM-fitted tanh(beta*Theta)
COLOR_NAIVE = "#555555"   # naive tanh(1*Theta)
COLOR_DATA = "#D55E00"    # empirical binned <h>


def build_trainer():
    key = jax.random.PRNGKey(SEED)
    key, model_key, sampler_key = jax.random.split(key, 3)

    rbm = FullyConnectedRBM(N, N, model_key)
    ising = TransverseFieldIsing1D(N, H_FIELD)
    sampler = ClassicalSampler(method="lsb")
    sampler._key = sampler_key

    config = {
        "learning_rate": LEARNING_RATE,
        "n_iterations": SNAPSHOT_ITERS[-1],
        "n_samples": N_SAMPLES_TRAIN,
        "regularization": 1e-5,
        "use_cem": True,
        "cem_interval": 5,
        "lsb_sigma": 1.0,
        "seed": SEED,
    }
    return Trainer(rbm, ising, sampler, config, args=None)


def take_snapshot(trainer):
    """Freeze current RBM state, draw a fresh large LSB batch, fit beta."""
    snap_config = {**trainer.config, "beta_x": trainer.beta_x}
    V, H = trainer.sampler.sample(
        trainer.rbm, SNAPSHOT_N_SAMPLES, snap_config, return_hidden=True
    )
    V, H = jnp.asarray(V, dtype=jnp.float64), jnp.asarray(H, dtype=jnp.float64)
    Theta = V @ trainer.rbm.W + trainer.rbm.b[None, :]
    beta_fit = estimate_beta_eff_cem(V, H, trainer.rbm)
    return {
        "beta_input": trainer.beta_x,
        "beta_fit": beta_fit,
        "theta": np.asarray(Theta).ravel(),
        "h": np.asarray(H).ravel(),
    }


def collect_snapshots():
    trainer = build_trainer()
    snapshots = []
    start = 0
    for target_iter in SNAPSHOT_ITERS:
        trainer.n_iterations = target_iter
        trainer.train(start_iteration=start)
        start = target_iter
        snap = take_snapshot(trainer)
        snap["iteration"] = target_iter
        snapshots.append(snap)
        print(
            f"  iter={target_iter:4d}  beta_input={snap['beta_input']:.3f}  "
            f"beta_fit={snap['beta_fit']:.3f}"
        )
    return snapshots


def _binned_response(theta, h, n_bins, min_count):
    edges = np.linspace(theta.min(), theta.max(), n_bins + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    centers, means, ses, counts = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c < min_count:
            continue
        counts.append(c)
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        m = float(h[mask].mean())
        means.append(m)
        ses.append(float(np.sqrt(max(1e-12, 1 - m**2) / c)))
    return np.array(centers), np.array(means), np.array(ses), np.array(counts)


def plot_mechanism(snapshots, fname):
    setup_style(fontsize=10, scale=1.0)
    fig, axes = plt.subplots(
        2, len(snapshots), figsize=(4.2 * len(snapshots), 4.6),
        sharex=False, height_ratios=[2.5, 1],
    )

    for col, snap in enumerate(snapshots):
        theta, h = snap["theta"], snap["h"]
        ax_curve, ax_hist = axes[0, col], axes[1, col]

        centers, means, ses, counts = _binned_response(theta, h, N_BINS, MIN_BIN_COUNT)
        ax_curve.errorbar(
            centers, means, yerr=ses, fmt="o", ms=3.5, color=COLOR_DATA,
            ecolor=COLOR_DATA, elinewidth=0.8, capsize=1.5, zorder=3,
            label=r"empirical $\langle h \rangle$",
        )

        grid = np.linspace(theta.min(), theta.max(), 300)
        ax_curve.plot(grid, np.tanh(snap["beta_fit"] * grid), color=COLOR_FIT,
                      lw=1.8, zorder=2, label=rf"CEM fit ($\beta$={snap['beta_fit']:.2f})")
        ax_curve.plot(grid, np.tanh(1.0 * grid), color=COLOR_NAIVE, lw=1.4,
                      ls="--", zorder=1, label=r"naive $\beta=1$")

        ax_curve.set_title(
            f"iteration {snap['iteration']}\n"
            rf"$\beta_x$ fed to sampler = {snap['beta_input']:.2f}"
        )
        ax_curve.set_ylim(-1.08, 1.08)
        if col == 0:
            ax_curve.set_ylabel(r"$\langle h \rangle$")
        ax_curve.legend(loc="lower right", fontsize=6.5, frameon=True, edgecolor="black")

        ax_hist.hist(theta, bins=60, color="#999999", edgecolor="none")
        ax_hist.set_xlabel(r"local field $\Theta = v \cdot W + b$")
        if col == 0:
            ax_hist.set_ylabel("count")
        ax_hist.set_xlim(ax_curve.get_xlim())

    fig.suptitle(
        f"What CEM fits: TFIM 1D, N={N}, h={H_FIELD:g} -- response curve and field density",
        y=1.04, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Training TFIM 1D N={N} h={H_FIELD} with CEM, snapshotting at {SNAPSHOT_ITERS}...")
    snapshots = collect_snapshots()
    plot_mechanism(snapshots, "cem_mechanism_tfim1d_n16")
