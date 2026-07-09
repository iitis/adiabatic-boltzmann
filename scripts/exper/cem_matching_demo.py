"""
cem_matching_demo.py -- isolates the "matching" step of CEM, nothing else.

Trains one TFIM 1D RBM (N=16, h=1) with CEM using the production LSB config
(lsb_steps=100, lsb_delta=1.0, lsb_sigma=1.0 -- matching results/tfim_1d/16/
custom/lsb/*.json.gz, NOT cem_mechanism.py's steps=1000/delta=0.1 defaults,
which drive the sampler into a different regime and give beta values ~4x
smaller than what training actually uses). Snapshots at iteration 300
(converged) and produces two SEPARATE standalone figures:

  cem_matching_candidates : the empirical <h> vs Theta response ("the
      distribution of observed hidden units") against several CANDIDATE
      tanh(beta*Theta) curves ("the calculated conditional expectation
      value") for different guesses of beta. Only one of them hugs the data.
  cem_matching_objective  : the matching objective itself,
      F(beta) = sum((h_observed - tanh(beta*Theta))^2), plotted over a grid
      of beta -- the literal loss landscape scipy.optimize.minimize_scalar
      searches inside estimate_beta_eff_cem(). Its minimum IS beta_eff.
      Colored dots mark where each candidate from the first figure sits on
      this curve.

Usage (from project root):
    python scripts/exper/cem_matching_demo.py
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

PLOTS_DIR = _REPO / "plots" / "cem"

N = 16
H_FIELD = 1.0
SEED = 1
LEARNING_RATE = 0.01
REGULARIZATION = 1e-5
N_SAMPLES_TRAIN = 1000
N_ITERATIONS = 300
CEM_INTERVAL = 5
LSB_STEPS = 100      # production config -- see results/tfim_1d/16/custom/lsb/*.json.gz
LSB_DELTA = 1.0
LSB_SIGMA = 1.0

SNAPSHOT_N_SAMPLES = 5000
N_BINS = 40
MIN_BIN_COUNT = 30

CANDIDATE_BETAS = [0.5, 1.0, 5.0]   # "wrong" guesses shown alongside the fit
COLOR_CANDIDATES = ["#cccccc", "#888888", "#444444"]  # light -> dark gray, magnitude-ordered
COLOR_DATA = "#D55E00"
COLOR_FIT = "#0072B2"


def build_trainer():
    key = jax.random.PRNGKey(SEED)
    key, model_key, sampler_key = jax.random.split(key, 3)
    rbm = FullyConnectedRBM(N, N, model_key)
    ising = TransverseFieldIsing1D(N, H_FIELD)
    sampler = ClassicalSampler(method="lsb")
    sampler._key = sampler_key
    config = {
        "learning_rate": LEARNING_RATE,
        "n_iterations": N_ITERATIONS,
        "n_samples": N_SAMPLES_TRAIN,
        "regularization": REGULARIZATION,
        "use_cem": True,
        "cem_interval": CEM_INTERVAL,
        "lsb_steps": LSB_STEPS,
        "lsb_delta": LSB_DELTA,
        "lsb_sigma": LSB_SIGMA,
        "seed": SEED,
    }
    return Trainer(rbm, ising, sampler, config, args=None)


def collect_snapshot():
    trainer = build_trainer()
    trainer.train()
    snap_config = {**trainer.config, "beta_x": trainer.beta_x}
    V, H = trainer.sampler.sample(
        trainer.rbm, SNAPSHOT_N_SAMPLES, snap_config, return_hidden=True
    )
    V, H = jnp.asarray(V, dtype=jnp.float64), jnp.asarray(H, dtype=jnp.float64)
    Theta = V @ trainer.rbm.W + trainer.rbm.b[None, :]
    beta_fit = estimate_beta_eff_cem(V, H, trainer.rbm)
    return {
        "beta_fit": beta_fit,
        "theta": np.asarray(Theta).ravel(),
        "h": np.asarray(H).ravel(),
    }


def _binned_response(theta, h, n_bins, min_count):
    edges = np.linspace(theta.min(), theta.max(), n_bins + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    centers, means, ses = [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c < min_count:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        m = float(h[mask].mean())
        means.append(m)
        ses.append(float(np.sqrt(max(1e-12, 1 - m**2) / c)))
    return np.array(centers), np.array(means), np.array(ses)


def _loss_curve(theta, h, beta_grid):
    return np.array([np.sum((h - np.tanh(b * theta)) ** 2) for b in beta_grid])


def _save(fig, fname):
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_candidates(snap):
    setup_style(fontsize=9)
    theta, h, beta_fit = snap["theta"], snap["h"], snap["beta_fit"]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))

    centers, means, ses = _binned_response(theta, h, N_BINS, MIN_BIN_COUNT)
    ax.errorbar(centers, means, yerr=ses, fmt="o", ms=3.0, color=COLOR_DATA,
                ecolor=COLOR_DATA, elinewidth=0.8, capsize=1.5, zorder=4,
                label="observed hidden units\n(binned average)")

    grid = np.linspace(theta.min(), theta.max(), 300)
    for beta, color in zip(CANDIDATE_BETAS, COLOR_CANDIDATES):
        ax.plot(grid, np.tanh(beta * grid), color=color, lw=1.3, ls="--", zorder=2,
                 label=rf"candidate $\beta={beta:g}$")
    ax.plot(grid, np.tanh(beta_fit * grid), color=COLOR_FIT, lw=2.2, zorder=3,
             label=rf"matched fit, $\beta_{{\rm eff}}={beta_fit:.2f}$")

    ax.set_xlabel(r"local field $\Theta = v \cdot W + b$")
    ax.set_ylabel(r"$\langle h \rangle$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(theta.min(), theta.max())
    ax.legend(loc="lower right", fontsize=6, frameon=True, edgecolor="black",
              handlelength=1.6, borderpad=0.4)

    _save(fig, "cem_matching_candidates")


def plot_objective(snap):
    setup_style(fontsize=9)
    theta, h, beta_fit = snap["theta"], snap["h"], snap["beta_fit"]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))

    beta_grid = np.geomspace(0.05, 10.0, 300)
    F = _loss_curve(theta, h, beta_grid)
    ax.plot(beta_grid, F, color="black", lw=1.6, zorder=2)
    ax.set_xscale("log")

    for beta, color in zip(CANDIDATE_BETAS, COLOR_CANDIDATES):
        f_val = np.sum((h - np.tanh(beta * theta)) ** 2)
        ax.scatter([beta], [f_val], color=color, s=28, zorder=4, edgecolors="black", linewidths=0.4)
        ax.annotate(rf"$\beta={beta:g}$", (beta, f_val), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6.5, color=color)

    f_fit = np.sum((h - np.tanh(beta_fit * theta)) ** 2)
    ax.scatter([beta_fit], [f_fit], color=COLOR_FIT, s=55, marker="*", zorder=5,
               edgecolors="black", linewidths=0.5, label=rf"minimum: $\beta_{{\rm eff}}={beta_fit:.2f}$")
    ax.axvline(beta_fit, color=COLOR_FIT, lw=0.8, ls=":", alpha=0.7, zorder=1)

    ax.set_xlabel(r"candidate $\beta$")
    ax.set_ylabel(r"$F(\beta) = \sum \left(h_{\rm observed} - \tanh(\beta\Theta)\right)^2$")
    ax.legend(loc="upper center", fontsize=7, frameon=True, edgecolor="black")

    _save(fig, "cem_matching_objective")


if __name__ == "__main__":
    print(f"Training TFIM 1D N={N} h={H_FIELD} to iteration {N_ITERATIONS}...")
    snap = collect_snapshot()
    print(f"  beta_eff (matched) = {snap['beta_fit']:.3f}")
    plot_candidates(snap)
    plot_objective(snap)
