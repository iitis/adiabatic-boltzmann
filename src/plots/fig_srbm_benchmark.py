"""
Reproduction of Figure 2 from Kubo & Goto (arXiv:2512.02323).

Compares LSB and Gibbs sampling accuracy on 10 random SRBM instances:
  - Panel (a): D_KL(P_S ∥ B_{β_eff}) per instance + mean ± SE
  - Panel (b): β_eff per instance (KL minimization and CEM)

SRBM (Semi-Restricted Boltzmann Machine):
  - V-V couplings (symmetric, zero diagonal): N_v × N_v matrix V
  - V-H couplings: N_v × N_h matrix W
  - No H-H couplings
  - Energy: E(v, h) = -1/2 v^T V v - v^T W h  (biases set to 0)

Run from src/:
    python plots/fig_srbm_benchmark.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar

from kl_utils import all_configs, log_boltzmann, empirical_dist, kl_divergence, estimate_beta_kl
from sampler import ClassicalSampler


# ---------------------------------------------------------------------------
# SRBM dataclass — exposes the same interface as RBM so ClassicalSampler
# (and its LSB code) can consume it directly.
# ---------------------------------------------------------------------------

@dataclass
class SRBM:
    """
    Semi-Restricted Boltzmann Machine.
      V: (n_visible, n_visible) symmetric, zero diagonal — visible-visible couplings
      W: (n_visible, n_hidden)  — visible-hidden couplings
      a: (n_visible,)           — visible biases  (zero for Fig. 2)
      b: (n_hidden,)            — hidden biases   (zero for Fig. 2)

    The ClassicalSampler._lsb_sample reads rbm.V, rbm.W, rbm.a, rbm.b directly,
    so no wrapper is needed.
    """
    V: np.ndarray          # (Nv, Nv)
    W: np.ndarray          # (Nv, Nh)
    a: np.ndarray = field(default=None)   # (Nv,), defaults to zeros
    b: np.ndarray = field(default=None)   # (Nh,), defaults to zeros

    def __post_init__(self):
        if self.a is None:
            self.a = np.zeros(self.W.shape[0])
        if self.b is None:
            self.b = np.zeros(self.W.shape[1])

    @property
    def n_visible(self): return self.W.shape[0]
    @property
    def n_hidden(self):  return self.W.shape[1]
    @property
    def N(self):         return self.n_visible + self.n_hidden

    def energy(self, s: np.ndarray) -> float:
        """E(v, h) = -1/2 v^T V v - v^T W h - a·v - b·h"""
        v, h = s[:self.n_visible], s[self.n_visible:]
        return -0.5 * v @ self.V @ v - v @ self.W @ h - self.a @ v - self.b @ h


def random_srbm(n_visible: int, n_hidden: int, rng: np.random.Generator) -> SRBM:
    """Random SRBM instance as in the paper: weights ~ N(0, 2/√N), biases = 0."""
    N = n_visible + n_hidden
    scale = 2.0 / np.sqrt(N)

    W = rng.normal(0, scale, (n_visible, n_hidden))

    V_raw = rng.normal(0, scale, (n_visible, n_visible))
    V = (V_raw + V_raw.T) / 2          # symmetrise
    np.fill_diagonal(V, 0.0)

    return SRBM(V=V, W=W)


# ---------------------------------------------------------------------------
# Gibbs sampler for SRBM (single-site for v, block for h)
# ---------------------------------------------------------------------------

def gibbs_sample_srbm(
    srbm: SRBM,
    n_samples: int,
    n_warmup: int = 200,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Gibbs sampler for SRBM targeting B_1(v, h).

    Runs n_samples independent chains — each sample starts from a fresh random
    init and is drawn after n_warmup sweeps, eliminating autocorrelation.

    p(h_j = +1 | v) = σ(2*(b_j + W[:,j]·v))         — block update
    p(v_i = +1 | v_{-i}, h) = σ(2*(a_i + V[i,:]·v + W[i,:]·h))  — single-site

    Returns joint (v, h) samples, shape (n_samples, N).
    """
    if rng is None:
        rng = np.random.default_rng()

    Nv, Nh = srbm.n_visible, srbm.n_hidden
    V, W, a, b = srbm.V, srbm.W, srbm.a, srbm.b

    def sample_h_given_v(v):
        prob = 1.0 / (1.0 + np.exp(-2.0 * (b + W.T @ v)))
        return np.where(rng.random(Nh) < prob, 1.0, -1.0)

    def sweep_v_given_h(v, h):
        v = v.copy()
        for i in range(Nv):
            field = a[i] + V[i, :] @ v + W[i, :] @ h
            prob_plus = 1.0 / (1.0 + np.exp(-2.0 * field))
            v[i] = 1.0 if rng.random() < prob_plus else -1.0
        return v

    samples = np.empty((n_samples, Nv + Nh))
    for i in range(n_samples):
        v = rng.choice([-1.0, 1.0], size=Nv)
        h = sample_h_given_v(v)
        for _ in range(n_warmup):
            v = sweep_v_given_h(v, h)
            h = sample_h_given_v(v)
        samples[i, :Nv] = v
        samples[i, Nv:] = h

    return samples


# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler for SRBM (single-spin-flip on joint (v,h))
# ---------------------------------------------------------------------------

def metropolis_sample_srbm(
    srbm: SRBM,
    n_samples: int,
    n_warmup: int = 200,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Single-spin-flip Metropolis-Hastings on joint (v,h) targeting B_1.

    Runs n_samples independent chains — each sample starts from a fresh random
    init and is drawn after n_warmup full sweeps, eliminating autocorrelation.

    ΔE for flipping spin k in joint vector s = (v, h):
      k < Nv: ΔE = 2·s[k]·(V[k,:]·v + W[k,:]·h + a[k])
      k ≥ Nv: ΔE = 2·s[k]·(W[:,k-Nv]·v + b[k-Nv])

    Returns joint samples shape (n_samples, Nv+Nh).
    """
    if rng is None:
        rng = np.random.default_rng()

    Nv, Nh = srbm.n_visible, srbm.n_hidden
    N = Nv + Nh
    V, W, a, b = srbm.V, srbm.W, srbm.a, srbm.b

    def delta_energy(s: np.ndarray, k: int) -> float:
        if k < Nv:
            return 2.0 * s[k] * (V[k, :] @ s[:Nv] + W[k, :] @ s[Nv:] + a[k])
        j = k - Nv
        return 2.0 * s[k] * (W[:, j] @ s[:Nv] + b[j])

    samples = np.empty((n_samples, N))
    for i in range(n_samples):
        s = rng.choice([-1.0, 1.0], size=N)
        for _ in range(n_warmup):
            for k in rng.permutation(N):
                dE = delta_energy(s, k)
                if dE <= 0.0 or rng.random() < np.exp(-dE):
                    s[k] = -s[k]
        samples[i] = s.copy()

    return samples


# ---------------------------------------------------------------------------
# CEM β_eff estimation (joint-samples variant, for LSB on SRBM)
# ---------------------------------------------------------------------------

def cem_beta_eff_srbm(srbm: SRBM, v_samples: np.ndarray, h_samples: np.ndarray) -> float:
    """
    Find β minimising Σ_{l,j} (h^(l)_j - tanh(β * a^(l)_j))²
    where a^(l)_j = b_j + W[:,j]·v^(l).
    """
    activation = v_samples @ srbm.W + srbm.b[None, :]  # (n_samples, Nh)

    def objective(beta):
        return float(np.sum((h_samples - np.tanh(beta * activation)) ** 2))

    result = minimize_scalar(objective, bounds=(1e-2, 10.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_figure2(
    n_instances: int = 10,
    n_visible: int = 10,
    n_hidden: int = 5,
    n_samples: int = 100,
    lsb_steps: int = 100,
    lsb_delta: float = 1.0,
    mcmc_warmup: int = 200,
    sigma_inv2_candidates: list | None = None,
    seed: int = 0,
):
    if sigma_inv2_candidates is None:
        sigma_inv2_candidates = [round(0.5 + 0.1 * k, 1) for k in range(16)]  # 0.5 … 2.0

    rng = np.random.default_rng(seed)
    N = n_visible + n_hidden
    configs = all_configs(N)

    lsb_sampler = ClassicalSampler(method="lsb")

    results = []

    for inst in range(n_instances):
        srbm = random_srbm(n_visible, n_hidden, rng)
        energy_fn = srbm.energy
        print(f"\n--- Instance {inst + 1}/{n_instances} ---")

        # --- Gibbs ---
        print("  Gibbs sampling...")
        gibbs_samples = gibbs_sample_srbm(srbm, n_samples, n_warmup=mcmc_warmup, rng=rng)
        gibbs_kl = kl_divergence(
            empirical_dist(gibbs_samples, configs),
            log_boltzmann(energy_fn, configs, 1.0),
        )
        print(f"  Gibbs: β_eff=1.000  KL={gibbs_kl:.4f}")

        # --- Metropolis ---
        print("  Metropolis sampling...")
        metro_samples = metropolis_sample_srbm(srbm, n_samples, n_warmup=mcmc_warmup, rng=rng)
        metro_kl = kl_divergence(
            empirical_dist(metro_samples, configs),
            log_boltzmann(energy_fn, configs, 1.0),
        )
        print(f"  Metropolis: β_eff=1.000  KL={metro_kl:.4f}")

        # --- LSB: optimise σ over candidates ---
        print("  LSB: optimising σ...")
        best_lsb_kl = float("inf")
        best_lsb_beta_eff_kl = None
        best_lsb_beta_eff_cem = None
        best_lsb_kl_cem = None
        best_joint = None

        for sigma_inv2 in sigma_inv2_candidates:
            sigma = 1.0 / np.sqrt(sigma_inv2)
            config = {"lsb_steps": lsb_steps, "lsb_delta": lsb_delta, "lsb_sigma": sigma, "beta_x": 1.0}
            v_s, h_s = lsb_sampler.sample(srbm, n_samples, config=config, return_hidden=True)
            joint = np.concatenate([v_s, h_s], axis=1)

            beta_eff_kl = estimate_beta_kl(energy_fn, configs, joint)
            kl = kl_divergence(
                empirical_dist(joint, configs),
                log_boltzmann(energy_fn, configs, beta_eff_kl),
            )

            if kl < best_lsb_kl:
                best_lsb_kl = kl
                best_lsb_beta_eff_kl = beta_eff_kl
                best_lsb_beta_eff_cem = cem_beta_eff_srbm(srbm, v_s, h_s)
                best_joint = joint

        # KL at CEM β_eff for the best sigma
        assert best_joint is not None and best_lsb_beta_eff_cem is not None
        best_lsb_kl_cem = kl_divergence(
            empirical_dist(best_joint, configs),
            log_boltzmann(energy_fn, configs, best_lsb_beta_eff_cem),
        )

        print(f"  LSB:  β_eff(KL)={best_lsb_beta_eff_kl:.3f}  KL(KL)={best_lsb_kl:.4f}"
              f"  β_eff(CEM)={best_lsb_beta_eff_cem:.3f}  KL(CEM)={best_lsb_kl_cem:.4f}")

        results.append({
            "gibbs_kl":          gibbs_kl,
            "metro_kl":          metro_kl,
            "lsb_kl":            best_lsb_kl,
            "lsb_kl_cem":        best_lsb_kl_cem,
            "lsb_beta_eff_kl":   best_lsb_beta_eff_kl,
            "lsb_beta_eff_cem":  best_lsb_beta_eff_cem,
        })

    return results


def plot_figure2(results: list, save_path: str | None = None, beff_ylim: tuple = (0.0, 4.0)):
    n = len(results)
    xs = np.arange(1, n + 1)

    gibbs_kls    = np.array([r["gibbs_kl"]         for r in results])
    metro_kls    = np.array([r["metro_kl"]          for r in results])
    lsb_kls      = np.array([r["lsb_kl"]            for r in results])
    lsb_kls_cem  = np.array([r["lsb_kl_cem"]        for r in results])
    lsb_beff_kl  = np.array([r["lsb_beta_eff_kl"]  for r in results])
    lsb_beff_cem = np.array([r["lsb_beta_eff_cem"] for r in results])

    bar_w = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # --- Panel (a): KL divergence — grouped bars (4 methods) ---
    ax = axes[0]
    ax.bar(xs - 1.5*bar_w, gibbs_kls,   width=bar_w, color="tab:blue",   label="Gibbs",       alpha=0.85)
    ax.bar(xs - 0.5*bar_w, metro_kls,   width=bar_w, color="tab:green",  label="Metropolis",  alpha=0.85)
    ax.bar(xs + 0.5*bar_w, lsb_kls,     width=bar_w, color="tab:orange", label="LSB (KL)",    alpha=0.85)
    ax.bar(xs + 1.5*bar_w, lsb_kls_cem, width=bar_w, color="tab:red",    label="LSB (CEM)",   alpha=0.85)

    for kls, color in [(gibbs_kls, "tab:blue"), (metro_kls, "tab:green"),
                       (lsb_kls, "tab:orange"), (lsb_kls_cem, "tab:red")]:
        mean = kls.mean()
        se   = kls.std() / np.sqrt(n)
        ax.axhline(mean, color=color, lw=1.5, ls="--")
        ax.axhspan(mean - se, mean + se, color=color, alpha=0.12)

    ax.set_xlabel("Instance")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P_S \| B_{\beta_{\mathrm{eff}}})$")
    ax.set_title("(a) Sampling accuracy")
    ax.legend()
    ax.set_xticks(xs)
    ax.set_xlim(0.5, n + 0.5)

    # --- Panel (b): β_eff — Gibbs and Metropolis fixed at 1, LSB estimated ---
    ax = axes[1]
    bar_w3 = 0.2
    ax.bar(xs - bar_w3,      np.ones(n), width=bar_w3, color="tab:blue",   label="Gibbs (fixed 1)", alpha=0.85)
    ax.bar(xs,               np.ones(n), width=bar_w3, color="tab:green",  label="Metropolis (fixed 1)", alpha=0.85)
    ax.bar(xs + bar_w3,      lsb_beff_kl,  width=bar_w3, color="tab:orange", label="LSB (KL)",   alpha=0.85)
    ax.bar(xs + 2 * bar_w3,  lsb_beff_cem, width=bar_w3, color="tab:red",    label="LSB (CEM)",  alpha=0.85)
    ax.axhline(1.0, color="black", lw=1.2, ls=":", label=r"$\beta=1$ (target)")

    ax.set_ylim(*beff_ylim)
    ax.set_xlabel("Instance")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}$")
    ax.set_title(r"(b) Effective inverse temperature $\beta_{\mathrm{eff}}$")
    ax.legend(fontsize=7)
    ax.set_xticks(xs)
    ax.set_xlim(0.5, n + 0.5)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    from pathlib import Path
    out_dir = Path(__file__).resolve().parent.parent.parent / "figures" / "fig_srbm_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_figure2()
    plot_figure2(results, save_path=str(out_dir / "fig_srbm_benchmark.png"))
