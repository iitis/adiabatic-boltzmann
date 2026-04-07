"""
KL divergence utilities for evaluating sampler quality.

Works for any model by accepting an energy function. Handles two cases:
  - Visible-only samples (Metropolis, SA): N = n_visible, enumerate 2^N_v states
  - Joint (v, h) samples (LSB, Gibbs):    N = n_visible + n_hidden, enumerate 2^N states

All spin configurations are over {-1, +1}.

All KL computations are done in log space to avoid underflow for large N.
"""

import numpy as np
from itertools import product
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


def all_configs(N: int) -> np.ndarray:
    """Return all 2^N spin configurations in {-1, +1}^N, shape (2^N, N)."""
    return np.array(list(product([-1, 1], repeat=N)), dtype=np.float64)


def log_boltzmann(energy_fn, configs: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Compute log B_β(s) for every configuration, normalised (log-probabilities).

    log B_β(s) = -β E(s) - log Z_β

    Computed stably via log-sum-exp; safe even when individual exp() would underflow.
    Returns: (2^N,) array of log-probabilities.
    """
    energies = np.array([energy_fn(s) for s in configs])
    log_unnorm = -beta * energies
    log_Z = _logsumexp(log_unnorm)
    return log_unnorm - log_Z


def empirical_dist(samples: np.ndarray, configs: np.ndarray) -> np.ndarray:
    """
    Build empirical distribution P_S from samples.

    samples: (n_samples, N) — observed spin vectors (float32 or float64)
    configs: (2^N, N)       — reference array from all_configs

    Returns: (2^N,) probability array. Unobserved states get probability 0.
    """
    # Cast to float64 for consistent hashing against configs
    samples = np.asarray(samples, dtype=np.float64)
    counts = np.zeros(len(configs))
    config_to_idx = {tuple(c): i for i, c in enumerate(configs)}
    for s in samples:
        key = tuple(s)
        if key in config_to_idx:
            counts[config_to_idx[key]] += 1
    n = counts.sum()
    if n == 0:
        raise ValueError("No samples matched any configuration — check spin convention (±1).")
    return counts / n


def kl_divergence(p_s: np.ndarray, log_b_beta: np.ndarray) -> float:
    """
    D_KL(P_S ∥ B_β) = Σ_s P_S(s) [log P_S(s) − log B_β(s)]

    Works entirely in log space — no underflow for large N.
    Terms where P_S(s) = 0 are skipped (0·log 0 = 0 by convention).

    p_s:        (2^N,) empirical probabilities
    log_b_beta: (2^N,) log-probabilities from log_boltzmann()
    """
    mask = p_s > 0
    return float(np.sum(p_s[mask] * (np.log(p_s[mask]) - log_b_beta[mask])))


# ---------------------------------------------------------------------------
# β_eff estimation
# ---------------------------------------------------------------------------


def estimate_beta_kl(
    energy_fn,
    configs: np.ndarray,
    samples: np.ndarray,
    beta_bounds: tuple = (1e-2, 50.0),
) -> float:
    """
    β_eff = argmin_{β>0} D_KL(P_S ∥ B_β), via bounded scalar minimisation.

    energy_fn:   callable (N,) → float
    configs:     (2^N, N) from all_configs(N)
    samples:     (n_samples, N) from sampler
    beta_bounds: search interval — upper bound of 50 avoids underflow
    """
    p_s = empirical_dist(samples, configs)
    energies = np.array([energy_fn(s) for s in configs])

    def objective(beta):
        log_unnorm = -beta * energies
        log_Z = _logsumexp(log_unnorm)
        log_b = log_unnorm - log_Z
        return kl_divergence(p_s, log_b)

    result = minimize_scalar(objective, bounds=beta_bounds, method="bounded")
    return float(result.x)


def sampling_accuracy(
    energy_fn,
    configs: np.ndarray,
    samples: np.ndarray,
    beta_eff: float = None,
) -> tuple[float, float]:
    """
    Compute (D_KL(P_S ∥ B_{β_eff}), β_eff).
    If beta_eff is None it is estimated by KL minimisation first.
    """
    if beta_eff is None:
        beta_eff = estimate_beta_kl(energy_fn, configs, samples)
    p_s = empirical_dist(samples, configs)
    log_b = log_boltzmann(energy_fn, configs, beta_eff)
    return kl_divergence(p_s, log_b), beta_eff


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _logsumexp(a: np.ndarray) -> float:
    c = a.max()
    return float(c + np.log(np.sum(np.exp(a - c))))
