"""
KL divergence / D_TV utilities for evaluating sampler quality (JAX, GPU-native).

Works for any model by accepting an RBM's log_psi. Handles two cases:
  - Visible-only samples (Metropolis, SA): N = n_visible, enumerate 2^N_v states
  - Joint (v, h) samples (Gibbs):         N = n_visible + n_hidden, enumerate 2^N states

All spin configurations are over {-1, +1}. Keeps data on GPU and avoids CPU
round-trips in hot loops.
"""

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# JAX variants — GPU-native, no CPU round-trips in hot loops
# ---------------------------------------------------------------------------


def all_configs_jax(N: int) -> jax.Array:
    """All 2^N spin configs in {-1, +1}^N via bit manipulation. Shape (2^N, N)."""
    idx = jnp.arange(2**N, dtype=jnp.int32)
    bits = ((idx[:, None] >> jnp.arange(N - 1, -1, -1, dtype=jnp.int32)) & 1).astype(
        jnp.float64
    )
    return 2.0 * bits - 1.0


def exact_psi_sq(rbm, N: int) -> jax.Array:
    """
    Exact |Ψ(v)|² / Z for all 2^N visible configs. Shape (2^N,).

    Uses jax.vmap over rbm.log_psi — works for FullyConnectedRBM and
    DWaveTopologyRBM without changes.
    """
    configs = all_configs_jax(N)
    log2psi = jax.vmap(rbm.log_psi)(configs) * 2.0
    log2psi -= jax.scipy.special.logsumexp(log2psi)
    return jnp.exp(log2psi)


def empirical_dist_jax(samples: jax.Array, N: int) -> jax.Array:
    """
    Empirical distribution from samples. Shape (2^N,).

    samples: (n_samples, N) JAX array of {-1, +1} spins.
    Encodes each row as a binary integer — no Python loop, no dict.
    """
    bits = ((samples + 1.0) / 2.0).astype(jnp.int32)
    powers = (2 ** jnp.arange(N - 1, -1, -1, dtype=jnp.int32))
    indices = bits @ powers
    counts = jnp.zeros(2**N, dtype=jnp.float64).at[indices].add(1.0)
    return counts / counts.sum()


def d_tv(p: jax.Array, q: jax.Array) -> jax.Array:
    """D_TV(p, q) = ½ Σ |p - q|. Returns a scalar JAX array."""
    return 0.5 * jnp.sum(jnp.abs(p - q))


def finite_sampling_floor(p_exact: jax.Array, n_samples: int, n_trials: int = 20) -> float:
    """
    Average D_TV achievable by drawing n_samples i.i.d. from p_exact itself.

    This is the irreducible floor set by finite sampling — the best any
    perfect sampler can do with this many samples.  Used as the dashed
    reference line in D_TV plots.
    """
    p_np = np.asarray(p_exact)
    n_states = len(p_np)
    vals = []
    for _ in range(n_trials):
        idx = np.random.choice(n_states, size=n_samples, p=p_np)
        counts = np.bincount(idx, minlength=n_states).astype(float)
        q = counts / counts.sum()
        vals.append(0.5 * float(np.sum(np.abs(p_np - q))))
    return float(np.mean(vals))
