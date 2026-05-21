"""
Deep Boltzmann Machine (DBM) wave function ansatz — JAX backend.

L hidden layers in sequence: v → h¹ → h² → ... → h^L

The joint distribution over (v, h¹, ..., h^L) is quadratic in all binary
variables, so it maps directly to a QUBO and can be sampled on D-Wave.

log|Ψ(v)| requires marginalising over all hidden layers, which is intractable
for L ≥ 2 and is approximated via mean-field:

    log|Ψ|_MF(v) = -a·v/2
                   + ½ Σ_l Σ_j log[2 cosh(α^l_j)]
                   - ½ Σ_l μ^l · W^{l+1} · μ^{l+1}

where α^l and μ^l = tanh(α^l) satisfy the fixed-point equations:

    α^1   = (W^1)ᵀv  + b^1 + W^2 μ^2
    α^l   = (W^l)ᵀμ^{l-1} + b^l + W^{l+1} μ^{l+1}   (1 < l < L)
    α^L   = (W^L)ᵀμ^{L-1} + b^L

converged after n_mf_steps Jacobi iterations from μ = 0.  For L = 1 the
approximation is exact and reduces to the standard RBM log_psi.

Derivation of the double-counting correction (½ μ^l · W^{l+1} · μ^{l+1}):
The variational free energy ELBO for factored q(h¹,...,h^L) = ∏ q^l simplifies to
  ELBO = -a·v/2 + Σ_l log 2cosh(α^l) - Σ_l μ^l · W^{l+1} · μ^{l+1}
The coupling μ^l · W^{l+1} · μ^{l+1} appears in α^l via W^{l+1}μ^{l+1} and in
α^{l+1} via (W^{l+1})ᵀμ^l, so the log-cosh sum double-counts it; the correction
removes one copy.

Interface matches ViTWaveFunction — drop-in compatible with TrainerGeneric,
SRLinearSystemGeneric, and GenericClassicalSampler.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


class DBMParams(NamedTuple):
    """
    Immutable DBM parameter container, automatically a JAX pytree.

    a : (n_v,)                          visible biases
    b : tuple of (n_hl,) arrays         one per hidden layer
    W : tuple of (n_{l}, n_{l+1}) arrays
        W[0] : (n_v,  n_h1)
        W[1] : (n_h1, n_h2)
        ...
    """

    a: jax.Array
    b: tuple
    W: tuple


# ---------------------------------------------------------------------------
# DBM model
# ---------------------------------------------------------------------------


class DeepBoltzmannMachine:
    """
    Deep Boltzmann Machine wave function ansatz.

    Parameters
    ----------
    n_visible    : number of visible spins
    hidden_sizes : list of hidden layer sizes, e.g. [8, 4] for two layers
    key          : JAX PRNG key for weight initialisation
    scale        : std-dev of initial weights (default 0.01)
    n_mf_steps   : mean-field fixed-point iterations (default 10)

    Interface matches ViTWaveFunction for drop-in use with TrainerGeneric.
    """

    def __init__(
        self,
        n_visible: int,
        hidden_sizes: list[int],
        key: jax.Array,
        scale: float = 0.01,
        n_mf_steps: int = 10,
    ):
        if not hidden_sizes:
            raise ValueError("hidden_sizes must have at least one layer.")

        self.n_visible = n_visible
        self.hidden_sizes = list(hidden_sizes)
        self.n_layers = len(hidden_sizes)
        self.n_hidden = hidden_sizes[-1]   # last-layer size; used for D-Wave node count
        self.scale = scale
        self.n_mf_steps = n_mf_steps

        sizes = [n_visible] + hidden_sizes
        keys = jax.random.split(key, self.n_layers + 1)

        a = jnp.zeros(n_visible, dtype=jnp.float64)
        b = tuple(
            jnp.zeros(hidden_sizes[l], dtype=jnp.float64)
            for l in range(self.n_layers)
        )
        W = tuple(
            jax.random.normal(
                keys[l + 1], (sizes[l], sizes[l + 1]), dtype=jnp.float64
            ) * scale
            for l in range(self.n_layers)
        )
        self.params = DBMParams(a=a, b=b, W=W)

        flat, self._unravel = ravel_pytree(self.params)
        self._n_params = int(flat.shape[0])

    # ── Core functional interface ─────────────────────────────────────────

    def log_psi_single(self, params: DBMParams, v: jax.Array) -> jax.Array:
        """
        Mean-field approximation of log|Ψ(v)|.

        params : DBMParams
        v      : (n_v,) spin config ∈ {-1, +1}

        Differentiable via jax.grad — the mean-field loop is unrolled
        statically at trace time (n_mf_steps is a Python int).
        """
        L = self.n_layers

        # Initialise mean-field marginals μ^l = 0
        mu = [jnp.zeros_like(params.b[l]) for l in range(L)]

        # Jacobi fixed-point iterations (all layers updated simultaneously)
        for _ in range(self.n_mf_steps):
            new_mu = []
            for l in range(L):
                prev = v if l == 0 else mu[l - 1]
                alpha = params.W[l].T @ prev + params.b[l]
                if l < L - 1:
                    alpha = alpha + params.W[l + 1] @ mu[l + 1]
                new_mu.append(jnp.tanh(alpha))
            mu = new_mu

        # Recompute converged local fields from final mu
        alphas = []
        for l in range(L):
            prev = v if l == 0 else mu[l - 1]
            alpha = params.W[l].T @ prev + params.b[l]
            if l < L - 1:
                alpha = alpha + params.W[l + 1] @ mu[l + 1]
            alphas.append(alpha)

        # -a·v/2 + ½ Σ_l log[2 cosh(α^l)] - ½ Σ_l μ^l · W^{l+1} · μ^{l+1}
        log_psi = -params.a @ v / 2
        for l in range(L):
            log_psi = log_psi + 0.5 * jnp.sum(
                jnp.log(2) + jnp.logaddexp(alphas[l], -alphas[l])
            )
        for l in range(L - 1):
            log_psi = log_psi - 0.5 * (mu[l] @ (params.W[l + 1] @ mu[l + 1]))

        return log_psi

    def log_psi_batch(self, V: jax.Array) -> jax.Array:
        """log|Ψ(v)| for a batch V : (ns, N) → (ns,)."""
        return jax.vmap(lambda v: self.log_psi_single(self.params, v))(V)

    # ── Weight serialisation ──────────────────────────────────────────────

    def get_flat_params(self) -> jax.Array:
        """Pack self.params pytree → 1D float64 array."""
        flat, _ = ravel_pytree(self.params)
        return flat

    def set_flat_params(self, w: jax.Array) -> None:
        """Unpack 1D array → self.params pytree."""
        self.params = self._unravel(w)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def n_parameters(self) -> int:
        return self._n_params

    def __repr__(self) -> str:
        layers = " → ".join(str(s) for s in [self.n_visible] + self.hidden_sizes)
        return (
            f"DeepBoltzmannMachine("
            f"{layers}, "
            f"n_params={self._n_params}, "
            f"n_mf_steps={self.n_mf_steps})"
        )
