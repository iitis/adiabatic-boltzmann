"""
Vision Transformer (ViT) wave function ansatz — JAX/Flax backend

Architecture:
    v ∈ {-1,+1}^N → Patch Embedding → Transformer Encoder → Output Head → log Ψ

The output layer is identical to the RBM cosh readout (Gardas Eq. 6-7):
    log Ψ = Σ_α log cosh(W_α · z + b_α)
where z is the sum-pooled transformer representation and the transformer stack
replaces the linear W·v of the RBM with a deep nonlinear feature extractor.

Geometry support
----------------
* 1D: patches are groups of `patch_size` consecutive spins.
* 2D: patches are `patch_size × patch_size` blocks on a square L×L lattice.

Reference: https://netket.readthedocs.io/en/latest/tutorials/ViT-wave-function.html
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen as nn
from flax.linen.initializers import xavier_uniform, zeros
from typing import Any


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------


def log_cosh(x: jax.Array) -> jax.Array:
    """Numerically stable log(cosh(x)) = logaddexp(x, -x)."""
    return jnp.logaddexp(x, -x)


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_patches_1d(x: jax.Array, patch_size: int) -> jax.Array:
    """
    Group 1D spin configs into non-overlapping patches.

    x          : (batch, N)   N divisible by patch_size
    patch_size : int

    Returns (batch, N//patch_size, patch_size).
    """
    batch, N = x.shape
    n_patches = N // patch_size
    return x.reshape(batch, n_patches, patch_size)


def extract_patches_2d(x: jax.Array, patch_size: int) -> jax.Array:
    """
    Extract 2D patches from a flattened square-lattice spin config.

    x          : (batch, N)   N = L², row-major ordering
    patch_size : int          patch side length (square)

    Returns (batch, (L//patch_size)², patch_size²).
    """
    batch = x.shape[0]
    N = x.shape[1]
    L = int(round(N**0.5))
    n_grid = L // patch_size
    x2d = x.reshape(batch, n_grid, patch_size, n_grid, patch_size)
    x2d = x2d.transpose(0, 1, 3, 2, 4)            # (B, n_grid, n_grid, ps, ps)
    return x2d.reshape(batch, n_grid * n_grid, -1)  # (B, n_patches, patch_size²)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class Embed(nn.Module):
    d_model: int
    patch_size: int
    geometry: str = "1d"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.geometry == "1d":
            patches = extract_patches_1d(x, self.patch_size)
        else:
            patches = extract_patches_2d(x, self.patch_size)
        return nn.Dense(
            self.d_model,
            kernel_init=xavier_uniform(),
            param_dtype=jnp.float64,
        )(patches)


# ---------------------------------------------------------------------------
# Factored Multi-Head Attention (FMHA)
# ---------------------------------------------------------------------------


class FMHA(nn.Module):
    """
    Factored Multi-Head Attention with position-only attention weights.

    Standard self-attention learns Q, K, V projections and computes
    softmax(QK^T/√d)V. Here the attention matrix α[h, i, j] depends only
    on patch positions (h=head, i=query patch, j=key patch) — not on the
    spin values. This reduces parameters and sidesteps softmax instabilities
    while still capturing long-range correlations between patches.

    alpha shape : (n_heads, n_patches, n_patches)
    v-projection: Dense(d_model → d_model), split into n_heads post-hoc.
    out-proj    : Dense(d_model → d_model).
    """

    d_model: int
    n_heads: int
    n_patches: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # x : (batch, n_patches, d_model)
        d_eff = self.d_model // self.n_heads

        # Value projection + head split → (B, n_heads, n_patches, d_eff)
        v = nn.Dense(
            self.d_model, kernel_init=xavier_uniform(), param_dtype=jnp.float64
        )(x)
        v = v.reshape(x.shape[0], self.n_patches, self.n_heads, d_eff)
        v = v.transpose(0, 2, 1, 3)  # (B, n_heads, n_patches, d_eff)

        # Positional attention weights (not data-dependent)
        alpha = self.param(
            "alpha",
            xavier_uniform(),
            (self.n_heads, self.n_patches, self.n_patches),
            jnp.float64,
        )

        # Weighted sum over key patches → (B, n_heads, n_patches, d_eff)
        attended = jnp.matmul(alpha, v)

        # Merge heads → (B, n_patches, d_model)
        attended = attended.transpose(0, 2, 1, 3)
        attended = attended.reshape(x.shape[0], self.n_patches, self.d_model)

        return nn.Dense(
            self.d_model, kernel_init=xavier_uniform(), param_dtype=jnp.float64
        )(attended)


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Pre-norm attention + residual
        x = x + FMHA(self.d_model, self.n_heads, self.n_patches)(
            nn.LayerNorm(param_dtype=jnp.float64)(x)
        )
        # Pre-norm feed-forward + residual (4× expansion, GELU)
        h = nn.LayerNorm(param_dtype=jnp.float64)(x)
        h = nn.Dense(
            4 * self.d_model, kernel_init=xavier_uniform(), param_dtype=jnp.float64
        )(h)
        h = nn.gelu(h)
        h = nn.Dense(
            self.d_model, kernel_init=xavier_uniform(), param_dtype=jnp.float64
        )(h)
        return x + h


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------


class OutputHead(nn.Module):
    """
    Sum-pool transformer output → RBM-style cosh readout.

        log Ψ(v) = Σ_α log cosh(W_α · z + b_α)

    z = LayerNorm(Σ_{patches} x_patch).  The Dense layers provide W_α, b_α.
    This is identical to the RBM output layer — the transformer is a nonlinear
    feature extractor that replaces the linear W·v.

    complex_output : bool
        If True, emit a complex log Ψ = Re + i·Im for models whose ground
        state has non-trivial phase structure (e.g. frustrated Heisenberg).
    """

    d_model: int
    complex_output: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # x : (batch, n_patches, d_model)
        z = nn.LayerNorm(param_dtype=jnp.float64)(x.sum(axis=1))  # (B, d_model)

        out_real = nn.LayerNorm(param_dtype=jnp.float64)(
            nn.Dense(
                self.d_model,
                kernel_init=xavier_uniform(),
                bias_init=zeros,
                param_dtype=jnp.float64,
            )(z)
        )  # (B, d_model)

        if self.complex_output:
            out_imag = nn.LayerNorm(param_dtype=jnp.float64)(
                nn.Dense(
                    self.d_model,
                    kernel_init=xavier_uniform(),
                    bias_init=zeros,
                    param_dtype=jnp.float64,
                )(z)
            )
            out = out_real.astype(jnp.complex128) + 1j * out_imag.astype(jnp.complex128)
        else:
            out = out_real

        return jnp.sum(log_cosh(out), axis=-1)  # (B,)


# ---------------------------------------------------------------------------
# Full ViT Flax module
# ---------------------------------------------------------------------------


class ViTModule(nn.Module):
    """
    Vision Transformer wave function.

    n_layers      : int   number of transformer encoder blocks
    d_model       : int   embedding dimension (must be divisible by n_heads)
    n_heads       : int   number of attention heads
    patch_size    : int   spins per patch (1D) or patch side length (2D)
    geometry      : str   "1d" or "2d"
    complex_output: bool  complex-valued log Ψ
    """

    n_layers: int
    d_model: int
    n_heads: int
    patch_size: int
    geometry: str = "1d"
    complex_output: bool = False

    @nn.compact
    def __call__(self, v: jax.Array) -> jax.Array:
        """
        v : (batch, N)   spin configs ∈ {-1, +1}^N

        Returns (batch,) log|Ψ(v)| values.
        """
        x = jnp.atleast_2d(v).astype(jnp.float64)
        N = x.shape[-1]

        if self.geometry == "1d":
            n_patches = N // self.patch_size
        else:
            L = int(round(N**0.5))
            n_patches = (L // self.patch_size) ** 2

        x = Embed(self.d_model, self.patch_size, self.geometry)(x)

        for _ in range(self.n_layers):
            x = EncoderBlock(self.d_model, self.n_heads, n_patches)(x)

        return OutputHead(self.d_model, self.complex_output)(x)


# ---------------------------------------------------------------------------
# ViTWaveFunction — flat-parameter wrapper for the SR trainer
# ---------------------------------------------------------------------------


class ViTWaveFunction:
    """
    Wraps ViTModule with the flat-parameter interface required by the generic
    SR trainer and MetropolisHastings sampler.

    Parameters are stored as a Flax param pytree in ``self.params`` and can
    be serialized to/from a flat 1D JAX array via ``get_flat_params`` /
    ``set_flat_params`` (using jax.flatten_util.ravel_pytree).

    ``log_psi_single(params, v)`` — the pure-functional form — is the entry
    point for ``jax.grad`` and ``jax.vmap`` in the SR trainer.
    """

    def __init__(
        self,
        n_visible: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        key: jax.Array,
        geometry: str = "1d",
        complex_output: bool = False,
    ):
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        if geometry == "1d":
            assert n_visible % patch_size == 0, (
                f"n_visible={n_visible} must be divisible by patch_size={patch_size}"
            )
        else:
            L = int(round(n_visible**0.5))
            assert L * L == n_visible, f"2D geometry requires n_visible = L², got {n_visible}"
            assert L % patch_size == 0, (
                f"L={L} must be divisible by patch_size={patch_size}"
            )

        self.n_visible = n_visible
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.patch_size = patch_size
        self.geometry = geometry
        self.complex_output = complex_output

        self._module = ViTModule(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            patch_size=patch_size,
            geometry=geometry,
            complex_output=complex_output,
        )

        # Initialize via a dummy forward pass
        dummy = jnp.zeros((1, n_visible), dtype=jnp.float64)
        variables = self._module.init(key, dummy)
        self.params = variables["params"]  # type: ignore[index]

        # Build the unravel function (shape is fixed after init)
        flat, self._unravel = ravel_pytree(self.params)
        self._n_params = int(flat.shape[0])

        if geometry == "1d":
            self._n_patches = n_visible // patch_size
        else:
            L = int(round(n_visible**0.5))
            self._n_patches = (L // patch_size) ** 2

    # ── Core functional interface ──────────────────────────────────────────

    def log_psi_single(self, params: Any, v: jax.Array) -> Any:
        """
        log|Ψ(v)| for a single configuration — pure function of params.

        params : Flax param pytree (same structure as self.params)
        v      : (N,) spin config ∈ {-1, +1}

        Used as:
            jax.grad(vit.log_psi_single, argnums=0)(params, v)
        to produce per-sample gradient pytrees for the SR optimizer.
        """
        out = self._module.apply({"params": params}, v[None, :])  # type: ignore[call-arg]
        return out[0]  # type: ignore[index]

    def log_psi_batch(self, V: jax.Array) -> Any:
        """
        log|Ψ(v)| for a batch of configurations using current self.params.

        V : (ns, N)   Returns (ns,).
        """
        return self._module.apply({"params": self.params}, V)  # type: ignore[call-arg]

    # ── Weight serialisation ───────────────────────────────────────────────

    def get_flat_params(self) -> jax.Array:
        """Pack self.params pytree → 1D float64 array."""
        flat, _ = ravel_pytree(self.params)
        return flat  # type: ignore[return-value]

    def set_flat_params(self, w: jax.Array) -> None:
        """Unpack 1D array → self.params pytree."""
        self.params = self._unravel(w)  # type: ignore[assignment]

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return self._n_params

    def n_parameters(self) -> int:
        return self._n_params

    def __repr__(self) -> str:
        return (
            f"ViTWaveFunction("
            f"n_visible={self.n_visible}, "
            f"geometry={self.geometry}, "
            f"patch_size={self.patch_size}, "
            f"n_patches={self._n_patches}, "
            f"n_layers={self.n_layers}, "
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_params={self.n_params})"
        )
