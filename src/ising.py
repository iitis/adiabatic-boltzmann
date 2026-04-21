"""
Transverse Field Ising Model — JAX backend

Local energy computation is JIT-compiled with jax.jit.
The 3D broadcast (ns, N, M) that was too large for CPU cache is a natural fit
for GPU: the entire tensor lives in VRAM and is processed as a single kernel.
On CPU, XLA is still faster than the Numba approach because it can use
BLAS-level vectorisation and avoids Python loop overhead entirely.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# JIT-compiled energy kernels (module-level so they compile once per session)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(4, 5))
def _local_energy_1d_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    h: float,
    N: int,
) -> jax.Array:
    """
    1D local energy for all ns samples simultaneously.

    V : (ns, N)   spin configs {-1, +1}
    W : (N, M)    RBM weights
    a : (N,)      visible biases
    b : (M,)      hidden biases
    h : float     transverse field strength
    N : int       n_visible (static — shapes must be known at compile time)

    Returns (ns,) local energies.

    Off-diagonal sum
    ----------------
    log_ratio(s, i) = a[i]*V[s,i]
                    + 0.5 * Σ_j [ logcosh(θ[s,j] - 2*V[s,i]*W[i,j])
                                  - logcosh(θ[s,j]) ]

    All N flips computed at once as a (ns, N, M) tensor — one XLA kernel.
    """
    theta = V @ W + b[None, :]                              # (ns, M)
    blc = jnp.logaddexp(theta, -theta)                      # logcosh base  (ns, M)

    # theta_flipped[s, i, j] = theta[s,j] - 2*V[s,i]*W[i,j]
    theta_flipped = theta[:, None, :] - 2.0 * V[:, :, None] * W[None, :, :]  # (ns, N, M)

    log_ratios = a[None, :] * V + 0.5 * jnp.sum(
        jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2
    )  # (ns, N)

    E_off = -h * jnp.sum(jnp.exp(log_ratios), axis=1)      # (ns,)

    right = (jnp.arange(N) + 1) % N
    E_diag = -jnp.sum(V * V[:, right], axis=1)             # (ns,)

    return E_diag + E_off


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def _local_energy_2d_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    h: float,
    N: int,
    L: int,
) -> jax.Array:
    """
    2D local energy for all ns samples simultaneously.

    V : (ns, N)   spin configs, N = L²
    L : int       linear lattice dimension (static)
    """
    theta = V @ W + b[None, :]
    blc = jnp.logaddexp(theta, -theta)
    theta_flipped = theta[:, None, :] - 2.0 * V[:, :, None] * W[None, :, :]

    log_ratios = a[None, :] * V + 0.5 * jnp.sum(
        jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2
    )
    E_off = -h * jnp.sum(jnp.exp(log_ratios), axis=1)

    # 2D diagonal bonds — vectorised neighbor index arrays
    i_idx = jnp.arange(N)
    cols = i_idx % L
    rows = i_idx // L
    right_idx = rows * L + (cols + 1) % L   # right neighbor (periodic within row)
    down_idx = ((rows + 1) % L) * L + cols  # down neighbor  (periodic across rows)

    E_diag = -jnp.sum(V * V[:, right_idx] + V * V[:, down_idx], axis=1)

    return E_diag + E_off


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class IsingModel(ABC):
    """Abstract Ising model base class."""

    def __init__(self, size: int, h: float = 1.0):
        self.size = size
        self.h = h

    @abstractmethod
    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        """Scalar local energy for a single configuration (uses Python loop)."""
        pass

    @abstractmethod
    def local_energy_batch(self, V, rbm) -> jax.Array:
        """
        Compute local energies for a batch of configurations.

        V   : (n_samples, n_visible) — NumPy or JAX array of ±1 spins
        rbm : RBM instance  (.a, .b, .W are JAX arrays)

        Returns (n_samples,) JAX array of local energies.
        """
        pass

    @abstractmethod
    def exact_ground_energy(self) -> float:
        pass

    @abstractmethod
    def get_neighbors(self, idx: int) -> list[int]:
        pass


# ---------------------------------------------------------------------------
# 1D chain
# ---------------------------------------------------------------------------


class TransverseFieldIsing1D(IsingModel):
    """1D transverse field Ising model with periodic boundary conditions."""

    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        E_diag = (
            -sum(
                v[i] * v[i_n]
                for i in range(self.size)
                for i_n in self.get_neighbors(i)
            )
            / 2
        )
        E_off_diag = -self.h * sum(psi_ratio_fn(v, i) for i in range(self.size))
        return E_diag + E_off_diag

    def local_energy_batch(self, V, rbm) -> jax.Array:
        """
        JIT-compiled batched local energy.

        Dispatches to _local_energy_1d_jit which compiles to a single XLA
        kernel — runs on GPU automatically when JAX is configured for CUDA.
        """
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        return _local_energy_1d_jit(V_jax, rbm.W, rbm.a, rbm.b, self.h, self.size)

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute
        return get_or_compute("1d", self.size, self.h, self._compute_exact_ground_energy)

    def _compute_exact_ground_energy(self) -> float:
        N, h = self.size, self.h
        m = np.arange(N)
        # Ramond sector: anti-periodic fermion BC (even-parity sector)
        k_R = np.pi * (2 * m + 1) / N
        E_R = -float(np.sum(np.sqrt(1.0 + h**2 - 2.0 * h * np.cos(k_R))))
        # Neveu-Schwarz sector: periodic fermion BC (odd-parity sector)
        k_NS = 2.0 * np.pi * m / N
        E_NS = -float(np.sum(np.sqrt(1.0 + h**2 - 2.0 * h * np.cos(k_NS))))
        return min(E_R, E_NS)

    def exact_ground_energy_netket(self):
        import netket as nk
        from scipy.sparse.linalg import eigsh

        N = self.size
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
        ha = nk.operator.LocalOperator(hilbert)
        for i in range(N):
            ha += (
                -1.0
                * nk.operator.spin.sigmaz(hilbert, i)
                @ nk.operator.spin.sigmaz(hilbert, (i + 1) % N)
            )
            ha += -self.h * nk.operator.spin.sigmax(hilbert, i)
        H_sparse = ha.to_sparse()
        vals, _ = eigsh(H_sparse, k=1, which="SA")
        return vals[0]

    def get_neighbors(self, idx: int) -> list[int]:
        left = (idx - 1) % self.size
        right = (idx + 1) % self.size
        return [left, right]


# ---------------------------------------------------------------------------
# 2D square lattice
# ---------------------------------------------------------------------------


class TransverseFieldIsing2D(IsingModel):
    """2D transverse field Ising model on square lattice with periodic BC."""

    def __init__(self, size: int, h: float = 1.0):
        """size: linear dimension L (total N = L² spins)."""
        super().__init__(size * size, h)
        self.linear_size = size

    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        E_diag = 0.0
        for i in range(self.size):
            right = (i % self.linear_size + 1) % self.linear_size + (
                i // self.linear_size
            ) * self.linear_size
            down = (i + self.linear_size) % self.size
            E_diag -= v[i] * v[right] + v[i] * v[down]
        E_off_diag = -self.h * sum(psi_ratio_fn(v, i) for i in range(self.size))
        return E_diag + E_off_diag

    def local_energy_batch(self, V, rbm) -> jax.Array:
        """JIT-compiled batched 2D local energy."""
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        return _local_energy_2d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.h, self.size, self.linear_size
        )

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute
        return get_or_compute("2d", self.linear_size, self.h, self._compute_exact_ground_energy)

    def _compute_exact_ground_energy(self) -> float:
        L = self.linear_size
        if L > 4:
            raise NotImplementedError(
                f"No exact reference energy available for 2D TFIM with L={L}. "
                "Exact diagonalization is only feasible for L ≤ 4 (2^16 states). "
                "No method in this codebase meets the 0.001 accuracy requirement for L > 4."
            )
        return self._exact_diag_2d()

    def _exact_diag_2d(self) -> float:
        import netket as nk
        from scipy.sparse.linalg import eigsh

        N = self.size
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
        ha = nk.operator.LocalOperator(hilbert, dtype=complex)
        for i in range(N):
            for j in self.get_neighbors(i):
                if i < j:
                    ha += (
                        -1.0
                        * nk.operator.spin.sigmaz(hilbert, i)
                        @ nk.operator.spin.sigmaz(hilbert, j)
                    )
            ha += -self.h * nk.operator.spin.sigmax(hilbert, i)
        vals, _ = eigsh(ha.to_sparse(), k=1, which="SA")
        return float(vals[0])

    def get_neighbors(self, idx: int) -> list[int]:
        i = idx // self.linear_size
        j = idx % self.linear_size
        neighbors_2d = [
            ((i - 1) % self.linear_size, j),
            ((i + 1) % self.linear_size, j),
            (i, (j - 1) % self.linear_size),
            (i, (j + 1) % self.linear_size),
        ]
        return [i * self.linear_size + j for i, j in neighbors_2d]
