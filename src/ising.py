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


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def _local_energy_xxz_1d_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    J: float,
    delta: float,
    N: int,
) -> jax.Array:
    """
    XXZ local energy for all ns samples simultaneously.

    V : (ns, N)   spin configs {-1, +1}
    W : (N, M)    RBM weights
    a : (N,)      visible biases
    b : (M,)      hidden biases
    J : float     coupling strength
    delta : float XXZ anisotropy Δ (1 = isotropic Heisenberg, 0 = XY, ∞ = Ising)
    N : int       n_visible (static)

    Returns (ns,) local energies.

    Hamiltonian (Pauli convention, ±1 eigenvalues):
        H = J Σᵢ [σˣᵢσˣᵢ₊₁ + σʸᵢσʸᵢ₊₁ + Δ σᶻᵢσᶻᵢ₊₁]

    Off-diagonal contribution (spin exchange on antiparallel bonds)
    ---------------------------------------------------------------
    For bond (i, i+1): contributes J * Ψ(v_{i,i+1-swapped})/Ψ(v) only when vᵢ ≠ vᵢ₊₁.
    Selector: (1 - vᵢ·vᵢ₊₁) / 2  = 1 iff spins are antiparallel, else 0.

    The two-spin log-ratio for bond i:
        log Ψ'/Ψ = aᵢvᵢ + aᵣvᵣ + ½ Σⱼ [logcosh(θⱼ - 2vᵢWᵢⱼ - 2vᵣWᵣⱼ) - logcosh(θⱼ)]
    where r = right[i].  Vectorised over all N bonds as a (ns, N, M) tensor.
    """
    theta = V @ W + b[None, :]                               # (ns, M)
    blc = jnp.logaddexp(theta, -theta)                        # logcosh base (ns, M)

    right = (jnp.arange(N) + 1) % N                          # (N,) right-neighbor indices

    # Combined hidden-unit shift for flipping both site i and its right neighbor
    delta_theta = -2.0 * (
        V[:, :, None] * W[None, :, :]            # (ns, N, M)  site-i contribution
        + V[:, right, None] * W[None, right, :]  # (ns, N, M)  right-neighbor contribution
    )                                             # (ns, N, M)

    theta_flipped = theta[:, None, :] + delta_theta           # (ns, N, M)

    log_ratios = (
        a[None, :] * V                            # (ns, N)  a[i]*v[i]
        + a[None, right] * V[:, right]            # (ns, N)  a[right[i]]*v[right[i]]
        + 0.5 * jnp.sum(
            jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2
        )
    )                                             # (ns, N)

    # (1 - vᵢ·v_right) = 2 for antiparallel, 0 for parallel — already encodes the 2J factor
    exchange = 1.0 - V * V[:, right]                                   # (ns, N)
    E_off = J * jnp.sum(exchange * jnp.exp(log_ratios), axis=1)        # (ns,)
    E_diag = J * delta * jnp.sum(V * V[:, right], axis=1)             # (ns,)

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
        """
        Build the 2D TFIM Hamiltonian as a scipy sparse matrix.
        No netket dependency — works on Python 3.13+.

        Bonds counted via right + down neighbors (each bond appears once).
        Encoding: bit (N-1-i) of integer s represents spin at site i.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        L = self.linear_size
        N = self.size  # L²
        h = self.h
        dim = 2 ** N

        def spin(s: int, i: int) -> int:
            return 1 - 2 * ((s >> (N - 1 - i)) & 1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s in range(dim):
            diag = 0.0
            for i in range(N):
                col_i = i % L
                row_i = i // L
                right = row_i * L + (col_i + 1) % L
                down  = ((row_i + 1) % L) * L + col_i
                diag -= spin(s, i) * spin(s, right) + spin(s, i) * spin(s, down)
            rows.append(s); cols.append(s); vals.append(diag)

            # Off-diagonal: -h σˣᵢ flips spin i, matrix element = -h
            for i in range(N):
                s_flip = s ^ (1 << (N - 1 - i))
                rows.append(s_flip); cols.append(s); vals.append(-h)

        H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float)
        eigenvalues, _ = eigsh(H, k=1, which="SA")
        return float(eigenvalues[0])

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


# ---------------------------------------------------------------------------
# Heisenberg XXZ chain
# ---------------------------------------------------------------------------


class HeisenbergXXZ1D(IsingModel):
    """
    1D XXZ Heisenberg chain with periodic boundary conditions.

        H = J Σᵢ [σˣᵢσˣᵢ₊₁ + σʸᵢσʸᵢ₊₁ + Δ σᶻᵢσᶻᵢ₊₁]

    where σ are Pauli matrices with ±1 eigenvalues.

    Special cases:
        Δ = 1   → isotropic XXX Heisenberg
        Δ = 0   → XY model
        Δ → ∞   → Ising model
        J > 0   → antiferromagnetic
        J < 0   → ferromagnetic

    The `size` argument is the chain length N (number of spins).
    The `h` slot in the base class is unused; coupling and anisotropy are
    stored as `self.J` and `self.delta`.

    Note: `local_energy(v, rbm)` takes an RBM instance (not a psi_ratio
    callable) because it needs `rbm.psi_ratio_pair` for two-spin flips.
    """

    def __init__(self, size: int, J: float = 1.0, delta: float = 1.0):
        super().__init__(size, h=0.0)  # h unused
        self.J = J
        self.delta = delta

    def local_energy(self, v: np.ndarray, rbm) -> float:
        """Scalar local energy for a single configuration.

        Args:
            v   : (N,) array of ±1 spins
            rbm : RBM instance (must have psi_ratio_pair method)
        """
        v_jax = jnp.asarray(v, dtype=jnp.float64)
        E_diag = self.J * self.delta * float(
            sum(v[i] * v[(i + 1) % self.size] for i in range(self.size))
        )
        E_off = 0.0
        for i in range(self.size):
            j = (i + 1) % self.size
            if v[i] != v[j]:  # only antiparallel bonds contribute; matrix element = 2J
                E_off += 2 * self.J * float(rbm.psi_ratio_pair(v_jax, i, j))
        return E_diag + E_off

    def local_energy_batch(self, V, rbm) -> jax.Array:
        """JIT-compiled batched XXZ local energy."""
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        return _local_energy_xxz_1d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.J, self.delta, self.size
        )

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute

        if self.size > 20:
            raise NotImplementedError(
                f"Exact diagonalization not feasible for Heisenberg N={self.size}. "
                "Implement Bethe ansatz or use N ≤ 20."
            )
        # Encode delta in the model key; J maps to the 'h' slot in the cache key.
        model_key = f"heisenberg_xxz_1d_delta{self.delta:.10g}"
        return get_or_compute(model_key, self.size, self.J, self._compute_exact_ground_energy)

    def _compute_exact_ground_energy(self) -> float:
        """
        Build the XXZ Hamiltonian as a scipy sparse matrix and find its ground
        state via Lanczos.  No netket dependency — works on Python 3.13+.

        Encoding: bit i of integer s = 0 → spin +1, bit i = 1 → spin -1.
        Bit ordering: bit (N-1-i) represents site i (MSB = site 0).
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        N = self.size
        dim = 2 ** N

        def spin(s: int, i: int) -> int:
            return 1 - 2 * ((s >> (N - 1 - i)) & 1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s in range(dim):
            diag = self.J * self.delta * sum(
                spin(s, i) * spin(s, (i + 1) % N) for i in range(N)
            )
            rows.append(s); cols.append(s); vals.append(diag)

            # Off-diagonal exchange: matrix element = 2J for antiparallel bonds
            # (σˣᵢσˣⱼ + σʸᵢσʸⱼ = 2(σ⁺ᵢσ⁻ⱼ + σ⁻ᵢσ⁺ⱼ), non-zero only when vᵢ ≠ vⱼ)
            for i in range(N):
                j = (i + 1) % N
                if spin(s, i) != spin(s, j):
                    s_flip = s ^ (1 << (N - 1 - i)) ^ (1 << (N - 1 - j))
                    rows.append(s_flip); cols.append(s); vals.append(2 * self.J)

        H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float)
        eigenvalues, _ = eigsh(H, k=1, which="SA")
        return float(eigenvalues[0])

    def get_neighbors(self, idx: int) -> list[int]:
        left = (idx - 1) % self.size
        right = (idx + 1) % self.size
        return [left, right]
