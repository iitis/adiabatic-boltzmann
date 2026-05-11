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


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _local_energy_lr1d_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    J_coupling: float,
    h: float,
    alpha: float,
    N: int,
) -> jax.Array:
    """
    Long-range 1D TFIM local energy for all ns samples simultaneously.

    V         : (ns, N)   spin configs {-1, +1}
    J_coupling: float     overall coupling scale
    h         : float     transverse field strength
    alpha     : float     power-law exponent
    N         : int       n_visible (static)

    Hamiltonian:
        H = -J Σᵢ<ⱼ σᶻᵢσᶻⱼ / d(i,j)^α  −  h Σᵢ σˣᵢ
    where d(i,j) = min(|i−j|, N−|i−j|)  (periodic chord distance).

    Off-diagonal term is identical to nearest-neighbor TFIM: single-spin flips.
    Diagonal term is O(N²) — all pairs weighted by 1/d^α.
    """
    # --- Coupling matrix (constant-folded into the kernel since alpha, N static) ---
    idx = jnp.arange(N)
    raw_dist = jnp.abs(idx[:, None] - idx[None, :])
    dist = jnp.minimum(raw_dist, N - raw_dist).astype(jnp.float64)   # (N, N) periodic
    safe_dist = jnp.where(dist > 0.0, dist, 1.0)
    J_mat = jnp.where(dist > 0.0, J_coupling / safe_dist ** alpha, 0.0)  # (N, N)

    # --- Off-diagonal (single-spin flips, same as 1D TFIM kernel) ---
    theta = V @ W + b[None, :]                                         # (ns, M)
    blc = jnp.logaddexp(theta, -theta)
    theta_flipped = theta[:, None, :] - 2.0 * V[:, :, None] * W[None, :, :]  # (ns, N, M)
    log_ratios = a[None, :] * V + 0.5 * jnp.sum(
        jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2
    )
    E_off = -h * jnp.sum(jnp.exp(log_ratios), axis=1)                 # (ns,)

    # --- Diagonal: -0.5 * Σᵢ Σⱼ J_mat[i,j] vᵢvⱼ  (factor 0.5 because J_mat symmetric) ---
    E_diag = -0.5 * jnp.einsum("si,ij,sj->s", V, J_mat, V)            # (ns,)

    return E_diag + E_off


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _local_energy_j1j2_1d_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    J1: float,
    J2: float,
    h: float,
    N: int,
) -> jax.Array:
    """
    J₁-J₂ frustrated 1D chain local energy.

    H = -J1 Σᵢ σᶻᵢσᶻᵢ₊₁ - J2 Σᵢ σᶻᵢσᶻᵢ₊₂ - h Σᵢ σˣᵢ

    Off-diagonal term is identical to nearest-neighbor TFIM (single-spin flips).
    Diagonal adds NNN bonds on top of NN bonds.
    """
    theta = V @ W + b[None, :]
    blc = jnp.logaddexp(theta, -theta)
    theta_flipped = theta[:, None, :] - 2.0 * V[:, :, None] * W[None, :, :]
    log_ratios = a[None, :] * V + 0.5 * jnp.sum(
        jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2
    )
    E_off = -h * jnp.sum(jnp.exp(log_ratios), axis=1)

    idx = jnp.arange(N)
    right1 = (idx + 1) % N
    right2 = (idx + 2) % N
    E_diag = (
        -J1 * jnp.sum(V * V[:, right1], axis=1)
        - J2 * jnp.sum(V * V[:, right2], axis=1)
    )
    return E_diag + E_off


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _local_energy_xxz_2d_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    J: float,
    delta: float,
    N: int,
    L: int,
) -> jax.Array:
    """
    2D XXZ Heisenberg local energy on L×L square lattice (periodic BC).

    V : (ns, N)   spin configs, N = L²
    W : (N, M)    RBM weights
    a : (N,)      visible biases
    b : (M,)      hidden biases
    J, delta, N, L : static compile-time constants

    Returns (ns,) local energies.

    Off-diagonal contribution for bond (i,j): matrix element = 2J only when v[i]≠v[j].
    The exchange selector (1 - v[i]·v[j]) equals 2 for antiparallel and 0 for parallel,
    so E_off = J * Σ_bonds (1-v[i]v[j]) * exp(log_ratio) naturally encodes the 2J factor.
    Each bond direction (right, down) is counted once — total 2N bonds per configuration.
    """
    theta = V @ W + b[None, :]            # (ns, M)
    blc   = jnp.logaddexp(theta, -theta)  # logcosh base (ns, M)

    # Neighbor indices — constant-folded by XLA since N, L are static
    i_idx     = jnp.arange(N)
    right_idx = (i_idx // L) * L + (i_idx % L + 1) % L   # (N,) right neighbor
    down_idx  = ((i_idx // L + 1) % L) * L + i_idx % L   # (N,) down  neighbor

    def _bond(partner_idx: jax.Array):
        """Diagonal + off-diagonal energy for all N bonds toward partner_idx."""
        dt = -2.0 * (
            V[:, :, None] * W[None, :, :]
            + V[:, partner_idx, None] * W[None, partner_idx, :]
        )                                                    # (ns, N, M)
        theta_f = theta[:, None, :] + dt                    # (ns, N, M)
        log_r = (
            a[None, :] * V
            + a[None, partner_idx] * V[:, partner_idx]
            + 0.5 * jnp.sum(
                jnp.logaddexp(theta_f, -theta_f) - blc[:, None, :], axis=2
            )
        )                                                    # (ns, N)
        exchange = 1.0 - V * V[:, partner_idx]              # (ns, N)
        e_off  = J * jnp.sum(exchange * jnp.exp(log_r), axis=1)          # (ns,)
        e_diag = J * delta * jnp.sum(V * V[:, partner_idx], axis=1)      # (ns,)
        return e_diag, e_off

    e_diag_r, e_off_r = _bond(right_idx)
    e_diag_d, e_off_d = _bond(down_idx)
    return e_diag_r + e_diag_d + e_off_r + e_off_d


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
# FBM off-diagonal kernel (shared by all single-flip Hamiltonians)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(5, 6))
def _fbm_off_diagonal_jit(
    V: jax.Array,
    W: jax.Array,
    a: jax.Array,
    b: jax.Array,
    J: jax.Array,
    h: float,
    N: int,
) -> jax.Array:
    """
    Off-diagonal FBM local energy  -h Σᵢ Ψ(vᶠˡⁱᵖⁱ)/Ψ(v)  for single-spin
    flip Hamiltonians (TFIM, J1J2, LR-TFIM).

    log_ratio(s, i) = a_i v_{s,i}
                    + ½ Σⱼ [logcosh(θ'ⱼ) - logcosh(θⱼ)]   [RBM part]
                    - v_{s,i} Σₖ J[i,k] v_{s,k}              [J_vv correction]

    The J_vv term vectorises as  V * (V @ J)  — elementwise product, (ns, N).
    """
    theta = V @ W + b[None, :]                                        # (ns, M)
    blc = jnp.logaddexp(theta, -theta)                                # (ns, M)
    theta_flipped = theta[:, None, :] - 2.0 * V[:, :, None] * W[None, :, :]  # (ns, N, M)
    log_ratios = a[None, :] * V + 0.5 * jnp.sum(
        jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2
    )                                                                  # (ns, N)
    log_ratios = log_ratios - V * (V @ J)                             # J_vv correction
    return -h * jnp.sum(jnp.exp(log_ratios), axis=1)                  # (ns,)


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

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """
        Generic local energy for arbitrary wave functions.

        V          : (ns, N)  spin configs ±1
        log_psi_fn : callable (N,) → scalar  — log|Ψ(v)|

        Default implementation raises NotImplementedError; subclasses that
        support non-RBM ansätze override this method.
        """
        raise NotImplementedError(  # noqa: PIE796
            f"{self.__class__.__name__} does not implement local_energy_batch_generic. "
            "Override it to support non-RBM ansätze."
        )

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
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        if hasattr(rbm, "J"):
            E_off = _fbm_off_diagonal_jit(V_jax, rbm.W, rbm.a, rbm.b, rbm.J, self.h, self.size)
            right = (jnp.arange(self.size) + 1) % self.size
            return -jnp.sum(V_jax * V_jax[:, right], axis=1) + E_off
        return _local_energy_1d_jit(V_jax, rbm.W, rbm.a, rbm.b, self.h, self.size)

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """
        Generic 1D TFIM local energy via log_psi_fn.

        Evaluates N single-spin-flip ratios per sample using jax.vmap.
        log_psi_fn : callable (N,) → scalar
        """
        N = self.size
        h = self.h
        log_p_V = jax.vmap(log_psi_fn)(V)                   # (ns,)
        right = (jnp.arange(N) + 1) % N
        E_diag = -jnp.sum(V * V[:, right], axis=1)           # (ns,)

        def ratio_for_site(i):
            mask = jax.nn.one_hot(i, N, dtype=jnp.float64)
            V_flip = V * (1.0 - 2.0 * mask[None, :])         # (ns, N)
            return jnp.exp(jax.vmap(log_psi_fn)(V_flip) - log_p_V)  # (ns,)

        all_ratios = jax.vmap(ratio_for_site)(jnp.arange(N))  # (N, ns)
        E_off = -h * jnp.sum(all_ratios, axis=0)              # (ns,)
        return E_diag + E_off

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
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        if hasattr(rbm, "J"):
            E_off = _fbm_off_diagonal_jit(V_jax, rbm.W, rbm.a, rbm.b, rbm.J, self.h, self.size)
            N, L = self.size, self.linear_size
            i_idx = jnp.arange(N)
            right_idx = (i_idx // L) * L + (i_idx % L + 1) % L
            down_idx = ((i_idx // L + 1) % L) * L + i_idx % L
            E_diag = -jnp.sum(V_jax * V_jax[:, right_idx] + V_jax * V_jax[:, down_idx], axis=1)
            return E_diag + E_off
        return _local_energy_2d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.h, self.size, self.linear_size
        )

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """
        Generic 2D TFIM local energy via log_psi_fn (single-spin flips).

        log_psi_fn : callable (N,) → scalar
        """
        N = self.size
        L = self.linear_size
        h = self.h
        log_p_V = jax.vmap(log_psi_fn)(V)                    # (ns,)

        i_idx = jnp.arange(N)
        right_idx = (i_idx // L) * L + (i_idx % L + 1) % L
        down_idx  = ((i_idx // L + 1) % L) * L + i_idx % L
        E_diag = -jnp.sum(V * V[:, right_idx] + V * V[:, down_idx], axis=1)  # (ns,)

        def ratio_for_site(i):
            mask = jax.nn.one_hot(i, N, dtype=jnp.float64)
            V_flip = V * (1.0 - 2.0 * mask[None, :])
            return jnp.exp(jax.vmap(log_psi_fn)(V_flip) - log_p_V)

        all_ratios = jax.vmap(ratio_for_site)(jnp.arange(N))  # (N, ns)
        E_off = -h * jnp.sum(all_ratios, axis=0)
        return E_diag + E_off

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
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        if hasattr(rbm, "J"):
            # XXZ uses pair flips; no single-flip JIT kernel for FBM — use generic
            return self.local_energy_batch_generic(V_jax, rbm.log_psi)
        return _local_energy_xxz_1d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.J, self.delta, self.size
        )

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """
        Generic 1D XXZ local energy via log_psi_fn (two-spin exchange).

        log_psi_fn : callable (N,) → scalar
        """
        N = self.size
        J, delta = self.J, self.delta
        log_p_V = jax.vmap(log_psi_fn)(V)                    # (ns,)

        right = (jnp.arange(N) + 1) % N
        E_diag = J * delta * jnp.sum(V * V[:, right], axis=1)  # (ns,)

        def exchange_ratio_for_bond(i):
            j = (i + 1) % N
            mask = (jax.nn.one_hot(i, N, dtype=jnp.float64)
                    + jax.nn.one_hot(j, N, dtype=jnp.float64))
            V_flip = V * (1.0 - 2.0 * mask[None, :])         # (ns, N)
            return jnp.exp(jax.vmap(log_psi_fn)(V_flip) - log_p_V)  # (ns,)

        all_ratios = jax.vmap(exchange_ratio_for_bond)(jnp.arange(N))  # (N, ns)
        exchange = (1.0 - V * V[:, right]).T                  # (N, ns)
        E_off = J * jnp.sum(exchange * all_ratios, axis=0)    # (ns,)
        return E_diag + E_off

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


# ---------------------------------------------------------------------------
# Long-range 1D TFIM
# ---------------------------------------------------------------------------


class LongRangeTFIM1D(IsingModel):
    """
    1D Transverse Field Ising Model with power-law interactions.

        H = −J Σᵢ<ⱼ σᶻᵢσᶻⱼ / d(i,j)^α  −  h Σᵢ σˣᵢ

    where d(i,j) = min(|i−j|, N−|i−j|)  (chord distance on a ring).

    Special cases:
        α → ∞  : nearest-neighbor TFIM
        α = 2  : dipolar-like
        α = 1  : Coulomb-like
        α = 0  : all-to-all (mean-field Ising)

    J > 0 : ferromagnetic.  Off-diagonal term is identical to the
    nearest-neighbor TFIM (single-spin flips); only E_diag becomes O(N²).
    """

    def __init__(self, size: int, h: float = 0.5, alpha: float = 2.0, J: float = 1.0):
        super().__init__(size, h)
        self.alpha = alpha
        self.J = J

    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        N = self.size
        E_diag = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = min(j - i, N - (j - i))
                E_diag -= self.J * v[i] * v[j] / d ** self.alpha
        E_off = -self.h * sum(psi_ratio_fn(v, i) for i in range(N))
        return E_diag + E_off

    def local_energy_batch(self, V, rbm) -> jax.Array:
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        if hasattr(rbm, "J"):
            E_off = _fbm_off_diagonal_jit(V_jax, rbm.W, rbm.a, rbm.b, rbm.J, self.h, self.size)
            N = self.size
            idx = jnp.arange(N)
            raw_dist = jnp.abs(idx[:, None] - idx[None, :])
            dist = jnp.minimum(raw_dist, N - raw_dist).astype(jnp.float64)
            safe_dist = jnp.where(dist > 0.0, dist, 1.0)
            J_mat = jnp.where(dist > 0.0, self.J / safe_dist ** self.alpha, 0.0)
            E_diag = -0.5 * jnp.einsum("si,ij,sj->s", V_jax, J_mat, V_jax)
            return E_diag + E_off
        return _local_energy_lr1d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.J, self.h, self.alpha, self.size
        )

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """Generic LR-TFIM local energy (single-spin flips, all-to-all diagonal)."""
        N = self.size
        J, h, alpha = self.J, self.h, self.alpha
        log_p_V = jax.vmap(log_psi_fn)(V)

        # All-pair diagonal coupling
        idx = jnp.arange(N)
        raw_dist = jnp.abs(idx[:, None] - idx[None, :])
        dist = jnp.minimum(raw_dist, N - raw_dist).astype(jnp.float64)
        safe_dist = jnp.where(dist > 0.0, dist, 1.0)
        J_mat = jnp.where(dist > 0.0, J / safe_dist**alpha, 0.0)
        E_diag = -0.5 * jnp.einsum("si,ij,sj->s", V, J_mat, V)

        def ratio_for_site(i):
            mask = jax.nn.one_hot(i, N, dtype=jnp.float64)
            V_flip = V * (1.0 - 2.0 * mask[None, :])
            return jnp.exp(jax.vmap(log_psi_fn)(V_flip) - log_p_V)

        all_ratios = jax.vmap(ratio_for_site)(jnp.arange(N))
        E_off = -h * jnp.sum(all_ratios, axis=0)
        return E_diag + E_off

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute

        if self.size > 16:
            raise NotImplementedError(
                f"Exact diagonalization not feasible for LR-TFIM N={self.size}. "
                "Only N ≤ 16 is supported (2^16 = 65 536 states)."
            )
        model_key = f"lr_tfim_1d_alpha{self.alpha:.10g}_J{self.J:.10g}"
        return get_or_compute(model_key, self.size, self.h, self._compute_exact_ground_energy)

    def _compute_exact_ground_energy(self) -> float:
        """
        Build the LR-TFIM Hamiltonian as a scipy sparse matrix.
        No netket dependency — works on Python 3.13+.

        Encoding: bit (N-1-i) of integer s represents site i.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        N, h, alpha, J = self.size, self.h, self.alpha, self.J
        dim = 2 ** N

        def spin(s: int, i: int) -> int:
            return 1 - 2 * ((s >> (N - 1 - i)) & 1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s in range(dim):
            diag = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = min(j - i, N - (j - i))
                    diag -= J * spin(s, i) * spin(s, j) / d ** alpha
            rows.append(s); cols.append(s); vals.append(diag)

            for i in range(N):
                s_flip = s ^ (1 << (N - 1 - i))
                rows.append(s_flip); cols.append(s); vals.append(-h)

        H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float)
        eigenvalues, _ = eigsh(H, k=1, which="SA")
        return float(eigenvalues[0])

    def get_neighbors(self, idx: int) -> list[int]:
        return [j for j in range(self.size) if j != idx]


# ---------------------------------------------------------------------------
# J1-J2 frustrated Ising chain
# ---------------------------------------------------------------------------


class J1J2Ising1D(IsingModel):
    """
    1D J₁-J₂ frustrated transverse-field Ising chain.

        H = -J1 Σᵢ σᶻᵢσᶻᵢ₊₁ - J2 Σᵢ σᶻᵢσᶻᵢ₊₂ - h Σᵢ σˣᵢ

    Classical Lifshitz point at J2/J1 = 0.5 (NN and NNN interactions compete).
    J2 = 0 reduces to the standard nearest-neighbor TFIM.
    """

    def __init__(self, size: int, J1: float = 1.0, J2: float = 0.5, h: float = 0.5):
        super().__init__(size, h)
        self.J1 = J1
        self.J2 = J2

    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        N = self.size
        E_diag = 0.0
        for i in range(N):
            E_diag -= self.J1 * v[i] * v[(i + 1) % N]
            E_diag -= self.J2 * v[i] * v[(i + 2) % N]
        E_off = -self.h * sum(psi_ratio_fn(v, i) for i in range(N))
        return E_diag + E_off

    def local_energy_batch(self, V, rbm) -> jax.Array:
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        if hasattr(rbm, "J"):
            E_off = _fbm_off_diagonal_jit(V_jax, rbm.W, rbm.a, rbm.b, rbm.J, self.h, self.size)
            idx = jnp.arange(self.size)
            right1 = (idx + 1) % self.size
            right2 = (idx + 2) % self.size
            E_diag = (
                -self.J1 * jnp.sum(V_jax * V_jax[:, right1], axis=1)
                - self.J2 * jnp.sum(V_jax * V_jax[:, right2], axis=1)
            )
            return E_diag + E_off
        return _local_energy_j1j2_1d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.J1, self.J2, self.h, self.size
        )

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """Generic J1-J2 1D local energy (single-spin flips)."""
        N = self.size
        J1, J2, h = self.J1, self.J2, self.h
        log_p_V = jax.vmap(log_psi_fn)(V)

        idx = jnp.arange(N)
        right1 = (idx + 1) % N
        right2 = (idx + 2) % N
        E_diag = (
            -J1 * jnp.sum(V * V[:, right1], axis=1)
            - J2 * jnp.sum(V * V[:, right2], axis=1)
        )

        def ratio_for_site(i):
            mask = jax.nn.one_hot(i, N, dtype=jnp.float64)
            V_flip = V * (1.0 - 2.0 * mask[None, :])
            return jnp.exp(jax.vmap(log_psi_fn)(V_flip) - log_p_V)

        all_ratios = jax.vmap(ratio_for_site)(jnp.arange(N))
        E_off = -h * jnp.sum(all_ratios, axis=0)
        return E_diag + E_off

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute
        if self.size > 16:
            raise NotImplementedError(
                f"Exact diagonalization not feasible for J1J2 N={self.size}. "
                "Only N ≤ 16 is supported (2^16 = 65 536 states)."
            )
        model_key = f"j1j2_1d_J1{self.J1:.10g}_J2{self.J2:.10g}"
        return get_or_compute(model_key, self.size, self.h, self._compute_exact_ground_energy)

    def _compute_exact_ground_energy(self) -> float:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        N, h, J1, J2 = self.size, self.h, self.J1, self.J2
        dim = 2 ** N

        def spin(s: int, i: int) -> int:
            return 1 - 2 * ((s >> (N - 1 - i)) & 1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s in range(dim):
            diag = 0.0
            for i in range(N):
                diag -= J1 * spin(s, i) * spin(s, (i + 1) % N)
                diag -= J2 * spin(s, i) * spin(s, (i + 2) % N)
            rows.append(s); cols.append(s); vals.append(diag)
            for i in range(N):
                s_flip = s ^ (1 << (N - 1 - i))
                rows.append(s_flip); cols.append(s); vals.append(-h)

        H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float)
        eigenvalues, _ = eigsh(H, k=1, which="SA")
        return float(eigenvalues[0])

    def get_neighbors(self, idx: int) -> list[int]:
        N = self.size
        return [
            (idx - 2) % N, (idx - 1) % N,
            (idx + 1) % N, (idx + 2) % N,
        ]


# ---------------------------------------------------------------------------
# Heisenberg XY chain
# ---------------------------------------------------------------------------


class HeisenbergXY1D(HeisenbergXXZ1D):
    """
    1D XY chain: Heisenberg XXZ with Δ = 0.

        H = J Σᵢ [σˣᵢσˣᵢ₊₁ + σʸᵢσʸᵢ₊₁]

    Uses the same JIT kernel and SR optimizer as HeisenbergXXZ1D.
    Exact energy via ED (inherited from XXZ, Δ=0 case).
    """

    def __init__(self, size: int, J: float = 1.0):
        super().__init__(size, J=J, delta=0.0)

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute
        # Share cache with HeisenbergXXZ1D(delta=0) to avoid redundant computation
        model_key = "heisenberg_xxz_1d_delta0"
        return get_or_compute(model_key, self.size, self.J, self._compute_exact_ground_energy)


# ---------------------------------------------------------------------------
# Heisenberg XXZ 2D square lattice
# ---------------------------------------------------------------------------


class HeisenbergXXZ2D(IsingModel):
    """
    2D XXZ Heisenberg model on an L×L square lattice with periodic BC.

        H = J Σ_bonds [σˣᵢσˣⱼ + σʸᵢσʸⱼ + Δ σᶻᵢσᶻⱼ]

    where bonds are nearest-neighbor pairs on the square lattice.

    Special cases:
        Δ = 1   → isotropic XXX Heisenberg (standard 2D benchmark)
        Δ = 0   → 2D XY model
        J > 0   → antiferromagnetic
        J < 0   → ferromagnetic

    The `size` argument is the linear dimension L (total N = L² spins).
    The `h` slot in the base class is unused (h=0).

    Note: `local_energy(v, rbm)` takes an RBM instance (needs psi_ratio_pair).
    """

    def __init__(self, size: int, J: float = 1.0, delta: float = 1.0):
        super().__init__(size * size, h=0.0)
        self.linear_size = size
        self.J = J
        self.delta = delta

    def local_energy(self, v: np.ndarray, rbm) -> float:
        v_jax = jnp.asarray(v, dtype=jnp.float64)
        L = self.linear_size
        E_diag = 0.0
        E_off = 0.0
        for i in range(self.size):
            right = (i // L) * L + (i % L + 1) % L
            down  = ((i // L + 1) % L) * L + i % L
            for j in [right, down]:
                E_diag += self.J * self.delta * v[i] * v[j]
                if v[i] != v[j]:
                    E_off += 2 * self.J * float(rbm.psi_ratio_pair(v_jax, i, j))
        return E_diag + E_off

    def local_energy_batch(self, V, rbm) -> jax.Array:
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        if hasattr(rbm, "J"):
            return self.local_energy_batch_generic(V_jax, rbm.log_psi)
        return _local_energy_xxz_2d_jit(
            V_jax, rbm.W, rbm.a, rbm.b, self.J, self.delta, self.size, self.linear_size
        )

    def local_energy_batch_generic(self, V: jax.Array, log_psi_fn) -> jax.Array:
        """Generic 2D XXZ local energy via log_psi_fn (two-spin exchange)."""
        N = self.size
        L = self.linear_size
        J, delta = self.J, self.delta
        log_p_V = jax.vmap(log_psi_fn)(V)                    # (ns,)

        i_idx = jnp.arange(N)
        right_idx = (i_idx // L) * L + (i_idx % L + 1) % L
        down_idx  = ((i_idx // L + 1) % L) * L + i_idx % L

        E_diag = J * delta * jnp.sum(
            V * V[:, right_idx] + V * V[:, down_idx], axis=1
        )  # (ns,)

        def exchange_ratio_for_bond(bond):
            # bond encodes (site_i, partner_j) as bond = i * N + j
            i = bond // N
            j = bond % N
            mask = (jax.nn.one_hot(i, N, dtype=jnp.float64)
                    + jax.nn.one_hot(j, N, dtype=jnp.float64))
            V_flip = V * (1.0 - 2.0 * mask[None, :])
            return jnp.exp(jax.vmap(log_psi_fn)(V_flip) - log_p_V)  # (ns,)

        # Build bond index array: right and down bonds
        right_bonds = i_idx * N + right_idx  # (N,)
        down_bonds  = i_idx * N + down_idx   # (N,)
        all_bonds   = jnp.concatenate([right_bonds, down_bonds])  # (2N,)

        all_ratios = jax.vmap(exchange_ratio_for_bond)(all_bonds)  # (2N, ns)

        right_exchange = (1.0 - V * V[:, right_idx]).T   # (N, ns)
        down_exchange  = (1.0 - V * V[:, down_idx]).T    # (N, ns)
        all_exchange = jnp.concatenate([right_exchange, down_exchange], axis=0)  # (2N, ns)

        E_off = J * jnp.sum(all_exchange * all_ratios, axis=0)  # (ns,)
        return E_diag + E_off

    def exact_ground_energy(self) -> float:
        from reference_energies import get_or_compute
        if self.linear_size > 4:
            raise NotImplementedError(
                f"Exact diagonalization not feasible for 2D Heisenberg L={self.linear_size}. "
                "Only L ≤ 4 (N ≤ 16) is supported."
            )
        model_key = f"heisenberg_xxz_2d_delta{self.delta:.10g}"
        return get_or_compute(model_key, self.linear_size, self.J, self._compute_exact_ground_energy)

    def _compute_exact_ground_energy(self) -> float:
        """
        Build the 2D XXZ Hamiltonian as a scipy sparse matrix.
        Bonds are right + down neighbors for each site (counted once each).
        Encoding: bit (N-1-i) of integer s represents site i.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        L = self.linear_size
        N = self.size
        dim = 2 ** N
        J, delta = self.J, self.delta

        def spin(s: int, i: int) -> int:
            return 1 - 2 * ((s >> (N - 1 - i)) & 1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s in range(dim):
            diag = 0.0
            for i in range(N):
                right = (i // L) * L + (i % L + 1) % L
                down  = ((i // L + 1) % L) * L + i % L
                diag += J * delta * (
                    spin(s, i) * spin(s, right) + spin(s, i) * spin(s, down)
                )
            rows.append(s); cols.append(s); vals.append(diag)

            for i in range(N):
                right = (i // L) * L + (i % L + 1) % L
                down  = ((i // L + 1) % L) * L + i % L
                for partner in [right, down]:
                    if spin(s, i) != spin(s, partner):
                        s_flip = s ^ (1 << (N - 1 - i)) ^ (1 << (N - 1 - partner))
                        rows.append(s_flip); cols.append(s); vals.append(2 * J)

        H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float)
        eigenvalues, _ = eigsh(H, k=1, which="SA")
        return float(eigenvalues[0])

    def get_neighbors(self, idx: int) -> list[int]:
        L = self.linear_size
        i = idx // L
        j = idx % L
        return [
            ((i - 1) % L) * L + j, ((i + 1) % L) * L + j,
            i * L + (j - 1) % L,   i * L + (j + 1) % L,
        ]
