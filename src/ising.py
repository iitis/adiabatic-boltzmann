import numpy as np
from abc import ABC, abstractmethod


class IsingModel(ABC):
    """Abstract Ising model base class."""

    def __init__(self, size: int, h: float = 1.0):
        """
        size: number of spins
        h: transverse field strength (or coupling strength)
        """
        self.size = size
        self.h = h

    @abstractmethod
    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        """
        Compute local energy E_loc(v) for configuration v.

        Parameters:
        - v: current spin configuration (±1 for each spin)
        - psi_ratio_fn: function that computes Ψ(v_flip) / Ψ(v)
                        Usage: ratio = psi_ratio_fn(v, flip_idx)

        Returns: scalar local energy
        """
        pass

    @abstractmethod
    def local_energy_batch(self, V: np.ndarray, rbm) -> np.ndarray:
        """
        Compute local energies for a batch of configurations.

        V   : (n_samples, n_visible)  spin configurations in {-1, +1}
        rbm : RBM instance (needs .a, .b, .W, .logcosh)

        Returns: (n_samples,) array of local energies.

        Vectorised over samples — avoids Python loop over configurations.
        """
        pass

    @abstractmethod
    def exact_ground_energy(self) -> float:
        """
        Return exact ground state energy (for validation).

        This is a reference value that the RBM should approach.
        """
        pass

    @abstractmethod
    def get_neighbors(self, idx: int) -> list[int]:
        """Return indices of spins coupled to spin idx."""
        pass


class TransverseFieldIsing1D(IsingModel):
    """
    1D transverse field Ising model with periodic boundary conditions.
    """

    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        """
        Parameters:
        - v: current spin configuration (±1 for each spin)
        - psi_ratio_fn: function that computes Ψ(v_flip) / Ψ(v)
        """
        # Diagonal part: Ising coupling between neighbors
        E_diag = (
            -sum(
                [
                    v[i] * v[i_n]
                    for i in range(self.size)
                    for i_n in self.get_neighbors(i)
                ]
            )
            / 2
        )

        # Off-diagonal part: transverse field
        E_off_diag = -self.h * sum([psi_ratio_fn(v, i) for i in range(self.size)])
        return E_diag + E_off_diag

    def exact_ground_energy_netket(self):
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
        from scipy.sparse.linalg import eigsh

        vals, _ = eigsh(H_sparse, k=1, which="SA")
        return vals[0]

    def local_energy_batch(self, V: np.ndarray, rbm) -> np.ndarray:
        """
        Batched local energy over all samples simultaneously.

        Diagonal term:  E_diag(v) = -Σ_{bonds} v_i * v_j
        Off-diagonal:   E_off(v)  = -h * Σ_i Ψ(v_flip_i)/Ψ(v)

        The psi_ratio for flipping spin i across all samples is:

            log_ratio_i(s) = a_i * V[s,i]
                        + 0.5 * Σ_j [logcosh(θ'_ij) - logcosh(θ_ij)]

        where θ'_ij = θ_ij - 2*V[s,i]*W[i,j]
        """
        ns, N = V.shape

        # θ[s, j] = b_j + Σ_i W_ij * v_si  —  shape (ns, n_hidden)
        theta = V @ rbm.W + rbm.b[None, :]  # (ns, n_hidden)
        base = rbm.logcosh(theta)  # (ns, n_hidden)

        # Off-diagonal: sum psi_ratio over all spin flips
        transverse = np.zeros(ns, dtype=np.float64)
        for i in range(N):
            # θ after flipping spin i:  θ' = θ - 2*v_i * W[i, :]
            # shape: (ns, n_hidden)
            theta_flipped = theta - 2.0 * V[:, i : i + 1] * rbm.W[i, :]

            log_ratio = rbm.a[i] * V[:, i] + 0.5 * np.sum(
                rbm.logcosh(theta_flipped) - base, axis=1
            )
            transverse += np.exp(log_ratio)

        E_off_diag = -self.h * transverse

        # Diagonal: vectorised bond sum using precomputed edge arrays
        # Build right-neighbour array once (periodic BC)
        right = (np.arange(N) + 1) % N  # shape (N,)
        E_diag = -np.sum(V * V[:, right], axis=1)  # (ns,)  one bond per site

        return E_diag + E_off_diag

    def exact_ground_energy(self) -> float:
        """
        TASK 2: Implement exact solution.
        """

        from scipy.integrate import quad
        import numpy as np

        def integrand(k):
            return np.sqrt((self.h - np.cos(k)) ** 2 + np.sin(k) ** 2)

        result, _ = quad(integrand, 0, np.pi)

        return -result / np.pi * self.size

    ###
    def get_neighbors(self, idx: int):
        """Return neighbor indices for spin idx (periodic BC)."""
        left = (idx - 1) % self.size
        right = (idx + 1) % self.size
        return [left, right]


class TransverseFieldIsing2D(IsingModel):
    """
    2D transverse field Ising model on square lattice.

    """

    def __init__(self, size: int, h: float = 1.0):
        """size: linear dimension (total N = size^2 spins)."""
        super().__init__(size * size, h)
        self.linear_size = size  # For 2D indexing
        print("2dddd")

    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        E_diag = (
            -sum(
                [
                    v[i] * v[i_n]
                    for i in range(self.size)
                    for i_n in self.get_neighbors(i)
                ]
            )
            / 2
        )

        # Off-diagonal part: transverse field
        E_off_diag = -self.h * sum([psi_ratio_fn(v, i) for i in range(self.size)])
        return E_diag + E_off_diag

    def exact_ground_energy(self) -> float:
        """
        For 2D, exact solution is not known in closed form.
        """
        # TODO: Replace with actual reference
        return -2 * self.size  # Placeholder

    def get_neighbors(self, idx: int):
        """
        Return neighbor indices on 2D square lattice (periodic BC).

        Return list of 4 neighbor indices.
        """
        i = idx // self.linear_size
        j = idx % self.linear_size

        neighbors_2d = [
            ((i - 1) % self.linear_size, j),  # up
            ((i + 1) % self.linear_size, j),  # down
            (i, (j - 1) % self.linear_size),  # left
            (i, (j + 1) % self.linear_size),  # right
        ]

        # Convert back to 1D indices
        return [i * self.linear_size + j for i, j in neighbors_2d]
