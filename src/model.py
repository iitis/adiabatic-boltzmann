"""
Restricted Boltzmann Machine Implementation

Key reference: Gardas et al., Eq. 6-7 (ansatz) and Eq. 15 (gradients)

Wave function: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
"""

import numpy as np
from abc import ABC, abstractmethod


class RBM(ABC):
    """Abstract RBM base class."""

    def __init__(self, n_visible: int, n_hidden: int):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        scale = 0.01

        self.a = np.random.normal(0, scale, n_visible)
        self.b = np.random.normal(0, scale, n_hidden)
        self.W = np.random.normal(0, scale, (n_visible, n_hidden))

    def logcosh(self, x):
        return np.logaddexp(x, -x)

    @abstractmethod
    def get_connectivity_mask(self):
        """Return (n_visible, n_hidden) binary mask indicating W[i,j] != 0."""
        pass

    def psi(self, v: np.ndarray) -> float:
        """Compute Ψ(v) - wave function value for configuration v.
        
        From paper: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
        """
        first_term = np.exp(-(self.a @ v) / 2)
        second_term = np.prod(
            [2 * np.cosh(self.b[i] + self.W.T[i] @ v) for i in range(self.n_hidden)]
        )
        return float(first_term * np.sqrt(second_term))

    def log_psi(self, v: np.ndarray) -> float:
        """
        Compute log(Ψ(v)) - log of the wave function.

        v: spin configuration, shape (n_visible,), values in {-1, +1}

        Returns: scalar log(Ψ(v))

        From Gardas et al., Eq. 6-7:
        Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
        log(Ψ) = -a·v/2 + (1/2) ∑_j log[2·cosh(b_j + W_j·v)]
        """

        first_term = -(self.a @ v) / 2
        theta = self.b + self.W.T @ v
        # log(2*cosh(x)) = log(2) + log(cosh(x))
        second_term = 0.5 * np.sum(np.log(2) + self.logcosh(theta))
        return first_term + second_term

    def psi_ratio_old(self, v: np.ndarray, flip_idx: int) -> float:
        """
        Compute Ψ(v_flip) / Ψ(v) when flipping spin at index flip_idx. Optimized version don't compute psi twice
        v: current configuration
        flip_idx: which spin to flip (0 to n_visible-1)

        Returns: Ψ(v with spin flip_idx flipped) / Ψ(v)

        """
        v_flipped = np.copy(v)
        v_flipped[flip_idx] *= -1
        return self.psi(v_flipped) / self.psi(v)

    def psi_ratio(self, v: np.ndarray, flip_idx: int) -> float:
        vi = v[flip_idx]
        # log change from visible bias: -a[i]*(-v_i)/2 - (-a[i]*v_i/2) = a[i]*v_i
        log_ratio_a = self.a[flip_idx] * vi
        theta = self.b + self.W.T @ v
        theta_flipped = theta - 2 * vi * self.W[flip_idx, :]

        # log change from hidden part: (1/2) * [log(2*cosh(theta')) - log(2*cosh(theta))]
        log_ratio_cosh = 0.5 * np.sum(self.logcosh(theta_flipped) - self.logcosh(theta))

        return np.exp(log_ratio_a + log_ratio_cosh)

    def gradient_log_psi(self, v: np.ndarray) -> dict:
        """
        Compute ∂log(Ψ)/∂p for all parameters p.

        This implements Equation 15 from the paper.

        From Gardas et al., Eq. 6-7:
        Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
        
        ∂log(Ψ)/∂a_i = -v_i/2
        ∂log(Ψ)/∂b_j = (1/2) tanh(θ_j)  
        ∂log(Ψ)/∂W_ij = (1/2) v_i tanh(θ_j)
        
        Returns dict with keys 'a', 'b', 'W' containing gradients of same shape as weights.
        """
        theta = self.b + self.W.T @ v  # Hidden unit activations

        grad_a = -0.5 * np.copy(v)
        grad_b = 0.5 * np.tanh(theta)
        grad_W = 0.5 * np.outer(v, np.tanh(theta))

        return {"a": grad_a, "b": grad_b, "W": grad_W}

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a 1D vector for SR algorithm."""
        return np.concatenate([self.a.flatten(), self.b.flatten(), self.W.flatten()])

    def set_weights(self, w: np.ndarray):
        """Unflatten 1D vector back to weights."""
        idx = 0
        n_a = self.n_visible
        n_b = self.n_hidden
        n_w = self.n_visible * self.n_hidden

        self.a = w[idx : idx + n_a].copy()
        idx += n_a
        self.b = w[idx : idx + n_b].copy()
        idx += n_b
        self.W = w[idx : idx + n_w].reshape(self.n_visible, self.n_hidden).copy()


class FullyConnectedRBM(RBM):
    """RBM with all visible-hidden connections (no topology constraint)."""

    def get_connectivity_mask(self):
        """Return mask of all ones (fully connected)."""
        return np.ones((self.n_visible, self.n_hidden))


class DWaveTopologyRBM(RBM):
    """
    RBM constrained to D-Wave Chimera/Pegasus-like topology.

    """

    def get_connectivity_mask(self):
        """Create sparse connectivity mask for D-Wave-like topology."""
        mask = np.zeros((self.n_visible, self.n_hidden))

        # TODO: Fill in sparse pattern
        # Pattern idea: each hidden unit j connects to ~2 visible units

        return mask

    def gradient_log_psi(self, v: np.ndarray) -> dict:
        """
        Override to apply connectivity mask to gradients.

        Compute gradients, then zero out entries not in the mask.
        This ensures sparse updates during training.
        """
        # TODO:
        # 1. Call parent's gradient_log_psi (or reimplement)
        # 2. Apply mask to grad_W: grad_W *= self.get_connectivity_mask()
        # 3. Return masked gradients
        gradients = super().gradient_log_psi(v)
        gradients["W"] *= self.get_connectivity_mask()
        pass
