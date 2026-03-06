"""
HOMEWORK: Restricted Boltzmann Machine Implementation

This is where you implement the wave function and its gradients.

Key reference: Gardas et al., Eq. 6-7 (ansatz) and Eq. 15 (gradients)

Wave function: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)

TASK 1: Implement log_psi() - the log of the wave function
TASK 2: Implement psi_ratio() - efficient ratio evaluation for flips
TASK 3: Implement gradient_log_psi() - Eq. 15 derivatives
"""

import numpy as np
from abc import ABC, abstractmethod


class RBM(ABC):
    """Abstract RBM base class."""
    
    def __init__(self, n_visible: int, n_hidden: int):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Initialize weights to small random values
        # HINT: Look at initialization scales in the paper's methods
        self.a = np.random.normal(0, 1, n_visible)
        self.b = np.random.normal(0, 1, n_hidden)
        self.W = np.random.normal(0, 1, (n_visible, n_hidden))
    
    @abstractmethod
    def get_connectivity_mask(self):
        """Return (n_visible, n_hidden) binary mask indicating W[i,j] != 0."""
        pass
    



    def log_psi(self, v: np.ndarray) -> float:
        """
        Compute log(Ψ(v)) - log of the wave function.
        
        v: spin configuration, shape (n_visible,), values in {-1, +1}
        
        Returns: scalar log(Ψ(v))
        
        HINTS:
        - First term: log(e^(-a·v/2)) = -a·v/2
        - Second term: sum over hidden units of log(2·cosh(...))
        - Use np.logcosh() for numerical stability!
        - What goes inside the cosh?
        """
        def logcosh(x):
            # s always has real part >= 0
            s = np.sign(x) * x
            p = np.exp(-2 * s)
            return s + np.log1p(p) - np.log(2)
        
        first_term = (self.a @ v)/2
        second_term = np.log(2)*self.n_hidden+sum([logcosh(self.b[j]+self.W.T[j]@v) for j in range(self.n_hidden)])

        return first_term + 0.5*second_term
    
    def psi_ratio(self, v: np.ndarray, flip_idx: int) -> float:
        """
        Compute Ψ(v_flip) / Ψ(v) when flipping spin at index flip_idx.
        
        This is critical for the local energy calculation in the Ising solver.
        
        v: current configuration
        flip_idx: which spin to flip (0 to n_visible-1)
        
        Returns: Ψ(v with spin flip_idx flipped) / Ψ(v)
        
        HINTS:
        - You can compute this as exp(log_psi(v_flip) - log_psi(v))
        - But there's a more efficient way!
        - Notice that changing one spin only affects:
          1. The a·v term
          2. The hidden unit activations involving that spin
        - Avoid computing full log_psi twice; instead compute log-ratio directly
        - After flip: v'[flip_idx] = -v[flip_idx]
        - What changes in the cosh terms?
        """
        v_flipped = np.copy(v)
        v_flipped[flip_idx]  *= -1
        return np.exp(self.log_psi(v_flipped) - self.log_psi(v))
    
    def gradient_log_psi(self, v: np.ndarray) -> dict:
        """
        Compute ∂log(Ψ)/∂p for all parameters p.
        
        This implements Equation 15 from the paper.
        
        Returns dict with keys 'a', 'b', 'W' containing gradients of same shape as weights.
        
        CRITICAL EQUATIONS (Eq. 15):
        For three parameter types:
        
        1. ∂log(Ψ)/∂a_i = ???  (HINT: First term in ansatz)
        
        2. ∂log(Ψ)/∂b_j = ???  (HINT: Derivative of log(cosh))
           - Remember: d/dx[log(cosh(x))] = tanh(x)
           - What x are we taking tanh of?
        
        3. ∂log(Ψ)/∂W_{ij} = ???  (HINT: Product rule)
           - Both the a term and the cosh term depend on W
           - Which term actually depends on W?
        
        Additional hints:
        - Precompute θ_j = b_j + W_j · v (activations)
        - All three gradients should use tanh(θ_j) somewhere
        - Be careful with the factor of 1/2 from the square root!
        """
        theta = self.b + self.W.T @ v  # Hidden unit activations
        
        grad_a = 0.5 * np.copy(v)
        grad_b = 0.5 * np.tanh(theta)
        grad_W = 0.5 * np.outer(v, np.tanh(theta))
        
        return {'a': grad_a, 'b': grad_b, 'W': grad_W}
    
    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a 1D vector for SR algorithm."""
        return np.concatenate([self.a.flatten(), self.b.flatten(), self.W.flatten()])
    
    def set_weights(self, w: np.ndarray):
        """Unflatten 1D vector back to weights."""
        idx = 0
        n_a = self.n_visible
        n_b = self.n_hidden
        n_w = self.n_visible * self.n_hidden
        
        self.a = w[idx:idx+n_a].copy()
        idx += n_a
        self.b = w[idx:idx+n_b].copy()
        idx += n_b
        self.W = w[idx:idx+n_w].reshape(self.n_visible, self.n_hidden).copy()


class FullyConnectedRBM(RBM):
    """RBM with all visible-hidden connections (no topology constraint)."""
    
    def get_connectivity_mask(self):
        """Return mask of all ones (fully connected)."""
        return np.ones((self.n_visible, self.n_hidden))


class DWaveTopologyRBM(RBM):
    """
    RBM constrained to D-Wave Chimera/Pegasus-like topology.
    
    TASK 4: Design a sparse connectivity pattern.
    
    HINTS:
    - D-Wave qubits have ~5 neighbors on average
    - Each hidden unit should connect to ~2 visible units
    - Use a simple pattern: hidden unit j connects to visible units j, (j+1)%n_visible
    - When computing gradients, apply the mask: only update connected W[i,j]
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
        gradients['W'] *= self.get_connectivity_mask()
        pass


# Test stubs (fill in your own tests!)

if __name__ == "__main__":
    # Test 1: Check gradient shapes
    rbm = FullyConnectedRBM(n_visible=4, n_hidden=3)
    v = np.array([1, -1, 1, -1])
    grad = rbm.gradient_log_psi(v)
    
    print("Gradient shapes:")
    print(f"  a: {grad['a'].shape}, should be (4,)")
    print(f"  b: {grad['b'].shape}, should be (3,)")
    print(f"  W: {grad['W'].shape}, should be (4, 3)")
    
    # Test 2: Check psi_ratio (critical for sampling!)
    # The ratio should always be positive (it's |Ψ_new/Ψ_old|)
    for flip_idx in range(v.shape[0]):
        ratio = rbm.psi_ratio(v, flip_idx)
        print(f"Flip {flip_idx}: ratio = {ratio:.4f}")
        assert ratio > 0, f"Ratio must be positive! Got {ratio}"
    
    # Test 3: Verify log_psi is real
    log_psi_val = rbm.log_psi(v)
    assert np.isreal(log_psi_val), f"log_psi should be real, got {log_psi_val}"
    print(f"\nlog_psi({v}) = {log_psi_val}")


    rbm = FullyConnectedRBM(4, 3)
    v = np.array([1., -1., 1., -1.])
    grad_analytical = rbm.gradient_log_psi(v)

    # Numerical gradient check
    eps = 1e-5
    grad_numerical = {'a': np.zeros(4), 'b': np.zeros(3), 'W': np.zeros((4, 3))}

    # Check 'a' gradients
    for i in range(rbm.n_visible):
        rbm.a[i] += eps
        f_plus = rbm.log_psi(v)
        rbm.a[i] -= 2*eps
        f_minus = rbm.log_psi(v)
        rbm.a[i] += eps
        grad_numerical['a'][i] = (f_plus - f_minus) / (2*eps)

    # Check 'b' gradients (similar)
    # Check 'W' gradients (similar)

    # Compare
    print("a gradient error:", np.max(np.abs(grad_analytical['a'] - grad_numerical['a'])))
    print("b gradient error:", np.max(np.abs(grad_analytical['b'] - grad_numerical['b'])))
    print("W gradient error:", np.max(np.abs(grad_analytical['W'] - grad_numerical['W'])))
