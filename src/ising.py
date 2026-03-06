"""
HOMEWORK: Quantum Spin Models

Implement Ising Hamiltonians that the RBM will learn to simulate.

Key concepts:
- Hamiltonian H tells us the energy of any configuration
- Local energy E_loc(v) = ⟨v|H|Ψ⟩ / ⟨v|Ψ⟩ (not just expectation!)
- Local energies drive training through stochastic reconfiguration
"""

import numpy as np
from abc import ABC, abstractmethod
import math

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
        
        This is CRITICAL for VMC training!
        
        Formula (general):
        E_loc(v) = ⟨v|H|Ψ⟩ / ⟨v|Ψ⟩
        
        In practice, for Ising model with RBM ansatz:
        
        H = -∑_i h·S_x^i - ∑_ij J_ij·S_z^i S_z^j
        
        The local energy splits into:
        E_loc = (diagonal part) + (off-diagonal part)
        
        Diagonal: contribution from same configuration v
        Off-diagonal: sum over "flip" operations (requires psi_ratio)
        
        Parameters:
        - v: current spin configuration (±1 for each spin)
        - psi_ratio_fn: function that computes Ψ(v_flip) / Ψ(v)
                        Usage: ratio = psi_ratio_fn(v, flip_idx)
        
        Returns: scalar local energy
        """
        pass
    
    @abstractmethod
    def exact_ground_energy(self) -> float:
        """
        Return exact ground state energy (for validation).
        
        This is a reference value that the RBM should approach.
        For 1D Ising, there's an analytic solution.
        For 2D, use DMRG or numerical reference.
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, idx: int)-> list[int]:
        """Return indices of spins coupled to spin idx."""
        pass


class TransverseFieldIsing1D(IsingModel):
    """
    1D transverse field Ising model with periodic boundary conditions.
    
    Hamiltonian:
    H = -∑_i h·σ_x^i - ∑_i σ_z^i σ_z^{i+1}
    
    TASK 1: Implement local_energy()
    
    HINTS for local energy computation:
    
    The Hamiltonian has two terms:
    
    1. Transverse field term: -h·∑_i σ_x^i
       - σ_x flips the spin
       - Contribution: diagonal part (v=v) is 0 if using eigenbasis
       - Off-diagonal: sum_i h · (Ψ(v with v_i flipped) / Ψ(v))
       - Need psi_ratio_fn for each i
    
    2. Ising coupling: -∑_i σ_z^i σ_z^{i+1}
       - σ_z measurement gives eigenvalue (±1)
       - v_i * v_{i+1} = ±1 directly
       - Diagonal contribution: -∑_i v_i * v_{i+1}
    
    Total:
    E_loc = -∑_{neighbors (i,i+1)} v_i * v_{i+1}  [diagonal]
            -h * ∑_i (Ψ(v_flip_i) / Ψ(v))          [off-diagonal]
    
    Algorithm:
    1. Diagonal part: sum over neighbors, multiply spins
    2. Off-diagonal part: loop over each spin i
       - Call psi_ratio_fn(v, i) to get Ψ(v_i flipped) / Ψ(v)
       - Multiply by -h
       - Sum over all i
    """
    
    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        """
        IMPLEMENT THIS.
        """
        # Diagonal part: Ising coupling between neighbors
        E_diag = -sum([v[i]* v[i_n] 
                       for i in range(self.size) 
                       for i_n in self.get_neighbors(i)]) / 2
        
        # Off-diagonal part: transverse field
        E_off_diag = -self.h * sum([psi_ratio_fn(v,i) for i in range(self.size)])

        return E_diag + E_off_diag
    
    def exact_ground_energy(self) -> float:
        """
        TASK 2: Implement exact solution.
        
        For 1D transverse field Ising, the ground state energy is:
        
        E_0 = -(1/π) ∫_0^π dk cos(k) √[1 - (1/h)^2 · (1 - cos(k))^2]
        
        Wait, that looks complex. Actually for the model:
        H = -∑_i σ_x^i - λ·∑_i σ_z^i σ_z^{i+1}
        
        The ground energy (per spin) is:
        e_0(λ) = -(1/π) ∫_0^π dk [λ^2 + 1 - 2λ·cos(k)]^{1/2}
        
        HINTS:
        - Use numerical integration (scipy.integrate.quad)
        - The integral is smooth, so simple quadrature works
        - Result scales with system size: E_0 = e_0 * N
        - h and λ are related (check coupling ratio)
        
        For this code, assume: λ = 1 (unit Ising coupling), h = transverse field
        The dispersion relation is: ω(k) = √[(1-h·cos(k))^2 + h^2·sin(k)^2]
        
        Actually, simpler: For h ≠ 1, use:
        E_0 / N = -2·∫_0^π dk cos(k) √[h^2 + 1 - 2h·cos(k)] / (2π)
        
        Or even simpler: just return reference value for common h values:
        """
        
        from scipy.integrate import quad
        import numpy as np

        def integrand(k):
            return np.sqrt((self.h - np.cos(k))**2 + np.sin(k)**2)

        result, _ = quad(integrand, 0, np.pi)

        return -result / np.pi * self.size
    
    def get_neighbors(self, idx: int):
        """Return neighbor indices for spin idx (periodic BC)."""
        left = (idx - 1) % self.size
        right = (idx + 1) % self.size
        return [left, right]


class TransverseFieldIsing2D(IsingModel):
    """
    2D transverse field Ising model on square lattice.
    
    Hamiltonian: H = -h·∑_i σ_x^i - ∑_ij σ_z^i σ_z^j
    
    TASK 3: Implement local_energy()
    
    Similar to 1D, but now each spin has 4 neighbors (up, down, left, right).
    Periodic boundary conditions.
    
    Local energy formula is the same structure:
    E_loc = (diagonal Ising interaction) + (transverse field off-diagonal)
    
    The main difference:
    - get_neighbors(idx) returns 4 indices instead of 2
    - Otherwise, implementation mirrors 1D
    """
    
    def __init__(self, size: int, h: float = 1.0):
        """size: linear dimension (total N = size^2 spins)."""
        super().__init__(size * size, h)
        self.linear_size = size  # For 2D indexing
    
    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
        """
        IMPLEMENT THIS (similar to 1D, but with 4 neighbors per spin).
        """
        E_diag = -sum([v[i]* v[i_n] 
                       for i in range(self.size) 
                       for i_n in self.get_neighbors(i)]) / 2
        
        # Off-diagonal part: transverse field
        E_off_diag = -self.h * sum([psi_ratio_fn(v,i) for i in range(self.size)])
        return E_diag + E_off_diag
    
    def exact_ground_energy(self) -> float:
        """
        For 2D, exact solution is not known in closed form.
        
        TASK 4 (Optional): Provide reference value.
        
        Options:
        1. Use DMRG library (expensive, but accurate)
        2. Hardcode reference values from literature for small systems
        3. Use simple bounds (e.g., mean-field approximation)
        
        For now, return a placeholder based on mean-field:
        E_0 ≈ -2N (very rough lower bound)
        """
        # TODO: Replace with actual reference
        return -2 * self.size  # Placeholder
    
    def get_neighbors(self, idx: int):
        """
        Return neighbor indices on 2D square lattice (periodic BC).
        
        IMPLEMENT THIS:
        
        Convert 1D index to 2D:
        - i, j = idx // linear_size, idx % linear_size
        
        Get 4 neighbors (periodic):
        - up: (i-1) % linear_size, j
        - down: (i+1) % linear_size, j
        - left: i, (j-1) % linear_size
        - right: i, (j+1) % linear_size
        
        Convert back to 1D: idx = i * linear_size + j
        
        Return list of 4 neighbor indices.
        """
        i = idx // self.linear_size
        j = idx % self.linear_size
        
        neighbors_2d = [
            ((i-1) % self.linear_size, j),  # up
            ((i+1) % self.linear_size, j),  # down
            (i, (j-1) % self.linear_size),  # left
            (i, (j+1) % self.linear_size),  # right
        ]
        
        # Convert back to 1D indices
        return [i*self.linear_size + j for i, j in neighbors_2d]


# Test: Verify local energy computation

if __name__ == "__main__":
    ising = TransverseFieldIsing1D(size=3, h=0.5)
    
    # Simple RBM mock
    class MockRBM:
        def psi_ratio(self, v, flip_idx):
            # Dummy: just return 1 (no actual ratio)
            return 1.0
    
    rbm = MockRBM()
    v = np.array([-1,1,1])
    
    E_loc = ising.local_energy(v, lambda v, i: rbm.psi_ratio(v, i))
    print(f"Local energy: {E_loc}")
    
    E_exact = ising.exact_ground_energy()
    print(f"Exact ground energy: {E_exact}")
    
    # 2D test
    ising2d = TransverseFieldIsing2D(size=4, h=1.0)
    v2d = np.random.choice([-1, 1], size=16)
    E_loc_2d = ising2d.local_energy(v2d, lambda v, i: 1.0)
    print(f"2D local energy: {E_loc_2d}")
