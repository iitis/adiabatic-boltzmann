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
    def exact_ground_energy(self) -> float:
        """
        Return exact ground state energy (for validation).
        
        This is a reference value that the RBM should approach.
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, idx: int)-> list[int]:
        """Return indices of spins coupled to spin idx."""
        pass


class TransverseFieldIsing1D(IsingModel):
    """
    1D transverse field Ising model with periodic boundary conditions.
    """
    
    def local_energy(self, v: np.ndarray, psi_ratio_fn) -> float:
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
            ((i-1) % self.linear_size, j),  # up
            ((i+1) % self.linear_size, j),  # down
            (i, (j-1) % self.linear_size),  # left
            (i, (j+1) % self.linear_size),  # right
        ]
        
        # Convert back to 1D indices
        return [i*self.linear_size + j for i, j in neighbors_2d]
