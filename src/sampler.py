"""
HOMEWORK: Sampling Algorithms

Implement two classical sampling methods:
1. Metropolis-Hastings (locally equilibrate)
2. Simulated Annealing (explore landscape with temperature schedule)

Key challenge: Connect the RBM (which gives Ψ) to Boltzmann statistics.
"""

import numpy as np
from abc import ABC, abstractmethod
from model import RBM


class Sampler(ABC):
    """Abstract sampling interface."""
    
    @abstractmethod
    def sample(self, rbm, n_samples: int, config: dict = None) -> np.ndarray:
        """
        Generate samples from the RBM distribution.
        
        rbm: the RBM instance (has log_psi, psi_ratio methods)
        n_samples: how many samples to draw
        config: optional configuration dict
        
        Returns: (n_samples, n_visible) array of spin configurations
        """
        pass


class ClassicalSampler(Sampler):
    """
    Classical sampling via Metropolis-Hastings or Simulated Annealing.
    
    TASK 1: Implement Metropolis-Hastings sampler
    
    Algorithm:
    - Start from random configuration
    - Propose: flip one random spin
    - Accept with probability: min(1, |Ψ_new/Ψ_old|^2)
    - Repeat until equilibrium, then collect samples
    
    Why |Ψ|^2? Because that's the probability distribution P(v) = |Ψ(v)|^2
    """
    
    def __init__(self):
        self.method = "metropolis"  # Can also be "simulated_annealing"
    
    def sample(self, rbm: RBM, n_samples: int, config: dict = None) -> np.ndarray:
        if config is None:
            config = {}
        
        method = config.get('method', 'metropolis')
        
        if method == 'metropolis':
            return self._metropolis_hastings(rbm, n_samples, config)
        elif method == 'simulated_annealing':
            return self._simulated_annealing(rbm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _metropolis_hastings(self, rbm: RBM, n_samples: int, config: dict) -> np.ndarray:
        """
        IMPLEMENT THIS:
        
        Metropolis-Hastings sampling using spin flips.
        
        Parameters from config (with defaults):
        - n_sweeps: int, equilibration steps (default ???)
        - n_between: int, steps between samples (default ???)
        
        PSEUDOCODE:
        1. Initialize v = random ±1 configuration
        2. Equilibrate: perform n_sweeps spin flips
           - For each sweep: try to flip each spin once
           - For each flip attempt:
             a. Compute ratio = |Ψ_new/Ψ_old|^2 (use psi_ratio!)
             b. Accept with probability min(1, ratio)
             c. If accept: flip the spin, else keep old
        3. Collect samples: repeat n_samples times
           - Do n_between flips
           - Store current v
        
        HINTS:
        - Use np.random.random() < probability for acceptance
        - psi_ratio is already Ψ_new/Ψ_old, so ratio = psi_ratio^2
        - Random spin choice: np.random.randint(0, n_visible)
        - For numerical stability: cap acceptance at 1.0
        
        Hyperparameters to try:
        - n_sweeps: 100-1000 (more = better equilibration)
        - n_between: 1-10 (for uncorrelated samples)
        """
        n_visible = rbm.n_visible
        n_sweeps = config.get('n_sweeps', 100)
        n_between = config.get('n_between', 5)
        
        # Initialize
        v = (2 * np.random.randint(0, 2, n_visible) - 1).astype(float)
        samples = []
        
        # Equilibrium
        spin_flip_array = np.random.randint(0,len(v), n_sweeps)
        for spin_flip_idx in spin_flip_array:
            ratio_squared = rbm.psi_ratio(v,spin_flip_idx) ** 2
            if np.random.random() < min(1,ratio_squared):
                v[spin_flip_idx] *= -1 

        # Sample collection
        for _ in range(n_samples):
            spin_flips_array_between = np.random.randint(0,len(v),n_between)
            for spin_flip_idx in spin_flips_array_between:
                ratio_squared = rbm.psi_ratio(v,spin_flip_idx) ** 2
                if np.random.random() < min(1,ratio_squared):
                    v[spin_flip_idx] *= -1 
        
            samples.append(np.copy(v))

        return np.array(samples)
    
    def _simulated_annealing(self, rbm, n_samples: int, config: dict) -> np.ndarray:
        """
        IMPLEMENT THIS:
        
        Simulated Annealing: gradually lower temperature as you sample.
        
        This is a temperature-based Metropolis variant:
        - Acceptance probability: min(1, (Ψ_new/Ψ_old)^2)^(1/T)
        - Or equivalently: min(1, exp(-β·ΔE)) where β=1/T
        
        Parameters from config:
        - T_initial: starting temperature (default ???)
        - T_final: final temperature (default ???)
        - n_steps: total annealing steps (default ???)
        
        PSEUDOCODE:
        1. Initialize v = random configuration
        2. Create temperature schedule: T(step) = T_initial * (T_final/T_initial)^(step/n_steps)
           (This is geometric cooling - exponential decay)
        3. For each step from 0 to n_steps:
           a. Get current T = schedule[step]
           b. Flip one random spin
           c. Compute ratio = psi_ratio^2
           d. Compute acceptance = min(1, ratio^(1/T))
           e. Accept with that probability
           f. Every k steps, store sample
        
        HINTS:
        - ratio^(1/T) = exp((1/T) * log(ratio))
        - With T high initially, acceptance is easier (exploration)
        - With T low finally, only good moves accepted (exploitation)
        - Use geometric schedule: T = T_init * (T_final/T_init)^(i/N)
        
        Hyperparameters:
        - T_initial: 1.0 - 10.0 (higher = more exploration)
        - T_final: 0.01 - 0.1 (lower = more greedy)
        - n_steps: 1000 - 10000
        """
        T_initial = config.get('T_initial', 10)
        T_final = config.get('T_final', 0.05)
        n_steps = config.get('n_steps', int(1e5))
        K = 10
        
        n_visible = rbm.n_visible
        v = (2 * np.random.randint(0, 2, n_visible) - 1).astype(float)
        samples = []
        
        # Create temperature schedule
        def schedule(step: int) -> float:
            return T_initial * (T_final/T_initial) ** (step/n_steps) # Geometric
        
        # Equilibrium
        spin_flip_array = np.random.randint(0,len(v), n_steps)
        for step,spin_flip_idx in enumerate(spin_flip_array):
            T = schedule(step)
            ratio_squared = rbm.psi_ratio(v,spin_flip_idx) ** 2
            if np.random.random() < min(1,ratio_squared**(1/T)):
                v[spin_flip_idx] *= -1

            if step % (n_steps // n_samples) == 0:
                samples.append(np.copy(v)) 

        
        return np.array(samples)


class DWaveSampler(Sampler):
    """
    TASK 2 (Advanced): D-Wave quantum annealer interface.
    
    This is a STUB for now. In reality:
    - Embed RBM onto D-Wave Chimera/Pegasus graph
    - Set couplings based on W matrix
    - Set fields based on a, b vectors
    - Run quantum annealing
    - Collect samples from results
    
    For homework, you can:
    1. Use ClassicalSampler as mock (no actual quantum hardware needed)
    2. Or implement a simple hybrid classical-quantum variant
    
    HINTS:
    - D-Wave SDK: dimod for problem specification, dwave.system for sampler
    - Problem: QUBO or Ising formulation
    - Embedding is nontrivial (chain strength, scaling)
    """
    
    def sample(self, rbm, n_samples: int, config: dict = None) -> np.ndarray:
        if config is None:
            config = {}
        
        use_hardware = config.get('use_hardware', False)
        
        if use_hardware:
            # TODO: Real D-Wave implementation
            # - Import dimod, dwave.system
            # - Construct QUBO from RBM parameters
            # - Embed onto hardware
            # - Submit problem
            # - Return samples
            raise NotImplementedError("Real D-Wave hardware not set up yet")
        else:
            # Fallback: use classical simulated annealing (simple mock)
            print("WARNING: Using classical SA as D-Wave mock")
            classical = ClassicalSampler()
            config['method'] = 'simulated_annealing'
            return classical.sample(rbm, n_samples, config)


# For proper unit tests, see test_sampler.py
# Run with: pytest test_sampler.py -v
    
