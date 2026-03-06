"""
HOMEWORK: Variational Training via Stochastic Reconfiguration

This is the core algorithm of the paper.

Key idea: Find RBM weights that minimize E = ⟨H⟩ = ⟨Ψ|H|Ψ⟩/⟨Ψ|Ψ⟩

Algorithm: Stochastic Reconfiguration (= Natural Gradient Descent)
- Compute energy gradient ∂E/∂w
- Use metric tensor (covariance matrix) to precondition update
- Solve linear system: S·x = F, then w ← w - γ·x
"""

import numpy as np


class Trainer:
    """
    Variational Monte Carlo trainer using Stochastic Reconfiguration.
    
    TASK 1: Implement the main training loop
    """
    
    def __init__(self, rbm, ising_model, sampler, config: dict = None):
        """
        rbm: RBM instance (has gradient_log_psi)
        ising_model: Ising Hamiltonian (has local_energy)
        sampler: Sampling algorithm (has sample method)
        config: training config (learning_rate, n_sweeps, etc.)
        """
        self.rbm = rbm
        self.ising = ising_model
        self.sampler = sampler
        
        if config is None:
            config = {}
        
        self.learning_rate = config.get('learning_rate', 0.1)
        self.n_iterations = config.get('n_iterations', 50)
        self.n_samples = config.get('n_samples', 1000)
        self.regularization = config.get('regularization', 1e-5)
        
        # History tracking
        self.history = {
            'energy': [],
            'error': [],
            'learning_rate': [],
        }
    
    def train(self) -> dict:
        """
        IMPLEMENT THIS:
        
        Main training loop.
        
        PSEUDOCODE:
        For each iteration:
            1. Sample configurations: v_1, v_2, ..., v_M from RBM
            2. Compute local energy for each: E_loc(v_i) = ⟨H⟩_{v_i}
            3. Compute RBM gradients for each: D_i = ∇log(Ψ(v_i))
            4. Assemble stochastic reconfiguration matrices (see below)
            5. Solve linear system: S·x = F
            6. Update weights: w ← w - γ·x
            7. Track: energy, error, other metrics
        
        Returns: history dict with convergence data
        """
        
        for iteration in range(self.n_iterations):
            # Sample from RBM
            samples = self.sampler.sample(self.rbm, self.n_samples)
            
            # Compute local energies and gradients
            local_energies = []
            gradients_list = []  # List of dicts
            
            for v in samples:
                # TODO: 
                # 1. Compute local energy: E_loc = ising.local_energy(v, self.rbm.psi_ratio)
                E_loc = self.ising.local_energy(v, self.rbm.psi_ratio)
                local_energies.append(E_loc)
                
                # 2. Compute gradient: grad = rbm.gradient_log_psi(v)
                grad = self.rbm.gradient_log_psi(v)
                gradients_list.append(grad)
            
            local_energies = np.array(local_energies)
            
            # TODO: Compute stochastic reconfiguration matrices S and F
            S, F = self._compute_sr_matrices(gradients_list, local_energies)
            
            # TODO: Solve S·x = F
            try:
                x = np.linalg.solve(S,F)  # np.linalg.solve(S, F)
            except np.linalg.LinAlgError:
                x = np.linalg.pinv(S) @ F  # np.linalg.pinv(S) @ F
            
            # TODO: Update weights
            w = self.rbm.get_weights()
            w_new = w - self.learning_rate * x  # w - learning_rate * x
            self.rbm.set_weights(w_new)
            
            # Track metrics
            E_mean = np.mean(local_energies)  # np.mean(local_energies)
            E_std = np.std(local_energies)  # np.std(local_energies)
            
            self.history['energy'].append(E_mean)
            self.history['error'].append(E_std)
            self.history['learning_rate'].append(self.learning_rate)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d}: E = {E_mean:.6f} ± {E_std:.6f}")
        
        return self.history
    
    def _compute_sr_matrices(self, gradients_list, local_energies) -> tuple:
        """
        IMPLEMENT THIS:
        
        Compute stochastic reconfiguration matrices from samples.
        
        Inputs:
        - gradients_list: list of dicts, each with keys 'a', 'b', 'W'
        - local_energies: array of energies for each sample
        
        Outputs:
        - S: (n_params, n_params) covariance matrix
        - F: (n_params,) force vector
        
        MATHEMATICAL FORMULAS (Eq. 10-13 in paper):
        
        First, flatten all gradients into a matrix D:
        - Each row i = sample i
        - Each column j = parameter j (flattened: a_0, a_1, ..., b_0, b_1, ..., W_00, W_01, ...)
        
        Then:
        
        1. Covariance matrix S:
           S_ij = ⟨⟨D_i* D_j⟩⟩_ρ - ⟨⟨D_i*⟩⟩ ⟨⟨D_j⟩⟩
           
           In code (for real gradients, no conjugate):
           S = (1/M) * D^T @ D - outer(mean_D, mean_D)
           
           Where:
           - D: (M, n_params) matrix of flattened gradients
           - mean_D: (n_params,) mean gradient over samples
           - M = number of samples
        
        2. Force vector F:
           F_j = ⟨⟨E * D_j⟩⟩_ρ - ⟨⟨E⟩⟩ ⟨⟨D_j⟩⟩
           
           In code:
           F = (1/M) * (E^T @ D) - mean_E * mean_D
           
           Where:
           - E: (M,) local energies
           - mean_E: scalar, mean energy
        
        3. Add regularization (numerical stability):
           S_reg = S + λ*I  where λ ~ 1e-5
        
        HINTS:
        - np.outer(u, v) computes outer product
        - D can be built by concatenating flattened gradients
        - Remember to normalize by 1/M
        - Regularization prevents singular S
        
        Returns: (S, F) where S is (n_params, n_params) and F is (n_params,)
        """
        
        # Convert list of gradient dicts to matrix
        D = []
        for grad_dict in gradients_list:
            # Flatten: [a, b, W]
            row = np.concatenate([
                grad_dict['a'].flatten(),
                grad_dict['b'].flatten(),
                grad_dict['W'].flatten(),
            ])
            D.append(row)
        
        D = np.array(D)  # Shape: (M, n_params)
        M = D.shape[0]
        
        # TODO: Compute S and F using formulas above
        
        mean_D = np.mean(D)  # Average over samples
        mean_E = np.mean(local_energies)  # Average energy
        
        # Covariance matrix
        S = (1/M) * D.T @ D - np.outer(mean_D, mean_D)
        
        # Force vector
        F = (1/M) * (local_energies.T @ D) - mean_E * mean_D
        
        # Regularization
        S += self.regularization * np.eye(S.shape[0])
        
        return S, F


class ExperimentRunner:
    """Utility for running multiple experiments and comparing architectures."""
    
    def run_experiment(self, ising_model, rbm, sampler, config: dict = None) -> dict:
        """Run single experiment and return results."""
        if config is None:
            config = {}
        
        trainer = Trainer(rbm, ising_model, sampler, config)
        history = trainer.train()
        
        return {
            'config': config,
            'history': history,
            'final_weights': {
                'a': rbm.a.copy(),
                'b': rbm.b.copy(),
                'W': rbm.W.copy(),
            }
        }
    
    def compare_architectures(self, ising_model, sampler, config: dict = None) -> dict:
        """
        OPTIONAL TASK: Compare fully-connected vs sparse RBM.
        
        Run two experiments:
        1. FullyConnectedRBM
        2. DWaveTopologyRBM
        
        Track which converges faster/better.
        """
        from model import FullyConnectedRBM, DWaveTopologyRBM
        
        results = {}
        
        for name, RBMClass in [('full', FullyConnectedRBM), ('dwave', DWaveTopologyRBM)]:
            print(f"\n{'='*50}")
            print(f"Training {name.upper()} architecture")
            print(f"{'='*50}")
            
            rbm = RBMClass(
                n_visible=ising_model.size,
                n_hidden=ising_model.size
            )
            
            results[name] = self.run_experiment(ising_model, rbm, sampler, config)
        
        return results


# Test: Verify matrix dimensions

if __name__ == "__main__":
    from model import FullyConnectedRBM
    from sampler import ClassicalSampler
    
    # Simple mock Ising model
    class MockIsing:
        def __init__(self):
            self.size = 4
        
        def local_energy(self, v, psi_ratio_fn):
            # Just return sum(v^2) as dummy energy
            return np.sum(v**2)
    
    rbm = FullyConnectedRBM(4, 3)
    ising = MockIsing()
    sampler = ClassicalSampler()
    
    trainer = Trainer(rbm, ising, sampler, {
        'learning_rate': 0.1,
        'n_iterations': 5,
        'n_samples': 50,
    })
    
    # Test SR matrix computation
    samples = sampler.sample(rbm, 50)
    gradients = [rbm.gradient_log_psi(v) for v in samples]
    energies = np.array([ising.local_energy(v, None) for v in samples])
    
    S, F = trainer._compute_sr_matrices(gradients, energies)
    
    n_params = 4 + 3 + 12  # a + b + W
    print(f"S shape: {S.shape}, should be ({n_params}, {n_params})")
    print(f"F shape: {F.shape}, should be ({n_params},)")
    print(f"S is symmetric? {np.allclose(S, S.T)}")
