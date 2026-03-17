import numpy as np


class Trainer:
    """
    Variational Monte Carlo trainer using Stochastic Reconfiguration.
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

        self.learning_rate = config.get("learning_rate", 0.1)
        self.n_iterations = config.get("n_iterations", 50)
        self.n_samples = config.get("n_samples", 1000)
        self.regularization = config.get("regularization", 1e-5)
        self.history = {
            "energy": [],  # mean local energy per iteration
            "error": [],  # std of local energies (variance proxy)
            "energy_error": [],  # std / sqrt(n_samples) — true statistical error of mean
            "learning_rate": [],  # actual lr used (useful if you add adaptive lr later)
            "grad_norm": [],  # ||x|| = ||S^-1 F|| — detects exploding/vanishing updates
            "s_condition_number": [],  # condition number of S — detects SR instability
            "weight_norm": [],  # ||w|| — detects weight explosion
        }

    def train(self) -> dict:
        """
        Returns: history dict with convergence data
        """

        for iteration in range(self.n_iterations):
            # Sample from RBM
            samples = self.sampler.sample(self.rbm, self.n_samples)

            # Compute local energies and gradients
            local_energies = []
            gradients_list = []  # List of dicts

            for v in samples:
                v = v.copy()
                # 1. Compute local energy: E_loc = ising.local_energy(v, self.rbm.psi_ratio)
                E_loc = self.ising.local_energy(v, self.rbm.psi_ratio)
                local_energies.append(E_loc)

                # 2. Compute gradient: grad = rbm.gradient_log_psi(v)
                grad = self.rbm.gradient_log_psi(v)
                gradients_list.append(grad)

            local_energies = np.array(local_energies)

            S, F = self._compute_sr_matrices(gradients_list, local_energies)

            x = np.linalg.solve(S, F)
            w = self.rbm.get_weights()
            w_new = w - self.learning_rate * x
            self.rbm.set_weights(w_new)

            # Track metrics
            E_mean = np.mean(local_energies)
            E_std = np.std(local_energies)

            self.history["energy"].append(float(E_mean))
            self.history["error"].append(float(E_std))
            self.history["energy_error"].append(
                float(E_std / np.sqrt(len(local_energies)))
            )
            self.history["learning_rate"].append(self.learning_rate)
            self.history["grad_norm"].append(float(np.linalg.norm(x)))
            self.history["s_condition_number"].append(float(np.linalg.cond(S)))
            self.history["weight_norm"].append(float(np.linalg.norm(w_new)))
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d}: E = {E_mean:.6f} ± {E_std:.6f}")

        return self.history

    def _compute_sr_matrices(self, gradients_list, local_energies) -> tuple:
        """
        Returns: (S, F) where S is (n_params, n_params) and F is (n_params,)
        """

        # Convert list of gradient dicts to matrix
        D = []
        for grad_dict in gradients_list:
            # Flatten: [a, b, W]
            row = np.concatenate(
                [
                    grad_dict["a"].flatten(),
                    grad_dict["b"].flatten(),
                    grad_dict["W"].flatten(),
                ]
            )
            D.append(row)
        D = np.array(D)  # Shape: (M, n_params)
        M = D.shape[0]

        # TODO: Compute S and F
        M = D.shape[0]
        mean_D = np.mean(D, axis=0)  # (n_params,)
        mean_E = np.mean(local_energies)  # scalar
        D_centered = D - mean_D  # (M, n_params)

        S = (1 / M) * D_centered.T @ D_centered  # (n_params, n_params)
        F = (1 / M) * D_centered.T @ local_energies  # (n_params,)

        S += self.regularization * np.eye(S.shape[0])

        return S, 2 * F
        mean_D = np.mean(D)

        mean_E = np.mean(local_energies)
        S = (1 / M) * D.T @ D - np.outer(mean_D, mean_D)
        F = (1 / M) * (local_energies.T @ D) - mean_E * mean_D

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
            "config": config,
            "history": history,
            "final_weights": {
                "a": rbm.a.copy(),
                "b": rbm.b.copy(),
                "W": rbm.W.copy(),
            },
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

        for name, RBMClass in [
            ("full", FullyConnectedRBM),
            ("dwave", DWaveTopologyRBM),
        ]:
            print(f"\n{'=' * 50}")
            print(f"Training {name.upper()} architecture")
            print(f"{'=' * 50}")

            rbm = RBMClass(n_visible=ising_model.size, n_hidden=ising_model.size)

            results[name] = self.run_experiment(ising_model, rbm, sampler, config)

        return results
