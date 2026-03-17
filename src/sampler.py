import numpy as np
from abc import ABC, abstractmethod
from model import RBM
import dimod
import neal
from dwave.samplers import TabuSampler
from veloxq_sdk import VeloxQSolver
from veloxq_sdk.config import load_config, VeloxQAPIConfig


class Sampler(ABC):
    """Abstract sampling interface."""

    def rbm_to_ising(self, rbm):
        """
        Convert RBM parameters to Ising model parameters (J, h).
        Args:
            rbm (RBM): An RBM instance
        """
        Nv = rbm.n_visible
        Nh = rbm.n_hidden

        linear = {}
        quadratic = {}

        # visible biases
        for i in range(Nv):
            linear[i] = -rbm.a[i]

        # hidden biases
        for j in range(Nh):
            linear[Nv + j] = -rbm.b[j]

        # RBM couplings
        for i in range(Nv):
            for j in range(Nh):
                quadratic[(i, Nv + j)] = -rbm.W[i, j]

        return quadratic, linear

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
    """

    def __init__(self, method: str):
        self.method = method

    def sample(self, rbm: RBM, n_samples: int, config: dict = None) -> np.ndarray:
        if config is None:
            config = {}

        print(rbm)
        if self.method == "metropolis":
            return self._metropolis_hastings(rbm, n_samples, config)
        elif self.method == "simulated_annealing":
            return self._simulated_annealing(rbm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _metropolis_hastings(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Metropolis-Hastings sampling using spin flips.

        Parameters from config (with defaults):
        - n_sweeps: int, equilibration steps (default ???)
        - n_between: int, steps between samples (default ???)
        """
        print("metropolis")
        n_visible = rbm.n_visible
        n_sweeps = config.get("n_sweeps", 100)
        n_between = config.get("n_between", 5)
        print("----W------")
        print(rbm.W.shape)
        print("----a------")
        print(rbm.a.shape)
        print("----b")
        print(rbm.b.shape)
        # Initialize
        v = (2 * np.random.randint(0, 2, n_visible) - 1).astype(float)
        samples = []

        # Equilibrium
        spin_flip_array = np.random.randint(0, len(v), n_sweeps)
        for spin_flip_idx in spin_flip_array:
            ratio_squared = rbm.psi_ratio(v, spin_flip_idx) ** 2
            if np.random.random() < min(1, ratio_squared):
                v[spin_flip_idx] *= -1

        # Sample collection
        for _ in range(n_samples):
            spin_flips_array_between = np.random.randint(0, len(v), n_between)
            for spin_flip_idx in spin_flips_array_between:
                ratio_squared = rbm.psi_ratio(v, spin_flip_idx) ** 2
                if np.random.random() < min(1, ratio_squared):
                    v[spin_flip_idx] *= -1

            samples.append(np.copy(v))

        return np.array(samples)

    def _simulated_annealing(self, rbm, n_samples: int, config: dict) -> np.ndarray:
        """
        Simulated Annealing: gradually lower temperature as you sample.

        Parameters from config:
        - T_initial: starting temperature (default ???)
        - T_final: final temperature (default ???)
        - n_steps: total annealing steps (default ???)
        """

        T_initial = config.get("T_initial", 10)
        T_final = config.get("T_final", 0.05)
        n_steps = config.get("n_steps", int(1e5))
        print("simulated annealing")
        n_visible = rbm.n_visible
        v = (2 * np.random.randint(0, 2, n_visible) - 1).astype(float)
        samples = []

        # Create temperature schedule
        def schedule(step: int) -> float:
            return T_initial * (T_final / T_initial) ** (step / n_steps)  # Geometric

        # Equilibrium
        spin_flip_array = np.random.randint(0, len(v), n_steps)
        for step, spin_flip_idx in enumerate(spin_flip_array):
            T = schedule(step)
            ratio_squared = rbm.psi_ratio(v, spin_flip_idx) ** 2
            if np.random.random() < min(1, ratio_squared ** (1 / T)):
                v[spin_flip_idx] *= -1

            if step % (n_steps // n_samples) == 0:
                samples.append(np.copy(v))

        return np.array(samples)


class VeloxSampler(Sampler):
    def __init__(self, method: str):
        self.method = method
        self.solver = VeloxQSolver()

        load_config("velox_api_config.py")
        api_config = VeloxQAPIConfig.instance()

        with open("velox_token.txt", "r") as file:
            api_config.token = file.read().strip()

    def sample(self, rbm, n_samples: int, config: dict = {}) -> np.ndarray:
        self.n_visible = rbm.n_visible
        J, h = self.rbm_to_ising(rbm)
        self.solver.parameters.num_rep = n_samples
        sampleset = self.solver.sample(h, J)

        df = sampleset.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(
            drop=True
        )  # expand
        # return visible only
        return df.loc[:, list(range(self.n_visible))].to_numpy()


class DimodSampler(Sampler):
    def __init__(self, method: str):
        self.method = method

    def sample(self, rbm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Sample from the RBM distribution using a classical/quantum sampler from the dimod library.
        Args:
            - rbm (RBM): An RBM instance
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the sampler
        """
        J, h = self.rbm_to_ising(rbm)
        self.n_visible = rbm.n_visible
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)

        if self.method == "simulated_annealing":
            return self.simulated_annealing(bqm, n_samples, config)
        elif self.method == "tabu":
            return self.tabu_search(bqm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def simulated_annealing(self, bqm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Run simulated annealing using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the annealing schedule
        """
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=n_samples)

        samples = sampleset.record.sample

        # return visible spins only
        return samples[:, : self.n_visible]

    def tabu_search(self, bqm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Run tabu search using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the tabu search
        """
        sampler = TabuSampler()
        sampleset = sampler.sample(bqm, num_reads=n_samples)

        samples = sampleset.record.sample

        # return visible spins only
        return samples[:, : self.n_visible]
