import fcntl
import json
import numpy as np
from abc import ABC, abstractmethod
from model import RBM
import dimod
import neal
from dwave.samplers import TabuSampler
from veloxq_sdk import VeloxQSolver
from veloxq_sdk.config import load_config, VeloxQAPIConfig
from pathlib import Path
from helpers import get_solver_name


class Sampler(ABC):
    """Abstract sampling interface."""

    def rbm_to_ising(self, rbm, beta_x: float = 1.0):
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
            linear[i] = -rbm.a[i] / beta_x

        # hidden biases
        for j in range(Nh):
            linear[Nv + j] = -rbm.b[j] / beta_x

        # RBM couplings
        for i in range(Nv):
            for j in range(Nh):
                if abs(rbm.W[i, j]) > 1e-6:
                    quadratic[(i, Nv + j)] = -rbm.W[i, j] / beta_x

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

    def __init__(
        self,
        method: str,
        n_warmup: int = 200,
        n_sweeps: int = 1,
        T_initial: float = 5.0,
        T_final: float = 1.0,
    ):
        """
        method:   'metropolis' | 'simulated_annealing' | 'gibbs'
        n_warmup: equilibration sweeps before collecting samples
        n_sweeps: full sweeps (n_visible flip attempts) between each sample
                  increase to 2-3 if acceptance rate drops below 0.2
        """
        self.method = method
        self.n_warmup = n_warmup
        self.n_sweeps = n_sweeps

        self.T_initial = T_initial
        self.T_final = T_final

    def sample(self, rbm: RBM, n_samples: int, config: dict = None) -> np.ndarray:
        if config is None:
            config = {}

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
        Metropolis-Hastings sampling targeting |Ψ(v)|².

        Proposal: flip a single random spin.
        Acceptance: min(1, |Ψ(v')/Ψ(v)|²)

        One sweep = n_visible attempted flips at randomly chosen sites.

        Parameters from config:
        - n_warmup: equilibration sweeps (overrides __init__ value)
        - n_sweeps: sweeps between collected samples (overrides __init__ value)
        """
        N = rbm.n_visible
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)
        rng = np.random.default_rng()

        v = rng.choice([-1.0, 1.0], size=N)

        n_accepted = 0
        n_proposed = 0

        def sweep(v):
            nonlocal n_accepted, n_proposed
            for flip_idx in rng.integers(0, N, size=N):
                ratio_sq = rbm.psi_ratio(v, flip_idx) ** 2
                n_proposed += 1
                if rng.random() < min(1.0, ratio_sq):
                    v[flip_idx] *= -1
                    n_accepted += 1
            return v

        # Warmup — equilibrate from random initial state
        for _ in range(n_warmup):
            sweep(v)

        # Reset counters so acceptance rate reflects collection phase only
        n_accepted = 0
        n_proposed = 0

        # Collect samples
        samples = []
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                sweep(v)
            samples.append(v.copy())

        acceptance_rate = n_accepted / max(n_proposed, 1)
        print(
            f"  [MH]    acceptance={acceptance_rate:.3f}  "
            f"unique={len(set(map(tuple, samples)))}/{n_samples}"
        )

        return np.array(samples)

    def _simulated_annealing(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Simulated Annealing sampling targeting |Ψ(v)|².

        Starts at high temperature (flat distribution, full exploration) and
        cools geometrically. At each step accepts a spin flip with probability:

            min(1, |Ψ(v')/Ψ(v)|^(2/T))

        At T→∞ this accepts everything (random walk).
        At T→0 this only accepts improvements (greedy).

        Unlike the Metropolis sampler which targets the fixed distribution
        |Ψ(v)|² at T=1, SA uses temperature to escape local modes early in
        training when the RBM is poorly initialised, then sharpens toward
        the true distribution as T→1 at the end of the schedule.

        One sweep = n_visible attempted spin flips at randomly chosen sites.

        Parameters from config (all optional):
        - T_initial:  starting temperature  (default: 5.0)
        - T_final:    ending temperature    (default: 1.0)
                    set to 1.0 so the final samples are from |Ψ|² exactly
        - n_warmup:   sweeps at T_initial before schedule starts (default: 50)
        - n_sweeps:   sweeps between collected samples during cooling (default: 1)
        """
        N = rbm.n_visible
        T_initial = config.get(
            "T_initial", self.T_initial if hasattr(self, "T_initial") else 5.0
        )
        T_final = config.get(
            "T_final", self.T_final if hasattr(self, "T_final") else 1.0
        )
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)
        rng = np.random.default_rng()

        v = rng.choice([-1.0, 1.0], size=N)

        # Geometric cooling schedule: T(step) = T_initial * (T_final/T_initial)^(step/n_steps)
        n_steps = n_samples * n_sweeps

        def schedule(step: int) -> float:
            if T_initial == T_final:
                return T_final
            return T_initial * (T_final / T_initial) ** (step / max(n_steps - 1, 1))

        n_accepted = 0
        n_proposed = 0

        def sweep(v, T):
            nonlocal n_accepted, n_proposed
            for flip_idx in rng.integers(0, N, size=N):
                ratio_sq = rbm.psi_ratio(v, flip_idx) ** 2
                n_proposed += 1
                # At T=1: standard Metropolis acceptance = min(1, ratio²)
                # At T>1: acceptance = min(1, ratio^(2/T)) — flatter, more exploratory
                accept_prob = min(1.0, ratio_sq ** (1.0 / T))
                if rng.random() < accept_prob:
                    v[flip_idx] *= -1
                    n_accepted += 1
            return v

        # Warmup at T_initial — equilibrate before cooling
        for _ in range(n_warmup):
            sweep(v, T_initial)

        n_accepted = 0
        n_proposed = 0

        # Collect samples while cooling
        samples = []
        step = 0
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                T = schedule(step)
                sweep(v, T)
                step += 1
            samples.append(v.copy())

        acceptance_rate = n_accepted / max(n_proposed, 1)
        T_now = schedule(step - 1)
        print(
            f"  [SA]    acceptance={acceptance_rate:.3f}  "
            f"T: {T_initial:.2f}→{T_now:.2f}  "
            f"unique={len(set(map(tuple, samples)))}/{n_samples}"
        )

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

        MAX_VELOX_RETRIES = 3
        for attempt in range(1, MAX_VELOX_RETRIES + 1):
            try:
                sampleset = self.solver.sample(h, J)
                break
            except Exception as e:
                print(f"  [VeloxQ] attempt {attempt}/{MAX_VELOX_RETRIES} failed: {e}")
                if attempt == MAX_VELOX_RETRIES:
                    raise RuntimeError(
                        f"VeloxQ sampling failed after {MAX_VELOX_RETRIES} attempts."
                    ) from e

        df = sampleset.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(
            drop=True
        )  # expand
        # return visible only
        return df.loc[:, list(range(self.n_visible))].to_numpy()


class DimodSampler(Sampler):
    def __init__(self, method: str):
        self.method = method
        self.time_path = Path("time.json")
        if not self.time_path.exists():
            with self.time_path.open("w") as f:
                json.dump({"time_ms": 0}, f)

        self._embedding_cache: dict = {}

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
        elif self.method == "pegasus" or self.method == "zephyr":
            config["solver"] = get_solver_name(self.method)
            return self.dwave(bqm, n_samples, config, rbm=rbm)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _log_access_time(self, access_time_us: float):
        """Log the D-Wave access time to time.json.

        Uses an exclusive flock for cross-process safety and an atomic
        write (temp file → rename) to prevent partial/corrupt writes.
        """
        with self.time_path.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                time_dict = json.load(f)
                time_dict["time_ms"] += access_time_us * 1e-3
                tmp = self.time_path.with_suffix(".tmp")
                with tmp.open("w") as tf:
                    json.dump(time_dict, tf)
                tmp.rename(self.time_path)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def simulated_annealing(self, bqm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Run simulated annealing using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the annealing schedule
        """
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=n_samples,
            beta_range=(0.01, 10.0),  # wider temperature range
            num_sweeps=1000,  # more sweeps per read
            beta_schedule_type="geometric",
        )
        samples = sampleset.record.sample
        unique_samples = len(set(map(tuple, samples)))
        print(f"  unique samples: {unique_samples}/{len(samples)}")
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

    def dwave(self, bqm, n_samples: int, config: dict = {}, rbm=None) -> np.ndarray:
        from dwave.system import (
            DWaveSampler,
            EmbeddingComposite,
            FixedEmbeddingComposite,
        )
        from model import DWaveTopologyRBM

        solver_name = config.get("solver", None)
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)
        cache_key = (self.n_visible, solver_name)

        # ── Build or retrieve cached composite ───────────────────────────────
        if cache_key not in self._embedding_cache:
            dwave_sampler = DWaveSampler(solver=solver_name)

            if rbm is not None and isinstance(rbm, DWaveTopologyRBM):
                # Trivial identity embedding — no minorminer needed
                assert rbm._qubit_mapping is not None, (
                    "DWaveTopologyRBM must be built from a solver to use "
                    "trivial embedding. rbm._qubit_mapping is None."
                )
                identity_embedding = {
                    logical: [phys] for phys, logical in rbm._qubit_mapping.items()
                }
                composite = FixedEmbeddingComposite(dwave_sampler, identity_embedding)
                print(
                    f"  [embedding] Trivial identity embedding cached for {cache_key}."
                )

            else:
                # Find embedding once with minorminer, then fix it
                print(
                    f"  [embedding] Running minorminer for {cache_key} — this may take a moment..."
                )
                import minorminer

                embedding = minorminer.find_embedding(
                    list(bqm.quadratic.keys()),
                    dwave_sampler.edgelist,
                )
                if not embedding:
                    raise RuntimeError(
                        f"minorminer failed to find an embedding for "
                        f"n_visible={self.n_visible} on solver '{solver_name}'."
                    )
                composite = FixedEmbeddingComposite(dwave_sampler, embedding)
                print(f"  [embedding] Embedding found and cached for {cache_key}.")

            self._embedding_cache[cache_key] = composite

        else:
            composite = self._embedding_cache[cache_key]

        # ── Build sample kwargs ───────────────────────────────────────────────
        is_trivial = (
            rbm is not None
            and isinstance(rbm, DWaveTopologyRBM)
            and rbm._qubit_mapping is not None
        )

        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
            answer_mode="raw",
            auto_scale=True,
        )

        # chain_strength only applies when there are actual chains
        if not is_trivial and chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength

        # ── Sample with retries ───────────────────────────────────────────────
        MAX_DWAVE_RETRIES = 3
        success = False
        tries = 0

        while not success and tries < MAX_DWAVE_RETRIES:
            tries += 1
            try:
                sampleset = composite.sample(bqm, **sample_kwargs)
                access_time_us = sampleset.info["timing"]["qpu_access_time"]
                self._log_access_time(access_time_us * tries)
                success = True
            except Exception as e:
                print(
                    f"  D-Wave sampling attempt {tries}/{MAX_DWAVE_RETRIES} failed: {e}"
                )
                if tries < MAX_DWAVE_RETRIES:
                    # Invalidate cache — composite may have stale connections after failure
                    self._embedding_cache.pop(cache_key, None)
                    dwave_sampler = DWaveSampler(solver=solver_name)
                    composite = (
                        FixedEmbeddingComposite(
                            dwave_sampler,
                            self._embedding_cache.get(cache_key, composite).embedding,
                        )
                        if cache_key in self._embedding_cache
                        else composite
                    )
                    # Rebuild from scratch on next cache miss
                    self._embedding_cache.pop(cache_key, None)

        if not success:
            raise RuntimeError(
                f"D-Wave sampling failed after {MAX_DWAVE_RETRIES} attempts."
            )

        df = sampleset.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(drop=True)
        return df.loc[:, list(range(self.n_visible))].to_numpy()
