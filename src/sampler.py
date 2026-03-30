import fcntl
import json
import numpy as np
from abc import ABC, abstractmethod
from model import RBM
import dimod
import neal
from dwave.samplers import TabuSampler
from veloxq_sdk import VeloxQSolver, SBMSolver, SBMParameters
from veloxq_sdk.config import load_config, VeloxQAPIConfig
from pathlib import Path
from helpers import get_solver_name
from scipy.optimize import minimize_scalar


def _cem_fit_beta(h_mean: np.ndarray, activation: np.ndarray) -> float:
    """
    CEM scalar fit (conditional variant): find β minimising
        Σ_j (⟨h_j⟩_{r,C} - tanh(β·a_j))²

    h_mean:     (n_hidden,) empirical conditional mean  ⟨h_j⟩_{r,C}
    activation: (n_hidden,) pre-activations  a_j = b_j + Σ_i r_i W_{ij}
    """
    def objective(beta):
        return float(np.sum((h_mean - np.tanh(beta * activation)) ** 2))

    result = minimize_scalar(objective, bounds=(1e-3, 1e3), method="bounded")
    return float(result.x)


def _cem_fit_beta_joint(v_samples: np.ndarray, h_samples: np.ndarray, rbm) -> float:
    """
    CEM scalar fit (joint-samples variant): find β minimising
        Σ_{l,j} (h^(l)_j - tanh(β · a^(l)_j))²
    where  a^(l)_j = b_j + Σ_i v^(l)_i W_{ij}.

    Used when the sampler returns joint (v, h) pairs from the full interacting
    (v-h) Ising problem.  Applicable to D-Wave and VeloxQ on RBMs, where the
    conditional h-only problem has no quadratic terms and conditional CEM is
    not meaningful.
    """
    activation = v_samples @ rbm.W + rbm.b[None, :]  # (n_samples, n_hidden)

    def objective(beta):
        return float(np.sum((h_samples - np.tanh(beta * activation)) ** 2))

    result = minimize_scalar(objective, bounds=(1e-3, 1e3), method="bounded")
    return float(result.x)


class Sampler(ABC):
    """Abstract sampling interface."""

    def rbm_to_ising(self, rbm, beta_x: float = 1.0):
        """
        Convert RBM parameters to Ising model parameters (J, h).
        Couplings are divided by beta_x so the sampler sees an effectively
        softer landscape; at β_eff_hardware / beta_x = 1 the samples match |Ψ|².

        Bottom-level check: prints beta_x whenever it changes by > 1 %,
        confirming the CEM-estimated value propagated all the way here.
        """
        _last = getattr(self, "_last_beta_x_logged", None)
        if _last is None or abs(beta_x - _last) / max(abs(_last), 1e-9) > 0.01:
            print(f"  [rbm_to_ising] beta_x = {beta_x:.4f}")
            self._last_beta_x_logged = beta_x

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
    def sample(self, rbm, n_samples: int, config: dict = None, return_hidden: bool = False):
        """
        Generate samples from the RBM distribution.

        rbm: the RBM instance (has log_psi, psi_ratio methods)
        n_samples: how many samples to draw
        config: optional configuration dict
        return_hidden: if True, return (v_samples, h_samples) tuple instead of v_samples only

        Returns: (n_samples, n_visible) array, or tuple of
                 ((n_samples, n_visible), (n_samples, n_hidden)) if return_hidden=True
        """
        pass

    def estimate_beta_eff(self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500) -> float:
        """
        Estimate the effective inverse temperature β_eff of this sampler via
        Conditional Expectation Matching (CEM).

        Procedure (Kubo & Goto 2025, Sec. II A 2):
          1. Pick condition vector r (random if not provided).
          2. Run the sampler conditionally: fix v = r, sample h.
          3. Compute empirical ⟨h_j⟩_{r,C} = mean of sampled h values.
          4. Fit β by minimising Σ_j (⟨h_j⟩_{r,C} - tanh(β·(b_j + Σ_i r_i W_{ij})))².

        Returns β_eff.  For an ideal β=1 sampler the result should be ≈ 1.0.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement estimate_beta_eff.")


class ClassicalSampler(Sampler):
    """
    Classical sampling: Metropolis-Hastings, Simulated Annealing, or
    Simulated Bifurcation (via the `simulated-bifurcation` package).
    """

    def __init__(
        self,
        method: str,
        n_warmup: int = 200,
        n_sweeps: int = 1,
        T_initial: float = 5.0,
        T_final: float = 1.0,
        sb_mode: str = "discrete",
        sb_heated: bool = False,
        sb_max_steps: int = 10000,
    ):
        """
        method:       'metropolis' | 'simulated_annealing' | 'sbm'
        n_warmup:     equilibration sweeps (metropolis / SA only)
        n_sweeps:     sweeps between samples (metropolis / SA only)
        sb_mode:      SBM algorithm variant — 'discrete' or 'ballistic'
        sb_heated:    enable heated variant of SBM
        sb_max_steps: max SBM iterations per agent
        """
        self.method = method
        self.n_warmup = n_warmup
        self.n_sweeps = n_sweeps
        self.T_initial = T_initial
        self.T_final = T_final
        self.sb_mode = sb_mode
        self.sb_heated = sb_heated
        self.sb_max_steps = sb_max_steps

    def sample(self, rbm: RBM, n_samples: int, config: dict = None, return_hidden: bool = False):
        if config is None:
            config = {}

        if self.method == "sbm":
            v, h = self._sbm_sample(rbm, n_samples, config)
            if return_hidden:
                return v, h
            return v

        if self.method == "metropolis":
            v = self._metropolis_hastings(rbm, n_samples, config)
        elif self.method == "simulated_annealing":
            v = self._simulated_annealing(rbm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_hidden:
            return v, self._sample_hidden(rbm, v)
        return v

    def _sbm_sample(self, rbm: RBM, n_samples: int, config: dict):
        """
        Sample from the full (v, h) RBM Boltzmann distribution using the
        simulated-bifurcation library (Ageron et al., 2025).

        Runs n_samples agents in parallel on the Ising problem defined by
        the full (v, h) coupling matrix and bias vector.

        Energy convention (library minimises):
            -(1/2) s^T M s + h_bias · s

        Mapping from RBM parameters scaled by beta_x:
            M[:Nv, Nv:] = W / beta_x    (off-diagonal, symmetric)
            h_bias       = -(a, b) / beta_x
        """
        import simulated_bifurcation as sb
        import torch

        beta_x    = config.get("beta_x", 1.0)
        mode      = config.get("sb_mode",      self.sb_mode)
        heated    = config.get("sb_heated",    self.sb_heated)
        max_steps = config.get("sb_max_steps", self.sb_max_steps)

        Nv, Nh = rbm.n_visible, rbm.n_hidden
        N = Nv + Nh

        M = np.zeros((N, N))
        M[:Nv, Nv:] = rbm.W / beta_x
        M[Nv:, :Nv] = rbm.W.T / beta_x

        h_bias = np.empty(N)
        h_bias[:Nv] = -rbm.a / beta_x
        h_bias[Nv:] = -rbm.b / beta_x

        M_t = torch.tensor(M, dtype=torch.float32)
        h_t = torch.tensor(h_bias, dtype=torch.float32)

        vectors, _ = sb.minimize(
            M_t, h_t, 0.0,
            domain="spin",
            agents=n_samples,
            best_only=False,
            mode=mode,
            heated=heated,
            max_steps=max_steps,
            verbose=False,
        )

        # vectors shape is (N, n_samples); transpose to (n_samples, N)
        s = vectors.numpy()
        if s.shape[0] == N and s.shape[1] == n_samples:
            s = s.T

        v = s[:, :Nv]
        h = s[:, Nv:]
        unique = len(set(map(tuple, v.tolist())))
        print(f"  [SBM]   mode={mode} heated={heated} unique={unique}/{n_samples}")
        return v, h

    def _sample_hidden(self, rbm: RBM, v_samples: np.ndarray) -> np.ndarray:
        """Sample h ~ p(h|v) at β=1 for each visible sample."""
        activation = rbm.b[None, :] + v_samples @ rbm.W  # (n_samples, n_hidden)
        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * activation))
        rng = np.random.default_rng()
        return np.where(rng.random(prob_plus.shape) < prob_plus, 1.0, -1.0)

    def estimate_beta_eff(self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500) -> float:
        """
        Estimate β_eff via CEM.

        SBM samples the full (v, h) state jointly → joint-samples variant.
        Metropolis / SA target |Ψ(v)|² at β=1 exactly → conditional variant
        (result should be ≈ 1.0).
        """
        if self.method == "sbm":
            v, h = self._sbm_sample(rbm, n_samples, config={})
            return _cem_fit_beta_joint(v, h, rbm)

        rng = np.random.default_rng()
        if r is None:
            r = rng.choice([-1.0, 1.0], size=rbm.n_visible)

        activation = rbm.b + r @ rbm.W  # (n_hidden,)
        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * activation))
        h_samples = np.where(
            rng.random((n_samples, rbm.n_hidden)) < prob_plus[None, :], 1.0, -1.0
        )
        h_mean = h_samples.mean(axis=0)
        return _cem_fit_beta(h_mean, activation)

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
    def __init__(self, method: str, sbm_steps: int = 5000, sbm_dt: float = 1.0,
                 sbm_discrete: bool = False):
        self.method = method

        load_config("velox_api_config.py")
        api_config = VeloxQAPIConfig.instance()
        with open("velox_token.txt", "r") as file:
            api_config.token = file.read().strip()

        if method == "sbm":
            params = SBMParameters(
                num_steps=sbm_steps,
                dt=sbm_dt,
                discrete_version=sbm_discrete,
            )
            self.solver = SBMSolver(parameters=params)
        else:
            self.solver = VeloxQSolver()

    def sample(self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False):
        self.n_visible = rbm.n_visible
        beta_x = config.get("beta_x", 1.0) if config else 1.0
        J, h = self.rbm_to_ising(rbm, beta_x)
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
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(drop=True)
        v = df.loc[:, list(range(self.n_visible))].to_numpy()
        if return_hidden:
            h_samples = df.loc[:, list(range(self.n_visible, self.n_visible + rbm.n_hidden))].to_numpy()
            return v, h_samples
        return v

    def estimate_beta_eff(self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500) -> float:
        """
        Estimate β_eff via the joint-samples variant of CEM.

        The conditional h problem has no h-h interactions in an RBM, so
        conditional sampling on VeloxQ would trivially return ground states.
        Instead, joint (v, h) samples are drawn from the full interacting
        problem and β is fit via _cem_fit_beta_joint.  r is unused.
        """
        v, h = self.sample(rbm, n_samples, return_hidden=True)
        return _cem_fit_beta_joint(v, h, rbm)


class LSBSampler(Sampler):
    """
    Langevin Simulated Bifurcation (LSB) sampler.

    Based on Kubo & Goto (2025), arXiv:2512.02323, Sec. II A 1.

    Runs n_samples parallel Ising chains on the full (v, h) state for n_steps
    steps. Each step applies the discretised LSB update with stochastic
    momentum re-initialisation (Eqs. 6-9 with Δ=1):

        g = x @ J + f               (local field at current binary state)
        ξ ~ N(0, σ)                 (fresh momentum noise every step)
        x = sgn(x + g + ξ)         (update + discretise)

    All n_samples chains run in parallel (fully vectorised).

    σ controls exploration: larger σ → more randomness → higher effective
    temperature.  β_eff is unknown a priori and must be estimated via CEM.
    """

    def __init__(self, sigma: float = 1.0, n_steps: int = 100):
        self.sigma = sigma
        self.n_steps = n_steps

    def _build_ising(self, rbm, beta_x: float = 1.0):
        """
        Build dense J matrix and bias vector f for the full (v, h) system.

        J is block off-diagonal (RBM has no v-v or h-h couplings):
            J[:Nv, Nv:] = W / beta_x
            J[Nv:, :Nv] = W.T / beta_x
        f = (a, b) / beta_x
        """
        Nv, Nh = rbm.n_visible, rbm.n_hidden
        N = Nv + Nh

        J = np.zeros((N, N))
        J[:Nv, Nv:] = rbm.W / beta_x
        J[Nv:, :Nv] = rbm.W.T / beta_x

        f = np.empty(N)
        f[:Nv] = rbm.a / beta_x
        f[Nv:] = rbm.b / beta_x

        return J, f

    def sample(self, rbm, n_samples: int, config: dict = None, return_hidden: bool = False):
        if config is None:
            config = {}
        beta_x = config.get("beta_x", 1.0)
        sigma   = config.get("lsb_sigma",  self.sigma)
        n_steps = config.get("lsb_steps",  self.n_steps)

        J, f = self._build_ising(rbm, beta_x)
        rng = np.random.default_rng()

        # Initialise all chains uniformly in {-1, +1}
        x = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, J.shape[0]))

        for _ in range(n_steps):
            g  = x @ J + f[None, :]                          # (L, N) local field
            xi = rng.normal(0, sigma, size=x.shape)          # (L, N) momentum noise
            x  = np.sign(x + g + xi)
            x[x == 0] = 1.0                                  # break ties (rare)

        v = x[:, :rbm.n_visible]
        unique = len(set(map(tuple, v.tolist())))
        print(f"  [LSB]   sigma={sigma:.3f}  steps={n_steps}  unique={unique}/{n_samples}")

        if return_hidden:
            return v, x[:, rbm.n_visible:]
        return v

    def estimate_beta_eff(self, rbm, r: np.ndarray = None, n_samples: int = 500) -> float:
        """
        Estimate β_eff via the joint-samples CEM variant.

        LSB samples the full (v, h) state jointly, so the conditional CEM
        variant (which fixes v=r) is not applicable.  Instead, fit β from
        joint (v, h) pairs — same approach as D-Wave and VeloxQ.
        r is unused.
        """
        v, h = self.sample(rbm, n_samples, return_hidden=True)
        return _cem_fit_beta_joint(v, h, rbm)


class DimodSampler(Sampler):
    def __init__(self, method: str):
        self.method = method
        self.time_path = Path("time.json")
        if not self.time_path.exists():
            with self.time_path.open("w") as f:
                json.dump({"time_ms": 0}, f)

        self._embedding_cache: dict = {}

    def sample(self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False):
        """
        Sample from the RBM distribution using a classical/quantum sampler from the dimod library.
        Args:
            - rbm (RBM): An RBM instance
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the sampler
            - return_hidden: if True, return (v_samples, h_samples) tuple
        """
        beta_x = config.get("beta_x", 1.0)
        J, h = self.rbm_to_ising(rbm, beta_x)
        self.n_visible = rbm.n_visible
        self.n_hidden = rbm.n_hidden
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)

        if self.method == "simulated_annealing":
            return self.simulated_annealing(bqm, n_samples, config, return_hidden)
        elif self.method == "tabu":
            return self.tabu_search(bqm, n_samples, config, return_hidden)
        elif self.method == "pegasus" or self.method == "zephyr":
            config["solver"] = get_solver_name(self.method)
            return self.dwave(bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def estimate_beta_eff(self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500) -> float:
        """
        Estimate β_eff via CEM (Kubo & Goto 2025, Sec. II A 2).

        SA / tabu  — conditional variant:
          Builds the h-only conditional BQM for a fixed visible configuration r
          (linear biases b_j + Σ_i r_i W_{ij}, no quadratic terms).
          Runs the same sampler on this reduced problem, computes ⟨h_j⟩_{r,C},
          then fits β via _cem_fit_beta.

        D-Wave (pegasus / zephyr) — joint-samples variant:
          The conditional h problem has no h-h interactions in an RBM, so
          D-Wave trivially finds the ground state.  Instead, joint (v, h)
          samples are drawn from the full interacting problem and β is fit via
          _cem_fit_beta_joint.  r is unused in this branch.
        """
        if self.method in ("pegasus", "zephyr"):
            config = {"solver": get_solver_name(self.method)}
            v, h = self.sample(rbm, n_samples, config=config, return_hidden=True)
            return _cem_fit_beta_joint(v, h, rbm)

        rng = np.random.default_rng()
        if r is None:
            r = rng.choice([-1.0, 1.0], size=rbm.n_visible)

        activation = rbm.b + r @ rbm.W  # (n_hidden,)
        linear = {j: -float(activation[j]) for j in range(rbm.n_hidden)}
        bqm = dimod.BinaryQuadraticModel.from_ising(linear, {}, 0.0)

        if self.method == "simulated_annealing":
            sampleset = neal.SimulatedAnnealingSampler().sample(
                bqm,
                num_reads=n_samples,
                beta_range=(0.01, 10.0),
                num_sweeps=1000,
                beta_schedule_type="geometric",
            )
        elif self.method == "tabu":
            sampleset = TabuSampler().sample(bqm, num_reads=n_samples)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        h_mean = sampleset.record.sample.mean(axis=0)  # (n_hidden,)
        return _cem_fit_beta(h_mean, activation)

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

    def simulated_annealing(self, bqm, n_samples: int, config: dict = {}, return_hidden: bool = False):
        """
        Run simulated annealing using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the annealing schedule
            - return_hidden: if True, return (v_samples, h_samples) tuple
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
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def tabu_search(self, bqm, n_samples: int, config: dict = {}, return_hidden: bool = False):
        """
        Run tabu search using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the tabu search
            - return_hidden: if True, return (v_samples, h_samples) tuple
        """
        sampler = TabuSampler()
        sampleset = sampler.sample(bqm, num_reads=n_samples)

        samples = sampleset.record.sample
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def dwave(self, bqm, n_samples: int, config: dict = {}, rbm=None, return_hidden: bool = False):
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
        v = df.loc[:, list(range(self.n_visible))].to_numpy()
        if return_hidden:
            h_cols = list(range(self.n_visible, self.n_visible + self.n_hidden))
            return v, df.loc[:, h_cols].to_numpy()
        return v
