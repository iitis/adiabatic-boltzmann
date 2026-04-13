import fcntl
import json
import math as _math
import numpy as np
from abc import ABC, abstractmethod
from model import RBM
import dimod

try:
    import numba as nb

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    import cupy as cp

    _xp = cp
    _DEVICE = "gpu"
except ImportError:
    _xp = np
    _DEVICE = "cpu"
import neal
from dwave.samplers import TabuSampler
try:
    from veloxq_sdk import VeloxQSolver, SBMSolver, SBMParameters
    from veloxq_sdk.config import load_config, VeloxQAPIConfig
except:
    print("Velox could not be imported")
from pathlib import Path
from helpers import get_solver_name
from scipy.optimize import minimize_scalar


def _logcosh_xp(xp, x):
    """Device-agnostic logcosh: works with numpy or cupy arrays."""
    ax = xp.abs(x)
    return ax + xp.log1p(xp.exp(-2.0 * ax))


if _HAS_NUMBA:

    @nb.njit(cache=True)
    def _mh_sweep_nb(v, theta, W, a, flip_indices, rand_u):
        """Numba-compiled MH sweep. Mutates v and theta in-place. Returns n_accepted."""
        n_accepted = 0
        Nh = theta.shape[0]
        for k in range(flip_indices.shape[0]):
            i = flip_indices[k]
            vi = v[i]
            log_ratio = a[i] * vi
            for j in range(Nh):
                tf = theta[j] - 2.0 * vi * W[i, j]
                t = theta[j]
                atf = abs(tf)
                at = abs(t)
                log_ratio += 0.5 * (
                    atf
                    + _math.log1p(_math.exp(-2.0 * atf))
                    - at
                    - _math.log1p(_math.exp(-2.0 * at))
                )
            log_accept = 2.0 * log_ratio
            if rand_u[k] < (1.0 if log_accept >= 0.0 else _math.exp(log_accept)):
                v[i] = -vi
                for j in range(Nh):
                    theta[j] -= 2.0 * vi * W[i, j]
                n_accepted += 1
        return n_accepted

    @nb.njit(cache=True)
    def _sa_sweep_nb(v, theta, W, a, flip_indices, rand_u, T):
        """Numba-compiled SA sweep. Mutates v and theta in-place. Returns n_accepted."""
        n_accepted = 0
        Nh = theta.shape[0]
        for k in range(flip_indices.shape[0]):
            i = flip_indices[k]
            vi = v[i]
            log_ratio = a[i] * vi
            for j in range(Nh):
                tf = theta[j] - 2.0 * vi * W[i, j]
                t = theta[j]
                atf = abs(tf)
                at = abs(t)
                log_ratio += 0.5 * (
                    atf
                    + _math.log1p(_math.exp(-2.0 * atf))
                    - at
                    - _math.log1p(_math.exp(-2.0 * at))
                )
            log_accept = 2.0 * log_ratio / T
            if rand_u[k] < (1.0 if log_accept >= 0.0 else _math.exp(log_accept)):
                v[i] = -vi
                for j in range(Nh):
                    theta[j] -= 2.0 * vi * W[i, j]
                n_accepted += 1
        return n_accepted


def _cem_fit_beta(h_mean: np.ndarray, activation: np.ndarray) -> float:
    """
    CEM scalar fit (conditional variant): find β minimising
        Σ_j (⟨h_j⟩_{r,C} - tanh(β·a_j))²

    h_mean:     (n_hidden,) empirical conditional mean  ⟨h_j⟩_{r,C}
    activation: (n_hidden,) pre-activations  a_j = b_j + Σ_i r_i W_{ij}
    """

    def objective(beta):
        return float(np.sum((h_mean - np.tanh(beta * activation)) ** 2))

    result = minimize_scalar(objective, bounds=(1e-2, 10.0), method="bounded")
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

    result = minimize_scalar(objective, bounds=(1e-2, 10.0), method="bounded")
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
    def sample(
        self, rbm, n_samples: int, config: dict = None, return_hidden: bool = False
    ):
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

    def estimate_beta_eff(
        self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500
    ) -> float:
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
        raise NotImplementedError(
            f"{type(self).__name__} does not implement estimate_beta_eff."
        )


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
        gibbs_collapse_threshold: float = 0.2,
        gibbs_reinit_fraction: float = 0.5,
    ):
        """
        method:       'metropolis' | 'simulated_annealing' | 'sbm' | 'gibbs'
        n_warmup:     equilibration sweeps (metropolis / SA / gibbs only)
        n_sweeps:     sweeps between samples (metropolis / SA only);
                      for gibbs this is the number of block sweeps per call (k)
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
        self.gibbs_collapse_threshold = gibbs_collapse_threshold
        self.gibbs_reinit_fraction = gibbs_reinit_fraction

        # Persistent chain state for Gibbs sampler (initialised on first call)
        self._gibbs_v = None  # (_xp array) (n_chains, n_visible)

    def sample(
        self, rbm: RBM, n_samples: int, config: dict = None, return_hidden: bool = False
    ):
        if config is None:
            config = {}

        if self.method == "lsb":
            v, h = self._lsb_sample(rbm, n_samples, config)
            if return_hidden:
                return v, h
            return v

        if self.method == "gibbs":
            v, h = self._gibbs_sample(rbm, n_samples, config)
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

    def _lsb_sample(self, rbm: RBM, n_samples: int, config: dict):
        import torch

        beta_x = config.get("beta_x", 2.0)
        steps = config.get("lsb_steps", 1000)
        delta = config.get("lsb_delta", 1.0)
        # sigma=0 → auto-scale to RMS of local fields (same as LSBSampler).
        # A fixed sigma risks collapse when weights grow: if ||force|| >> sigma
        # all spins align with the field and samples become degenerate.
        sigma = config.get("lsb_sigma", 0.0)
        Nv, Nh = rbm.n_visible, rbm.n_hidden
        N = Nv + Nh

        M = np.zeros((N, N))
        if getattr(rbm, "V", None) is not None:
            M[:Nv, :Nv] = rbm.V / beta_x   # visible-visible couplings (SRBM)
        M[:Nv, Nv:] = rbm.W / beta_x
        M[Nv:, :Nv] = rbm.W.T / beta_x
        f = np.empty(N)
        f[:Nv] = rbm.a / beta_x
        f[Nv:] = rbm.b / beta_x

        # --- Adaptive sigma: set σ = RMS(force) over random probe configs ---
        if sigma <= 0:
            rng_probe = np.random.default_rng()
            x_probe = rng_probe.choice([-1.0, 1.0], size=(200, N))
            g_probe = x_probe @ M + f[None, :]
            sigma = float(np.sqrt(np.mean(g_probe**2))) + 1e-6

        device = "cuda" if torch.cuda.is_available() else "cpu"
        M_t = torch.tensor(M, dtype=torch.float32, device=device)
        f_t = torch.tensor(f, dtype=torch.float32, device=device)

        # --- Initialize: x ~ Uniform{-1,+1},  y ~ N(0, σ) ---
        x = torch.randint(0, 2, (n_samples, N), device=device).float() * 2 - 1
        y = sigma * torch.randn(n_samples, N, device=device)

        # --- Dynamics: symplectic Euler + discretize + re-init momentum ---
        for _ in range(steps):
            s = torch.sign(x)
            s[s == 0] = 1.0
            force = torch.matmul(s, M_t.T) + f_t  # Σ_j J_ij sgn(x_j) + f_i
            y = y + delta * force  # y(k+1) = y(k) + Δ·force
            x = x + delta * y  # x(k+1) = x(k) + Δ·y(k+1)
            x = torch.sign(x)
            x[x == 0] = 1.0
            y = sigma * torch.randn_like(y)  # re-init momentum

        # --- Extract final spins (OUTSIDE loop) ---
        s = torch.sign(x).cpu().numpy()
        v = s[:, :Nv]
        h = s[:, Nv:]

        unique = len(set(map(tuple, v.tolist())))
        print(
            f"  [LSB] steps={steps} delta={delta} sigma={sigma:.4f} unique={unique}/{n_samples}"
        )
        return v, h

    def _gibbs_sample(self, rbm: RBM, n_samples: int, config: dict):
        """
        Persistent block Gibbs sampler (PCD-k) targeting |Ψ(v)|².

        Samples the joint (v, h) distribution with ±1 spins:
            p(v, h) ∝ exp(-a·v + b·h + v·W·h)

        Block conditionals (all units independent within each block):
            p(h_j = +1 | v) = σ( 2(b_j + W[:,j]·v) )
            p(v_i = +1 | h) = σ( 2(W[i,:]·h  - a_i) )

        Note: no beta_x scaling — Gibbs is exact at T=1 by construction.
        GPU: if CuPy is available all arrays live on the GPU; otherwise
        NumPy is used transparently (same code path via _xp).
        """
        n_sweeps = config.get("n_sweeps", self.n_sweeps)
        n_warmup = config.get("n_warmup", self.n_warmup)

        Nv, Nh = rbm.n_visible, rbm.n_hidden
        W = _xp.asarray(rbm.W)
        a = _xp.asarray(rbm.a)
        b = _xp.asarray(rbm.b)

        rng = np.random.default_rng()

        def _h_given_v(V):
            """Sample H ~ p(h|v).  V: (C, Nv) → H: (C, Nh)"""
            prob = 1.0 / (1.0 + _xp.exp(-2.0 * (V @ W + b[None, :])))
            u = _xp.asarray(rng.random((V.shape[0], Nh)))
            return _xp.where(u < prob, 1.0, -1.0)

        def _v_given_h(H):
            """Sample V ~ p(v|h).  H: (C, Nh) → V: (C, Nv)"""
            prob = 1.0 / (1.0 + _xp.exp(-2.0 * (H @ W.T - a[None, :])))
            u = _xp.asarray(rng.random((H.shape[0], Nv)))
            return _xp.where(u < prob, 1.0, -1.0)

        def _init_chains(n: int):
            """Random ±1 init, warmed up for n_warmup sweeps."""
            V_ = _xp.asarray(rng.choice([-1.0, 1.0], size=(n, Nv)))
            for _ in range(n_warmup):
                V_ = _v_given_h(_h_given_v(V_))
            return V_

        # Initialise or reinitialise persistent chains when shape changes
        if self._gibbs_v is None or self._gibbs_v.shape != (n_samples, Nv):
            self._gibbs_v = _init_chains(n_samples)

        V = self._gibbs_v

        # PCD-k: k block Gibbs sweeps from current chain state
        for _ in range(n_sweeps):
            V = _v_given_h(_h_given_v(V))

        # Collapse detection: reinitialise stuck chains
        v_np = _xp.asnumpy(V) if _xp is not np else np.asarray(V)
        unique = len(set(map(tuple, v_np.tolist())))
        restarted = 0
        if unique < self.gibbs_collapse_threshold * n_samples:
            n_reinit = int(self.gibbs_reinit_fraction * n_samples)
            idx = rng.choice(n_samples, n_reinit, replace=False)
            V[idx] = _init_chains(n_reinit)
            restarted = n_reinit
            v_np = _xp.asnumpy(V) if _xp is not np else np.asarray(V)
            unique = len(set(map(tuple, v_np.tolist())))

        self._gibbs_v = V

        # Sample H once from final V to return joint samples
        H = _h_given_v(V)
        h_np = _xp.asnumpy(H) if _xp is not np else np.asarray(H)

        restart_str = f"  restarted={restarted}" if restarted else ""
        print(
            f"  [Gibbs] device={_DEVICE}  k={n_sweeps}  unique={unique}/{n_samples}{restart_str}"
        )
        return v_np, h_np

    def _sample_hidden(self, rbm: RBM, v_samples: np.ndarray) -> np.ndarray:
        """Sample h ~ p(h|v) at β=1 for each visible sample."""
        activation = rbm.b[None, :] + v_samples @ rbm.W  # (n_samples, n_hidden)
        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * activation))
        rng = np.random.default_rng()
        return np.where(rng.random(prob_plus.shape) < prob_plus, 1.0, -1.0)

    def estimate_beta_eff(
        self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500
    ) -> float:
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
        if _DEVICE == "gpu":
            return self._metropolis_hastings_batched(rbm, n_samples, config)

        N = rbm.n_visible
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)
        rng = np.random.default_rng()

        v = rng.choice([-1.0, 1.0], size=N)
        # Ensure C-contiguous float64 for Numba
        W_cont = np.ascontiguousarray(rbm.W, dtype=np.float64)
        a_cont = np.ascontiguousarray(rbm.a, dtype=np.float64)

        n_accepted = 0
        n_proposed = 0

        def sweep(v, theta):
            """One sweep with cached theta = b + W.T @ v, updated incrementally."""
            nonlocal n_accepted, n_proposed
            n_proposed += N
            flip_indices = rng.integers(0, N, size=N).astype(np.int64)
            rand_u = rng.random(N)
            if _HAS_NUMBA:
                n_accepted += _mh_sweep_nb(
                    v, theta, W_cont, a_cont, flip_indices, rand_u
                )
            else:
                for k in range(N):
                    flip_idx = flip_indices[k]
                    vi = v[flip_idx]
                    theta_flip = theta - 2.0 * vi * rbm.W[flip_idx, :]
                    log_ratio = rbm.a[flip_idx] * vi + 0.5 * np.sum(
                        rbm.logcosh(theta_flip) - rbm.logcosh(theta)
                    )
                    if rand_u[k] < min(1.0, np.exp(2.0 * log_ratio)):
                        v[flip_idx] *= -1
                        theta[:] = theta_flip
                        n_accepted += 1
            return v, theta

        # Warmup — equilibrate from random initial state
        theta = np.ascontiguousarray(rbm.b + rbm.W.T @ v, dtype=np.float64)
        v = v.astype(np.float64)
        for _ in range(n_warmup):
            v, theta = sweep(v, theta)

        # Reset counters so acceptance rate reflects collection phase only
        n_accepted = 0
        n_proposed = 0

        # Collect samples
        samples = []
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                v, theta = sweep(v, theta)
            samples.append(v.copy())

        acceptance_rate = n_accepted / max(n_proposed, 1)
        print(
            f"  [MH]    acceptance={acceptance_rate:.3f}  "
            f"unique={len(set(map(tuple, samples)))}/{n_samples}"
        )

        return np.array(samples)

    def _metropolis_hastings_batched(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Batched Metropolis-Hastings: all n_samples chains run in parallel on the GPU.

        Each chain is independent and starts from a random state, runs n_warmup sweeps
        to equilibrate, then n_sweeps more sweeps before its final state is collected
        as a sample.  One sweep = N single-spin-flip proposals per chain.

        Parallelism: at each proposal step, all C chains propose a (possibly different)
        spin flip simultaneously → (C, Nh) tensor operations, mapped to GPU cores.

        Device: uses CuPy (_xp = cp) when available, NumPy otherwise — same code path.
        """
        xp = _xp
        N, Nh = rbm.n_visible, rbm.n_hidden
        C = n_samples
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)

        W = xp.asarray(rbm.W, dtype=np.float64)  # (N, Nh)
        a = xp.asarray(rbm.a, dtype=np.float64)  # (N,)
        b = xp.asarray(rbm.b, dtype=np.float64)  # (Nh,)

        # Initialise C independent chains with random ±1 spins
        v = (xp.random.randint(0, 2, (C, N)) * 2 - 1).astype(np.float64)  # (C, N)
        theta = b[None, :] + v @ W  # (C, Nh)
        ci = xp.arange(C)

        n_accepted_total = 0
        n_proposed_total = 0

        def sweep(v, theta):
            nonlocal n_accepted_total, n_proposed_total
            n_proposed_total += C * N
            for _ in range(N):
                # Draw one random flip site per chain
                flip_idx = xp.random.randint(0, N, (C,))  # (C,)
                vi = v[ci, flip_idx]  # (C,)
                W_row = W[flip_idx]  # (C, Nh)

                # Compute theta if this flip were accepted (O(C*Nh), fully vectorised)
                theta_flip = theta - 2.0 * vi[:, None] * W_row  # (C, Nh)

                # Log acceptance ratio: 2 * [ a_i * v_i + 0.5 * Σ_j Δlogcosh_j ]
                lc_diff = 0.5 * xp.sum(
                    _logcosh_xp(xp, theta_flip) - _logcosh_xp(xp, theta), axis=1
                )  # (C,)
                log_ratio = a[flip_idx] * vi + lc_diff  # (C,)

                # Accept with probability min(1, exp(2*log_ratio))
                accept = xp.log(xp.random.rand(C)) < 2.0 * log_ratio  # (C,) bool

                # Update v and theta only for accepted chains
                v[ci, flip_idx] = xp.where(accept, -vi, vi)
                theta = xp.where(accept[:, None], theta_flip, theta)
                n_accepted_total += int(xp.sum(accept))
            return v, theta

        for _ in range(n_warmup):
            v, theta = sweep(v, theta)

        # Reset acceptance counters — only measure collection phase
        n_accepted_total = 0
        n_proposed_total = 0

        for _ in range(n_sweeps):
            v, theta = sweep(v, theta)

        acceptance_rate = n_accepted_total / max(n_proposed_total, 1)
        # Transfer to CPU
        v_np = xp.asnumpy(v) if xp is not np else np.asarray(v)
        unique = len(set(map(tuple, v_np.tolist())))
        print(
            f"  [MH-batch] device={_DEVICE}  acceptance={acceptance_rate:.3f}  "
            f"unique={unique}/{n_samples}"
        )
        return v_np

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
        if _DEVICE == "gpu":
            return self._simulated_annealing_batched(rbm, n_samples, config)

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
        # Ensure C-contiguous float64 for Numba
        W_cont = np.ascontiguousarray(rbm.W, dtype=np.float64)
        a_cont = np.ascontiguousarray(rbm.a, dtype=np.float64)

        # Geometric cooling schedule: T(step) = T_initial * (T_final/T_initial)^(step/n_steps)
        n_steps = n_samples * n_sweeps

        def schedule(step: int) -> float:
            if T_initial == T_final:
                return T_final
            return T_initial * (T_final / T_initial) ** (step / max(n_steps - 1, 1))

        n_accepted = 0
        n_proposed = 0

        def sweep(v, theta, T):
            """One sweep with cached theta = b + W.T @ v, updated incrementally."""
            nonlocal n_accepted, n_proposed
            n_proposed += N
            flip_indices = rng.integers(0, N, size=N).astype(np.int64)
            rand_u = rng.random(N)
            if _HAS_NUMBA:
                n_accepted += _sa_sweep_nb(
                    v, theta, W_cont, a_cont, flip_indices, rand_u, T
                )
            else:
                for k in range(N):
                    flip_idx = flip_indices[k]
                    vi = v[flip_idx]
                    theta_flip = theta - 2.0 * vi * rbm.W[flip_idx, :]
                    log_ratio = rbm.a[flip_idx] * vi + 0.5 * np.sum(
                        rbm.logcosh(theta_flip) - rbm.logcosh(theta)
                    )
                    if rand_u[k] < min(1.0, np.exp(2.0 * log_ratio / T)):
                        v[flip_idx] *= -1
                        theta[:] = theta_flip
                        n_accepted += 1
            return v, theta

        # Warmup at T_initial — equilibrate before cooling
        theta = np.ascontiguousarray(rbm.b + rbm.W.T @ v, dtype=np.float64)
        v = v.astype(np.float64)
        for _ in range(n_warmup):
            v, theta = sweep(v, theta, T_initial)

        n_accepted = 0
        n_proposed = 0

        # Collect samples while cooling
        samples = []
        step = 0
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                T = schedule(step)
                v, theta = sweep(v, theta, T)
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

    def _simulated_annealing_batched(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Batched SA: C chains cooled in parallel on GPU (or CPU via NumPy).

        All chains share the same geometric cooling schedule.  After n_warmup
        sweeps at T_initial, the schedule runs for n_samples * n_sweeps sweeps
        and the final state of each chain is returned as a sample.
        """
        xp = _xp
        N, Nh = rbm.n_visible, rbm.n_hidden
        C = n_samples
        T_initial = config.get(
            "T_initial", self.T_initial if hasattr(self, "T_initial") else 5.0
        )
        T_final = config.get(
            "T_final", self.T_final if hasattr(self, "T_final") else 1.0
        )
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)

        W = xp.asarray(rbm.W, dtype=np.float64)
        a = xp.asarray(rbm.a, dtype=np.float64)
        b = xp.asarray(rbm.b, dtype=np.float64)

        v = (xp.random.randint(0, 2, (C, N)) * 2 - 1).astype(np.float64)
        theta = b[None, :] + v @ W
        ci = xp.arange(C)

        n_total_steps = n_samples * n_sweeps

        def schedule(step):
            if T_initial == T_final:
                return T_final
            return T_initial * (T_final / T_initial) ** (
                step / max(n_total_steps - 1, 1)
            )

        def sweep(v, theta, T):
            for _ in range(N):
                flip_idx = xp.random.randint(0, N, (C,))
                vi = v[ci, flip_idx]
                W_row = W[flip_idx]
                theta_flip = theta - 2.0 * vi[:, None] * W_row
                lc_diff = 0.5 * xp.sum(
                    _logcosh_xp(xp, theta_flip) - _logcosh_xp(xp, theta), axis=1
                )
                log_ratio = a[flip_idx] * vi + lc_diff
                accept = xp.log(xp.random.rand(C)) < 2.0 * log_ratio / T
                v[ci, flip_idx] = xp.where(accept, -vi, vi)
                theta = xp.where(accept[:, None], theta_flip, theta)
            return v, theta

        for _ in range(n_warmup):
            v, theta = sweep(v, theta, T_initial)

        step = 0
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                v, theta = sweep(v, theta, schedule(step))
                step += 1

        v_np = xp.asnumpy(v) if xp is not np else np.asarray(v)
        unique = len(set(map(tuple, v_np.tolist())))
        T_now = schedule(step - 1)
        print(
            f"  [SA-batch] device={_DEVICE}  T: {T_initial:.2f}→{T_now:.2f}  "
            f"unique={unique}/{n_samples}"
        )
        return v_np


class VeloxSampler(Sampler):
    def __init__(
        self,
        method: str,
        sbm_steps: int = 5000,
        sbm_dt: float = 1.0,
        sbm_discrete: bool = False,
    ):
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

    def sample(
        self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
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
            h_samples = df.loc[
                :, list(range(self.n_visible, self.n_visible + rbm.n_hidden))
            ].to_numpy()
            return v, h_samples
        return v

    def estimate_beta_eff(
        self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500
    ) -> float:
        """
        Estimate β_eff via the joint-samples variant of CEM.

        The conditional h problem has no h-h interactions in an RBM, so
        conditional sampling on VeloxQ would trivially return ground states.
        Instead, joint (v, h) samples are drawn from the full interacting
        problem and β is fit via _cem_fit_beta_joint.  r is unused.
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

    def sample(
        self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
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
            return self.dwave(
                bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def estimate_beta_eff(
        self, rbm: RBM, r: np.ndarray = None, n_samples: int = 500
    ) -> float:
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

    def simulated_annealing(
        self, bqm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
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
        # Sort columns by variable index — sampleset.variables is not
        # guaranteed to be ordered, so raw slicing [:, :n_visible] would
        # silently mix visible and hidden units.
        sort_idx = np.argsort(list(sampleset.variables))
        samples = sampleset.record.sample[:, sort_idx]
        unique_samples = len(set(map(tuple, samples)))
        print(f"  unique samples: {unique_samples}/{len(samples)}")
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def tabu_search(
        self, bqm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
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

        # Sort columns by variable index — same reason as in simulated_annealing.
        sort_idx = np.argsort(list(sampleset.variables))
        samples = sampleset.record.sample[:, sort_idx]
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def dwave(
        self,
        bqm,
        n_samples: int,
        config: dict = {},
        rbm=None,
        return_hidden: bool = False,
    ):
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
                self._log_access_time(access_time_us)
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
