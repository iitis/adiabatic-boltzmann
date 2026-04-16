"""
Samplers — JAX backend

Key changes vs NumPy/CuPy/Numba version
-----------------------------------------
* CuPy and the _xp device-dispatch abstraction are removed.
  JAX dispatches to GPU automatically via XLA — no code changes needed.
* Numba @njit kernels (_mh_sweep_nb, _sa_sweep_nb) replaced by
  jax.lax.scan-based kernels JIT-compiled once per unique (C, N, n_steps).
* LSB migrated from PyTorch to JAX lax.scan.
* All np.random calls replaced by jax.random with explicit PRNG keys.
  ClassicalSampler maintains self._key as stateful key; call _next_key()
  to get a fresh subkey and advance the state.
* Gibbs persistent chain (self._gibbs_v) is now a JAX array.
* In-place mutations like v[idx] = x become v = v.at[idx].set(x).
* DimodSampler, VeloxSampler, FPGASampler are unchanged — they interface
  with external hardware and return NumPy arrays.
"""

import fcntl
import functools
import json
import math as _math
import os
import subprocess
import tempfile
import threading
import time
import uuid
import sys as _sys
import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from model import RBM
import dimod
from pathlib import Path
from helpers import get_solver_name
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# JIT-compiled sampling kernels (module-level — compiled once per process)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def _mh_sweep_jit(
    v: jax.Array,
    theta: jax.Array,
    W: jax.Array,
    a: jax.Array,
    key: jax.Array,
    C: int,
    N: int,
    total_steps: int,
) -> tuple:
    """
    Batched Metropolis-Hastings: C independent chains for total_steps flips.

    v     : (C, N)   spin configs ±1
    theta : (C, Nh)  pre-activations b + v @ W
    W     : (N, Nh)
    a     : (N,)
    key   : JAX PRNG key

    One step = one random single-spin-flip proposal for every chain simultaneously.
    All C chains are updated in a single XLA kernel via jax.lax.scan.
    """
    ci = jnp.arange(C)

    def one_flip(carry, _):
        v, theta, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        flip_idx = jax.random.randint(k1, (C,), 0, N)   # (C,) — different site per chain
        vi = v[ci, flip_idx]                              # (C,)
        W_row = W[flip_idx]                               # (C, Nh)
        theta_flip = theta - 2.0 * vi[:, None] * W_row   # (C, Nh)
        lc_diff = 0.5 * jnp.sum(
            jnp.logaddexp(theta_flip, -theta_flip)
            - jnp.logaddexp(theta, -theta),
            axis=1,
        )  # (C,)
        log_ratio = a[flip_idx] * vi + lc_diff            # (C,)
        rand_u = jax.random.uniform(k2, (C,), dtype=jnp.float64)
        accept = jnp.log(rand_u) < 2.0 * log_ratio       # (C,) bool
        v = v.at[ci, flip_idx].set(jnp.where(accept, -vi, vi))
        theta = jnp.where(accept[:, None], theta_flip, theta)
        return (v, theta, key), None

    (v, theta, _), _ = jax.lax.scan(one_flip, (v, theta, key), None, length=total_steps)
    return v, theta


@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def _sa_sweep_jit(
    v: jax.Array,
    theta: jax.Array,
    W: jax.Array,
    a: jax.Array,
    key: jax.Array,
    C: int,
    N: int,
    total_steps: int,
    T_initial: float,
    T_final: float,
) -> tuple:
    """
    Batched Simulated Annealing: C chains cooled along a geometric schedule.

    xs = jnp.arange(total_steps) carries the step index into each scan body
    so the temperature can be computed without Python-level control flow.
    """
    ci = jnp.arange(C)
    n_steps_f = jnp.float64(max(total_steps - 1, 1))

    def one_flip(carry, step):
        v, theta, key = carry
        T = T_initial * (T_final / T_initial) ** (step.astype(jnp.float64) / n_steps_f)
        key, k1, k2 = jax.random.split(key, 3)
        flip_idx = jax.random.randint(k1, (C,), 0, N)
        vi = v[ci, flip_idx]
        W_row = W[flip_idx]
        theta_flip = theta - 2.0 * vi[:, None] * W_row
        lc_diff = 0.5 * jnp.sum(
            jnp.logaddexp(theta_flip, -theta_flip)
            - jnp.logaddexp(theta, -theta),
            axis=1,
        )
        log_ratio = a[flip_idx] * vi + lc_diff
        rand_u = jax.random.uniform(k2, (C,), dtype=jnp.float64)
        accept = jnp.log(rand_u) < 2.0 * log_ratio / T
        v = v.at[ci, flip_idx].set(jnp.where(accept, -vi, vi))
        theta = jnp.where(accept[:, None], theta_flip, theta)
        return (v, theta, key), None

    steps = jnp.arange(total_steps)
    (v, theta, _), _ = jax.lax.scan(one_flip, (v, theta, key), steps)
    return v, theta


@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def _lsb_jit(
    key: jax.Array,
    M: jax.Array,
    f: jax.Array,
    sigma: float,
    delta: float,
    n_samples: int,
    steps: int,
    N_total: int,
) -> jax.Array:
    """
    Langevin Simulated Bifurcation (Kubo & Goto 2025, Sec. II B 1).

    Symplectic Euler integration (lax.scan over `steps` iterations):
        y[k+1] = y[k] + δ·(M·x[k] + f) + σ·ξ    ξ ~ N(0,1)
        x[k+1] = x[k] + δ·y[k+1]
        x      ← clip(x, −1, +1)

    Init: x ~ U[−1,1],  y ~ N(0, σ²).
    Discretise once at the end: s = sgn(x).
    """
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (n_samples, N_total), dtype=jnp.float64) * 2.0 - 1.0
    y = sigma * jax.random.normal(k2, (n_samples, N_total), dtype=jnp.float64)

    def step_fn(carry, _):
        x, y, key = carry
        key, noise_key = jax.random.split(key)
        force = x @ M.T + f
        noise = sigma * jax.random.normal(noise_key, y.shape, dtype=jnp.float64)
        y = y + delta * force + noise
        x = x + delta * y
        x = jnp.clip(x, -1.0, 1.0)
        return (x, y, key), None

    (x, _, _), _ = jax.lax.scan(step_fn, (x, y, k3), None, length=steps)
    s = jnp.sign(x)
    s = jnp.where(s == 0, 1.0, s)
    return s


# ---------------------------------------------------------------------------
# Abstract sampler
# ---------------------------------------------------------------------------


class Sampler(ABC):
    """Abstract sampling interface."""

    def rbm_to_ising(self, rbm, beta_x: float = 1.0):
        """
        Convert RBM parameters to Ising (J, h) for external solvers.
        rbm.W / .a / .b are JAX arrays; float() conversion is safe for scalars.
        """
        _last = getattr(self, "_last_beta_x_logged", None)
        if _last is None or abs(beta_x - _last) / max(abs(_last), 1e-9) > 0.01:
            print(f"  [rbm_to_ising] beta_x = {beta_x:.4f}")
            self._last_beta_x_logged = beta_x

        Nv, Nh = rbm.n_visible, rbm.n_hidden
        linear = {}
        quadratic = {}

        # Use np.asarray once for whole-array access (avoids repeated scalar transfers)
        a_np = np.asarray(rbm.a)
        b_np = np.asarray(rbm.b)
        W_np = np.asarray(rbm.W)

        for i in range(Nv):
            linear[i] = -float(a_np[i]) / beta_x
        for j in range(Nh):
            linear[Nv + j] = -float(b_np[j]) / beta_x
        for i in range(Nv):
            for j in range(Nh):
                if abs(W_np[i, j]) > 1e-6:
                    quadratic[(i, Nv + j)] = -float(W_np[i, j]) / beta_x

        return quadratic, linear

    @abstractmethod
    def sample(
        self, rbm, n_samples: int, config: dict = None, return_hidden: bool = False
    ):
        pass


# ---------------------------------------------------------------------------
# Classical (CPU/GPU) sampler
# ---------------------------------------------------------------------------


class ClassicalSampler(Sampler):
    """
    Classical sampling via Metropolis-Hastings, Simulated Annealing,
    Gibbs, or Langevin SB — all JAX-accelerated.
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

        self._gibbs_v = None          # persistent chain state (JAX array)
        self._key = None              # JAX PRNG key — initialised lazily
        self._last_sample_config: dict = {}

    def _next_key(self) -> jax.Array:
        """Advance self._key and return a fresh subkey."""
        if self._key is None:
            # Lazy init: derive from numpy so the caller doesn't have to set it
            seed = int(np.random.randint(0, 2**31))
            self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def sample(
        self, rbm: RBM, n_samples: int, config: dict = None, return_hidden: bool = False
    ):
        if config is None:
            config = {}
        self._last_sample_config = dict(config)

        if self.method == "lsb":
            v, h = self._lsb_sample(rbm, n_samples, config)
            return (v, h) if return_hidden else v

        if self.method == "gibbs":
            v, h = self._gibbs_sample(rbm, n_samples, config)
            return (v, h) if return_hidden else v

        if self.method == "metropolis":
            v = self._metropolis_hastings(rbm, n_samples, config)
        elif self.method == "simulated_annealing":
            v = self._simulated_annealing(rbm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_hidden:
            return v, self._sample_hidden(rbm, v)
        return v

    # ── Langevin SB ──────────────────────────────────────────────────────

    def _lsb_sample(self, rbm: RBM, n_samples: int, config: dict):
        """
        Langevin Simulated Bifurcation — pure JAX, GPU-accelerated via lax.scan.
        """
        beta_x = config.get("beta_x", 1.0)
        steps = config.get("lsb_steps", 1000)
        delta = config.get("lsb_delta", 1.0)
        sigma_inv2 = config.get("lsb_sigma", 1.0)
        sigma = float(1.0 / np.sqrt(sigma_inv2))

        Nv, Nh = rbm.n_visible, rbm.n_hidden
        N_total = Nv + Nh

        # Build interaction matrix (JAX)
        M = jnp.zeros((N_total, N_total), dtype=jnp.float64)
        M = M.at[:Nv, Nv:].set(rbm.W / beta_x)
        M = M.at[Nv:, :Nv].set(rbm.W.T / beta_x)
        f = jnp.concatenate([rbm.a / beta_x, rbm.b / beta_x])

        key = self._next_key()
        s = _lsb_jit(key, M, f, sigma, delta, n_samples, steps, N_total)

        s_np = np.asarray(s)
        v = s_np[:, :Nv]
        h = s_np[:, Nv:]

        unique = len(set(map(tuple, v.tolist())))
        print(
            f"  [LSB] steps={steps} delta={delta} sigma={sigma:.4f}"
            f" unique={unique}/{n_samples}"
        )
        return v, h

    # ── Gibbs ─────────────────────────────────────────────────────────────

    def _gibbs_sample(self, rbm: RBM, n_samples: int, config: dict):
        """
        Persistent block Gibbs (PCD-k) targeting |Ψ(v)|².

        Block conditionals (all units independent within each block):
            p(h_j = +1 | v) = σ(2(b_j + W[:,j]·v))
            p(v_i = +1 | h) = σ(2(W[i,:]·h − a_i))
        """
        n_sweeps = config.get("n_sweeps", self.n_sweeps)
        n_warmup = config.get("n_warmup", self.n_warmup)
        Nv, Nh = rbm.n_visible, rbm.n_hidden
        W, a, b = rbm.W, rbm.a, rbm.b  # JAX arrays

        def h_given_v(V, key):
            prob = 1.0 / (1.0 + jnp.exp(-2.0 * (V @ W + b[None, :])))
            u = jax.random.uniform(key, (V.shape[0], Nh), dtype=jnp.float64)
            return jnp.where(u < prob, 1.0, -1.0)

        def v_given_h(H, key):
            prob = 1.0 / (1.0 + jnp.exp(-2.0 * (H @ W.T - a[None, :])))
            u = jax.random.uniform(key, (H.shape[0], Nv), dtype=jnp.float64)
            return jnp.where(u < prob, 1.0, -1.0)

        def gibbs_sweep(V, key):
            k1, k2 = jax.random.split(key)
            return v_given_h(h_given_v(V, k1), k2)

        def init_chains(n, key):
            k1, k2 = jax.random.split(key)
            V_ = jax.random.choice(
                k1, jnp.array([-1.0, 1.0]), shape=(n, Nv)
            ).astype(jnp.float64)
            for _ in range(n_warmup):
                k2, k = jax.random.split(k2)
                V_ = gibbs_sweep(V_, k)
            return V_

        # Initialise or reinitialise persistent chains when shape changes
        key = self._next_key()
        if self._gibbs_v is None or self._gibbs_v.shape != (n_samples, Nv):
            self._gibbs_v = init_chains(n_samples, key)
            key = self._next_key()

        V = self._gibbs_v
        for _ in range(n_sweeps):
            key = self._next_key()
            V = gibbs_sweep(V, key)

        # Collapse detection
        v_np = np.asarray(V)
        unique = len(set(map(tuple, v_np.tolist())))
        restarted = 0
        if unique < self.gibbs_collapse_threshold * n_samples:
            n_reinit = int(self.gibbs_reinit_fraction * n_samples)
            key = self._next_key()
            k1, k2 = jax.random.split(key)
            # Select which chains to reinitialise
            idx = np.random.choice(n_samples, n_reinit, replace=False)
            new_chains = init_chains(n_reinit, k2)
            V = V.at[jnp.array(idx)].set(new_chains)
            restarted = n_reinit
            v_np = np.asarray(V)
            unique = len(set(map(tuple, v_np.tolist())))

        self._gibbs_v = V

        # Sample hidden once from final V
        key = self._next_key()
        H = h_given_v(V, key)
        h_np = np.asarray(H)

        restart_str = f"  restarted={restarted}" if restarted else ""
        print(f"  [Gibbs] k={n_sweeps}  unique={unique}/{n_samples}{restart_str}")
        return v_np, h_np

    def _sample_hidden(self, rbm: RBM, v_samples) -> np.ndarray:
        """Sample h ~ p(h|v) at β=1 for each visible sample."""
        V = jnp.asarray(v_samples, dtype=jnp.float64)
        activation = rbm.b[None, :] + V @ rbm.W
        prob_plus = 1.0 / (1.0 + jnp.exp(-2.0 * activation))
        key = self._next_key()
        u = jax.random.uniform(key, prob_plus.shape, dtype=jnp.float64)
        return np.asarray(jnp.where(u < prob_plus, 1.0, -1.0))

    # ── Metropolis-Hastings ───────────────────────────────────────────────

    def _metropolis_hastings(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Batched MH: n_samples independent chains run in parallel.

        Each chain runs n_warmup + n_sweeps sweeps.  A sweep = N single-spin-
        flip proposals.  All C chains are updated simultaneously via lax.scan.
        """
        N, Nh = rbm.n_visible, rbm.n_hidden
        C = n_samples
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)

        W, a, b = rbm.W, rbm.a, rbm.b

        key = self._next_key()
        k1, k2 = jax.random.split(key)
        v = jax.random.choice(
            k1, jnp.array([-1.0, 1.0]), shape=(C, N)
        ).astype(jnp.float64)
        theta = b[None, :] + v @ W  # (C, Nh)

        total_steps = N * (n_warmup + n_sweeps)
        v, _ = _mh_sweep_jit(v, theta, W, a, k2, C, N, total_steps)

        v_np = np.asarray(v)
        unique = len(set(map(tuple, v_np.tolist())))
        print(f"  [MH]    unique={unique}/{n_samples}")
        return v_np

    # ── Simulated Annealing ───────────────────────────────────────────────

    def _simulated_annealing(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Batched SA: n_samples chains cooled in parallel along a geometric schedule.
        """
        N, Nh = rbm.n_visible, rbm.n_hidden
        C = n_samples
        T_initial = config.get("T_initial", self.T_initial)
        T_final = config.get("T_final", self.T_final)
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)

        W, a, b = rbm.W, rbm.a, rbm.b

        key = self._next_key()
        k1, k2, k3 = jax.random.split(key, 3)

        # Warmup at T_initial
        v = jax.random.choice(
            k1, jnp.array([-1.0, 1.0]), shape=(C, N)
        ).astype(jnp.float64)
        theta = b[None, :] + v @ W
        warmup_steps = N * n_warmup
        v, theta = _sa_sweep_jit(
            v, theta, W, a, k2, C, N, warmup_steps, T_initial, T_initial
        )

        # Cooling sweep
        cool_steps = N * n_sweeps
        v, _ = _sa_sweep_jit(v, theta, W, a, k3, C, N, cool_steps, T_initial, T_final)

        v_np = np.asarray(v)
        unique = len(set(map(tuple, v_np.tolist())))
        print(
            f"  [SA]    T: {T_initial:.2f}→{T_final:.2f}  unique={unique}/{n_samples}"
        )
        return v_np


# ---------------------------------------------------------------------------
# Velox sampler (unchanged — external hardware, NumPy I/O)
# ---------------------------------------------------------------------------


class VeloxSampler(Sampler):
    def __init__(
        self,
        method: str,
        sbm_steps: int = 5000,
        sbm_dt: float = 1.0,
        sbm_discrete: bool = False,
    ):
        from veloxq_sdk import VeloxQSolver, SBMSolver, SBMParameters
        from veloxq_sdk.config import load_config, VeloxQAPIConfig

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
            from veloxq_sdk import VeloxQSolver
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


# ---------------------------------------------------------------------------
# FPGA sampler (unchanged — subprocess + Julia bridge, NumPy I/O)
# ---------------------------------------------------------------------------


class FPGASampler(Sampler):
    """
    FPGA sampler wrapper that delegates sampling to the VeloxQFPGA Julia stack.
    """

    _ENV_MAP = {
        "fpga_syscon_path": "FPGA_SYSCON_PATH",
        "fpga_bulk_dir": "FPGA_BULK_DIR",
        "fpga_bulk_load": "FPGA_BULK_LOAD",
        "fpga_bitstream": "FPGA_BITSTREAM",
        "fpga_quartus_root": "FPGA_QUARTUS_ROOT",
        "fpga_pcie_device": "FPGA_PCIE_DEVICE",
        "fpga_pcie_bar_size": "FPGA_PCIE_BAR_SIZE",
        "fpga_core_clock_hz": "FPGA_CORE_CLOCK_HZ",
        "fpga_timeout_s": "FPGA_TIMEOUT_S",
        "fpga_verbose": "FPGA_VERBOSE",
    }

    def __init__(
        self,
        transport: str = "jtag",
        julia_cmd: str = "julia",
        project_path=None,
        script_path=None,
        num_rep: int = 1024,
        num_steps: int = 100,
        num_sweeps: int = 1,
        start_temp: float = -1.0,
        stop_temp: float = -1.0,
        schedule_type: str = "geometric",
        keep_files: bool = False,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        default_project = (repo_root.parent / "veloxQFPGA").resolve()
        default_script = repo_root / "scripts" / "fpga_sa_bridge.jl"

        self.transport = transport
        self.julia_cmd = julia_cmd
        self.project_path = (
            Path(project_path).resolve() if project_path else default_project
        )
        self.script_path = (
            Path(script_path).resolve() if script_path else default_script
        )
        self.num_rep = int(num_rep)
        self.num_steps = int(num_steps)
        self.num_sweeps = int(num_sweeps)
        self.start_temp = float(start_temp)
        self.stop_temp = float(stop_temp)
        self.schedule_type = str(schedule_type)
        self.keep_files = keep_files

        self._tmpdir = tempfile.TemporaryDirectory(prefix="fpga_sampler_")
        self._tmp_root = Path(self._tmpdir.name)

        if not self.project_path.exists():
            raise FileNotFoundError(
                f"VeloxQFPGA project not found at {self.project_path}."
            )
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"FPGA Julia bridge script not found at {self.script_path}."
            )
        self.last_sampling_time_s = None

    def _write_ising_csv(self, path, linear, quadratic, n_vars):
        with path.open("w") as f:
            for i in range(n_vars):
                val = float(linear.get(i, 0.0)) * 0.5
                f.write(f"{i + 1},{i + 1},{val:.16g}\n")
            for (i, j), val in sorted(quadratic.items()):
                f.write(f"{i + 1},{j + 1},{float(val):.16g}\n")

    def _apply_env_overrides(self, env, config):
        for key, env_key in self._ENV_MAP.items():
            if key not in config:
                continue
            val = config[key]
            if isinstance(val, bool):
                env[env_key] = "true" if val else "false"
            else:
                env[env_key] = str(val)

    def _bool_from_env(self, env, key):
        val = env.get(key, "")
        return str(val).strip().lower() in ("1", "true", "yes", "on") if val else False

    def _timeout_from_env(self, env, key):
        val = env.get(key, "")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def sample(self, rbm, n_samples: int, config: dict = None, return_hidden: bool = False):
        if config is None:
            config = {}
        self.last_sampling_time_s = None
        beta_x = config.get("beta_x", 1.0)
        quadratic, linear = self.rbm_to_ising(rbm, beta_x)
        n_vars = rbm.n_visible + rbm.n_hidden
        model_path = self._tmp_root / f"ising_{uuid.uuid4().hex}.csv"
        out_path = self._tmp_root / f"states_{uuid.uuid4().hex}.txt"
        meta_path = self._tmp_root / f"meta_{uuid.uuid4().hex}.txt"
        self._write_ising_csv(model_path, linear, quadratic, n_vars)

        num_rep = int(config.get("fpga_num_rep", self.num_rep))
        if n_samples > num_rep:
            raise ValueError(
                f"Requested n_samples={n_samples} exceeds FPGA num_rep={num_rep}."
            )

        num_steps = int(config.get("fpga_num_steps", self.num_steps))
        num_sweeps = int(config.get("fpga_num_sweeps", self.num_sweeps))
        start_temp = float(config.get("fpga_start_temp", self.start_temp))
        stop_temp = float(config.get("fpga_stop_temp", self.stop_temp))
        schedule_type = str(config.get("fpga_schedule", self.schedule_type))
        transport = str(config.get("fpga_transport", self.transport))

        cmd = [
            self.julia_cmd, f"--project={self.project_path}", str(self.script_path),
            str(model_path), str(out_path), str(num_rep), str(num_steps),
            str(num_sweeps), str(start_temp), str(stop_temp),
            schedule_type, transport, str(meta_path),
        ]

        env = os.environ.copy()
        self._apply_env_overrides(env, config)
        stream_output = bool(
            config.get("fpga_stream_output")
            if "fpga_stream_output" in config
            else self._bool_from_env(env, "FPGA_STREAM_OUTPUT")
            or self._bool_from_env(env, "FPGA_VERBOSE")
        )
        timeout_s = config.get("fpga_timeout_s", None)
        if timeout_s is None:
            timeout_s = self._timeout_from_env(env, "FPGA_TIMEOUT_S")

        if stream_output:
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            def _reader(stream, sink, collector):
                try:
                    for line in iter(stream.readline, ""):
                        if not line:
                            break
                        sink.write(line)
                        sink.flush()
                        collector.append(line)
                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env,
            )
            t_out = threading.Thread(
                target=_reader, args=(proc.stdout, _sys.stdout, stdout_lines)
            )
            t_err = threading.Thread(
                target=_reader, args=(proc.stderr, _sys.stderr, stderr_lines)
            )
            t_out.daemon = True
            t_err.daemon = True
            t_out.start()
            t_err.start()
            try:
                if timeout_s is None:
                    returncode = proc.wait()
                else:
                    returncode = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait()
                msg = f"FPGA sampler timed out after {timeout_s} seconds."
                if stdout_lines:
                    msg += "\nstdout:\n" + "".join(stdout_lines[-200:])
                if stderr_lines:
                    msg += "\nstderr:\n" + "".join(stderr_lines[-200:])
                raise RuntimeError(msg)
            finally:
                t_out.join(timeout=2)
                t_err.join(timeout=2)
            if returncode != 0:
                msg = f"FPGA sampler failed (exit {returncode})."
                if stdout_lines:
                    msg += "\nstdout:\n" + "".join(stdout_lines[-200:])
                if stderr_lines:
                    msg += "\nstderr:\n" + "".join(stderr_lines[-200:])
                raise RuntimeError(msg)
        else:
            try:
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, env=env, timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as e:
                msg = f"FPGA sampler timed out after {timeout_s} seconds."
                if e.stdout:
                    msg += f"\nstdout:\n{e.stdout}"
                if e.stderr:
                    msg += f"\nstderr:\n{e.stderr}"
                raise RuntimeError(msg) from e
            if result.returncode != 0:
                msg = f"FPGA sampler failed (exit {result.returncode})."
                if result.stdout:
                    msg += f"\nstdout:\n{result.stdout}"
                if result.stderr:
                    msg += f"\nstderr:\n{result.stderr}"
                raise RuntimeError(msg)

        samples = np.loadtxt(out_path, dtype=np.int8)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.shape[1] != n_vars:
            raise RuntimeError(
                f"FPGA sampler returned {samples.shape[1]} vars, expected {n_vars}."
            )
        if samples.shape[0] < n_samples:
            raise RuntimeError(
                f"FPGA sampler returned {samples.shape[0]} samples, expected {n_samples}."
            )
        if samples.shape[0] > n_samples:
            samples = samples[:n_samples]

        if meta_path.exists():
            try:
                self.last_sampling_time_s = float(meta_path.read_text().strip())
            except Exception:
                self.last_sampling_time_s = None

        if not self.keep_files:
            for p in (model_path, out_path, meta_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        v = samples[:, : rbm.n_visible]
        if return_hidden:
            h_samples = samples[:, rbm.n_visible : rbm.n_visible + rbm.n_hidden]
            return v, h_samples
        return v


# ---------------------------------------------------------------------------
# Dimod sampler (unchanged — D-Wave QPU, NumPy I/O)
# ---------------------------------------------------------------------------


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
        beta_x = config.get("beta_x", 1.0)
        J, h = self.rbm_to_ising(rbm, beta_x)
        self.n_visible = rbm.n_visible
        self.n_hidden = rbm.n_hidden
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)

        if self.method == "simulated_annealing":
            return self.simulated_annealing(bqm, n_samples, config, return_hidden)
        elif self.method == "tabu":
            return self.tabu_search(bqm, n_samples, config, return_hidden)
        elif self.method in ("pegasus", "zephyr"):
            config["solver"] = get_solver_name(self.method)
            return self.dwave(bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _log_access_time(self, access_time_us: float):
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

    def simulated_annealing(self, bqm, n_samples, config={}, return_hidden=False):
        import neal
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=n_samples,
            beta_range=(0.01, 10.0),
            num_sweeps=1000,
            beta_schedule_type="geometric",
        )
        sort_idx = np.argsort(list(sampleset.variables))
        samples = sampleset.record.sample[:, sort_idx]
        print(f"  unique samples: {len(set(map(tuple, samples)))}/{len(samples)}")
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def tabu_search(self, bqm, n_samples, config={}, return_hidden=False):
        from dwave.samplers import TabuSampler
        sampler = TabuSampler()
        sampleset = sampler.sample(bqm, num_reads=n_samples)
        sort_idx = np.argsort(list(sampleset.variables))
        samples = sampleset.record.sample[:, sort_idx]
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def dwave(self, bqm, n_samples, config={}, rbm=None, return_hidden=False):
        from dwave.system import (
            DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite,
        )
        from model import DWaveTopologyRBM

        solver_name = config.get("solver", None)
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)
        cache_key = (self.n_visible, solver_name)

        if cache_key not in self._embedding_cache:
            dwave_sampler = DWaveSampler(solver=solver_name)
            if rbm is not None and isinstance(rbm, DWaveTopologyRBM):
                assert rbm._qubit_mapping is not None
                identity_embedding = {
                    logical: [phys] for phys, logical in rbm._qubit_mapping.items()
                }
                composite = FixedEmbeddingComposite(dwave_sampler, identity_embedding)
                print(f"  [embedding] Trivial identity embedding cached for {cache_key}.")
            else:
                print(f"  [embedding] Running minorminer for {cache_key}...")
                import minorminer
                embedding = minorminer.find_embedding(
                    list(bqm.quadratic.keys()), dwave_sampler.edgelist,
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

        is_trivial = (
            rbm is not None
            and isinstance(rbm, DWaveTopologyRBM)
            and rbm._qubit_mapping is not None
        )
        sample_kwargs = dict(
            num_reads=num_reads, annealing_time=annealing_time,
            answer_mode="raw", auto_scale=True,
        )
        if not is_trivial and chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength

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
                print(f"  D-Wave sampling attempt {tries}/{MAX_DWAVE_RETRIES} failed: {e}")
                if tries < MAX_DWAVE_RETRIES:
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
