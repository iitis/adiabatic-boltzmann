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
import atexit
import json
import os
import socket as _socket
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
import sys as _sys
import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from model import RBM
import dimod
from pathlib import Path
from helpers import get_solver_name


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
        flip_idx = jax.random.randint(k1, (C,), 0, N)  # (C,) — different site per chain
        vi = v[ci, flip_idx]  # (C,)
        W_row = W[flip_idx]  # (C, Nh)
        theta_flip = theta - 2.0 * vi[:, None] * W_row  # (C, Nh)
        lc_diff = 0.5 * jnp.sum(
            jnp.logaddexp(theta_flip, -theta_flip) - jnp.logaddexp(theta, -theta),
            axis=1,
        )  # (C,)
        log_ratio = a[flip_idx] * vi + lc_diff  # (C,)
        rand_u = jax.random.uniform(k2, (C,), dtype=jnp.float64)
        accept = jnp.log(rand_u) < 2.0 * log_ratio  # (C,) bool
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
            jnp.logaddexp(theta_flip, -theta_flip) - jnp.logaddexp(theta, -theta),
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
def _exchange_mh_sweep_jit(
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
    Batched spin-exchange MH: C independent chains for total_steps proposals.

    Each proposal: pick two random sites i, j.  If spins are antiparallel
    (vᵢ ≠ vⱼ), attempt a simultaneous flip of both — this conserves total S_z.
    If spins are parallel, the proposal is skipped (no move).

    The log-ratio for simultaneously flipping sites i and j is:
        log Ψ'/Ψ = aᵢvᵢ + aⱼvⱼ + ½Σₖ[logcosh(θₖ−2vᵢWᵢₖ−2vⱼWⱼₖ) − logcosh(θₖ)]

    Picking random non-adjacent pairs (not just nearest-neighbour) is essential
    for Heisenberg antiferromagnets: it lets the chain hop between degenerate
    Néel configurations that are far apart in Hamming distance.
    """
    ci = jnp.arange(C)

    def one_step(carry, _):
        v, theta, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        i_idx = jax.random.randint(k1, (C,), 0, N)
        j_raw = jax.random.randint(k2, (C,), 0, N)
        # Guarantee i ≠ j by shifting j when they coincide
        j_idx = jnp.where(j_raw == i_idx, (i_idx + 1) % N, j_raw)

        vi = v[ci, i_idx]  # (C,)
        vj = v[ci, j_idx]  # (C,)

        Wi_row = W[i_idx]  # (C, Nh) — gathered rows
        Wj_row = W[j_idx]  # (C, Nh)

        theta_flip = theta - 2.0 * vi[:, None] * Wi_row - 2.0 * vj[:, None] * Wj_row

        lc_diff = 0.5 * jnp.sum(
            jnp.logaddexp(theta_flip, -theta_flip) - jnp.logaddexp(theta, -theta),
            axis=1,
        )  # (C,)
        log_ratio = a[i_idx] * vi + a[j_idx] * vj + lc_diff  # (C,)

        rand_u = jax.random.uniform(k3, (C,), dtype=jnp.float64)
        accept_mh = jnp.log(rand_u) < 2.0 * log_ratio  # (C,)

        # Only exchange antiparallel pairs — parallel pairs skip (preserves S_z)
        antiparallel = vi != vj  # (C,)
        accept = accept_mh & antiparallel

        v = v.at[ci, i_idx].set(jnp.where(accept, -vi, vi))
        v = v.at[ci, j_idx].set(jnp.where(accept, -vj, vj))
        theta = jnp.where(accept[:, None], theta_flip, theta)
        return (v, theta, key), None

    (v, theta, _), _ = jax.lax.scan(one_step, (v, theta, key), None, length=total_steps)
    return v, theta


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def _lsb_jit(
    key: jax.Array,
    M: jax.Array,
    f: jax.Array,
    sigma: float,
    delta: float,
    gamma: float,
    n_samples: int,
    steps: int,
    N_total: int,
) -> jax.Array:
    """
    Langevin Simulated Bifurcation (Kubo & Goto 2025, Sec. II B 1).

    Symplectic Euler integration (lax.scan over `steps` iterations):
        y[k+1] = (1−γ)·y[k] + δ·(M·x[k] + f) + σ·ξ    ξ ~ N(0,1)
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
        y = (1.0 - gamma) * y + delta * force + noise
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

        a_np = np.asarray(rbm.a)
        b_np = np.asarray(rbm.b)
        W_np = np.asarray(rbm.W)

        for i in range(Nv):
            linear[i] = float(a_np[i]) / beta_x
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
    ):
        self.method = method
        self.n_warmup = n_warmup
        self.n_sweeps = n_sweeps
        self.T_initial = T_initial
        self.T_final = T_final
        self.sb_mode = sb_mode
        self.sb_heated = sb_heated
        self.sb_max_steps = sb_max_steps

        self._gibbs_v = None  # persistent chain state (JAX array)
        self._key = None  # JAX PRNG key — initialised lazily
        self._last_sample_config: dict = {}
        self._qubo_h_min = float("inf")
        self._qubo_h_max = 0.0
        self._qubo_J_min = float("inf")
        self._qubo_J_max = 0.0
        self._qubo_tracked = False

    def _next_key(self) -> jax.Array:
        """Advance self._key and return a fresh subkey."""
        if self._key is None:
            # Lazy init
            seed = int(np.random.randint(0, 2**31))
            self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def sample(
        self,
        rbm: RBM,
        n_samples: int,
        config: dict = None,
        return_hidden: bool = False,
        return_jax: bool = False,
    ):
        if config is None:
            config = {}
        self._last_sample_config = dict(config)

        if self.method == "lsb":
            v, h = self._lsb_sample(rbm, n_samples, config)
            if not return_jax:
                v, h = np.asarray(v), np.asarray(h)
            return (v, h) if return_hidden else v

        if self.method == "gibbs":
            v, h = self._gibbs_sample(rbm, n_samples, config)
            if not return_jax:
                v, h = np.asarray(v), np.asarray(h)
            return (v, h) if return_hidden else v

        if self.method == "metropolis":
            v = self._metropolis_hastings(rbm, n_samples, config)
        elif self.method == "simulated_annealing":
            v = self._simulated_annealing(rbm, n_samples, config)
        elif self.method == "exchange":
            v = self._exchange_metropolis(rbm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if not return_jax:
            v = np.asarray(v)

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
        delta = config.get("lsb_delta", 0.1)
        gamma = config.get("lsb_gamma", 0.1)
        sigma_inv2 = config.get("lsb_sigma", 1.0)
        sigma = float(1.0 / np.sqrt(sigma_inv2))

        Nv, Nh = rbm.n_visible, rbm.n_hidden
        N_total = Nv + Nh

        M = jnp.zeros((N_total, N_total), dtype=jnp.float64)
        M = M.at[:Nv, Nv:].set(rbm.W / beta_x)
        M = M.at[Nv:, :Nv].set(rbm.W.T / beta_x)
        f = jnp.concatenate([-rbm.a / beta_x, rbm.b / beta_x])

        key = self._next_key()
        s = _lsb_jit(key, M, f, sigma, delta, gamma, n_samples, steps, N_total)

        v = s[:, :Nv]
        h = s[:, Nv:]

        unique = len(np.unique(np.asarray(v), axis=0))
        print(
            f"  [LSB] steps={steps} delta={delta} gamma={gamma} sigma={sigma:.4f}"
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
            V_ = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(n, Nv)).astype(
                jnp.float64
            )
            for _ in range(n_warmup):
                k2, k = jax.random.split(k2)
                V_ = gibbs_sweep(V_, k)
            return V_

        # Reinit chains if shape changed
        key = self._next_key()
        if self._gibbs_v is None or self._gibbs_v.shape != (n_samples, Nv):
            self._gibbs_v = init_chains(n_samples, key)
            key = self._next_key()

        V = self._gibbs_v
        for _ in range(n_sweeps):
            key = self._next_key()
            V = gibbs_sweep(V, key)

        self._gibbs_v = V

        key = self._next_key()
        H = h_given_v(V, key)

        unique = len(np.unique(np.asarray(V), axis=0))
        print(f"  [Gibbs] k={n_sweeps}  unique={unique}/{n_samples}")

        quadratic, linear = self.rbm_to_ising(rbm)
        if linear:
            hs = np.array(list(linear.values()))
            self._qubo_h_min = min(self._qubo_h_min, float(hs.min()))
            self._qubo_h_max = max(self._qubo_h_max, float(hs.max()))
            self._qubo_tracked = True
        if quadratic:
            Js = np.array(list(quadratic.values()))
            self._qubo_J_min = min(self._qubo_J_min, float(Js.min()))
            self._qubo_J_max = max(self._qubo_J_max, float(Js.max()))
            self._qubo_tracked = True

        return V, H

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
        v = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(C, N)).astype(
            jnp.float64
        )
        theta = b[None, :] + v @ W  # (C, Nh)

        total_steps = N * (n_warmup + n_sweeps)
        v, _ = _mh_sweep_jit(v, theta, W, a, k2, C, N, total_steps)

        unique = len(np.unique(np.asarray(v), axis=0))
        print(f"  [MH]    unique={unique}/{n_samples}")
        return v

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

        v = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(C, N)).astype(
            jnp.float64
        )
        theta = b[None, :] + v @ W
        warmup_steps = N * n_warmup
        v, theta = _sa_sweep_jit(
            v, theta, W, a, k2, C, N, warmup_steps, T_initial, T_initial
        )

        cool_steps = N * n_sweeps
        v, _ = _sa_sweep_jit(v, theta, W, a, k3, C, N, cool_steps, T_initial, T_final)

        unique = len(np.unique(np.asarray(v), axis=0))
        print(f"  [SA]    T: {T_initial:.2f}→{T_final:.2f}  unique={unique}/{n_samples}")
        return v

    # ── Spin-exchange Metropolis ──────────────────────────────────────────

    def _exchange_metropolis(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Batched spin-exchange MH: n_samples chains run in parallel.

        Chains are initialised in the S_z ≈ 0 sector (N//2 up-spins, rest
        down) and proposals are always antiparallel-pair swaps, so total S_z
        is conserved throughout.  This is the right sampler for Heisenberg
        antiferromagnets whose ground state lives in S_z=0 and whose Néel
        configurations are Hamming-distance ≈ N/2 apart — unreachable by
        standard single-spin MH.
        """
        N, Nh = rbm.n_visible, rbm.n_hidden
        C = n_samples
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)

        W, a, b = rbm.W, rbm.a, rbm.b

        key = self._next_key()
        k1, k2 = jax.random.split(key)

        n_up = N // 2
        base = jnp.concatenate(
            [jnp.ones(n_up, dtype=jnp.float64), -jnp.ones(N - n_up, dtype=jnp.float64)]
        )
        perms = jax.vmap(lambda k: jax.random.permutation(k, N))(
            jax.random.split(k1, C)
        )
        v = base[perms]  # (C, N)
        theta = b[None, :] + v @ W  # (C, Nh)

        total_steps = N * (n_warmup + n_sweeps)
        v, _ = _exchange_mh_sweep_jit(v, theta, W, a, k2, C, N, total_steps)

        unique = len(np.unique(np.asarray(v), axis=0))
        print(f"  [Exchange MH] unique={unique}/{n_samples}")
        return v


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
        num_steps: int = 1000,
        num_sweeps: int = 5,
        start_temp: float = -1.0,
        stop_temp: float = -1.0,
        schedule_type: str = "geometric",
        keep_files: bool = False,
        server_ready_timeout_s: float = 120.0,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        default_project = (repo_root.parent / "veloxQFPGA").resolve()
        default_script = repo_root / "scripts" / "fpga" / "fpga_sa_server.jl"

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
        self.server_ready_timeout_s = float(server_ready_timeout_s)

        self._tmpdir = tempfile.TemporaryDirectory(prefix="fpga_sampler_")
        self._tmp_root = Path(self._tmpdir.name)

        if not self.project_path.exists():
            raise FileNotFoundError(
                f"VeloxQFPGA project not found at {self.project_path}."
            )
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"FPGA Julia server script not found at {self.script_path}."
            )
        self.last_sampling_time_s = None

        self._proc: subprocess.Popen | None = None
        self._sock: _socket.socket | None = None
        self._sock_file = None
        self._socket_path: str | None = None
        self._stdout_thread: threading.Thread | None = None
        self._shutdown_lock = threading.Lock()

        weak_self = weakref.ref(self)

        def _atexit_shutdown():
            obj = weak_self()
            if obj is not None:
                obj._shutdown_server()

        self._atexit_callback = _atexit_shutdown
        atexit.register(self._atexit_callback)

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

    def _ensure_server(self, env_overrides: dict):
        if self._proc is not None and self._proc.poll() is None and self._sock is not None:
            return
        self._socket_path = str(self._tmp_root / f"server_{uuid.uuid4().hex}.sock")
        env = os.environ.copy()
        self._apply_env_overrides(env, env_overrides)
        env["FPGA_SOCKET"] = self._socket_path
        env["FPGA_TRANSPORT"] = str(self.transport)
        cmd = [
            self.julia_cmd, f"--project={self.project_path}", str(self.script_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        deadline = time.monotonic() + self.server_ready_timeout_s
        ready_marker = "[server] READY"
        while True:
            if self._proc.poll() is not None:
                tail = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(
                    f"FPGA SA server exited before READY (code {self._proc.returncode}).\n{tail}"
                )
            if time.monotonic() > deadline:
                self._shutdown_server()
                raise RuntimeError(
                    f"FPGA SA server did not become ready within {self.server_ready_timeout_s}s."
                )
            line = self._proc.stdout.readline() if self._proc.stdout else ""
            if not line:
                time.sleep(0.05)
                continue
            _sys.stdout.write(line)
            _sys.stdout.flush()
            if ready_marker in line:
                break

        def _drain():
            try:
                for line in iter(self._proc.stdout.readline, ""):
                    if not line:
                        break
                    _sys.stdout.write(line)
                    _sys.stdout.flush()
            except Exception:
                pass

        self._stdout_thread = threading.Thread(target=_drain, daemon=True)
        self._stdout_thread.start()

        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.connect(self._socket_path)
        sock.settimeout(None)
        self._sock = sock
        self._sock_file = sock.makefile("rwb", buffering=0)

    def _shutdown_server(self):
        with self._shutdown_lock:
            if self._sock is not None:
                try:
                    self._sock.sendall(b"shutdown\n")
                except OSError:
                    pass
                try:
                    if self._sock_file is not None:
                        self._sock_file.close()
                except OSError:
                    pass
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
                self._sock_file = None
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
                self._proc = None
            if self._socket_path:
                try:
                    os.unlink(self._socket_path)
                except FileNotFoundError:
                    pass
                self._socket_path = None

    def close(self):
        self._shutdown_server()
        cb = getattr(self, "_atexit_callback", None)
        if cb is not None:
            try:
                atexit.unregister(cb)
            except Exception:
                pass
            self._atexit_callback = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _send_request(self, parts: list) -> None:
        if self._sock_file is None:
            raise RuntimeError("FPGA SA server socket is not open.")
        line = ("\t".join(parts) + "\n").encode()
        self._sock_file.write(line)
        response = self._sock_file.readline()
        if not response:
            raise RuntimeError("FPGA SA server closed the connection.")
        text = response.decode().rstrip("\n")
        if text == "ok":
            return
        if text.startswith("err\t"):
            raise RuntimeError(f"FPGA SA server error: {text[4:]}")
        raise RuntimeError(f"Unexpected server response: {text!r}")

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

        self._ensure_server(config)

        self._send_request([
            "sample",
            str(model_path), str(out_path),
            str(num_rep), str(num_steps), str(num_sweeps),
            str(start_temp), str(stop_temp),
            schedule_type, str(meta_path),
        ])

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
        if not hasattr(self, "_subsample_rng"):
            self._subsample_rng = np.random.default_rng(
                config.get("fpga_subsample_seed")
            )
        idx = self._subsample_rng.choice(samples.shape[0], size=n_samples, replace=False)
        samples = samples[idx]

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
# VeloxQstandard SimulatedAnnealing sampler — subprocess + Julia bridge
# ---------------------------------------------------------------------------


class VeloxQStandardSASampler(Sampler):
    """
    Sampler that delegates sampling to VeloxQstandard.SimulatedAnnealing via a
    Julia bridge subprocess. Mirrors FPGASampler's I/O model (CSV in, states +
    meta out) but targets the classical SA solver instead of FPGA.

    Unbiased subsampling
    --------------------
    VeloxQstandard.SimulatedAnnealing is a *spectrum optimizer*: it returns the
    ``num_rep`` replica states **sorted by energy, lowest first** (see
    VeloxQtoolbox ``sort_spectrum``). When ``num_rep > n_samples`` we therefore
    must NOT keep the head ``states[:n_samples]`` — that would skim the
    lowest-energy tail of the replicas, over-weight the modes of |Ψ(v)|², and
    bias ⟨E_loc⟩ *below* the true thermal average (violating the variational
    bound). Because each replica is an independent β-Gibbs Metropolis chain, we
    instead draw a uniform random subset of the returned states, which restores
    an unbiased |Ψ(v)|² (β-Gibbs) ensemble. Seed via the ``veloxq_subsample_seed``
    config key for reproducibility. (FPGASampler has the same sort-and-skim
    behaviour and would need the same fix if used for sampling rather than
    optimisation.)
    """

    _ENV_MAP = {
        "veloxq_timeout_s": "VELOXQ_TIMEOUT_S",
        "veloxq_verbose": "VELOXQ_VERBOSE",
        "veloxq_scale_model": "VELOXQ_SCALE_MODEL",
        "veloxq_compress": "VELOXQ_COMPRESS",
        "veloxq_th_per_block": "VELOXQ_TH_PER_BLOCK",
    }

    def __init__(
        self,
        julia_cmd: str = "julia",
        project_path=None,
        script_path=None,
        num_rep: int = 1024,
        num_steps: int = 1000,
        num_sweeps: int = 5,
        start_temp: float = -1.0,
        stop_temp: float = -1.0,
        schedule_type: str = "geometric",
        keep_files: bool = False,
        server_ready_timeout_s: float = 120.0,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        default_project = (repo_root / "scripts" / "fpga" / "julia").resolve()
        default_script = repo_root / "scripts" / "fpga" / "veloxq_sa_server.jl"

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
        self.server_ready_timeout_s = float(server_ready_timeout_s)

        self._tmpdir = tempfile.TemporaryDirectory(prefix="veloxq_sa_sampler_")
        self._tmp_root = Path(self._tmpdir.name)

        if not self.project_path.exists():
            raise FileNotFoundError(
                f"VeloxQstandard project not found at {self.project_path}."
            )
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"VeloxQ SA Julia server script not found at {self.script_path}."
            )
        self.last_sampling_time_s = None

        self._proc: subprocess.Popen | None = None
        self._sock: _socket.socket | None = None
        self._sock_file = None
        self._socket_path: str | None = None
        self._stdout_thread: threading.Thread | None = None
        self._shutdown_lock = threading.Lock()

        weak_self = weakref.ref(self)

        def _atexit_shutdown():
            obj = weak_self()
            if obj is not None:
                obj._shutdown_server()

        self._atexit_callback = _atexit_shutdown
        atexit.register(self._atexit_callback)

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

    def _ensure_server(self, env_overrides: dict):
        if self._proc is not None and self._proc.poll() is None and self._sock is not None:
            return
        self._socket_path = str(self._tmp_root / f"server_{uuid.uuid4().hex}.sock")
        env = os.environ.copy()
        self._apply_env_overrides(env, env_overrides)
        env["VELOXQ_SOCKET"] = self._socket_path
        cmd = [
            self.julia_cmd, f"--project={self.project_path}", str(self.script_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        deadline = time.monotonic() + self.server_ready_timeout_s
        ready_marker = "[server] READY"
        while True:
            if self._proc.poll() is not None:
                tail = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(
                    f"VeloxQ SA server exited before READY (code {self._proc.returncode}).\n{tail}"
                )
            if time.monotonic() > deadline:
                self._shutdown_server()
                raise RuntimeError(
                    f"VeloxQ SA server did not become ready within {self.server_ready_timeout_s}s."
                )
            line = self._proc.stdout.readline() if self._proc.stdout else ""
            if not line:
                time.sleep(0.05)
                continue
            _sys.stdout.write(line)
            _sys.stdout.flush()
            if ready_marker in line:
                break

        def _drain():
            try:
                for line in iter(self._proc.stdout.readline, ""):
                    if not line:
                        break
                    _sys.stdout.write(line)
                    _sys.stdout.flush()
            except Exception:
                pass

        self._stdout_thread = threading.Thread(target=_drain, daemon=True)
        self._stdout_thread.start()

        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.connect(self._socket_path)
        sock.settimeout(None)
        self._sock = sock
        self._sock_file = sock.makefile("rwb", buffering=0)

    def _shutdown_server(self):
        with self._shutdown_lock:
            if self._sock is not None:
                try:
                    self._sock.sendall(b"shutdown\n")
                except OSError:
                    pass
                try:
                    if self._sock_file is not None:
                        self._sock_file.close()
                except OSError:
                    pass
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
                self._sock_file = None
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
                self._proc = None
            if self._socket_path:
                try:
                    os.unlink(self._socket_path)
                except FileNotFoundError:
                    pass
                self._socket_path = None

    def close(self):
        self._shutdown_server()
        cb = getattr(self, "_atexit_callback", None)
        if cb is not None:
            try:
                atexit.unregister(cb)
            except Exception:
                pass
            self._atexit_callback = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _send_request(self, parts: list) -> None:
        if self._sock_file is None:
            raise RuntimeError("VeloxQ SA server socket is not open.")
        line = ("\t".join(parts) + "\n").encode()
        self._sock_file.write(line)
        response = self._sock_file.readline()
        if not response:
            raise RuntimeError("VeloxQ SA server closed the connection.")
        text = response.decode().rstrip("\n")
        if text == "ok":
            return
        if text.startswith("err\t"):
            raise RuntimeError(f"VeloxQ SA server error: {text[4:]}")
        raise RuntimeError(f"Unexpected server response: {text!r}")

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

        num_rep = int(config.get("veloxq_num_rep", self.num_rep))
        if n_samples > num_rep:
            raise ValueError(
                f"Requested n_samples={n_samples} exceeds VeloxQ num_rep={num_rep}."
            )

        num_steps = int(config.get("veloxq_num_steps", self.num_steps))
        num_sweeps = int(config.get("veloxq_num_sweeps", self.num_sweeps))
        start_temp = float(config.get("veloxq_start_temp", self.start_temp))
        stop_temp = float(config.get("veloxq_stop_temp", self.stop_temp))
        schedule_type = str(config.get("veloxq_schedule", self.schedule_type))

        self._ensure_server(config)

        self._send_request([
            "sample",
            str(model_path), str(out_path),
            str(num_rep), str(num_steps), str(num_sweeps),
            str(start_temp), str(stop_temp),
            schedule_type, str(meta_path),
        ])

        samples = np.loadtxt(out_path, dtype=np.int8)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.shape[1] != n_vars:
            raise RuntimeError(
                f"VeloxQ SA sampler returned {samples.shape[1]} vars, expected {n_vars}."
            )
        if samples.shape[0] < n_samples:
            raise RuntimeError(
                f"VeloxQ SA sampler returned {samples.shape[0]} samples, expected {n_samples}."
            )
        if samples.shape[0] > n_samples:
            # Energy-sorted; subsample randomly to avoid bias (see class docstring)
            if not hasattr(self, "_subsample_rng"):
                self._subsample_rng = np.random.default_rng(
                    config.get("veloxq_subsample_seed")
                )
            idx = self._subsample_rng.choice(
                samples.shape[0], size=n_samples, replace=False
            )
            samples = samples[idx]

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
        self.last_sampleset = None  # raw dimod SampleSet from last QPU call
        self.last_embedding_info = None  # chain stats for the embedding used

    def sample(
        self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
        self.__dict__.pop("last_sampling_time_s", None)
        beta_x = config.get("beta_x", 1.0)
        J, h = self.rbm_to_ising(rbm, beta_x)
        self.n_visible = rbm.n_visible
        self.n_hidden = rbm.n_hidden
        self._n_cache = self.n_visible
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)

        if self.method == "simulated_annealing":
            return self.simulated_annealing(bqm, n_samples, config, return_hidden)
        elif self.method in ("pegasus", "zephyr"):
            config["solver"] = get_solver_name(self.method)
            return self.dwave(
                bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden
            )
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

        num_sweeps = config.get("num_sweeps", 1000)
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=n_samples,
            beta_range=(0.01, 10.0),
            num_sweeps=num_sweeps,
            beta_schedule_type="geometric",
        )
        sort_idx = np.argsort(list(sampleset.variables))
        samples = sampleset.record.sample[:, sort_idx]
        print(f"  unique samples: {len(set(map(tuple, samples)))}/{len(samples)}")
        v = samples[:, : self.n_visible]
        if return_hidden:
            return v, samples[:, self.n_visible : self.n_visible + self.n_hidden]
        return v

    def _get_composite(self, bqm, solver_name, rbm):
        """Build or return a cached FixedEmbeddingComposite. Returns (composite, is_trivial, cache_key)."""
        from dwave.system import DWaveSampler, FixedEmbeddingComposite
        from model import DWaveTopologyRBM

        cache_key = (getattr(self, "_n_cache", self.n_visible), solver_name)
        if cache_key not in self._embedding_cache:
            dwave_sampler = DWaveSampler(solver=solver_name)
            if rbm is not None and isinstance(rbm, DWaveTopologyRBM):
                assert rbm._qubit_mapping is not None
                if not rbm._live:
                    raise RuntimeError(
                        "This DWaveTopologyRBM was built with live=False (an idealized, "
                        "defect-free fabric graph, not this specific chip's real qubit "
                        "yield). Its qubit mapping is not valid for real QPU sampling — "
                        "reconstruct the RBM with live=True to submit to hardware."
                    )
                identity_embedding = {
                    logical: [phys] for phys, logical in rbm._qubit_mapping.items()
                }
                composite = FixedEmbeddingComposite(dwave_sampler, identity_embedding)
                print(
                    f"  [embedding] Trivial identity embedding cached for {cache_key}."
                )
                self.last_embedding_info = {
                    "type": "trivial",
                    "solver": solver_name,
                    "n_visible": self.n_visible,
                    "n_hidden": self.n_hidden,
                    "max_chain": 1,
                    "mean_chain": 1.0,
                    "qubits": len(identity_embedding),
                }
            else:
                print(f"  [embedding] Running busclique biclique for {cache_key}...")
                import minorminer.busclique as bc

                hw_graph = dwave_sampler.to_networkx_graph()
                cache_bc = bc.busgraph_cache(hw_graph)
                embedding = cache_bc.find_biclique_embedding(self.n_visible, self.n_hidden)
                if not embedding:
                    raise RuntimeError(
                        f"busclique failed to find a biclique embedding for "
                        f"K_{{{self.n_visible},{self.n_hidden}}} on solver '{solver_name}'."
                    )
                composite = FixedEmbeddingComposite(dwave_sampler, embedding)
                chains = [len(v) for v in embedding.values()]
                print(
                    f"  [embedding] Biclique K_{{{self.n_visible},{self.n_hidden}}} embedded: "
                    f"max_chain={max(chains)}, mean_chain={sum(chains)/len(chains):.1f}, "
                    f"qubits={sum(chains)}."
                )
                self.last_embedding_info = {
                    "type": "biclique",
                    "solver": solver_name,
                    "n_visible": self.n_visible,
                    "n_hidden": self.n_hidden,
                    "max_chain": max(chains),
                    "mean_chain": sum(chains) / len(chains),
                    "qubits": sum(chains),
                }
            self._embedding_cache[cache_key] = composite
        else:
            composite = self._embedding_cache[cache_key]

        is_trivial = (
            rbm is not None
            and isinstance(rbm, DWaveTopologyRBM)
            and rbm._qubit_mapping is not None
        )
        return composite, is_trivial, cache_key

    def _get_parallel_composite(self, source_bqm, solver_name, rbms, n_parallel):
        """Build or return a cached ParallelEmbeddingComposite for n_parallel independent runs.

        Cache key is separate from _get_composite to avoid collisions.
        Raises on any inconsistency rather than falling back silently.
        """
        from dwave.system import DWaveSampler, ParallelEmbeddingComposite
        from model import DWaveTopologyRBM

        cache_key = ("parallel", self.n_visible, self.n_hidden, solver_name, n_parallel)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        dwave_sampler = DWaveSampler(solver=solver_name)

        if all(isinstance(r, DWaveTopologyRBM) for r in rbms):
            # Identity embeddings — disjoint per RBM
            embeddings = []
            seen_phys: set = set()
            for k, rbm in enumerate(rbms):
                if rbm._qubit_mapping is None:
                    raise RuntimeError(
                        f"RBM {k} has no qubit mapping. "
                        "DWaveTopologyRBM must be constructed with a live solver."
                    )
                if not rbm._live:
                    raise RuntimeError(
                        f"RBM {k} was built with live=False (idealized fabric graph, "
                        "not this chip's real qubit yield) and is not valid for real "
                        "QPU sampling. Reconstruct it with live=True to submit to hardware."
                    )
                phys_set = set(rbm._qubit_mapping.keys())
                overlap = seen_phys & phys_set
                if overlap:
                    raise RuntimeError(
                        f"RBM {k} shares physical qubits {overlap} with a previous RBM. "
                        "All RBMs in a parallel call must use disjoint qubit subsets."
                    )
                seen_phys |= phys_set
                embeddings.append(
                    {logical: [phys] for phys, logical in rbm._qubit_mapping.items()}
                )
            composite = ParallelEmbeddingComposite(dwave_sampler, embeddings=embeddings)
            print(
                f"  [parallel] Identity embeddings cached for {n_parallel} runs "
                f"({cache_key})."
            )
        else:
            # Find n_parallel disjoint biclique embeddings
            import minorminer.busclique as bc

            hw_graph = dwave_sampler.to_networkx_graph()
            embeddings = []
            remaining = hw_graph.copy()
            for k in range(n_parallel):
                cache_bc = bc.busgraph_cache(remaining)
                emb = cache_bc.find_biclique_embedding(self.n_visible, self.n_hidden)
                if not emb:
                    raise RuntimeError(
                        f"busclique found only {k} disjoint biclique embeddings for "
                        f"K_{{{self.n_visible},{self.n_hidden}}} but {n_parallel} were "
                        f"requested. Reduce n_parallel or use a smaller problem."
                    )
                embeddings.append(emb)
                used = {q for chain in emb.values() for q in chain}
                remaining = remaining.copy()
                remaining.remove_nodes_from(used)
            composite = ParallelEmbeddingComposite(dwave_sampler, embeddings=embeddings)
            chains_all = [len(v) for emb in embeddings for v in emb.values()]
            print(
                f"  [parallel] {n_parallel} disjoint biclique K_{{{self.n_visible},{self.n_hidden}}} "
                f"embeddings found: max_chain={max(chains_all)}, "
                f"qubits_each={sum(len(v) for v in embeddings[0].values())} "
                f"({cache_key})."
            )

        self._embedding_cache[cache_key] = composite
        return composite

    def dwave_parallel(
        self, bqms, n_samples, config, rbms, n_parallel, return_hidden=False
    ):
        """Single QPU call sampling n_parallel independent BQMs via sample_multiple.

        All n_parallel BQMs are merged into one combined problem and solved with
        a single physical QPU submission, so info["timing"]["qpu_access_time"]
        is already the total real cost of this call — it is logged in full, not
        divided, so the shared time.json budget counter reflects actual QPU usage
        regardless of how many logical problems were packed into the anneal.

        Returns list[np.ndarray] of shape (n_samples, n_visible), one per run
        (or list of (v, h) tuples if return_hidden).
        """
        solver_name = config.get("solver")
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)

        cache_key = ("parallel", self.n_visible, self.n_hidden, solver_name, n_parallel)
        composite = self._get_parallel_composite(bqms[0], solver_name, rbms, n_parallel)

        chain_strengths = [chain_strength] * n_parallel
        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
            answer_mode="raw",
            auto_scale=False,
        )

        MAX_DWAVE_RETRIES = 3
        for tries in range(1, MAX_DWAVE_RETRIES + 1):
            try:
                samplesets, info = composite.sample_multiple(
                    bqms, chain_strengths=chain_strengths, **sample_kwargs
                )
                access_time_us = info["timing"]["qpu_access_time"]
                self._log_access_time(access_time_us)
                self.last_sampling_time_s = access_time_us * 1e-6
                self.last_n_parallel = n_parallel
                break
            except Exception as e:
                print(
                    f"  D-Wave parallel attempt {tries}/{MAX_DWAVE_RETRIES} failed: {e}"
                )
                if tries == MAX_DWAVE_RETRIES:
                    raise RuntimeError(
                        f"D-Wave parallel sampling failed after {MAX_DWAVE_RETRIES} "
                        f"attempts."
                    )
                self._embedding_cache.pop(cache_key, None)
                composite = self._get_parallel_composite(
                    bqms[0], solver_name, rbms, n_parallel
                )

        results = []
        for ss in samplesets:
            df = ss.to_pandas_dataframe()
            df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(drop=True)
            v = df.loc[:, list(range(self.n_visible))].to_numpy()
            if return_hidden:
                h_cols = list(range(self.n_visible, self.n_visible + self.n_hidden))
                results.append((v, df.loc[:, h_cols].to_numpy()))
            else:
                results.append(v)
        return results

    def sample_parallel(
        self, rbms, n_samples, config={}, n_parallel=None, return_hidden=False
    ):
        """Sample from n_parallel independent RBMs in a single QPU call.

        All RBMs must share the same n_visible and n_hidden. For DWaveTopologyRBM,
        each must carry a disjoint qubit mapping (built from different subgraphs).

        n_parallel defaults to len(rbms) and must equal len(rbms).
        Only supported for QPU methods ('pegasus', 'zephyr'). Raises for all others.

        Returns list[np.ndarray] of shape (n_samples, n_visible), one per RBM,
        in the same order as the input list.
        """
        if n_parallel is None:
            n_parallel = len(rbms)
        if n_parallel != len(rbms):
            raise ValueError(
                f"n_parallel={n_parallel} must equal len(rbms)={len(rbms)}."
            )
        if n_parallel < 1:
            raise ValueError(f"n_parallel must be >= 1, got {n_parallel}.")

        if self.method not in ("pegasus", "zephyr"):
            raise ValueError(
                f"sample_parallel requires a QPU method ('pegasus' or 'zephyr'), "
                f"got '{self.method}'. Use sample() for classical methods."
            )

        n_vis_0 = rbms[0].n_visible
        n_hid_0 = rbms[0].n_hidden
        for k, rbm in enumerate(rbms[1:], 1):
            n_hid_k = rbm.n_hidden
            if rbm.n_visible != n_vis_0 or n_hid_k != n_hid_0:
                raise ValueError(
                    f"RBM {k} has architecture (n_visible={rbm.n_visible}, "
                    f"n_hidden={n_hid_k}) but RBM 0 has ({n_vis_0}, {n_hid_0}). "
                    "All RBMs must share the same architecture."
                )

        self.__dict__.pop("last_sampling_time_s", None)
        self.__dict__.pop("last_n_parallel", None)

        beta_x = config.get("beta_x", 1.0)
        bqms = []
        for rbm in rbms:
            J, h = self.rbm_to_ising(rbm, beta_x)
            bqms.append(dimod.BinaryQuadraticModel.from_ising(h, J, 0.0))

        self.n_visible = n_vis_0
        self.n_hidden = n_hid_0

        config = dict(config)
        config["solver"] = get_solver_name(self.method)
        return self.dwave_parallel(
            bqms, n_samples, config, rbms, n_parallel, return_hidden=return_hidden
        )

    def dwave(self, bqm, n_samples, config={}, rbm=None, return_hidden=False):
        solver_name = config.get("solver", None)
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)

        composite, is_trivial, cache_key = self._get_composite(bqm, solver_name, rbm)

        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
            answer_mode="raw",
            auto_scale=False,
        )
        if not is_trivial and chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength

        MAX_DWAVE_RETRIES = 3
        for tries in range(1, MAX_DWAVE_RETRIES + 1):
            try:
                sampleset = composite.sample(bqm, **sample_kwargs)
                access_time_us = sampleset.info["timing"]["qpu_access_time"]
                self._log_access_time(access_time_us)
                self.last_sampling_time_s = access_time_us * 1e-6
                self.last_sampleset = sampleset
                break
            except Exception as e:
                print(
                    f"  D-Wave sampling attempt {tries}/{MAX_DWAVE_RETRIES} failed: {e}"
                )
                if tries == MAX_DWAVE_RETRIES:
                    raise RuntimeError(
                        f"D-Wave sampling failed after {MAX_DWAVE_RETRIES} attempts."
                    )
                self._embedding_cache.pop(cache_key, None)
                composite, is_trivial, cache_key = self._get_composite(
                    bqm, solver_name, rbm
                )

        df = sampleset.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(drop=True)
        v = df.loc[:, list(range(self.n_visible))].to_numpy()
        if return_hidden:
            h_cols = list(range(self.n_visible, self.n_visible + self.n_hidden))
            return v, df.loc[:, h_cols].to_numpy()
        return v
