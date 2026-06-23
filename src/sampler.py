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
import math as _math
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
from model import RBM, FullyConnectedRBM
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

        # Hidden activations after simultaneously flipping i and j
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

        # FBM: add visible-visible couplings (chain-free QUBO edges)
        if hasattr(rbm, "J"):
            J_np = np.asarray(rbm.J)
            for i in range(Nv):
                for k in range(i + 1, Nv):
                    if abs(J_np[i, k]) > 1e-6:
                        quadratic[(i, k)] = -float(J_np[i, k]) / beta_x

        return quadratic, linear

    def dbm_to_ising(self, dbm, beta_x: float = 1.0):
        """
        Convert DBM parameters to Ising (J, h) for external solvers.

        Node IDs:
          visible v    : 0 .. n_v - 1
          hidden  h^1  : n_v .. n_v + n_h1 - 1
          hidden  h^2  : n_v + n_h1 .. n_v + n_h1 + n_h2 - 1
          ...

        Couplings: W[l] connects layer l to layer l+1; no skip-layer edges.
        """
        _last = getattr(self, "_last_beta_x_logged", None)
        if _last is None or abs(beta_x - _last) / max(abs(_last), 1e-9) > 0.01:
            print(f"  [dbm_to_ising] beta_x = {beta_x:.4f}")
            self._last_beta_x_logged = beta_x

        Nv = dbm.n_visible
        hidden_sizes = dbm.hidden_sizes

        # Cumulative offsets within the hidden-node block
        # layer_starts[l] = first node ID for hidden layer l (relative to Nv)
        layer_starts = []
        offset = 0
        for nh in hidden_sizes:
            layer_starts.append(offset)
            offset += nh

        linear = {}
        quadratic = {}

        # Visible biases
        a_np = np.asarray(dbm.params.a)
        for i in range(Nv):
            linear[i] = -float(a_np[i]) / beta_x

        # Per-layer: biases for hidden layer l, couplings W[l] from layer l to l+1
        for l, (b_l, W_l) in enumerate(zip(dbm.params.b, dbm.params.W)):
            b_np = np.asarray(b_l)
            W_np = np.asarray(W_l)

            # Source layer: visible (l=0) or hidden layer l-1 (l>0)
            src_start = 0 if l == 0 else Nv + layer_starts[l - 1]

            # Destination: hidden layer l
            dst_start = Nv + layer_starts[l]
            n_dst = int(b_np.shape[0])

            # Hidden biases
            for j in range(n_dst):
                linear[dst_start + j] = -float(b_np[j]) / beta_x

            # Inter-layer couplings
            n_src = int(W_np.shape[0])
            for i in range(n_src):
                for j in range(n_dst):
                    if abs(W_np[i, j]) > 1e-6:
                        quadratic[(src_start + i, dst_start + j)] = (
                            -float(W_np[i, j]) / beta_x
                        )

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
            # Lazy init: derive from numpy so the caller doesn't have to set it
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

        # Build interaction matrix (JAX)
        M = jnp.zeros((N_total, N_total), dtype=jnp.float64)
        M = M.at[:Nv, Nv:].set(rbm.W / beta_x)
        M = M.at[Nv:, :Nv].set(rbm.W.T / beta_x)
        f = jnp.concatenate([rbm.a / beta_x, rbm.b / beta_x])

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

        # Initialise or reinitialise persistent chains when shape changes
        key = self._next_key()
        if self._gibbs_v is None or self._gibbs_v.shape != (n_samples, Nv):
            self._gibbs_v = init_chains(n_samples, key)
            key = self._next_key()

        V = self._gibbs_v
        for _ in range(n_sweeps):
            key = self._next_key()
            V = gibbs_sweep(V, key)

        self._gibbs_v = V

        # Sample hidden once from final V
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

        # Warmup at T_initial
        v = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(C, N)).astype(
            jnp.float64
        )
        theta = b[None, :] + v @ W
        warmup_steps = N * n_warmup
        v, theta = _sa_sweep_jit(
            v, theta, W, a, k2, C, N, warmup_steps, T_initial, T_initial
        )

        # Cooling sweep
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

        # Initialise each chain as a random shuffle of N//2 up-spins, rest down.
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
        default_script = repo_root / "scripts" / "fpga_sa_server.jl"

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
        default_project = (repo_root / "scripts" / "julia").resolve()
        default_script = repo_root / "scripts" / "veloxq_sa_server.jl"

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
            # `samples` is energy-sorted (lowest first) by the SA solver's
            # sort_spectrum. Taking the head would skim the lowest-energy
            # replicas and bias the |Ψ|² estimate (see class docstring). Draw a
            # uniform random subset instead so the kept states are an unbiased
            # β-Gibbs ensemble. Seedable via config["veloxq_subsample_seed"].
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


class LangevinSampler(Sampler):
    """
    Sampler that delegates sampling to the Langevin SB kernel via a Julia
    bridge subprocess. Mirrors VeloxQStandardSASampler's I/O model (CSV in,
    states + meta out), running on CPU.
    """

    _ENV_MAP = {
        "langevin_timeout_s": "LANGEVIN_TIMEOUT_S",
        "langevin_verbose": "LANGEVIN_VERBOSE",
        "langevin_scale_model": "LANGEVIN_SCALE_MODEL",
        "langevin_compress": "LANGEVIN_COMPRESS",
        "langevin_th_per_block": "LANGEVIN_TH_PER_BLOCK",
    }

    def __init__(
        self,
        julia_cmd: str = "julia",
        project_path=None,
        script_path=None,
        num_rep: int = 1024,
        num_steps: int = 1000,
        dt: float = 0.25,
        sigma: float = 1.0,
        detuning: float = 1.0,
        scale: float = 1.0,
        keep_files: bool = False,
        server_ready_timeout_s: float = 180.0,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        default_project = (repo_root / "scripts" / "julia_chaotic").resolve()
        default_script = repo_root / "scripts" / "langevin_server.jl"

        self.julia_cmd = julia_cmd
        self.project_path = (
            Path(project_path).resolve() if project_path else default_project
        )
        self.script_path = (
            Path(script_path).resolve() if script_path else default_script
        )
        self.num_rep = int(num_rep)
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.sigma = float(sigma)
        self.detuning = float(detuning)
        self.scale = float(scale)
        self.keep_files = keep_files
        self.server_ready_timeout_s = float(server_ready_timeout_s)

        self._tmpdir = tempfile.TemporaryDirectory(prefix="langevin_sampler_")
        self._tmp_root = Path(self._tmpdir.name)

        if not self.project_path.exists():
            raise FileNotFoundError(
                f"VeloxQchaotic project not found at {self.project_path}."
            )
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"Langevin Julia server script not found at {self.script_path}."
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

    def _ensure_server(self, env_overrides: dict):
        if self._proc is not None and self._proc.poll() is None and self._sock is not None:
            return
        self._socket_path = str(self._tmp_root / f"server_{uuid.uuid4().hex}.sock")
        env = os.environ.copy()
        self._apply_env_overrides(env, env_overrides)
        env["LANGEVIN_SOCKET"] = self._socket_path
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
                    f"Langevin server exited before READY (code {self._proc.returncode}).\n{tail}"
                )
            if time.monotonic() > deadline:
                self._shutdown_server()
                raise RuntimeError(
                    f"Langevin server did not become ready within {self.server_ready_timeout_s}s."
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
            raise RuntimeError("Langevin server socket is not open.")
        line = ("\t".join(parts) + "\n").encode()
        self._sock_file.write(line)
        response = self._sock_file.readline()
        if not response:
            raise RuntimeError("Langevin server closed the connection.")
        text = response.decode().rstrip("\n")
        if text == "ok":
            return
        if text.startswith("err\t"):
            raise RuntimeError(f"Langevin server error: {text[4:]}")
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

        num_rep = int(config.get("langevin_num_rep", self.num_rep))
        if n_samples > num_rep:
            raise ValueError(
                f"Requested n_samples={n_samples} exceeds Langevin num_rep={num_rep}."
            )

        num_steps = int(config.get("langevin_num_steps", self.num_steps))
        dt = float(config.get("langevin_dt", self.dt))
        sigma = float(config.get("langevin_sigma", self.sigma))
        detuning = float(config.get("langevin_detuning", self.detuning))
        scale = float(config.get("langevin_scale", self.scale))

        self._ensure_server(config)

        self._send_request([
            "sample",
            str(model_path), str(out_path),
            str(num_rep), str(num_steps),
            f"{dt:.8g}", f"{sigma:.8g}",
            f"{detuning:.8g}", f"{scale:.8g}",
            str(meta_path),
        ])

        samples = np.loadtxt(out_path, dtype=np.int8)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.shape[1] != n_vars:
            raise RuntimeError(
                f"Langevin sampler returned {samples.shape[1]} vars, expected {n_vars}."
            )
        if samples.shape[0] < n_samples:
            raise RuntimeError(
                f"Langevin sampler returned {samples.shape[0]} samples, expected {n_samples}."
            )
        if not hasattr(self, "_subsample_rng"):
            self._subsample_rng = np.random.default_rng(
                config.get("langevin_subsample_seed")
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
        self.last_sampleset = None  # set after every QPU call; holds the raw dimod SampleSet

    def sample(
        self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
        from model_dbm import DeepBoltzmannMachine

        self.__dict__.pop("last_sampling_time_s", None)
        beta_x = config.get("beta_x", 1.0)
        if isinstance(rbm, DeepBoltzmannMachine):
            J, h = self.dbm_to_ising(rbm, beta_x)
            self.n_visible = rbm.n_visible
            self.n_hidden = sum(rbm.hidden_sizes)
            self._n_cache = self.n_visible + self.n_hidden
        else:
            J, h = self.rbm_to_ising(rbm, beta_x)
            self.n_visible = rbm.n_visible
            self.n_hidden = rbm.n_hidden
            self._n_cache = self.n_visible
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)

        if self.method == "simulated_annealing":
            return self.simulated_annealing(bqm, n_samples, config, return_hidden)
        elif self.method == "tabu":
            return self.tabu_search(bqm, n_samples, config, return_hidden)
        elif self.method in ("pegasus", "zephyr"):
            config["solver"] = get_solver_name(self.method)
            return self.dwave(
                bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden
            )
        elif self.method in ("pegasus_ra", "zephyr_ra"):
            config["solver"] = get_solver_name(self.method.replace("_ra", ""))
            return self.reverse_annealing(
                bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden
            )
        elif self.method in ("pegasus_fast", "zephyr_fast"):
            config["solver"] = get_solver_name(self.method.replace("_fast", ""))
            return self.fast_anneal(
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

    def _get_composite(self, bqm, solver_name, rbm):
        """Build or return a cached FixedEmbeddingComposite. Returns (composite, is_trivial, cache_key)."""
        from dwave.system import DWaveSampler, FixedEmbeddingComposite
        from model import DWaveTopologyRBM

        cache_key = (getattr(self, "_n_cache", self.n_visible), solver_name)
        if cache_key not in self._embedding_cache:
            dwave_sampler = DWaveSampler(solver=solver_name)
            if rbm is not None and isinstance(rbm, DWaveTopologyRBM):
                assert rbm._qubit_mapping is not None
                identity_embedding = {
                    logical: [phys] for phys, logical in rbm._qubit_mapping.items()
                }
                composite = FixedEmbeddingComposite(dwave_sampler, identity_embedding)
                print(
                    f"  [embedding] Trivial identity embedding cached for {cache_key}."
                )
            else:
                print(f"  [embedding] Running busclique biclique for {cache_key}...")
                import minorminer.busclique as bc
                import dwave_networkx as dnx

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
            # Identity embeddings: each RBM supplies its own disjoint qubit mapping.
            embeddings = []
            seen_phys: set = set()
            for k, rbm in enumerate(rbms):
                if rbm._qubit_mapping is None:
                    raise RuntimeError(
                        f"RBM {k} has no qubit mapping. "
                        "DWaveTopologyRBM must be constructed with a live solver."
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
            # FullyConnectedRBM: find n_parallel disjoint biclique embeddings.
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

    def dwave_parallel(self, bqms, n_samples, config, rbms, n_parallel):
        """Single QPU call sampling n_parallel independent BQMs via sample_multiple.

        QPU access time is divided by n_parallel for per-run budget attribution.
        Returns list[np.ndarray] of shape (n_samples, n_visible), one per run.
        """
        solver_name = config.get("solver")
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)
        auto_scale = bool(config.get("auto_scale", True))

        cache_key = ("parallel", self.n_visible, self.n_hidden, solver_name, n_parallel)
        composite = self._get_parallel_composite(bqms[0], solver_name, rbms, n_parallel)

        chain_strengths = [chain_strength] * n_parallel
        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
            answer_mode="raw",
            auto_scale=auto_scale,
        )

        MAX_DWAVE_RETRIES = 3
        for tries in range(1, MAX_DWAVE_RETRIES + 1):
            try:
                samplesets, info = composite.sample_multiple(
                    bqms, chain_strengths=chain_strengths, **sample_kwargs
                )
                access_time_us = info["timing"]["qpu_access_time"]
                # Divide QPU time equally across parallel runs for budget attribution.
                self._log_access_time(access_time_us / n_parallel)
                self.last_sampling_time_s = access_time_us * 1e-6 / n_parallel
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
            results.append(v)
        return results

    def sample_parallel(self, rbms, n_samples, config={}, n_parallel=None):
        """Sample from n_parallel independent RBMs in a single QPU call.

        All RBMs must share the same n_visible and n_hidden. For DWaveTopologyRBM,
        each must carry a disjoint qubit mapping (built from different subgraphs).

        n_parallel defaults to len(rbms) and must equal len(rbms).
        Only supported for QPU methods ('pegasus', 'zephyr'). Raises for all others.

        Returns list[np.ndarray] of shape (n_samples, n_visible), one per RBM,
        in the same order as the input list.
        """
        from model_dbm import DeepBoltzmannMachine

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

        # Validate uniform architecture.
        n_vis_0 = rbms[0].n_visible
        n_hid_0 = (
            sum(rbms[0].hidden_sizes)
            if isinstance(rbms[0], DeepBoltzmannMachine)
            else rbms[0].n_hidden
        )
        for k, rbm in enumerate(rbms[1:], 1):
            n_hid_k = (
                sum(rbm.hidden_sizes)
                if isinstance(rbm, DeepBoltzmannMachine)
                else rbm.n_hidden
            )
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
            if isinstance(rbm, DeepBoltzmannMachine):
                J, h = self.dbm_to_ising(rbm, beta_x)
            else:
                J, h = self.rbm_to_ising(rbm, beta_x)
            bqms.append(dimod.BinaryQuadraticModel.from_ising(h, J, 0.0))

        self.n_visible = n_vis_0
        self.n_hidden = n_hid_0

        config = dict(config)
        config["solver"] = get_solver_name(self.method)
        return self.dwave_parallel(bqms, n_samples, config, rbms, n_parallel)

    def dwave(self, bqm, n_samples, config={}, rbm=None, return_hidden=False):
        solver_name = config.get("solver", None)
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)

        composite, is_trivial, cache_key = self._get_composite(bqm, solver_name, rbm)

        auto_scale = bool(config.get("auto_scale", True))
        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
            answer_mode="raw",
            auto_scale=auto_scale,
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

    def reverse_annealing(
        self, bqm, n_samples, config={}, rbm=None, return_hidden=False
    ):
        ra_initial_state = config.get("ra_initial_state")
        if ra_initial_state is None:
            print(
                "  [RA] No initial state for iteration 0 — falling back to forward anneal."
            )
            return self.dwave(
                bqm, n_samples, config, rbm=rbm, return_hidden=return_hidden
            )

        solver_name = config.get("solver", None)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)
        s_target = config.get("ra_s_target", 0.45)
        t_rev = float(config.get("ra_anneal_time", 10))
        t_paus = float(config.get("ra_pause_time", 10))
        anneal_schedule = [
            (0.0, 1.0),
            (t_rev, s_target),
            (t_rev + t_paus, s_target),
            (2.0 * t_rev + t_paus, 1.0),
        ]

        composite, is_trivial, cache_key = self._get_composite(bqm, solver_name, rbm)
        sample_kwargs = dict(
            num_reads=num_reads,
            anneal_schedule=anneal_schedule,
            initial_state=ra_initial_state,
            reinitialize_state=True,
            answer_mode="raw",
            auto_scale=True,
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
                    f"  D-Wave RA sampling attempt {tries}/{MAX_DWAVE_RETRIES} failed: {e}"
                )
                if tries == MAX_DWAVE_RETRIES:
                    raise RuntimeError(
                        f"D-Wave RA sampling failed after {MAX_DWAVE_RETRIES} attempts."
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

    def fast_anneal(self, bqm, n_samples, config={}, rbm=None, return_hidden=False):
        """
        D-Wave fast anneal in the coherent regime (default 7 ns).

        fast_anneal=True requires all linear (h) biases to be zero.
        The h biases from the RBM are silently dropped; only J couplings are used.
        """
        solver_name = config.get("solver", None)
        anneal_time_ns = float(config.get("fast_anneal_time_ns", 7.0))
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)

        # Zero out linear biases — fast_anneal requires h=0 on all qubits
        bqm_no_h = bqm.copy()
        for var in list(bqm_no_h.variables):
            bqm_no_h.set_linear(var, 0.0)
        n_dropped = sum(1 for b in bqm.linear.values() if abs(b) > 1e-9)
        if n_dropped:
            print(
                f"  [FastAnneal] dropping {n_dropped} non-zero h biases"
                f" (fast_anneal requires h=0)"
            )

        composite, is_trivial, cache_key = self._get_composite(bqm_no_h, solver_name, rbm)

        auto_scale = bool(config.get("auto_scale", True))
        sample_kwargs = dict(
            num_reads=num_reads,
            fast_anneal=True,
            annealing_time=anneal_time_ns,
            answer_mode="raw",
            auto_scale=auto_scale,
        )
        if not is_trivial and chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength

        MAX_DWAVE_RETRIES = 3
        for tries in range(1, MAX_DWAVE_RETRIES + 1):
            try:
                sampleset = composite.sample(bqm_no_h, **sample_kwargs)
                access_time_us = sampleset.info["timing"]["qpu_access_time"]
                self._log_access_time(access_time_us)
                self.last_sampling_time_s = access_time_us * 1e-6
                self.last_sampleset = sampleset
                break
            except Exception as e:
                print(
                    f"  D-Wave fast anneal attempt {tries}/{MAX_DWAVE_RETRIES} failed: {e}"
                )
                if tries == MAX_DWAVE_RETRIES:
                    raise RuntimeError(
                        f"D-Wave fast anneal failed after {MAX_DWAVE_RETRIES} attempts."
                    )
                self._embedding_cache.pop(cache_key, None)
                composite, is_trivial, cache_key = self._get_composite(
                    bqm_no_h, solver_name, rbm
                )

        df = sampleset.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(drop=True)
        v = df.loc[:, list(range(self.n_visible))].to_numpy()
        if return_hidden:
            h_cols = list(range(self.n_visible, self.n_visible + self.n_hidden))
            return v, df.loc[:, h_cols].to_numpy()
        return v


# ---------------------------------------------------------------------------
# D-Wave Metropolis-Hastings sampler — JIT kernels
# ---------------------------------------------------------------------------


@jax.jit
def _rbm_log_psi2_batch(
    V: jax.Array,
    a: jax.Array,
    b: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """
    Batch log|Ψ(v)|² for a standard RBM (no visible-visible J couplings).

    V : (ns, N)  spin configs ±1
    Returns (ns,) log|Ψ|² = -a·v + Σ_j logaddexp(θ_j, -θ_j)
    """
    Theta = V @ W + b[None, :]  # (ns, M)
    # logcosh convention in this codebase: self.logcosh(x) = logaddexp(x,-x) = log(2·cosh(x)),
    # so log|Ψ|² = -a·v + Σ_j [log(2) + logaddexp(θ_j,-θ_j)] matches 2·log_psi exactly.
    return -(V @ a) + jnp.sum(jnp.log(2.0) + jnp.logaddexp(Theta, -Theta), axis=1)


@jax.jit
def _fbm_log_psi2_batch(
    V: jax.Array,
    a: jax.Array,
    b: jax.Array,
    W: jax.Array,
    J: jax.Array,
) -> jax.Array:
    """
    Batch log|Ψ(v)|² for a Full Boltzmann Machine (includes visible-visible J).

    V : (ns, N), J : (N, N) symmetric, zero diagonal
    Returns (ns,) log|Ψ|² = -a·v + ½vᵀJv + Σ_j logaddexp(θ_j, -θ_j)
    """
    Theta = V @ W + b[None, :]  # (ns, M)
    J_term = 0.5 * jnp.einsum("si,ij,sj->s", V, J, V)  # (ns,)  ½vᵀJv
    return (
        -(V @ a) + J_term + jnp.sum(jnp.log(2.0) + jnp.logaddexp(Theta, -Theta), axis=1)
    )


@jax.jit
def _dwave_mh_accept_jit(
    v_curr: jax.Array,
    v_prop: jax.Array,
    lp_curr: jax.Array,
    lp_prop: jax.Array,
    key: jax.Array,
) -> tuple:
    """
    Batched Metropolis-Hastings accept/reject for D-Wave proposals.

    v_curr  : (ns, N)  current chain states
    v_prop  : (ns, N)  proposed states from D-Wave
    lp_curr : (ns,)    log|Ψ(v_curr)|²
    lp_prop : (ns,)    log|Ψ(v_prop)|²
    key     : JAX PRNG key

    Acceptance: α = min(1, |Ψ(v')|² / |Ψ(v)|²)
    Returns (v_new (ns, N), accept_rate scalar).
    """
    log_alpha = lp_prop - lp_curr  # (ns,)
    u = jax.random.uniform(key, (v_curr.shape[0],), dtype=jnp.float64)
    accept = jnp.log(u) < log_alpha  # (ns,) bool
    v_new = jnp.where(accept[:, None], v_prop, v_curr)
    accept_rate = jnp.mean(accept.astype(jnp.float64))
    return v_new, accept_rate


# ---------------------------------------------------------------------------
# D-Wave Metropolis-Hastings sampler
# ---------------------------------------------------------------------------


class DWaveMHSampler(Sampler):
    """
    D-Wave quantum annealer used as a Metropolis-Hastings proposal generator.

    At each call to sample(), D-Wave produces n_samples candidate configurations.
    These are used as MH proposals against persistent chains targeting |Ψ(v)|².
    Acceptance: α = min(1, |Ψ(v')|² / |Ψ(v)|²).

    This differs from the Gardas 2018 approach (direct use of D-Wave samples,
    violating detailed balance) in that the MH filter restores stationarity:
    the persistent chains converge to the correct VMC distribution |Ψ|².

    Parameters
    ----------
    method : 'pegasus_mh' or 'zephyr_mh'
    n_warmup : D-Wave query rounds before collecting (default 0 — no extra QPU cost)
    n_sweeps : D-Wave query rounds per sample() call collected into chain (default 1)
    """

    def __init__(self, method: str, n_warmup: int = 0, n_sweeps: int = 1):
        self.method = method
        dwave_arch = method.replace("_mh", "")  # 'pegasus' or 'zephyr'
        self._dwave = DimodSampler(method=dwave_arch)
        self.n_warmup = n_warmup
        self.n_sweeps = n_sweeps
        self._chains: jax.Array | None = None  # persistent MH chain state
        self._key = jax.random.PRNGKey(0)
        self.last_sampling_time_s: float | None = None
        self.last_acceptance_rate: float | None = None

    def _next_key(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _log_psi2_batch(self, V: jax.Array, rbm) -> jax.Array:
        """Dispatch to RBM or FBM batch log|Ψ|² kernel."""
        from model import FullBoltzmannMachine

        if isinstance(rbm, FullBoltzmannMachine):
            return _fbm_log_psi2_batch(V, rbm.a, rbm.b, rbm.W, rbm.J)
        return _rbm_log_psi2_batch(V, rbm.a, rbm.b, rbm.W)

    def _mh_round(
        self, rbm, n_samples: int, config: dict
    ) -> tuple[jax.Array, float, float]:
        """
        One D-Wave query + MH accept/reject step.

        Returns (v_new, accept_rate, qpu_time_s).
        """
        V_proposals_np = self._dwave.sample(rbm, n_samples, config)
        qpu_time = getattr(self._dwave, "last_sampling_time_s", 0.0) or 0.0

        V_proposals = jnp.asarray(V_proposals_np, dtype=jnp.float64)

        # Randomly shuffle proposals among chains (independence sampler).
        perm_key, accept_key = jax.random.split(self._next_key())
        perm = jax.random.permutation(perm_key, n_samples)
        V_proposals = V_proposals[perm]

        lp_curr = self._log_psi2_batch(self._chains, rbm)
        lp_prop = self._log_psi2_batch(V_proposals, rbm)
        v_new, accept_rate = _dwave_mh_accept_jit(
            self._chains, V_proposals, lp_curr, lp_prop, accept_key
        )
        return v_new, float(accept_rate), qpu_time

    def sample(
        self, rbm, n_samples: int, config: dict = {}, return_hidden: bool = False
    ):
        N = rbm.n_visible

        # Bootstrap chains on first call or if shape changed.
        if self._chains is None or self._chains.shape != (n_samples, N):
            V_init = self._dwave.sample(rbm, n_samples, config)
            self._chains = jnp.asarray(V_init, dtype=jnp.float64)

        total_qpu_time = 0.0
        sweep_accept_rates: list[float] = []

        # Warmup rounds (D-Wave cost but not counted in acceptance statistics).
        for _ in range(self.n_warmup):
            self._chains, _, t = self._mh_round(rbm, n_samples, config)
            total_qpu_time += t

        # Collection rounds.
        for _ in range(self.n_sweeps):
            self._chains, ar, t = self._mh_round(rbm, n_samples, config)
            total_qpu_time += t
            sweep_accept_rates.append(ar)

        self.last_sampling_time_s = total_qpu_time
        self.last_acceptance_rate = (
            float(np.mean(sweep_accept_rates)) if sweep_accept_rates else None
        )

        V_out = np.asarray(self._chains)
        unique = len(set(map(tuple, V_out.tolist())))
        print(
            f"  [DWave-MH] accept={self.last_acceptance_rate:.3f}"
            f"  unique={unique}/{n_samples}"
        )

        if return_hidden:
            V_jax = jnp.asarray(V_out, dtype=jnp.float64)
            activation = rbm.b[None, :] + V_jax @ rbm.W
            prob_plus = 1.0 / (1.0 + jnp.exp(-2.0 * activation))
            u = jax.random.uniform(self._next_key(), prob_plus.shape, dtype=jnp.float64)
            H_out = np.asarray(jnp.where(u < prob_plus, 1.0, -1.0))
            return V_out, H_out

        return V_out


# ---------------------------------------------------------------------------
# Generic Metropolis-Hastings sampler (works with any log_psi callable)
# ---------------------------------------------------------------------------


def _make_generic_mh_jit(log_psi_fn, C: int, N: int, n_steps: int):
    """
    JIT-compile a batched MH sweep for a fixed (log_psi_fn, C, N, n_steps).

    log_psi_fn(params, v) must be a pure JAX function where params is a
    traced pytree and v is a (N,) array.

    Returns a compiled function sweep(v, log_p_v, params, key).
    The sweep runs n_steps single-spin-flip proposals on C parallel chains.
    """

    @jax.jit
    def sweep(
        v: jax.Array,
        log_p_v: jax.Array,
        params,
        key: jax.Array,
    ):
        def one_step(carry, _):
            v, log_p_v, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            flip_idx = jax.random.randint(k1, (C,), 0, N)
            # Build flipped configs: flip one site per chain
            mask = jax.nn.one_hot(flip_idx, N, dtype=jnp.float64)  # (C, N)
            v_flip = v * (1.0 - 2.0 * mask)  # (C, N)
            # Evaluate log|Ψ| for all flipped configs in one batched call
            log_p_flip = jax.vmap(lambda vi: log_psi_fn(params, vi))(v_flip)  # (C,)
            log_ratio = 2.0 * (log_p_flip - log_p_v)
            u = jax.random.uniform(k2, (C,), dtype=jnp.float64)
            accept = jnp.log(u) < log_ratio
            v = jnp.where(accept[:, None], v_flip, v)
            log_p_v = jnp.where(accept, log_p_flip, log_p_v)
            return (v, log_p_v, key), None

        (v, log_p_v, _), _ = jax.lax.scan(
            one_step, (v, log_p_v, key), None, length=n_steps
        )
        return v, log_p_v

    return sweep


class GenericClassicalSampler:
    """
    Metropolis-Hastings sampler for arbitrary wave functions.

    Uses log_psi_fn(params, v) for acceptance ratios — no analytical
    psi_ratio required. Compatible with ViTWaveFunction and any model
    implementing the (params, v) → scalar interface.

    Because each MH step requires a full forward pass (vs. the RBM's O(M)
    analytical ratio), use fewer warmup steps than ClassicalSampler:
        n_warmup=20, n_sweeps=1  is a good starting point for ViT.

    Parameters
    ----------
    n_warmup : int   MH steps before collecting the sample (default: 20)
    n_sweeps : int   MH steps between successive collected samples (default: 1)
    """

    def __init__(self, n_warmup: int = 20, n_sweeps: int = 1):
        self.n_warmup = n_warmup
        self.n_sweeps = n_sweeps
        self._key = None
        self._sweep_jit = {}  # cache compiled sweeps keyed by (C, N, n_steps)

    def _next_key(self) -> jax.Array:
        if self._key is None:
            seed = int(np.random.randint(0, 2**31))
            self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def sample(self, model, n_samples: int, config: dict | None = None, **_):
        """
        Draw n_samples spin configurations targeting |Ψ(v)|².

        model    : ViTWaveFunction (must have log_psi_single and params)
        n_samples: number of chains = number of samples returned
        config   : optional dict with 'n_warmup', 'n_sweeps' overrides

        Returns (n_samples, N) array of ±1 spins.
        """
        if config is None:
            config = {}
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)

        C = n_samples
        N = model.n_visible
        n_steps_total = N * (n_warmup + n_sweeps)

        cache_key = (C, N, n_steps_total)
        if cache_key not in self._sweep_jit:
            self._sweep_jit[cache_key] = _make_generic_mh_jit(
                model.log_psi_single, C, N, n_steps_total
            )
        sweep = self._sweep_jit[cache_key]

        params = model.params
        log_psi_fn = model.log_psi_single

        # Initialise chains uniformly at random
        key = self._next_key()
        k1, k2 = jax.random.split(key)
        v = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(C, N)).astype(
            jnp.float64
        )
        log_p_v = jax.vmap(lambda vi: log_psi_fn(params, vi))(v)  # (C,)

        # Run the compiled sweep
        v, _ = sweep(v, log_p_v, params, k2)

        v_np = np.asarray(v)
        unique = len(set(map(tuple, v_np.tolist())))
        print(f"  [GenericMH] n_steps={n_steps_total}  unique={unique}/{n_samples}")
        return v_np


# ---------------------------------------------------------------------------
# D-Wave MH proposal sampler for ViT wave functions
# ---------------------------------------------------------------------------


class DWaveProposalSampler:
    """
    Metropolis-Hastings sampler for ViT wave functions using D-Wave as proposal.

    An auxiliary FullyConnectedRBM is fitted online to the current ViT
    distribution via contrastive divergence:
        positive phase : accepted ViT samples from the previous VMC iteration
        negative phase : D-Wave QPU samples drawn from the auxiliary RBM

    One QPU call per VMC iteration — same cost as the standard RBM+D-Wave mode.
    Acceptance corrects for the mismatch between the RBM proposal and the true
    ViT target:

        α(v → v') = min(1, |Ψ_ViT(v')|² · p_RBM(v)
                          / (|Ψ_ViT(v)|²  · p_RBM(v')))

    where p_RBM(v) ∝ |Ψ_RBM(v)|² (normalization cancels).

    Parameters
    ----------
    n_visible    : number of visible spins
    n_hidden     : hidden units in the auxiliary RBM
    key          : JAX PRNG key (for RBM init and MH acceptance)
    dwave_method : "pegasus" or "zephyr"
    rbm_lr       : learning rate for the online CD update (default 0.01)
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        key: jax.Array,
        dwave_method: str = "pegasus",
        rbm_lr: float = 0.01,
    ):
        key, rbm_key = jax.random.split(key)
        self.rbm = FullyConnectedRBM(n_visible, n_hidden, rbm_key)
        self._dwave = DimodSampler(method=dwave_method)
        self.rbm_lr = rbm_lr
        self._key = key
        self._v_current: np.ndarray | None = None  # persistent chain (ns, N) ±1

    def _next_key(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _rbm_log_psi_batch(self, V: jax.Array) -> jax.Array:
        """log|Ψ_RBM(v)| for a batch V : (ns, N) → (ns,)."""
        theta = self.rbm.b[None, :] + V @ self.rbm.W  # (ns, nh)
        return -V @ self.rbm.a / 2 + 0.5 * jnp.sum(jnp.logaddexp(theta, -theta), axis=1)

    def _update_rbm(self, V_pos: np.ndarray, V_neg: np.ndarray) -> None:
        """
        One CD gradient step.

        V_pos : accepted ViT samples  (positive phase / data)
        V_neg : D-Wave RBM samples    (negative phase / model)

        For an RBM with log p(v) ∝ 2 log|Ψ_RBM(v)| = -a·v + Σⱼ log 2cosh(θⱼ):

            ∂ log L / ∂a   =  <v>_neg  − <v>_pos
            ∂ log L / ∂b   =  <tanh θ>_pos − <tanh θ>_neg
            ∂ log L / ∂Wᵢⱼ =  <vᵢ tanh θⱼ>_pos − <vᵢ tanh θⱼ>_neg
        """
        Vp = jnp.asarray(V_pos, dtype=jnp.float64)
        Vn = jnp.asarray(V_neg, dtype=jnp.float64)

        tanh_pos = jnp.tanh(self.rbm.b[None, :] + Vp @ self.rbm.W)  # (ns, nh)
        tanh_neg = jnp.tanh(self.rbm.b[None, :] + Vn @ self.rbm.W)  # (ns, nh)

        lr = self.rbm_lr
        self.rbm.a = self.rbm.a + lr * (Vn.mean(0) - Vp.mean(0))
        self.rbm.b = self.rbm.b + lr * (tanh_pos.mean(0) - tanh_neg.mean(0))
        self.rbm.W = self.rbm.W + lr * (
            Vp.T @ tanh_pos / Vp.shape[0] - Vn.T @ tanh_neg / Vn.shape[0]
        )

    def sample(
        self, model, n_samples: int, config: dict | None = None, **_
    ) -> np.ndarray:
        """
        Draw n_samples spin configurations targeting |Ψ_ViT(v)|².

        model    : ViTWaveFunction (must expose log_psi_batch)
        n_samples: number of parallel chains = number of returned samples
        config   : forwarded to DimodSampler (beta_x, annealing_time, …)

        Returns (n_samples, N) numpy array of ±1 spins.
        """
        if config is None:
            config = {}

        # ── 1. One QPU call: draw proposals from the auxiliary RBM ───────
        V_prop = self._dwave.sample(self.rbm, n_samples, config)  # (ns, N) numpy

        # ── 2. First iteration: no persistent state, accept all proposals ─
        if self._v_current is None:
            self._v_current = V_prop
            return V_prop

        # ── 3. MH acceptance (RBM density evaluated BEFORE parameter update)
        V_curr_j = jnp.asarray(self._v_current, dtype=jnp.float64)
        V_prop_j = jnp.asarray(V_prop, dtype=jnp.float64)

        log_psi_vit_curr = model.log_psi_batch(V_curr_j).real  # (ns,)
        log_psi_vit_prop = model.log_psi_batch(V_prop_j).real  # (ns,)
        log_psi_rbm_curr = self._rbm_log_psi_batch(V_curr_j)  # (ns,)
        log_psi_rbm_prop = self._rbm_log_psi_batch(V_prop_j)  # (ns,)

        log_alpha = 2.0 * (log_psi_vit_prop - log_psi_vit_curr) + 2.0 * (
            log_psi_rbm_curr - log_psi_rbm_prop
        )
        u = jax.random.uniform(self._next_key(), (n_samples,), dtype=jnp.float64)
        accept = jnp.log(u) < log_alpha  # (ns,) bool

        V_accepted = jnp.where(accept[:, None], V_prop_j, V_curr_j)
        self._v_current = np.asarray(V_accepted)

        n_acc = int(accept.sum())
        unique = len(set(map(tuple, self._v_current.tolist())))
        print(
            f"  [DWaveMH] accept={n_acc / n_samples:.3f}  unique={unique}/{n_samples}"
        )

        # ── 4. Update auxiliary RBM: accepted ViT samples vs D-Wave proposals
        self._update_rbm(self._v_current, V_prop)

        return self._v_current
