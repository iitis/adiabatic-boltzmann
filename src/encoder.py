"""
SR trainer — JAX backend

Key changes vs the NumPy version
---------------------------------
* All array ops use jax.numpy (jnp); JAX dispatches to GPU automatically.
* SRLinearSystem.matvec is wrapped in jax.jit so the XLA kernel is compiled
  once per unique (N, M, ns) shape and reused for every CG iteration.
* RBM parameters flow functionally: rbm.set_weights() returns a new RBMParams
  and updates rbm.params; no mutation of individual arrays.
* The CG outer loop stays as a Python for-loop — control flow is cheap and
  keeping it out of jax.lax.while_loop avoids XLA recompilation on tolerance
  changes.  Each matvec call runs fully on the accelerator.
* scipy.optimize.minimize_scalar (CEM β-fit) runs on CPU scalars; the JAX
  arrays passed through it are just read, not traced.
"""

import math
import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from helpers import save_rbm_checkpoint, save_dwave_samples
from sampler import ClassicalSampler
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# JIT-compiled SR matvec kernel
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def _sr_matvec_jit(
    V: jax.Array,
    H: jax.Array,
    mu_a: jax.Array,
    mu_b: jax.Array,
    mu_W: jax.Array,
    diag_shift: float,
    N: int,
    M: int,
    ns: int,
    x: jax.Array,
) -> jax.Array:
    """
    Compute  S·x  without forming the (n_params × n_params) S matrix.

    S·x = (1/ns) Ō^T Ō x  +  diag_shift · x

    N, M, ns are static (shapes, not values) so JAX compiles once per unique
    problem size and reuses the kernel for every CG iteration and training step.
    """
    xa = x[:N]
    xb = x[N : N + M]
    xW = x[N + M :].reshape(M, N)

    # Step 1+2: z_s = (O_s - ⟨O⟩)·x  for every sample s
    z = -0.5 * (V @ xa)                                    # a-block
    z = z + 0.5 * (H @ xb)                                 # b-block
    z = z + 0.5 * jnp.einsum("sm,mn,sn->s", H, xW, V)     # W-block
    z = z - (mu_a @ xa + mu_b @ xb + jnp.sum(mu_W * xW))  # centre

    # Step 3: back-project
    out_a = -0.5 * (z @ V) / ns + diag_shift * xa
    out_b =  0.5 * (z @ H) / ns + diag_shift * xb
    out_W = (0.5 * (H.T @ (z[:, None] * V)) / ns + diag_shift * xW)

    return jnp.concatenate([out_a.ravel(), out_b.ravel(), out_W.ravel()])


# ---------------------------------------------------------------------------
# Matrix-free SR linear system
# ---------------------------------------------------------------------------


class SRLinearSystem:
    """
    Matrix-free representation of the SR system  S·x = F.

    Memory: O(n_samples * (N + M))  instead of O(n_params²).
    Matvec: O(n_samples * n_params) per CG iteration  (JIT-compiled, runs on GPU).

    Sign convention (Gardas ansatz):
        ∂log Ψ / ∂a_i  = -v_i / 2
        ∂log Ψ / ∂b_j  =  tanh(θ_j) / 2
        ∂log Ψ / ∂W_ij =  v_i · tanh(θ_j) / 2
    """

    def __init__(
        self,
        V: jax.Array,
        H: jax.Array,
        E: jax.Array,
        diag_shift: float,
    ):
        self.V = jnp.asarray(V, dtype=jnp.float64)
        self.H = jnp.asarray(H, dtype=jnp.float64)
        self.E = jnp.asarray(E, dtype=jnp.float64)
        self.ns = int(self.V.shape[0])
        self.N = int(self.V.shape[1])   # n_visible
        self.M = int(self.H.shape[1])   # n_hidden
        self.diag_shift = float(diag_shift)

        # Mean gradients ⟨O_k⟩
        self.mu_a = -0.5 * jnp.mean(self.V, axis=0)             # (N,)
        self.mu_b =  0.5 * jnp.mean(self.H, axis=0)             # (M,)
        self.mu_W =  0.5 * (self.H.T @ self.V) / self.ns        # (M, N)

        # Force vector F_k = ⟨O_k · E_loc⟩ − ⟨O_k⟩⟨E_loc⟩
        centered_E = self.E - jnp.mean(self.E)
        self.F_a = -0.5 * (centered_E @ self.V) / self.ns       # (N,)
        self.F_b =  0.5 * (centered_E @ self.H) / self.ns       # (M,)
        self.F_W =  0.5 * (self.H.T @ (centered_E[:, None] * self.V)) / self.ns  # (M, N)

    def pack(self, a: jax.Array, b: jax.Array, W: jax.Array) -> jax.Array:
        """Flatten (a, b, W) → 1-D.  W expected shape (M, N) here."""
        return jnp.concatenate([a.ravel(), b.ravel(), W.ravel()])

    def unpack(self, x: jax.Array):
        """Split 1-D vector → (a, b, W) where W has shape (M, N)."""
        a = x[: self.N]
        b = x[self.N : self.N + self.M]
        W = x[self.N + self.M :].reshape(self.M, self.N)
        return a, b, W

    @property
    def force(self) -> jax.Array:
        return self.pack(self.F_a, self.F_b, self.F_W)

    def matvec(self, x: jax.Array) -> jax.Array:
        """S·x  — dispatches to the JIT-compiled XLA kernel."""
        return _sr_matvec_jit(
            self.V, self.H, self.mu_a, self.mu_b, self.mu_W,
            self.diag_shift, self.N, self.M, self.ns, x,
        )


# ---------------------------------------------------------------------------
# Conjugate gradient
# ---------------------------------------------------------------------------


def conjugate_gradient(
    matvec,
    b: jax.Array,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> tuple:
    """
    Solve  A·x = b  for symmetric positive-definite A, given only matvec.

    The loop runs on the Python side; each matvec call executes on the
    accelerator (the result is a JAX array kept on device until float()
    pulls the scalar for the convergence check — a tiny ~8-byte transfer).

    Returns (x, info) where info = {'iterations': int, 'residual_norm': float}.
    """
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    p = r
    rs_old = float(r @ r)
    info = {"iterations": 0, "residual_norm": math.sqrt(rs_old)}

    if rs_old <= tol * tol:
        return x, info

    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if abs(denom) < 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        info = {"iterations": it, "residual_norm": math.sqrt(rs_new)}
        if rs_new <= tol * tol:
            return x, info
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, info


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


KL_EXACT_MAX_N = 16


def estimate_beta_eff_cem(V: jax.Array, H: jax.Array, rbm) -> float:
    """
    CEM estimate of β_eff from joint LSB samples (V, H).

    scipy.optimize.minimize_scalar calls F(beta) with Python floats.
    The JAX computation inside F runs on device; float() pulls the scalar.
    """
    Theta = V @ rbm.W + rbm.b[None, :]  # (ns, M)

    def F(beta):
        return float(jnp.sum((H - jnp.tanh(beta * Theta)) ** 2))

    result = minimize_scalar(F, bounds=(0.01, 50.0), method="bounded")
    return float(result.x)


class Trainer:
    """
    Variational Monte Carlo trainer using Stochastic Reconfiguration.

    SR system S·x = F solved matrix-free via conjugate gradient.
    Local energies and gradients computed in a single vectorised JIT-compiled
    pass — no Python loops over samples inside the training iteration.
    """

    def __init__(self, rbm, ising_model, sampler, config: dict = None, args=None):
        """
        Config keys
        -----------
        learning_rate  : float  (default 0.1)
        n_iterations   : int    (default 50)
        n_samples      : int    (default 1000)
        regularization : float  (default 1e-3)
        cg_tol         : float  (default 1e-8)
        cg_maxiter     : int    (default 200)
        beta_x_init    : float  (default 1.0)
        beta_adapt     : float  (default 0.05)
        beta_min       : float  (default 0.05)
        beta_max       : float  (default 20.0)
        param_clip     : float  (default 3.0)   None = off
        seed           : int    (default 0)      PRNG seed for beta adaptation
        """
        self.rbm = rbm
        self.ising = ising_model
        self.sampler = sampler
        self.args = args
        print(self.rbm)
        print(self.ising)
        print(self.args)
        if config is None:
            config = {}
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.1)
        self.n_iterations = config.get("n_iterations", 50)
        self.n_samples = config.get("n_samples", 1000)
        self.regularization = config.get("regularization", 1e-3)
        self.cg_tol = config.get("cg_tol", 1e-8)
        self.cg_maxiter = config.get("cg_maxiter", 200)

        _method = (
            getattr(sampler, "method", "")
            if isinstance(sampler, ClassicalSampler)
            else ""
        )
        self._beta_fixed = _method in ("metropolis", "gibbs")
        self.beta_x = 1.0 if self._beta_fixed else config.get("beta_x_init", 1.0)
        self.beta_adapt = config.get("beta_adapt", 0.05)
        self.beta_min = config.get("beta_min", 0.05)
        self.beta_max = config.get("beta_max", 20.0)

        self.use_cem = config.get("use_cem", False)
        self.cem_interval = config.get("cem_interval", 1)
        self.cem_ema_alpha = config.get("cem_ema_alpha", 0.3)

        if self.use_cem:
            print(
                f"  [CEM] β scheduling ENABLED — estimating β_eff every "
                f"{self.cem_interval} iteration(s) from joint LSB samples"
                f", EMA α={self.cem_ema_alpha}"
            )
        else:
            print("  [CEM] β scheduling disabled — using heuristic beta_x adaptation")

        self.stop_at_convergence = config.get("stop_at_convergence", False)
        self.conv_var_threshold = config.get("conv_var_threshold", 1e-4)
        self.conv_window = config.get("conv_window", 10)
        self.param_clip = config.get("param_clip", 3.0)
        self.save_checkpoints = config.get("save_checkpoints", False)
        self.checkpoint_interval = config.get("checkpoint_interval", 10)
        print("Checkpoint interval:", self.checkpoint_interval)

        # JAX PRNG key for beta-adaptation coin flip
        _seed = config.get("seed", 0)
        self._key = jax.random.PRNGKey(_seed)

        self.history = {
            "energy": [],
            "error": [],
            "energy_error": [],
            "learning_rate": [],
            "grad_norm": [],
            "weight_norm": [],
            "s_condition_number": [],
            "beta_x": [],
            "beta_eff_cem": [],
            "cg_iterations": [],
            "cg_residual": [],
            "sampling_time_s": [],
            "ess": [],
            "kl_exact": [],
            "n_unique_ratio": [],
        }

        self._kl_all_v = None
        self._kl_config_idx = None

    def _build_kl_cache(self):
        """Pre-compute all 2^N configs and index map for exact KL. Called once."""
        N = self.rbm.n_visible
        indices = np.arange(2**N, dtype=np.int32)
        all_v = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(
            np.float64
        ) * 2 - 1
        config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
        self._kl_all_v = jnp.asarray(all_v)
        self._kl_config_idx = config_idx

    def _compute_sample_metrics(self, V: jax.Array, Theta: jax.Array):
        """
        Compute ESS, unique-sample ratio, and (optionally) exact KL.

        V     : (ns, N)  visible spin configs
        Theta : (ns, M)  pre-activations b + V @ W

        Returns (ess_norm, kl, n_unique_ratio).
        """
        ns = V.shape[0]

        # Unique samples (needs a Python set, so we pull to CPU)
        v_np = np.asarray(V)
        n_unique_ratio = float(len(np.unique(v_np, axis=0))) / ns

        # ESS: log|Ψ|² = -a·v + Σ_j logcosh(θ_j)
        log_psi2 = -(V @ self.rbm.a) + jnp.sum(jnp.logaddexp(Theta, -Theta), axis=1)
        lw = log_psi2 - jnp.max(log_psi2)
        w = jnp.exp(lw)
        w = w / jnp.sum(w)
        ess_norm = float(1.0 / jnp.sum(w**2)) / ns

        # Exact KL (only for small N)
        if self.rbm.n_visible > KL_EXACT_MAX_N:
            return ess_norm, None, n_unique_ratio

        if self._kl_all_v is None:
            self._build_kl_cache()

        all_v = self._kl_all_v
        Theta_all = all_v @ self.rbm.W + self.rbm.b[None, :]
        log_psi2_all = -(all_v @ self.rbm.a) + jnp.sum(
            jnp.logaddexp(Theta_all, -Theta_all), axis=1
        )
        lw_all = log_psi2_all - jnp.max(log_psi2_all)
        p_true = jnp.exp(lw_all)
        p_true = p_true / jnp.sum(p_true)

        counts = np.zeros(len(all_v))
        for row in v_np.astype(int).tolist():
            idx = self._kl_config_idx.get(tuple(row))
            if idx is not None:
                counts[idx] += 1
        q_emp = counts / ns

        mask = q_emp > 0
        p_true_np = np.asarray(p_true)
        kl = float(
            np.sum(q_emp[mask] * (np.log(q_emp[mask]) - np.log(p_true_np[mask])))
        )
        return ess_norm, kl, n_unique_ratio

    def train(self) -> dict:
        prev_energy = None
        consecutive_converged = 0

        for iteration in range(self.n_iterations):
            # ── 1. Sample ──────────────────────────────────────────────────
            _need_hidden = self.use_cem and not self._beta_fixed
            try:
                _t0 = time.perf_counter()
                _result = self.sampler.sample(
                    self.rbm,
                    self.n_samples,
                    config={**self.config, "beta_x": self.beta_x},
                    return_hidden=_need_hidden,
                )
                elapsed = time.perf_counter() - _t0
                fpga_time = getattr(self.sampler, "last_sampling_time_s", None)
                sample_time_s = float(fpga_time) if fpga_time is not None else elapsed
                self.history["sampling_time_s"].append(sample_time_s)
            except Exception as e:
                print(f"  [Trainer] Sampling failed at iteration {iteration}: {e}")
                print("  [Trainer] Aborting this experiment.")
                raise

            if _need_hidden and isinstance(_result, tuple):
                _V_raw, _H_raw = _result
            else:
                _V_raw, _H_raw = _result, None

            V = jnp.asarray(_V_raw, dtype=jnp.float64)   # (ns, N)
            ns = int(V.shape[0])

            # ── 2. Batch local energies (JIT-compiled, runs on GPU) ────────
            local_energies = self.ising.local_energy_batch(V, self.rbm)  # (ns,)

            # ── 3. Batch gradients ─────────────────────────────────────────
            Theta = V @ self.rbm.W + self.rbm.b[None, :]  # (ns, M)
            TanH = jnp.tanh(Theta)                         # (ns, M)

            # ── Sample quality metrics ─────────────────────────────────────
            ess_norm, kl, n_unique_ratio = self._compute_sample_metrics(V, Theta)

            # ── D-Wave sample logging ──────────────────────────────────────
            if self.args and getattr(self.args, "sampling_method", "") in (
                "pegasus",
                "zephyr",
            ):
                save_dwave_samples(np.asarray(V), self.args, iteration)

            # ── 3b. CEM β estimate (before weight update) ─────────────────
            _cem_beta_raw = None
            if (
                self.use_cem
                and not self._beta_fixed
                and iteration % self.cem_interval == 0
                and _H_raw is not None
            ):
                H_cem = jnp.asarray(_H_raw, dtype=jnp.float64)
                _cem_beta_raw = estimate_beta_eff_cem(V, H_cem, self.rbm)

            # ── 4. Build SR system and solve with CG ──────────────────────
            sr = SRLinearSystem(V, TanH, local_energies, self.regularization)
            x, cg_info = conjugate_gradient(
                sr.matvec,
                sr.force,
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
            )

            # ── 5. Apply parameter update ──────────────────────────────────
            xa, xb, xW = sr.unpack(x)
            # xW is (M, N) — transpose to (N, M) to match rbm.W layout
            w = self.rbm.get_weights()
            update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
            w_new = w - self.learning_rate * update

            if self.param_clip is not None:
                w_new = jnp.clip(w_new, -self.param_clip, self.param_clip)

            self.rbm.set_weights(w_new)

            # ── 6. Adapt beta_x ────────────────────────────────────────────
            E_mean = float(jnp.mean(local_energies))
            beta_eff_this_iter = None

            if self._beta_fixed:
                pass
            elif self.use_cem and _cem_beta_raw is not None:
                self.beta_x = (
                    (1.0 - self.cem_ema_alpha) * self.beta_x
                    + self.cem_ema_alpha * _cem_beta_raw
                )
                beta_eff_this_iter = self.beta_x
                print(
                    f"  [CEM iter {iteration:3d}] β_eff = {_cem_beta_raw:.4f}"
                    f" → beta_x = {self.beta_x:.4f} (EMA α={self.cem_ema_alpha})"
                )
            elif not self.use_cem:
                if prev_energy is not None and E_mean > prev_energy:
                    self._key, subkey = jax.random.split(self._key)
                    flip = bool(jax.random.bernoulli(subkey))
                    factor = (1.0 + self.beta_adapt) if flip else (1.0 - self.beta_adapt)
                    self.beta_x = float(
                        jnp.clip(self.beta_x * factor, self.beta_min, self.beta_max)
                    )

            prev_energy = E_mean

            # ── 7. Metrics ─────────────────────────────────────────────────
            E_std = float(jnp.std(local_energies))
            E_error = E_std / math.sqrt(ns)
            E_var = float(jnp.var(local_energies))

            self.history["energy"].append(E_mean)
            self.history["error"].append(E_std)
            self.history["energy_error"].append(E_error)
            self.history["learning_rate"].append(self.learning_rate)
            self.history["grad_norm"].append(float(jnp.linalg.norm(x)))
            self.history["weight_norm"].append(float(jnp.linalg.norm(w_new)))
            self.history["s_condition_number"].append(float(cg_info["residual_norm"]))
            self.history["beta_x"].append(self.beta_x)
            self.history["beta_eff_cem"].append(beta_eff_this_iter)
            self.history["cg_iterations"].append(int(cg_info["iterations"]))
            self.history["cg_residual"].append(float(cg_info["residual_norm"]))
            self.history["ess"].append(ess_norm)
            self.history["kl_exact"].append(kl)
            self.history["n_unique_ratio"].append(n_unique_ratio)

            if iteration % 10 == 0:
                time_label = "fpga_time" if fpga_time is not None else "sample_time"
                print(
                    f"Iter {iteration:3d}: "
                    f"E = {E_mean:.6f} ± {E_error:.6f}  "
                    f"β_x = {self.beta_x:.3f}  "
                    f"CG {cg_info['iterations']}it "
                    f"res={cg_info['residual_norm']:.2e}  "
                    f"{time_label}={sample_time_s:.3f}s  "
                    f"‖x‖={float(jnp.linalg.norm(x)):.4f}"
                )

            # ── 8. Save checkpoint ─────────────────────────────────────────
            if (
                self.save_checkpoints
                and self.args
                and iteration % self.checkpoint_interval == 0
            ):
                checkpoint_path = save_rbm_checkpoint(self.rbm, self.args, iteration)
                print(f"  → Checkpoint saved: {checkpoint_path.name}")

            # ── 9. Convergence check ───────────────────────────────────────
            if self.stop_at_convergence:
                if E_var < self.conv_var_threshold:
                    consecutive_converged += 1
                else:
                    consecutive_converged = 0

                if consecutive_converged >= self.conv_window:
                    print(
                        f"\n[Converged] Iter {iteration}: "
                        f"Var(E_loc) = {E_var:.2e} < {self.conv_var_threshold:.2e} "
                        f"for {self.conv_window} consecutive iterations. "
                        f"Final E = {E_mean:.6f}"
                    )
                    break

        return self.history
