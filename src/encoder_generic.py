"""
Generic SR trainer for arbitrary wave function ansätze — JAX backend

Replaces the RBM-specific analytical gradient formulas in encoder.py with
jax.vmap(jax.grad), producing an explicit (ns, n_params) O matrix that is
then used for matrix-free CG in the same way as SRLinearSystem.

Memory: O(ns × n_params) for the O matrix.
Matvec: O(ns × n_params) per CG iteration, JIT-compiled.

Entry point: TrainerGeneric — a drop-in replacement for Trainer that works
with ViTWaveFunction (or any model implementing log_psi_single / get_flat_params
/ set_flat_params).
"""

import math
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from encoder import conjugate_gradient  # reuse existing CG solver


# ---------------------------------------------------------------------------
# Generic SR linear system
# ---------------------------------------------------------------------------


@jax.jit
def _sr_matvec_generic_jit(O_c: jax.Array, diag_shift: float, x: jax.Array) -> jax.Array:
    """
    Generic SR matvec: S·x = (O_c^T O_c x) / ns + λx

    O_c : (ns, n_params)  centered per-sample gradient matrix
    x   : (n_params,)     current CG iterate
    """
    ns = O_c.shape[0]
    Ox = O_c @ x        # (ns,)
    return O_c.T @ Ox / ns + diag_shift * x


class SRLinearSystemGeneric:
    """
    Matrix-free SR system S·x = F built from a generic gradient matrix O.

    O[s, k] = ∂ log Ψ(v_s) / ∂θ_k  computed via jax.vmap(jax.grad).

    Construction
    ------------
    O          : (ns, n_params) per-sample gradients
    E_loc      : (ns,)          local energies
    diag_shift : float          regularization λ (adds λI to S)

    Force vector
    ------------
    F_k = ⟨O_k (E_loc − ⟨E_loc⟩)⟩ = (O^T E_centered) / ns
    """

    def __init__(self, O: jax.Array, E_loc: jax.Array, diag_shift: float):
        self.ns = int(O.shape[0])
        self.n_params = int(O.shape[1])
        self.diag_shift = float(diag_shift)

        self.mu_O = jnp.mean(O, axis=0)          # (n_params,)
        self.O_c = O - self.mu_O[None, :]         # (ns, n_params) centered

        E_c = E_loc - jnp.mean(E_loc)             # (ns,)
        self.F = self.O_c.T @ E_c / self.ns       # (n_params,) force

    @property
    def force(self) -> jax.Array:
        return self.F

    def matvec(self, x: jax.Array) -> jax.Array:
        return _sr_matvec_generic_jit(self.O_c, self.diag_shift, x)


# ---------------------------------------------------------------------------
# O matrix computation
# ---------------------------------------------------------------------------


def compute_O_matrix(model, V: jax.Array) -> jax.Array:
    """
    Compute the (ns, n_params) per-sample gradient matrix.

    O[s, k] = ∂ log Ψ(v_s) / ∂θ_k

    Uses jax.vmap(jax.grad) — the grad pytree for each sample is flattened
    with the same ravel_pytree used by model.get_flat_params / set_flat_params,
    so the ordering is consistent with the parameter vector used in the update.

    model : ViTWaveFunction (must have log_psi_single and params attributes)
    V     : (ns, N)  visible spin configs

    Returns (ns, n_params) float64 array.
    """
    params = model.params
    grad_fn = jax.grad(model.log_psi_single, argnums=0)

    def per_sample_grad(v: jax.Array) -> jax.Array:
        g = grad_fn(params, v)
        flat, _ = ravel_pytree(g)
        return flat  # type: ignore[return-value]

    return jax.vmap(per_sample_grad)(V)  # (ns, n_params)


# ---------------------------------------------------------------------------
# Generic local energy helpers (used by TrainerGeneric)
# ---------------------------------------------------------------------------


def local_energy_batch_generic(ising_model, model, V: jax.Array) -> jax.Array:
    """
    Dispatch local energy computation to the model-specific generic method.

    ising_model : IsingModel subclass (must implement local_energy_batch_generic)
    model       : ViTWaveFunction
    V           : (ns, N) spin configs

    Returns (ns,) local energies.
    """
    log_psi_fn = lambda v: model.log_psi_single(model.params, v)
    return ising_model.local_energy_batch_generic(V, log_psi_fn)


# ---------------------------------------------------------------------------
# Generic trainer
# ---------------------------------------------------------------------------


KL_EXACT_MAX_N = 16


class TrainerGeneric:
    """
    VMC trainer using Stochastic Reconfiguration for arbitrary wave functions.

    Replaces the RBM-specific analytical gradients of Trainer with
    jax.vmap(jax.grad) on model.log_psi_single. Compatible with any model
    that implements:
        log_psi_single(params, v)  → scalar
        log_psi_batch(V)           → (ns,)
        get_flat_params()          → 1D array
        set_flat_params(w)         → updates model.params
        n_visible                  int

    Config keys (same subset as Trainer that apply to ViT)
    -------------------------------------------------------
    learning_rate  : float  (default 0.05)
    n_iterations   : int    (default 50)
    n_samples      : int    (default 500)
    regularization : float  (default 1e-3)
    cg_tol         : float  (default 1e-8)
    cg_maxiter     : int    (default 200)
    stop_at_convergence : bool  (default False)
    conv_var_threshold  : float (default 1e-4)
    conv_window         : int   (default 10)
    """

    def __init__(self, model, ising_model, sampler, config: dict | None = None, args=None):
        self.model = model
        self.ising = ising_model
        self.sampler = sampler
        self.args = args
        print(self.model)
        print(self.ising)
        print(self.args)

        if config is None:
            config = {}
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.05)
        self.n_iterations = config.get("n_iterations", 50)
        self.n_samples = config.get("n_samples", 500)
        self.regularization = config.get("regularization", 1e-3)
        self.cg_tol = config.get("cg_tol", 1e-8)
        self.cg_maxiter = config.get("cg_maxiter", 200)
        self.stop_at_convergence = config.get("stop_at_convergence", False)
        self.conv_var_threshold = config.get("conv_var_threshold", 1e-4)
        self.conv_window = config.get("conv_window", 10)

        self.history = {
            "energy": [],
            "error": [],
            "energy_error": [],
            "learning_rate": [],
            "grad_norm": [],
            "weight_norm": [],
            "s_condition_number": [],
            "cg_iterations": [],
            "cg_residual": [],
            "sampling_time_s": [],
            "ess": [],
            "kl_exact": [],
            "n_unique_ratio": [],
            # Placeholders kept for result-file compatibility
            "beta_x": [],
            "beta_eff_cem": [],
            "cem_time_s": [],
            "total_sampling_time_s": [],
        }

        self._kl_all_v = None
        self._kl_config_idx = None

    # ── KL cache ──────────────────────────────────────────────────────────

    def _build_kl_cache(self):
        N = self.model.n_visible
        indices = np.arange(2**N, dtype=np.int32)
        all_v = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(
            np.float64
        ) * 2 - 1
        config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
        self._kl_all_v = jnp.asarray(all_v)
        self._kl_config_idx = config_idx

    # ── Sample quality metrics ─────────────────────────────────────────────

    def _compute_sample_metrics(self, V: jax.Array):
        ns = V.shape[0]
        v_np = np.asarray(V)
        n_unique_ratio = float(len(np.unique(v_np, axis=0))) / ns

        # ESS via log|Ψ|² weights
        log_p2 = 2.0 * self.model.log_psi_batch(V).real  # (ns,)
        lw = log_p2 - jnp.max(log_p2)
        w = jnp.exp(lw)
        w = w / jnp.sum(w)
        ess_norm = float(1.0 / jnp.sum(w**2)) / ns

        if self.model.n_visible > KL_EXACT_MAX_N:
            return ess_norm, None, n_unique_ratio

        if self._kl_all_v is None:
            self._build_kl_cache()

        assert self._kl_all_v is not None and self._kl_config_idx is not None
        all_v = self._kl_all_v
        log_p2_all = 2.0 * self.model.log_psi_batch(all_v).real
        lw_all = log_p2_all - jnp.max(log_p2_all)
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

    # ── Training loop ──────────────────────────────────────────────────────

    def train(self, start_iteration: int = 0) -> dict:
        if start_iteration >= self.n_iterations:
            raise RuntimeError(
                f"start_iteration={start_iteration} >= n_iterations={self.n_iterations}."
            )

        consecutive_converged = 0

        for iteration in range(start_iteration, self.n_iterations):
            # ── 1. Sample ─────────────────────────────────────────────────
            try:
                _t0 = time.perf_counter()
                _V_raw = self.sampler.sample(
                    self.model, self.n_samples, config=self.config
                )
                sample_time_s = time.perf_counter() - _t0
            except Exception as e:
                print(f"  [TrainerGeneric] Sampling failed at iteration {iteration}: {e}")
                raise

            V = jnp.asarray(_V_raw, dtype=jnp.float64)  # (ns, N)
            ns = int(V.shape[0])
            self.history["sampling_time_s"].append(sample_time_s)
            self.history["cem_time_s"].append(0.0)
            self.history["total_sampling_time_s"].append(sample_time_s)
            self.history["beta_x"].append(1.0)
            self.history["beta_eff_cem"].append(None)

            # ── 2. Local energies ─────────────────────────────────────────
            local_energies = local_energy_batch_generic(self.ising, self.model, V)

            # ── 3. Sample quality metrics ─────────────────────────────────
            ess_norm, kl, n_unique_ratio = self._compute_sample_metrics(V)

            # ── 4. Per-sample gradient matrix O ──────────────────────────
            O = compute_O_matrix(self.model, V)  # (ns, n_params)

            # ── 5. Build SR system and solve with CG ──────────────────────
            sr = SRLinearSystemGeneric(O, local_energies, self.regularization)
            x, cg_info = conjugate_gradient(
                sr.matvec,
                sr.force,
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
            )

            # ── 6. Apply parameter update ─────────────────────────────────
            w = self.model.get_flat_params()
            w_new = w - self.learning_rate * x
            self.model.set_flat_params(w_new)

            # ── 7. Record metrics ─────────────────────────────────────────
            E_mean = float(jnp.mean(local_energies))
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
            self.history["cg_iterations"].append(int(cg_info["iterations"]))
            self.history["cg_residual"].append(float(cg_info["residual_norm"]))
            self.history["ess"].append(ess_norm)
            self.history["kl_exact"].append(kl)
            self.history["n_unique_ratio"].append(n_unique_ratio)

            if iteration % 10 == 0:
                print(
                    f"Iter {iteration:3d}: "
                    f"E = {E_mean:.6f} ± {E_error:.6f}  "
                    f"CG {cg_info['iterations']}it "
                    f"res={cg_info['residual_norm']:.2e}  "
                    f"sample={sample_time_s:.3f}s  "
                    f"‖x‖={float(jnp.linalg.norm(x)):.4f}  "
                    f"ESS={ess_norm:.3f}"
                )

            # ── 8. Convergence check ──────────────────────────────────────
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
