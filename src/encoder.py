import math
import time
import numpy as np
from helpers import save_rbm_checkpoint, save_dwave_samples
from sampler import ClassicalSampler
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Matrix-free SR linear system
# ---------------------------------------------------------------------------


class SRLinearSystem:
    """
    Matrix-free representation of the SR system  S·x = F.

    Avoids forming the (n_params × n_params) S matrix explicitly.
    Memory cost: O(n_samples * (N + M))  instead of  O(n_params²).
    Solve cost per CG iteration: O(n_samples * n_params)  instead of  O(n_params³).

    Sign convention matches our RBM (Gardas ansatz):
        ∂log Ψ / ∂a_i  = -v_i / 2
        ∂log Ψ / ∂b_j  =  tanh(θ_j) / 2
        ∂log Ψ / ∂W_ij =  v_i · tanh(θ_j) / 2

    Parameters
    ----------
    V          : (n_samples, n_visible)   spin configs {-1, +1}
    H          : (n_samples, n_hidden)    tanh(θ) activations
    E          : (n_samples,)             local energies
    diag_shift : regularization added to diagonal of S
    """

    def __init__(self, V: np.ndarray, H: np.ndarray, E: np.ndarray, diag_shift: float):
        self.V = np.asarray(V, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.E = np.asarray(E, dtype=np.float64)
        self.ns = self.V.shape[0]
        self.N = self.V.shape[1]  # n_visible
        self.M = self.H.shape[1]  # n_hidden
        self.diag_shift = float(diag_shift)

        # Mean gradients  ⟨O_k⟩  — note sign on a block
        self.mu_a = -0.5 * self.V.mean(axis=0)  # (N,)
        self.mu_b = 0.5 * self.H.mean(axis=0)  # (M,)
        self.mu_W = 0.5 * (self.H.T @ self.V) / self.ns  # (M, N)  W stored (M,N) here

        # Force vector  F_k = ⟨O_k · E_loc⟩ − ⟨O_k⟩⟨E_loc⟩
        centered_E = self.E - self.E.mean()
        self.F_a = -0.5 * (centered_E @ self.V) / self.ns  # (N,)
        self.F_b = 0.5 * (centered_E @ self.H) / self.ns  # (M,)
        self.F_W = 0.5 * (self.H.T @ (centered_E[:, None] * self.V)) / self.ns  # (M, N)

    def pack(self, a: np.ndarray, b: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Flatten (a, b, W) → 1-D.  W expected shape (M, N) here."""
        return np.concatenate([a.ravel(), b.ravel(), W.ravel()])

    def unpack(self, x: np.ndarray):
        """Split 1-D vector → (a, b, W) where W has shape (M, N)."""
        a = x[: self.N]
        b = x[self.N : self.N + self.M]
        W = x[self.N + self.M :].reshape(self.M, self.N)
        return a, b, W

    @property
    def force(self) -> np.ndarray:
        return self.pack(self.F_a, self.F_b, self.F_W)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Compute  S·x  without forming S.

        S·x = (1/ns) Ō^T Ō x  +  diag_shift · x

        Steps:
          1. z_s = O_s · x  (inner product of sample-s gradient with x)
          2. z_s -= ⟨O⟩ · x   (centre)
          3. out = (1/ns) Ō^T z  +  diag_shift · x
        """
        xa, xb, xW = self.unpack(x)

        # Step 1+2: z_s = (O_s - ⟨O⟩) · x  for each sample s
        # a-block: O_a = -v/2
        z = -0.5 * (self.V @ xa)
        # b-block: O_b = h/2
        z += 0.5 * (self.H @ xb)
        # W-block: O_W[s] = outer(h_s, v_s)/2  with W in (M,N) layout
        z += 0.5 * np.einsum("sm,mn,sn->s", self.H, xW, self.V)
        # subtract mean component
        z -= float(self.mu_a @ xa + self.mu_b @ xb + np.sum(self.mu_W * xW))

        # Step 3: back-project
        out_a = -0.5 * (z @ self.V) / self.ns + self.diag_shift * xa
        out_b = 0.5 * (z @ self.H) / self.ns + self.diag_shift * xb
        out_W = (
            0.5 * (self.H.T @ (z[:, None] * self.V)) / self.ns + self.diag_shift * xW
        )

        return self.pack(out_a, out_b, out_W)


# ---------------------------------------------------------------------------
# Conjugate gradient
# ---------------------------------------------------------------------------


def conjugate_gradient(
    matvec,
    b: np.ndarray,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> tuple:
    """
    Solve  A·x = b  for symmetric positive-definite A, given only matvec.

    Returns (x, info) where info = {'iterations': int, 'residual_norm': float}.
    """
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
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


KL_EXACT_MAX_N = 16  # enumerate all 2^N configs for exact KL when N ≤ this


def estimate_beta_eff_cem(V: np.ndarray, H: np.ndarray, rbm) -> float:
    """
    CEM estimate of β_eff from joint LSB samples (V, H).

    For each sample s and hidden unit j, the analytical conditional expectation is:
        ⟨h_j⟩_{v_s, β} = tanh(β · (b_j + Σ_i v_{si} W_{ij}))

    Find β_eff ∈ (0, 50] minimising the squared discrepancy against observed h_j:
        F(β) = Σ_{s,j} (H_{sj} − tanh(β · Θ_{sj}))²
    where Θ_{sj} = b_j + Σ_i v_{si} W_{ij}  (pre-activations of the hidden units).

    Parameters
    ----------
    V   : (ns, n_visible)  visible spin configs from LSB
    H   : (ns, n_hidden)   hidden spin configs from the same LSB batch
    rbm : RBM with .W (n_visible, n_hidden) and .b (n_hidden,)

    Returns
    -------
    β_eff : float
    """
    Theta = V @ rbm.W + rbm.b[None, :]  # (ns, n_hidden)

    def F(beta):
        return float(np.sum((H - np.tanh(beta * Theta)) ** 2))

    result = minimize_scalar(F, bounds=(0.01, 50.0), method="bounded")
    return float(result.x)


class Trainer:
    """
    Variational Monte Carlo trainer using Stochastic Reconfiguration.

    The SR system S·x = F is solved matrix-free via conjugate gradient.
    Local energies and gradients are computed in a single vectorised pass
    — no Python loops over samples inside the training iteration.
    """

    def __init__(self, rbm, ising_model, sampler, config: dict = None, args=None):
        """
        Config keys
        -----------
        learning_rate  : float  (default 0.1)
        n_iterations   : int    (default 50)
        n_samples      : int    (default 1000)
        regularization : float  (default 1e-3)   diag_shift for SR
        cg_tol         : float  (default 1e-8)   CG convergence tolerance
        cg_maxiter     : int    (default 200)    CG iteration limit
        beta_x_init    : float  (default 2.0)    initial sampler temperature scale
        beta_adapt     : float  (default 0.05)   fractional beta_x adjustment
        beta_min       : float  (default 0.05)   lower bound on beta_x
        beta_max       : float  (default 20.0)   upper bound on beta_x
        param_clip     : float  (default 3.0)    weight clipping bound (None = off)
        """
        self.rbm = rbm
        self.ising = ising_model
        self.sampler = sampler
        self.args = args  # For checkpoint saving
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

        # For samplers that ignore beta_x (metropolis, gibbs) lock it to 1.0
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
        self.checkpoint_interval = config.get(
            "checkpoint_interval", 10
        )  # Save every N iterations
        print("Checkpoint interval:", self.checkpoint_interval)
        self.history = {
            "energy": [],
            "error": [],
            "energy_error": [],
            "learning_rate": [],
            "grad_norm": [],
            "weight_norm": [],
            "s_condition_number": [],
            "beta_x": [],
            "beta_eff_cem": [],  # β_eff estimated by CEM; None on iterations where CEM didn't run
            "cg_iterations": [],
            "cg_residual": [],
            "sampling_time_s": [],
            "ess": [],  # effective sample size, normalised to [0, 1]
            "kl_exact": [],  # KL(q_empirical ‖ p_exact); None when N > KL_EXACT_MAX_N
            "n_unique_ratio": [],  # unique configs / n_samples  [0, 1]
        }

        # Pre-build exact enumeration for small N (cached across iterations)
        self._kl_all_v = None  # (2^N, N) array of all configs
        self._kl_config_idx = None  # tuple → index mapping

    def _build_kl_cache(self):
        """Pre-compute all 2^N configs and index map for exact KL. Called once."""
        N = self.rbm.n_visible
        indices = np.arange(2**N, dtype=np.int32)
        # Vectorised binary → ±1 spin: bit k → 1 if set, else -1
        all_v = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(
            np.float64
        ) * 2 - 1
        config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
        self._kl_all_v = all_v
        self._kl_config_idx = config_idx

    def _compute_sample_metrics(self, V: np.ndarray, Theta: np.ndarray):
        """
        Compute ESS, unique-sample ratio, and (optionally) exact KL from a batch of samples.

        V     : (ns, N)  visible spin configs
        Theta : (ns, M)  pre-activations b + W^T v, already computed in train()

        Returns (ess_norm, kl, n_unique_ratio) where
          ess_norm      ∈ [0, 1] — ESS / n_samples
          kl                     — KL(q_empirical ‖ p_exact) or None when N > KL_EXACT_MAX_N
          n_unique_ratio ∈ [0, 1] — distinct configs / n_samples
        """
        ns = V.shape[0]

        # ── Unique samples ────────────────────────────────────────────────────
        n_unique_ratio = float(len(np.unique(V, axis=0))) / ns

        # ── ESS ──────────────────────────────────────────────────────────────
        # log |Ψ(v)|^2 = -a·v + Σ_j logcosh(θ_j)
        # logcosh = logaddexp(x, -x) = log(e^x + e^{-x})
        log_psi2 = -(V @ self.rbm.a) + np.sum(np.logaddexp(Theta, -Theta), axis=1)
        # Subtract max for numerical stability before normalising
        lw = log_psi2 - log_psi2.max()
        w = np.exp(lw)
        w /= w.sum()
        ess_norm = float(1.0 / np.sum(w**2)) / ns  # normalised to [0, 1]

        # ── Exact KL ─────────────────────────────────────────────────────────
        if self.rbm.n_visible > KL_EXACT_MAX_N:
            return ess_norm, None, n_unique_ratio

        if self._kl_all_v is None:
            self._build_kl_cache()

        all_v = self._kl_all_v
        Theta_all = all_v @ self.rbm.W + self.rbm.b[None, :]
        log_psi2_all = -(all_v @ self.rbm.a) + np.sum(
            np.logaddexp(Theta_all, -Theta_all), axis=1
        )
        lw_all = log_psi2_all - log_psi2_all.max()
        p_true = np.exp(lw_all)
        p_true /= p_true.sum()

        counts = np.zeros(len(all_v))
        for row in V.astype(int).tolist():
            idx = self._kl_config_idx.get(tuple(row))
            if idx is not None:
                counts[idx] += 1
        q_emp = counts / ns

        mask = q_emp > 0
        kl = float(np.sum(q_emp[mask] * (np.log(q_emp[mask]) - np.log(p_true[mask]))))
        return ess_norm, kl, n_unique_ratio

    def train(self) -> dict:
        prev_energy = None
        rng = np.random.default_rng()
        consecutive_converged = 0  # ← add this
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
                if fpga_time is None:
                    sample_time_s = elapsed
                else:
                    sample_time_s = float(fpga_time)
                self.history["sampling_time_s"].append(sample_time_s)
            except Exception as e:
                print(f"  [Trainer] Sampling failed at iteration {iteration}: {e}")
                print("  [Trainer] Aborting this experiment.")
                raise
            if _need_hidden and isinstance(_result, tuple):
                _V_raw, _H_raw = _result
            else:
                _V_raw, _H_raw = _result, None
            V = np.asarray(_V_raw, dtype=np.float64)  # (ns, N)
            ns = V.shape[0]

            # ── 2. Batch local energies ────────────────────────────────────
            local_energies = self.ising.local_energy_batch(V, self.rbm)  # (ns,)

            # ── 3. Batch gradients (no Python loop) ───────────────────────
            # θ[s, j] = b_j + Σ_i W_ij v_si
            Theta = V @ self.rbm.W + self.rbm.b[None, :]  # (ns, M)
            TanH = np.tanh(Theta)  # (ns, M)

            # ── Sample quality metrics (ESS + KL + unique) — reuse Theta ──
            ess_norm, kl, n_unique_ratio = self._compute_sample_metrics(V, Theta)

            # ── Save D-Wave samples to disk for post-hoc analysis ──────────
            if self.args and getattr(self.args, "sampling_method", "") in (
                "pegasus",
                "zephyr",
            ):
                save_dwave_samples(V, self.args, iteration)

            # ── 3b. CEM β estimate — must happen before weight update ──────
            # estimate_beta_eff_cem uses self.rbm.W/b; calling it after
            # set_weights() would fit β against new weights but old samples.
            _cem_beta_raw = None
            if (
                self.use_cem
                and not self._beta_fixed
                and iteration % self.cem_interval == 0
                and _H_raw is not None
            ):
                H_cem = np.asarray(_H_raw, dtype=np.float64)
                _cem_beta_raw = estimate_beta_eff_cem(V, H_cem, self.rbm)

            # ── 4. Build SR system and solve with CG ──────────────────────
            # SRLinearSystem expects H = tanh(θ),  W layout (M, N)
            sr = SRLinearSystem(V, TanH, local_energies, self.regularization)
            x, cg_info = conjugate_gradient(
                sr.matvec,
                sr.force,
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
            )

            # ── 5. Unpack and apply update ─────────────────────────────────
            xa, xb, xW = sr.unpack(x)
            # xW is (M, N) — transpose to (N, M) to match rbm.W layout
            # pack in same order as get_weights(): [a, b, W.flatten()]
            w = self.rbm.get_weights()
            w_new = w - self.learning_rate * np.concatenate(
                [xa.ravel(), xb.ravel(), xW.T.ravel()]
            )

            if self.param_clip is not None:
                w_new = np.clip(w_new, -self.param_clip, self.param_clip)

            self.rbm.set_weights(w_new)

            # ── 6. Adapt beta_x ────────────────────────────────────────────
            E_mean = float(np.mean(local_energies))
            beta_eff_this_iter = None

            if self._beta_fixed:
                pass  # metropolis/gibbs: beta_x is meaningless, keep at 1.0
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
                    factor = (
                        (1.0 + self.beta_adapt)
                        if rng.random() < 0.5
                        else (1.0 - self.beta_adapt)
                    )
                    self.beta_x = float(
                        np.clip(self.beta_x * factor, self.beta_min, self.beta_max)
                    )

            prev_energy = E_mean

            # ── 7. Metrics ─────────────────────────────────────────────────
            E_std = float(np.std(local_energies))
            E_error = E_std / math.sqrt(ns)
            E_var = float(np.var(local_energies))

            self.history["energy"].append(E_mean)
            self.history["error"].append(E_std)
            self.history["energy_error"].append(E_error)
            self.history["learning_rate"].append(self.learning_rate)
            self.history["grad_norm"].append(float(np.linalg.norm(x)))
            self.history["weight_norm"].append(float(np.linalg.norm(w_new)))
            self.history["s_condition_number"].append(
                float(cg_info["residual_norm"])
            )  # CG residual proxy
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
                    f"‖x‖={np.linalg.norm(x):.4f}"
                )

            # ── 8. Save checkpoint ────────────────────────────────────────
            if (
                self.save_checkpoints
                and self.args
                and iteration % self.checkpoint_interval == 0
            ):
                checkpoint_path = save_rbm_checkpoint(self.rbm, self.args, iteration)
                print(f"  → Checkpoint saved: {checkpoint_path.name}")

            # ── 9. Convergence check ──────────────────────────────────────
            if self.stop_at_convergence:
                if E_var < self.conv_var_threshold:
                    consecutive_converged += 1
                else:
                    consecutive_converged = 0  # reset on any bad iteration

                if consecutive_converged >= self.conv_window:
                    print(
                        f"\n[Converged] Iter {iteration}: "
                        f"Var(E_loc) = {E_var:.2e} < {self.conv_var_threshold:.2e} "
                        f"for {self.conv_window} consecutive iterations. "
                        f"Final E = {E_mean:.6f}"
                    )
                    break
        return self.history
