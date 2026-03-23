import math
import numpy as np


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


class Trainer:
    """
    Variational Monte Carlo trainer using Stochastic Reconfiguration.

    The SR system S·x = F is solved matrix-free via conjugate gradient.
    Local energies and gradients are computed in a single vectorised pass
    — no Python loops over samples inside the training iteration.
    """

    def __init__(self, rbm, ising_model, sampler, config: dict = None):
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

        if config is None:
            config = {}

        self.learning_rate = config.get("learning_rate", 0.1)
        self.n_iterations = config.get("n_iterations", 50)
        self.n_samples = config.get("n_samples", 1000)
        self.regularization = config.get("regularization", 1e-3)
        self.cg_tol = config.get("cg_tol", 1e-8)
        self.cg_maxiter = config.get("cg_maxiter", 200)

        self.beta_x = config.get("beta_x_init", 2.0)
        self.beta_adapt = config.get("beta_adapt", 0.05)
        self.beta_min = config.get("beta_min", 0.05)
        self.beta_max = config.get("beta_max", 20.0)

        self.stop_at_convergence = config.get("stop_at_convergence", True)
        self.conv_var_threshold = config.get("conv_var_threshold", 1e-4)
        self.conv_window = config.get("conv_window", 10)
        self.param_clip = config.get("param_clip", 3.0)

        self.history = {
            "energy": [],
            "error": [],
            "energy_error": [],
            "learning_rate": [],
            "grad_norm": [],
            "weight_norm": [],
            "s_condition_number": [],
            "beta_x": [],
            "cg_iterations": [],
            "cg_residual": [],
        }

    def train(self) -> dict:
        prev_energy = None
        rng = np.random.default_rng()
        consecutive_converged = 0  # ← add this
        for iteration in range(self.n_iterations):
            # ── 1. Sample ──────────────────────────────────────────────────
            samples = self.sampler.sample(
                self.rbm,
                self.n_samples,
                config={"beta_x": self.beta_x},
            )
            V = np.array([v.copy() for v in samples], dtype=np.float64)  # (ns, N)
            ns = V.shape[0]

            # ── 2. Batch local energies ────────────────────────────────────
            local_energies = self.ising.local_energy_batch(V, self.rbm)  # (ns,)

            # ── 3. Batch gradients (no Python loop) ───────────────────────
            # θ[s, j] = b_j + Σ_i W_ij v_si
            Theta = V @ self.rbm.W + self.rbm.b[None, :]  # (ns, M)
            TanH = np.tanh(Theta)  # (ns, M)

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
            self.history["cg_iterations"].append(int(cg_info["iterations"]))
            self.history["cg_residual"].append(float(cg_info["residual_norm"]))

            if iteration % 10 == 0:
                print(
                    f"Iter {iteration:3d}: "
                    f"E = {E_mean:.6f} ± {E_error:.6f}  "
                    f"β_x = {self.beta_x:.3f}  "
                    f"CG {cg_info['iterations']}it "
                    f"res={cg_info['residual_norm']:.2e}  "
                    f"‖x‖={np.linalg.norm(x):.4f}"
                )
            # ── 8. Convergence check ──────────────────────────────────────
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
