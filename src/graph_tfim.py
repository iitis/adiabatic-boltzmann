"""
Transverse-field Ising model on an arbitrary coupling graph — JAX backend.

    H = - sum_{i<j} J_ij sz_i sz_j - h sum_i sx_i        (J_ij > 0 ferro, J_ij < 0 antiferro)

The off-diagonal term is -h sx with h >= 0, so H is stoquastic in the z basis for ANY couplings: the
ground state is positive (Perron-Frobenius) and the positive-amplitude RBM of model.py can represent it,
frustrated or not. Constructors:
  chain(N)              ferromagnetic periodic chain (same H as ising.TransverseFieldIsing1D, for validation)
  triangular(L1, L2)    antiferromagnetic triangular lattice, periodic in both directions (L1, L2 >= 3);
                        multiples of 3 keep the 3-sublattice order unfrustrated by the boundary
  triangular(L1, L2, pbc_x=False)   the same on a cylinder: open along x (length L1), periodic around y (width
                        L2) — the standard 2D DMRG geometry (an entanglement cut crosses ~L2 bonds, not ~2 L2)

Also: exact ground energy by sparse Lanczos (N <= 24), and the best product state with one angle per site
(not uniform: on frustrated lattices the optimum breaks translation symmetry), from many random starts.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnums=(4,))
def _local_energy_graph_jit(V, W, a, b, h, J):
    """Local energies of the RBM Psi(v) = e^(-a.v/2) prod_j (2cosh(b_j + W_j.v))^(1/2) for all rows of V.

    J: (N, N) symmetric coupling matrix, zero diagonal; E_diag = -1/2 v.J.v."""
    theta = V @ W + b[None, :]
    blc = jnp.logaddexp(theta, -theta)
    theta_flipped = theta[:, None, :] - 2.0 * V[:, :, None] * W[None, :, :]
    log_ratios = a[None, :] * V + 0.5 * jnp.sum(
        jnp.logaddexp(theta_flipped, -theta_flipped) - blc[:, None, :], axis=2)
    E_off = -h * jnp.sum(jnp.exp(log_ratios), axis=1)
    E_diag = -0.5 * jnp.sum((V @ J) * V, axis=1)
    return E_diag + E_off


class GraphTFIM:
    """TFIM with coupling matrix J (N x N, symmetric, zero diagonal) and uniform transverse field h."""

    def __init__(self, J: np.ndarray, h: float, name: str):
        J = np.asarray(J, dtype=float)
        assert J.shape[0] == J.shape[1] and np.allclose(J, J.T) and not np.any(np.diag(J))
        self.J_np, self.J = J, jnp.asarray(J)
        self.size, self.h, self.name = J.shape[0], float(h), name

    def local_energy_batch(self, V, rbm, chunk=None):
        """Local energies of rbm for configurations V (rows, +-1). chunk: rows per kernel call (memory);
        default keeps the (rows, N, M) flip tensor near 2^27 entries (~1 GB)."""
        V = jnp.asarray(V, dtype=jnp.float64)
        if chunk is None:
            chunk = max(256, (1 << 27) // (self.size * rbm.W.shape[1]))
        if V.shape[0] <= chunk:
            return _local_energy_graph_jit(V, rbm.W, rbm.a, rbm.b, self.h, self.J)
        return jnp.concatenate([_local_energy_graph_jit(V[i:i + chunk], rbm.W, rbm.a, rbm.b, self.h, self.J)
                                for i in range(0, V.shape[0], chunk)])

    # ── exact references ──────────────────────────────────────────────────
    def sparse_hamiltonian(self):
        import scipy.sparse as sp

        N = self.size
        assert N <= 24, "sparse ED limited to N <= 24"
        dim = 2 ** N
        idx = np.arange(dim, dtype=np.int64)
        diag = np.zeros(dim)
        iu, ju = np.nonzero(np.triu(self.J_np))
        for i, j in zip(iu, ju):  # -J_ij s_i s_j, s = +1 for bit 0
            si = 1 - 2 * ((idx >> i) & 1)
            sj = 1 - 2 * ((idx >> j) & 1)
            diag -= self.J_np[i, j] * si * sj
        rows = np.concatenate([idx] + [idx] * N)
        cols = np.concatenate([idx] + [idx ^ (1 << i) for i in range(N)])
        vals = np.concatenate([diag] + [np.full(dim, -self.h)] * N)
        return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))

    def exact_ground_energy(self):
        """Lanczos ground energy (cached in src/reference_energies.json under the model name)."""
        from reference_energies import get_or_compute
        import scipy.sparse.linalg as spla

        def compute():
            if self.size > 24:
                raise NotImplementedError(f"no exact reference for N={self.size}")
            return float(spla.eigsh(self.sparse_hamiltonian(), k=1, which="SA", tol=1e-12)[0][0])

        return get_or_compute(self.name, self.size, self.h, compute)

    def product_state_energy(self, theta):
        """Energy of prod_i (cos(t_i/2)|+1> + sin(t_i/2)|-1>): <sz_i> = cos t_i, <sx_i> = sin t_i."""
        c, s = np.cos(theta), np.sin(theta)
        return float(-0.5 * c @ self.J_np @ c - self.h * s.sum())

    def mean_field(self, n_starts=64, seed=0):
        """Best product state over per-site angles (L-BFGS from random starts): (energy, angles)."""
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)

        def f(t):
            c, s = np.cos(t), np.sin(t)
            Jc = self.J_np @ c
            return -0.5 * c @ Jc - self.h * s.sum(), s * Jc - self.h * c  # value, gradient

        best = None
        # Uniform starts (both signs) reach the translation-invariant optimum directly; random starts alone relax into
        # multi-domain local minima at large N (chain N = 64, h = 0.5: two domain walls). theta = pi/2 exactly is a
        # saddle (zero gradient), so the x-polarised start is perturbed.
        uniform = [np.full(self.size, t) for t in (0.2, 0.6, 1.0, 1.4, -0.2, -0.6, -1.0, -1.4)]
        starts = ([np.pi / 2 + 1e-3 * rng.standard_normal(self.size)] + uniform
                  + [rng.uniform(-np.pi, np.pi, self.size) for _ in range(n_starts)])
        for t0 in starts:
            r = minimize(f, t0, jac=True, method="L-BFGS-B", options=dict(gtol=1e-12, ftol=1e-15, maxiter=5000))
            if best is None or r.fun < best.fun:
                best = r
        return float(best.fun), best.x


def chain(N: int, h: float) -> GraphTFIM:
    J = np.zeros((N, N))
    for i in range(N):
        J[i, (i + 1) % N] = J[(i + 1) % N, i] = 1.0
    return GraphTFIM(J, h, "graph_chain")


def triangular(L1: int, L2: int, h: float, pbc_x: bool = True) -> GraphTFIM:
    """Antiferromagnetic (J = -1) triangular lattice, L1 x L2, bonds along a1, a2 and a1 - a2; periodic along y,
    and along x too unless pbc_x=False (cylinder: the bonds a1, a1 - a2 leaving the last column are dropped)."""
    assert L2 >= 3 and (L1 >= 3 or not pbc_x), "periodic lengths >= 3 so that the six neighbours are distinct"
    N = L1 * L2
    J = np.zeros((N, N))
    site = lambda x, y: (x % L1) * L2 + (y % L2)
    for x in range(L1):
        for y in range(L2):
            for dx, dy in ((1, 0), (0, 1), (1, -1)):
                if not pbc_x and x + dx >= L1:
                    continue
                i, j = site(x, y), site(x + dx, y + dy)
                J[i, j] = J[j, i] = -1.0
    return GraphTFIM(J, h, f"tri_af_{L1}x{L2}" if pbc_x else f"tri_af_cyl_{L1}x{L2}")


def make(model: str, size: str, h: float) -> GraphTFIM:
    """model 'chain' with size 'N', 'tri' (torus) or 'tricyl' (cylinder, open along L1) with size 'L1xL2'."""
    if model == "chain":
        return chain(int(size), h)
    if model in ("tri", "tricyl"):
        L1, L2 = map(int, size.lower().split("x"))
        return triangular(L1, L2, h, pbc_x=(model == "tri"))
    raise ValueError(model)
