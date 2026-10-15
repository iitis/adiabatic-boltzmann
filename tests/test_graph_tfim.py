"""
Correctness tests for src/graph_tfim.py (general-graph TFIM).

  chain local energies     == ising.TransverseFieldIsing1D (same H)
  <H> from local energies   == psi^T H psi / psi^T psi with the sparse Hamiltonian (chain, triangular)
  chain Lanczos energy      == free-fermion energy
  triangular geometry       6 distinct neighbours per site, 3N bonds, classical ground energy -N at h = 0
  mean field                variational (>= exact), chain value -(1 + h^2/4) N

    python -m pytest tests/test_graph_tfim.py -v
"""

import itertools
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_tfim import chain, triangular
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM, RBMParams


def random_rbm(N, M, seed, scale=0.3):
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(seed))
    rbm.params = RBMParams(*(jax.numpy.asarray(scale * rng.standard_normal(s)) for s in ((N,), (M,), (N, M))))
    return rbm


def all_configs(N):
    return np.array(list(itertools.product([-1.0, 1.0], repeat=N)))


def psi(rbm, V):
    phi = V @ np.asarray(rbm.W) + np.asarray(rbm.b)
    return np.exp(-0.5 * V @ np.asarray(rbm.a)) * np.sqrt(np.prod(2 * np.cosh(phi), axis=1))


def config_index(V):
    """Row of the sparse Hamiltonian: bit i set <=> spin i = -1."""
    return ((V < 0).astype(np.int64) * (1 << np.arange(V.shape[1]))).sum(1)


@pytest.mark.parametrize("h", [0.3, 1.0, 2.5])
def test_chain_matches_tfim1d(h):
    N = 8
    rbm = random_rbm(N, N, 1)
    V = all_configs(N)
    a = np.asarray(chain(N, h).local_energy_batch(V, rbm))
    b = np.asarray(TransverseFieldIsing1D(N, h).local_energy_batch(V, rbm))
    np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("model", [chain(8, 0.7), triangular(3, 3, 1.3), triangular(3, 4, 0.6)])
def test_local_energy_vs_sparse_H(model):
    N = model.size
    rbm = random_rbm(N, 2 * N, 2)
    V = all_configs(N)
    p = psi(rbm, V)
    E_loc = np.asarray(model.local_energy_batch(V, rbm, chunk=1000))
    vec = np.zeros(2 ** N)
    vec[config_index(V)] = p
    H = model.sparse_hamiltonian()
    exact = vec @ (H @ vec) / (vec @ vec)
    np.testing.assert_allclose(np.sum(p ** 2 * E_loc) / np.sum(p ** 2), exact, rtol=1e-12)


@pytest.mark.parametrize("N,h", [(8, 0.5), (12, 1.0), (10, 2.0)])
def test_chain_lanczos_vs_free_fermions(N, h):
    m = chain(N, h)
    import scipy.sparse.linalg as spla
    e = spla.eigsh(m.sparse_hamiltonian(), k=1, which="SA", tol=1e-12)[0][0]
    assert e == pytest.approx(TransverseFieldIsing1D(N, h)._compute_exact_ground_energy(), abs=1e-9)


@pytest.mark.parametrize("L1,L2", [(3, 3), (3, 6), (6, 6), (4, 5)])
def test_triangular_geometry(L1, L2):
    m = triangular(L1, L2, 1.0)
    nn = (m.J_np != 0).sum(1)
    assert np.all(nn == 6) and np.all(m.J_np[m.J_np != 0] == -1.0)
    assert (m.J_np != 0).sum() // 2 == 3 * m.size


def test_triangular_classical_ground_energy():
    m = triangular(3, 3, 0.0)
    import scipy.sparse.linalg as spla
    assert spla.eigsh(m.sparse_hamiltonian(), k=1, which="SA")[0][0] == pytest.approx(-m.size, abs=1e-9)
    assert m.mean_field(n_starts=16)[0] == pytest.approx(-m.size, abs=1e-8)


@pytest.mark.parametrize("model", [chain(10, 0.5), chain(10, 1.0), triangular(3, 3, 1.0), triangular(3, 3, 3.0)])
def test_mean_field_variational(model):
    import scipy.sparse.linalg as spla
    e0 = spla.eigsh(model.sparse_hamiltonian(), k=1, which="SA", tol=1e-12)[0][0]
    emf = model.mean_field(n_starts=16)[0]
    assert emf >= e0 - 1e-9
    if model.name == "graph_chain" and model.h <= 2:
        assert emf == pytest.approx(-(1 + model.h ** 2 / 4) * model.size, abs=1e-8)


@pytest.mark.parametrize("N,h", [(64, 0.5), (128, 0.5), (64, 1.5)])
def test_chain_mean_field_large_N_is_uniform(N, h):
    """Regression: random starts alone returned a two-domain-wall state at N = 64, h = 0.5."""
    E, theta = chain(N, h).mean_field(n_starts=16)
    assert E == pytest.approx(-(1 + h ** 2 / 4) * N, abs=1e-8)
    assert np.all(np.sign(np.cos(theta)) == np.sign(np.cos(theta[0])))


@pytest.mark.parametrize("L", [3, 6])
def test_triangular_mean_field_matches_three_sublattice(L):
    """Best product state on the triangular torus = best 3-sublattice product state (brute force over 3 angles)."""
    from scipy.optimize import minimize
    m = triangular(L, L, 1.0)
    s = np.array([(x - y) % 3 for x in range(L) for y in range(L)])
    f = lambda t3: m.product_state_energy(np.asarray(t3)[s])
    rng = np.random.default_rng(0)
    brute = min(minimize(f, rng.uniform(-np.pi, np.pi, 3), method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000)).fun for _ in range(30))
    assert m.mean_field(n_starts=16)[0] == pytest.approx(brute, abs=1e-7)


@pytest.mark.parametrize("L1,L2", [(3, 6), (6, 6), (9, 6)])
def test_cylinder_geometry(L1, L2):
    m = triangular(L1, L2, 1.0, pbc_x=False)
    nn = (m.J_np != 0).sum(1).reshape(L1, L2)
    assert np.all(nn[1:-1] == 6) and np.all(nn[0] == 4) and np.all(nn[-1] == 4)  # open edges lose two bonds
    assert (m.J_np != 0).sum() // 2 == 3 * m.size - 2 * L2
    s = np.array([(x - y) % 3 for x in range(L1) for y in range(L2)])
    i, j = np.nonzero(np.triu(m.J_np))
    assert np.all(s[i] != s[j])  # 3-colouring stays proper


def test_cylinder_local_energy_vs_sparse_H():
    m = triangular(2, 6, 0.8, pbc_x=False)
    rbm = random_rbm(m.size, m.size, 3)
    V = all_configs(m.size)
    p = psi(rbm, V)
    vec = np.zeros(2 ** m.size)
    vec[config_index(V)] = p
    exact = vec @ (m.sparse_hamiltonian() @ vec) / (vec @ vec)
    E = np.sum(p ** 2 * np.asarray(m.local_energy_batch(V, rbm))) / np.sum(p ** 2)
    assert E == pytest.approx(exact, rel=1e-12)
