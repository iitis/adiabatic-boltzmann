"""
Tests for the Heisenberg XXZ 1D model and psi_ratio_pair.

All tests use N=4 (16 configurations, fully enumerable).
Run from src/:
    python -m pytest test_heisenberg.py -v
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from ising import HeisenbergXXZ1D
from model import FullyConnectedRBM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 4
RNG_KEY = jax.random.PRNGKey(7)


@pytest.fixture(scope="module")
def rbm():
    return FullyConnectedRBM(N, N, RNG_KEY)


@pytest.fixture(scope="module")
def ising_xxx():
    """Antiferromagnetic XXX chain (Δ=1, J=1)."""
    return HeisenbergXXZ1D(N, J=1.0, delta=1.0)


@pytest.fixture(scope="module")
def ising_xxz():
    """Anisotropic XXZ chain (Δ=0.5, J=1)."""
    return HeisenbergXXZ1D(N, J=1.0, delta=0.5)


def _all_configs(n):
    """Enumerate all 2^n spin configurations as ±1 arrays."""
    configs = []
    for i in range(2**n):
        bits = [(1 - 2 * int(b)) for b in f"{i:0{n}b}"]
        configs.append(np.array(bits, dtype=np.float64))
    return configs


# ---------------------------------------------------------------------------
# psi_ratio_pair correctness
# ---------------------------------------------------------------------------


def test_psi_ratio_pair_consistent_with_log_psi(rbm):
    """psi_ratio_pair(v, i, j) == exp(log_psi(v') - log_psi(v)) for all configs."""
    configs = _all_configs(N)
    for v in configs:
        v_jax = jnp.asarray(v)
        for i in range(N):
            for j in range(i + 1, N):
                v_flip = v.copy()
                v_flip[i] = -v_flip[i]
                v_flip[j] = -v_flip[j]
                v_flip_jax = jnp.asarray(v_flip)

                ratio_fast = float(rbm.psi_ratio_pair(v_jax, i, j))
                ratio_exact = float(
                    jnp.exp(rbm.log_psi(v_flip_jax) - rbm.log_psi(v_jax))
                )
                assert abs(ratio_fast - ratio_exact) < 1e-10, (
                    f"psi_ratio_pair mismatch at v={v}, i={i}, j={j}: "
                    f"fast={ratio_fast:.10f}, exact={ratio_exact:.10f}"
                )


# ---------------------------------------------------------------------------
# local_energy scalar vs batch consistency
# ---------------------------------------------------------------------------


def test_local_energy_batch_consistent_with_scalar(rbm, ising_xxx):
    """local_energy_batch must match scalar local_energy for every config."""
    configs = _all_configs(N)
    V = np.stack(configs)

    E_batch = np.array(ising_xxx.local_energy_batch(V, rbm))

    for idx, v in enumerate(configs):
        E_scalar = ising_xxx.local_energy(v, rbm)
        assert abs(E_scalar - E_batch[idx]) < 1e-8, (
            f"Scalar/batch mismatch at config {idx}: "
            f"scalar={E_scalar:.10f}, batch={E_batch[idx]:.10f}"
        )


def test_local_energy_batch_xxz_consistent_with_scalar(rbm, ising_xxz):
    """Same check for anisotropic XXZ (Δ=0.5)."""
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(ising_xxz.local_energy_batch(V, rbm))
    for idx, v in enumerate(configs):
        E_scalar = ising_xxz.local_energy(v, rbm)
        assert abs(E_scalar - E_batch[idx]) < 1e-8, (
            f"XXZ scalar/batch mismatch at config {idx}: "
            f"scalar={E_scalar:.10f}, batch={E_batch[idx]:.10f}"
        )


# ---------------------------------------------------------------------------
# Variational bound
# ---------------------------------------------------------------------------


def test_variational_bound_heisenberg(rbm, ising_xxx):
    """VMC energy estimate must be >= exact ground energy (variational principle)."""
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(ising_xxx.local_energy_batch(V, rbm))

    log_psis = np.array([float(rbm.log_psi(jnp.asarray(v))) for v in configs])
    weights = np.exp(2 * log_psis)
    weights /= weights.sum()

    E_vmc = float(np.dot(weights, E_batch))
    E_exact = ising_xxx.exact_ground_energy()

    assert E_vmc >= E_exact - 1e-8, (
        f"Variational bound violated: E_VMC={E_vmc:.6f} < E_exact={E_exact:.6f}"
    )


# ---------------------------------------------------------------------------
# Hamiltonian matrix row check
# ---------------------------------------------------------------------------


def test_e_loc_sum_equals_hamiltonian_row(rbm, ising_xxx):
    """
    For any config v, sum over all configs v' of |Ψ(v')|² E_loc(v')
    must equal <Ψ|H|v> / Ψ(v) (inner product of H with the row v).

    Equivalently: Σ_v' p(v') E_loc(v') = <Ψ|H|Ψ> / <Ψ|Ψ>
    (the VMC energy estimator is unbiased).
    """
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(ising_xxx.local_energy_batch(V, rbm))

    log_psis = np.array([float(rbm.log_psi(jnp.asarray(v))) for v in configs])
    psi_vals = np.exp(log_psis)

    # <Ψ|H|Ψ> via matrix elements
    N_conf = len(configs)
    H = np.zeros((N_conf, N_conf))
    ising = ising_xxx
    J, delta, size = ising.J, ising.delta, ising.size

    for row_idx, v in enumerate(configs):
        # Diagonal
        for i in range(size):
            j = (i + 1) % size
            H[row_idx, row_idx] += J * delta * v[i] * v[j]
        # Off-diagonal exchange
        for i in range(size):
            j = (i + 1) % size
            if v[i] != v[j]:
                v_flip = v.copy()
                v_flip[i] = -v_flip[i]
                v_flip[j] = -v_flip[j]
                for col_idx, v2 in enumerate(configs):
                    if np.allclose(v2, v_flip):
                        H[row_idx, col_idx] += J
                        break

    E_matrix = float(psi_vals @ H @ psi_vals) / float(psi_vals @ psi_vals)
    E_vmc = float(np.dot(psi_vals**2 / (psi_vals**2).sum(), E_batch))

    assert abs(E_matrix - E_vmc) < 1e-8, (
        f"VMC estimator inconsistent with H matrix: "
        f"E_matrix={E_matrix:.10f}, E_vmc={E_vmc:.10f}"
    )
