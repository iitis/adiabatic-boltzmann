"""
Correctness tests for new model variants.

All tests use N=4 (16 configs, fully enumerable) or L=2 (2D, N=4).
Run from src/:
    python -m pytest test_new_models.py -v
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from ising import J1J2Ising1D, HeisenbergXY1D, HeisenbergXXZ2D, HeisenbergXXZ1D
from model import FullyConnectedRBM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 4
RNG_KEY = jax.random.PRNGKey(13)


@pytest.fixture(scope="module")
def rbm():
    return FullyConnectedRBM(N, N, RNG_KEY)


def _all_configs(n):
    configs = []
    for i in range(2**n):
        bits = [(1 - 2 * int(b)) for b in f"{i:0{n}b}"]
        configs.append(np.array(bits, dtype=np.float64))
    return configs


# ---------------------------------------------------------------------------
# J1-J2 chain
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def j1j2():
    return J1J2Ising1D(N, J1=1.0, J2=0.5, h=0.5)


@pytest.fixture(scope="module")
def j1j2_nn():
    """J2=0 should reduce to nearest-neighbor TFIM."""
    return J1J2Ising1D(N, J1=1.0, J2=0.0, h=0.5)


def test_j1j2_batch_consistent_with_scalar(rbm, j1j2):
    """local_energy_batch must match scalar local_energy for every config."""
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(j1j2.local_energy_batch(V, rbm))
    for idx, v in enumerate(configs):
        E_scalar = j1j2.local_energy(
            v, lambda v_, i: float(rbm.psi_ratio(jnp.asarray(v_), i))
        )
        assert abs(E_scalar - E_batch[idx]) < 1e-8, (
            f"J1J2 scalar/batch mismatch at config {idx}: "
            f"scalar={E_scalar:.10f}, batch={E_batch[idx]:.10f}"
        )


def test_j1j2_variational_bound(rbm, j1j2):
    """VMC energy >= exact ground energy (variational principle)."""
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(j1j2.local_energy_batch(V, rbm))
    log_psis = np.array([float(rbm.log_psi(jnp.asarray(v))) for v in configs])
    weights = np.exp(2 * log_psis)
    weights /= weights.sum()
    E_vmc = float(np.dot(weights, E_batch))
    E_exact = j1j2.exact_ground_energy()
    assert E_vmc >= E_exact - 1e-8, (
        f"J1J2 variational bound violated: E_VMC={E_vmc:.6f} < E_exact={E_exact:.6f}"
    )


def test_j1j2_nn_limit_matches_tfim(rbm, j1j2_nn):
    """J2=0 J1J2 must give same local energies as TFIM-1D (same Hamiltonian)."""
    from ising import TransverseFieldIsing1D
    tfim = TransverseFieldIsing1D(N, h=0.5)
    configs = _all_configs(N)
    V = np.stack(configs)
    E_j1j2 = np.array(j1j2_nn.local_energy_batch(V, rbm))
    E_tfim  = np.array(tfim.local_energy_batch(V, rbm))
    np.testing.assert_allclose(E_j1j2, E_tfim, atol=1e-10,
        err_msg="J1J2 with J2=0 must equal TFIM-1D")


def test_j1j2_exact_energy_finite(j1j2):
    E = j1j2.exact_ground_energy()
    assert np.isfinite(E), f"J1J2 exact energy is not finite: {E}"
    assert E < 0, f"J1J2 AFM exact energy should be negative, got {E}"


# ---------------------------------------------------------------------------
# Heisenberg XY chain
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xy():
    return HeisenbergXY1D(N, J=1.0)


def test_xy_batch_consistent_with_scalar(rbm, xy):
    """local_energy_batch must match scalar local_energy for every config."""
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(xy.local_energy_batch(V, rbm))
    for idx, v in enumerate(configs):
        E_scalar = xy.local_energy(v, rbm)
        assert abs(E_scalar - E_batch[idx]) < 1e-8, (
            f"XY scalar/batch mismatch at config {idx}: "
            f"scalar={E_scalar:.10f}, batch={E_batch[idx]:.10f}"
        )


def test_xy_variational_bound(rbm, xy):
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(xy.local_energy_batch(V, rbm))
    log_psis = np.array([float(rbm.log_psi(jnp.asarray(v))) for v in configs])
    weights = np.exp(2 * log_psis)
    weights /= weights.sum()
    E_vmc = float(np.dot(weights, E_batch))
    E_exact = xy.exact_ground_energy()
    assert E_vmc >= E_exact - 1e-8, (
        f"XY variational bound violated: E_VMC={E_vmc:.6f} < E_exact={E_exact:.6f}"
    )


def test_xy_equals_xxz_delta0(rbm, xy):
    """HeisenbergXY1D must give identical energies to HeisenbergXXZ1D(delta=0)."""
    xxz0 = HeisenbergXXZ1D(N, J=1.0, delta=0.0)
    configs = _all_configs(N)
    V = np.stack(configs)
    E_xy  = np.array(xy.local_energy_batch(V, rbm))
    E_xxz = np.array(xxz0.local_energy_batch(V, rbm))
    np.testing.assert_allclose(E_xy, E_xxz, atol=1e-10,
        err_msg="HeisenbergXY1D must match HeisenbergXXZ1D(delta=0)")


def test_xy_exact_energy_shared_cache():
    """XY exact energy must equal XXZ(delta=0) exact energy (same physics)."""
    xy  = HeisenbergXY1D(N, J=1.0)
    xxz = HeisenbergXXZ1D(N, J=1.0, delta=0.0)
    assert abs(xy.exact_ground_energy() - xxz.exact_ground_energy()) < 1e-10, (
        "XY and XXZ(delta=0) exact energies must agree"
    )


# ---------------------------------------------------------------------------
# Heisenberg XXZ 2D
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def heis2d_xxx():
    """L=2 (N=4) isotropic Heisenberg (Δ=1)."""
    return HeisenbergXXZ2D(2, J=1.0, delta=1.0)


@pytest.fixture(scope="module")
def heis2d_xy():
    """L=2 (N=4) XY limit (Δ=0)."""
    return HeisenbergXXZ2D(2, J=1.0, delta=0.0)


@pytest.fixture(scope="module")
def rbm2d():
    return FullyConnectedRBM(N, N, jax.random.PRNGKey(99))


def test_heis2d_batch_consistent_with_scalar(rbm2d, heis2d_xxx):
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(heis2d_xxx.local_energy_batch(V, rbm2d))
    for idx, v in enumerate(configs):
        E_scalar = heis2d_xxx.local_energy(v, rbm2d)
        assert abs(E_scalar - E_batch[idx]) < 1e-8, (
            f"2D Heisenberg scalar/batch mismatch at config {idx}: "
            f"scalar={E_scalar:.10f}, batch={E_batch[idx]:.10f}"
        )


def test_heis2d_xy_batch_consistent_with_scalar(rbm2d, heis2d_xy):
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(heis2d_xy.local_energy_batch(V, rbm2d))
    for idx, v in enumerate(configs):
        E_scalar = heis2d_xy.local_energy(v, rbm2d)
        assert abs(E_scalar - E_batch[idx]) < 1e-8, (
            f"2D XY scalar/batch mismatch at config {idx}: "
            f"scalar={E_scalar:.10f}, batch={E_batch[idx]:.10f}"
        )


def test_heis2d_variational_bound(rbm2d, heis2d_xxx):
    configs = _all_configs(N)
    V = np.stack(configs)
    E_batch = np.array(heis2d_xxx.local_energy_batch(V, rbm2d))
    log_psis = np.array([float(rbm2d.log_psi(jnp.asarray(v))) for v in configs])
    weights = np.exp(2 * log_psis)
    weights /= weights.sum()
    E_vmc = float(np.dot(weights, E_batch))
    E_exact = heis2d_xxx.exact_ground_energy()
    assert E_vmc >= E_exact - 1e-8, (
        f"2D Heisenberg variational bound violated: "
        f"E_VMC={E_vmc:.6f} < E_exact={E_exact:.6f}"
    )


def test_heis2d_exact_energy_finite(heis2d_xxx):
    E = heis2d_xxx.exact_ground_energy()
    assert np.isfinite(E), f"2D Heisenberg exact energy is not finite: {E}"
    assert E < 0, f"2D AFM Heisenberg exact energy should be negative, got {E}"


def test_heis2d_diagonal_only_when_delta1_h0(rbm2d):
    """With Δ=1 and an all-parallel config, off-diagonal contribution is zero.

    For L=2 with periodic BC, right[i] == left[i] and down[i] == up[i], so
    each unique bond is traversed twice by the right+down indexing scheme
    (the same pre-existing behaviour as _local_energy_2d_jit for TFIM).
    L=2 has 4 unique bonds; the kernel counts 8 bond-slots → E_diag = 8.
    Scalar and batch are consistent with each other and with the ED.
    """
    m = HeisenbergXXZ2D(2, J=1.0, delta=1.0)
    ferro = np.array([1.0, 1.0, 1.0, 1.0])  # all parallel → no exchange term
    V = ferro[None, :]
    E_batch = float(m.local_energy_batch(V, rbm2d)[0])
    E_scalar = m.local_energy(ferro, rbm2d)
    # Both methods count 8 bond contributions for L=2 (4 unique bonds × 2)
    assert abs(E_batch - 8.0) < 1e-8, f"L=2 all-parallel 2D Heisenberg E should be 8, got {E_batch}"
    assert abs(E_scalar - 8.0) < 1e-8, f"L=2 all-parallel scalar E should be 8, got {E_scalar}"
    # Scalar and batch must agree regardless
    assert abs(E_batch - E_scalar) < 1e-10, "Scalar and batch must agree"
