"""
End-to-end tests for the FBM (and baseline RBM) VMC implementation.

All tests use N=4 (16 enumerable configurations) so we can check against
exact results without relying on sampling.

Run:
    python -m pytest test_e2e.py -v
    python test_e2e.py          # direct execution, no pytest
"""

import sys
import math
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from model import FullyConnectedRBM, FullBoltzmannMachine
from ising import TransverseFieldIsing1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 4
H_FIELD = 0.5
RNG = np.random.default_rng(42)


def _all_configs(N: int) -> jax.Array:
    """Enumerate all 2^N spin configurations ±1."""
    indices = np.arange(2**N, dtype=np.int32)
    bits = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.float64)
    return jnp.asarray(bits * 2 - 1)


def _random_fbm(N: int, M: int, seed: int = 0) -> FullBoltzmannMachine:
    """FBM with non-trivial random J (scaled to avoid numerical issues)."""
    key = jax.random.PRNGKey(seed)
    fbm = FullBoltzmannMachine(N, M, key)
    # Randomise all parameters
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N) * 0.3
    b = rng.standard_normal(M) * 0.3
    W = rng.standard_normal((N, M)) * 0.3
    J_raw = rng.standard_normal((N, N)) * 0.2
    J_sym = (J_raw + J_raw.T) / 2
    np.fill_diagonal(J_sym, 0.0)
    fbm.a = jnp.asarray(a)
    fbm.b = jnp.asarray(b)
    fbm.W = jnp.asarray(W)
    fbm.J = jnp.asarray(J_sym)
    return fbm


def _hamiltonian_matrix(N: int, h: float) -> np.ndarray:
    """
    Build the 2^N × 2^N Hamiltonian matrix for 1D TFIM (periodic BC).

    Basis ordering: config i ↔ spin pattern of binary(i) with ±1 encoding.
    """
    n_states = 2**N
    all_v = np.array(_all_configs(N))
    H = np.zeros((n_states, n_states))

    # Diagonal: -Σ_i σᶻᵢ σᶻᵢ₊₁
    for s in range(n_states):
        v = all_v[s]
        E_diag = -sum(v[i] * v[(i + 1) % N] for i in range(N))
        H[s, s] = E_diag

    # Off-diagonal: -h Σ_i σˣᵢ  (flips spin i)
    config_to_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
    for s in range(n_states):
        v = all_v[s]
        for i in range(N):
            v_flip = v.copy()
            v_flip[i] = -v_flip[i]
            t = config_to_idx[tuple(v_flip.astype(int).tolist())]
            H[s, t] -= h

    return H


def _exact_expval(model, all_v: jax.Array, H: np.ndarray) -> float:
    """<Ψ|H|Ψ> / <Ψ|Ψ> using the full Hamiltonian matrix."""
    psi_vec = np.array([float(model.psi(v)) for v in all_v])
    return float(psi_vec @ H @ psi_vec) / float(psi_vec @ psi_vec)


def _vmc_energy(model, all_v: jax.Array, H: np.ndarray) -> float:
    """VMC energy via local energies summed over all configs, weighted by |Ψ|²."""
    ising = TransverseFieldIsing1D(N, H_FIELD)
    psi_vec = np.array([float(model.psi(v)) for v in all_v])
    weights = psi_vec**2
    weights /= weights.sum()
    e_locs = np.array([
        ising.local_energy(np.array(v), lambda v_, i: float(model.psi_ratio(jnp.asarray(v_), i)))
        for v in np.array(all_v)
    ])
    return float(np.sum(weights * e_locs))


# ===========================================================================
# FBM-specific tests
# ===========================================================================


def test_fbm_psi_ratio_consistent_with_log_psi():
    """psi_ratio(v, i) must equal exp(log_psi(v') - log_psi(v)) for every site."""
    fbm = _random_fbm(N, M=N)
    all_v = _all_configs(N)

    mismatches = []
    for v in all_v:
        for i in range(N):
            v_flip = v.at[i].set(-v[i])
            expected = float(jnp.exp(fbm.log_psi(v_flip) - fbm.log_psi(v)))
            got = float(fbm.psi_ratio(v, i))
            if not math.isclose(expected, got, rel_tol=1e-9):
                mismatches.append((v, i, expected, got))

    assert not mismatches, f"psi_ratio mismatch in {len(mismatches)} cases: {mismatches[:3]}"


def test_fbm_psi_ratio_pair_consistent_with_log_psi():
    """psi_ratio_pair(v, i, j) must equal exp(log_psi(v') - log_psi(v))."""
    fbm = _random_fbm(N, M=N)
    all_v = _all_configs(N)

    mismatches = []
    for v in all_v:
        for i in range(N):
            for j in range(i + 1, N):
                v_flip = v.at[i].set(-v[i]).at[j].set(-v[j])
                expected = float(jnp.exp(fbm.log_psi(v_flip) - fbm.log_psi(v)))
                got = float(fbm.psi_ratio_pair(v, i, j))
                if not math.isclose(expected, got, rel_tol=1e-9):
                    mismatches.append((i, j, expected, got))

    assert not mismatches, (
        f"psi_ratio_pair mismatch in {len(mismatches)} cases: {mismatches[:3]}"
    )


def test_fbm_local_energy_consistent_with_hamiltonian():
    """
    VMC energy from local_energy must match <Ψ|H|Ψ>/<Ψ|Ψ> from the full matrix.
    """
    fbm = _random_fbm(N, M=N)
    all_v = _all_configs(N)
    H = _hamiltonian_matrix(N, H_FIELD)

    matrix_energy = _exact_expval(fbm, all_v, H)
    vmc_energy = _vmc_energy(fbm, all_v, H)

    assert math.isclose(matrix_energy, vmc_energy, rel_tol=1e-7), (
        f"Matrix energy {matrix_energy:.8f} != VMC energy {vmc_energy:.8f}"
    )


def test_fbm_variational_bound():
    """<E>_FBM >= E_exact  (variational principle)."""
    fbm = _random_fbm(N, M=N)
    all_v = _all_configs(N)
    H = _hamiltonian_matrix(N, H_FIELD)

    vmc_energy = _vmc_energy(fbm, all_v, H)
    exact = TransverseFieldIsing1D(N, H_FIELD).exact_ground_energy()

    assert vmc_energy >= exact - 1e-8, (
        f"Variational bound violated: <E>={vmc_energy:.6f} < E_exact={exact:.6f}"
    )


def test_fbm_get_set_weights_roundtrip():
    """get_weights / set_weights must be a lossless roundtrip."""
    fbm = _random_fbm(N, M=N)
    w = fbm.get_weights()
    fbm2 = FullBoltzmannMachine(N, N, jax.random.PRNGKey(99))
    fbm2.set_weights(w)
    w2 = fbm2.get_weights()
    assert jnp.allclose(w, w2, atol=1e-12), "get_weights/set_weights roundtrip failed"


def test_fbm_n_parameters():
    """n_parameters() must equal N + M + N*M + N*(N-1)//2."""
    fbm = FullBoltzmannMachine(N, N, jax.random.PRNGKey(0))
    expected = N + N + N * N + N * (N - 1) // 2
    assert fbm.n_parameters() == expected, (
        f"n_parameters={fbm.n_parameters()}, expected {expected}"
    )


def test_fbm_j_zero_reduces_to_rbm():
    """With J=0, FBM log_psi must equal FullyConnectedRBM log_psi for all configs."""
    key = jax.random.PRNGKey(7)
    rbm = FullyConnectedRBM(N, N, key)
    fbm = FullBoltzmannMachine(N, N, key)
    # Share same a, b, W
    fbm.a = rbm.a
    fbm.b = rbm.b
    fbm.W = rbm.W
    # J is zero by default

    all_v = _all_configs(N)
    for v in all_v:
        lp_rbm = float(rbm.log_psi(v))
        lp_fbm = float(fbm.log_psi(v))
        assert math.isclose(lp_rbm, lp_fbm, rel_tol=1e-12), (
            f"log_psi mismatch: RBM={lp_rbm}, FBM={lp_fbm}"
        )


# ===========================================================================
# Baseline RBM tests (sanity)
# ===========================================================================


def test_rbm_psi_ratio_consistent_with_log_psi():
    """psi_ratio for FullyConnectedRBM must match exp(log_psi diff)."""
    key = jax.random.PRNGKey(0)
    rbm = FullyConnectedRBM(N, N, key)
    all_v = _all_configs(N)

    for v in all_v:
        for i in range(N):
            v_flip = v.at[i].set(-v[i])
            expected = float(jnp.exp(rbm.log_psi(v_flip) - rbm.log_psi(v)))
            got = float(rbm.psi_ratio(v, i))
            assert math.isclose(expected, got, rel_tol=1e-9), (
                f"RBM psi_ratio mismatch at site {i}: expected={expected}, got={got}"
            )


def test_rbm_local_energy_consistent_with_hamiltonian():
    """VMC energy must match full matrix expectation value for RBM."""
    key = jax.random.PRNGKey(1)
    rbm = FullyConnectedRBM(N, N, key)
    all_v = _all_configs(N)
    H = _hamiltonian_matrix(N, H_FIELD)

    matrix_energy = _exact_expval(rbm, all_v, H)
    vmc_energy = _vmc_energy(rbm, all_v, H)

    assert math.isclose(matrix_energy, vmc_energy, rel_tol=1e-7), (
        f"Matrix energy {matrix_energy:.8f} != VMC energy {vmc_energy:.8f}"
    )


def test_rbm_variational_bound():
    """<E>_RBM >= E_exact."""
    key = jax.random.PRNGKey(2)
    rbm = FullyConnectedRBM(N, N, key)
    all_v = _all_configs(N)
    H = _hamiltonian_matrix(N, H_FIELD)

    vmc_energy = _vmc_energy(rbm, all_v, H)
    exact = TransverseFieldIsing1D(N, H_FIELD).exact_ground_energy()

    assert vmc_energy >= exact - 1e-8, (
        f"Variational bound violated: <E>={vmc_energy:.6f} < E_exact={exact:.6f}"
    )


# ---------------------------------------------------------------------------
# Direct execution (no pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_fbm_psi_ratio_consistent_with_log_psi,
        test_fbm_psi_ratio_pair_consistent_with_log_psi,
        test_fbm_local_energy_consistent_with_hamiltonian,
        test_fbm_variational_bound,
        test_fbm_get_set_weights_roundtrip,
        test_fbm_n_parameters,
        test_fbm_j_zero_reduces_to_rbm,
        test_rbm_psi_ratio_consistent_with_log_psi,
        test_rbm_local_energy_consistent_with_hamiltonian,
        test_rbm_variational_bound,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
