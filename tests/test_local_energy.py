import numpy as np
import pytest
import netket as nk
import jax.numpy as jnp
import jax
from functools import partial
import flax

from src.ising import TransverseFieldIsing1D
from src.model import FullyConnectedRBM


# ---------------------------------------------------------------------------
# RBM construction
# ---------------------------------------------------------------------------


def make_rbm_with_known_weights(N: int, seed: int) -> FullyConnectedRBM:
    """
    Create RBM with deterministic, explicitly controlled weights.
    Uses default_rng (not global numpy state) so it's fully isolated.
    Small std avoids tanh saturation from the first iteration.
    """
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.01, N)
    rbm.b = rng.normal(0, 0.01, N)
    rbm.W = rng.normal(0, 0.01, (N, N))
    return rbm


# ---------------------------------------------------------------------------
# NetKet helpers
# ---------------------------------------------------------------------------


def build_hamiltonian(N: int, h: float):
    """Build the 1D TFIM Hamiltonian in NetKet."""
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    ha = nk.operator.LocalOperator(hilbert, dtype=complex)
    for i in range(N):
        ha += (
            -1.0
            * nk.operator.spin.sigmaz(hilbert, i)
            @ nk.operator.spin.sigmaz(hilbert, (i + 1) % N)
        )
        ha += -h * nk.operator.spin.sigmax(hilbert, i)
    return hilbert, ha


def build_netket_params(nk_rbm, rbm: FullyConnectedRBM, N: int):
    """
    Initialise NetKet RBM and overwrite with our RBM's weights.
    Asserts round-trip correctness before returning so any convention
    mismatch (wrong key, transposed kernel, etc.) fails here, not silently
    inside an energy calculation.
    """
    dummy_input = jnp.ones((1, N))
    params = nk_rbm.init(jax.random.PRNGKey(0), dummy_input)
    params = flax.core.unfreeze(params)

    # NetKet RBM kernel shape: (n_visible, n_hidden) — same as our W
    params["params"]["kernel"] = jnp.array(rbm.W)
    params["params"]["visible_bias"] = jnp.array(rbm.a)
    params["params"]["hidden_bias"] = jnp.array(rbm.b)

    params = flax.core.freeze(params)

    # Explicit parameter identity checks
    assert np.allclose(np.array(params["params"]["kernel"]), rbm.W, atol=1e-10), (
        f"kernel mismatch after assignment:\n{np.array(params['params']['kernel'])}\nvs\n{rbm.W}"
    )
    assert np.allclose(np.array(params["params"]["visible_bias"]), rbm.a, atol=1e-10), (
        f"visible_bias mismatch: {np.array(params['params']['visible_bias'])} vs {rbm.a}"
    )
    assert np.allclose(np.array(params["params"]["hidden_bias"]), rbm.b, atol=1e-10), (
        f"hidden_bias mismatch: {np.array(params['params']['hidden_bias'])} vs {rbm.b}"
    )

    return params


@partial(jax.jit, static_argnames="model")
def compute_local_energies(model, parameters, hamiltonian_jax, sigma):
    """
    Local energy for a batch of configurations.
    sigma: (batch, N)
    returns: (batch,) local energies
    """
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(
        sigma
    )  # (batch, n_conn, N), (batch, n_conn)
    logpsi_sigma = model.apply(parameters, sigma)  # (batch,)
    logpsi_eta = model.apply(parameters, eta)  # (batch, n_conn)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)  # (batch, 1)
    return jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


@partial(jax.jit, static_argnames="model")
def estimate_energy(model, parameters, hamiltonian_jax, sigma):
    """
    Estimate energy and statistical error from a batch of samples.
    Returns nk.stats.Stats(mean, error_of_mean, variance).
    """
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)
    return nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)


def _make_nk_rbm_and_params(N: int, h: float, rbm: FullyConnectedRBM):
    """Shared setup: builds Hamiltonian, Jax operator, NetKet RBM, and params."""
    assert rbm.n_hidden % N == 0, (
        f"n_hidden ({rbm.n_hidden}) must be a multiple of N ({N})"
    )
    _, ha = build_hamiltonian(N, h)
    ha_jax = ha.to_jax_operator()
    nk_rbm = nk.models.RBM(
        alpha=rbm.n_hidden // N,
        use_visible_bias=True,
        use_hidden_bias=True,
    )
    params = build_netket_params(nk_rbm, rbm, N)
    return nk_rbm, params, ha_jax


def netket_local_energy(
    N: int, h: float, v: np.ndarray, rbm: FullyConnectedRBM
) -> float:
    """Local energy for a single configuration via the Jax operator."""
    nk_rbm, params, ha_jax = _make_nk_rbm_and_params(N, h, rbm)
    sigma = jnp.array(v).reshape(1, -1)
    E_loc = compute_local_energies(nk_rbm, params, ha_jax, sigma)
    return float(jnp.real(E_loc[0]))


def netket_energy_from_samples(
    N: int, h: float, samples: np.ndarray, rbm: FullyConnectedRBM
) -> nk.stats.Stats:
    """
    Estimate energy + error from a batch of samples.
    samples: (n_samples, N) in {-1, +1}
    """
    nk_rbm, params, ha_jax = _make_nk_rbm_and_params(N, h, rbm)
    sigma = jnp.array(samples)
    return estimate_energy(nk_rbm, params, ha_jax, sigma)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,h",
    [
        (4, 0.1),
        (4, 1.0),
        (4, 2.0),
        (8, 0.5),
        (8, 1.5),
        (12, 1.0),
        (16, 0.5),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_matches_netket(N: int, h: float, seed: int):
    """Per-configuration local energy must agree with NetKet to 1e-5."""
    rng = np.random.default_rng(seed)
    ising = TransverseFieldIsing1D(N, h)
    rbm = make_rbm_with_known_weights(N, seed)

    for _ in range(3):
        v = rng.choice([-1, 1], size=N).astype(float)

        E_ours = ising.local_energy(v, rbm.psi_ratio)
        E_ref = netket_local_energy(N, h, v, rbm)

        assert np.allclose(E_ours, E_ref, atol=1e-5), (
            f"N={N}, h={h}, seed={seed}, v={v}\n"
            f"  ours={E_ours:.8f}, netket={E_ref:.8f}, "
            f"diff={abs(E_ours - E_ref):.2e}"
        )


@pytest.mark.parametrize("N,h", [(4, 1.0), (8, 0.5), (8, 1.5)])
@pytest.mark.parametrize("seed", [0, 1])
def test_energy_estimate_is_finite_and_consistent(N: int, h: float, seed: int):
    """
    Energy estimated from samples should be finite and self-consistent.
    We don't assert closeness to exact energy here — the RBM is untrained.
    That belongs in a training integration test.
    """
    rng = np.random.default_rng(seed)
    ising = TransverseFieldIsing1D(N, h)
    rbm = make_rbm_with_known_weights(N, seed)
    exact = ising.exact_ground_energy()

    samples = rng.choice([-1, 1], size=(500, N)).astype(float)
    stats = netket_energy_from_samples(N, h, samples, rbm)

    E_mean = float(jnp.real(stats.mean))
    E_error = float(jnp.real(stats.error_of_mean))
    E_variance = float(jnp.real(stats.variance))

    assert np.isfinite(E_mean), f"Energy mean is not finite: {E_mean}"
    assert np.isfinite(E_error), f"Energy error is not finite: {E_error}"
    assert E_variance >= 0, f"Variance is negative: {E_variance}"
    assert E_error >= 0, f"Error of mean is negative: {E_error}"

    print(
        f"N={N}, h={h}, seed={seed}: "
        f"E={E_mean:.4f} ± {E_error:.4f}  (exact={exact:.4f})"
    )


@pytest.mark.parametrize("N,h", [(4, 1.0), (8, 1.0)])
def test_error_of_mean_decreases_with_samples(N: int, h: float):
    """
    Error of mean must shrink as ~1/sqrt(n_samples).
    Validates that estimate_energy wires up E_error = sqrt(var/n) correctly.
    """
    rbm = make_rbm_with_known_weights(N, seed=0)
    rng = np.random.default_rng(0)

    errors = []
    for n_samples in [50, 200, 800]:
        samples = rng.choice([-1, 1], size=(n_samples, N)).astype(float)
        stats = netket_energy_from_samples(N, h, samples, rbm)
        errors.append(float(jnp.real(stats.error_of_mean)))

    assert errors[0] > errors[1] > errors[2], (
        f"Error of mean not monotonically decreasing with samples: {errors}"
    )


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_netket_params_match_our_rbm(N: int, seed: int):
    """
    Explicit check that weights are transferred correctly to NetKet params.
    Isolates parameter convention bugs from energy computation bugs.
    """
    rbm = make_rbm_with_known_weights(N, seed)
    nk_rbm = nk.models.RBM(alpha=1, use_visible_bias=True, use_hidden_bias=True)
    params = build_netket_params(nk_rbm, rbm, N)  # asserts fire here if mismatch

    assert np.allclose(np.array(params["params"]["kernel"]), rbm.W, atol=1e-10)
    assert np.allclose(np.array(params["params"]["visible_bias"]), rbm.a, atol=1e-10)
    assert np.allclose(np.array(params["params"]["hidden_bias"]), rbm.b, atol=1e-10)
