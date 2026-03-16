import numpy as np
import pytest
from src.model import FullyConnectedRBM


def numerical_gradient(
    rbm: FullyConnectedRBM, v: np.ndarray, eps: float = 1e-5
) -> dict:
    """Compute numerical gradients of log_psi via central finite differences."""
    grad_a = np.zeros_like(rbm.a)
    grad_b = np.zeros_like(rbm.b)
    grad_W = np.zeros_like(rbm.W)

    for i in range(rbm.n_visible):
        rbm.a[i] += eps
        fp = rbm.log_psi(v)
        rbm.a[i] -= 2 * eps
        fm = rbm.log_psi(v)
        rbm.a[i] += eps
        grad_a[i] = (fp - fm) / (2 * eps)

    for j in range(rbm.n_hidden):
        rbm.b[j] += eps
        fp = rbm.log_psi(v)
        rbm.b[j] -= 2 * eps
        fm = rbm.log_psi(v)
        rbm.b[j] += eps
        grad_b[j] = (fp - fm) / (2 * eps)

    for i in range(rbm.n_visible):
        for j in range(rbm.n_hidden):
            rbm.W[i, j] += eps
            fp = rbm.log_psi(v)
            rbm.W[i, j] -= 2 * eps
            fm = rbm.log_psi(v)
            rbm.W[i, j] += eps
            grad_W[i, j] = (fp - fm) / (2 * eps)

    return {"a": grad_a, "b": grad_b, "W": grad_W}


@pytest.mark.parametrize("N", [4, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_log_psi(N: int, seed: int):
    """
    Analytical gradients must match central finite differences for all
    parameters (a, b, W) across multiple spin configurations.
    """
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.01, N)
    rbm.b = rng.normal(0, 0.01, N)
    rbm.W = rng.normal(0, 0.01, (N, N))

    for config_idx in range(3):
        v = rng.choice([-1, 1], size=N).astype(float)
        grad = rbm.gradient_log_psi(v)
        num = numerical_gradient(rbm, v)

        err_a = np.max(np.abs(grad["a"] - num["a"]))
        err_b = np.max(np.abs(grad["b"] - num["b"]))
        err_W = np.max(np.abs(grad["W"] - num["W"]))

        assert err_a < 1e-7, (
            f"N={N}, seed={seed}, config={config_idx}: "
            f"a gradient max error={err_a:.2e}\n"
            f"  analytical: {grad['a']}\n"
            f"  numerical:  {num['a']}"
        )
        assert err_b < 1e-7, (
            f"N={N}, seed={seed}, config={config_idx}: "
            f"b gradient max error={err_b:.2e}\n"
            f"  analytical: {grad['b']}\n"
            f"  numerical:  {num['b']}"
        )
        assert err_W < 1e-7, (
            f"N={N}, seed={seed}, config={config_idx}: "
            f"W gradient max error={err_W:.2e}\n"
            f"  analytical:\n{grad['W']}\n"
            f"  numerical:\n{num['W']}"
        )


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_log_psi_saturated_weights(N: int, seed: int):
    """
    Test gradients with large weights where tanh is saturated (close to ±1).
    This catches cancellation errors in the gradient formula.
    """
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 1.0, N)
    rbm.b = rng.normal(0, 1.0, N)
    rbm.W = rng.normal(0, 1.0, (N, N))

    v = rng.choice([-1, 1], size=N).astype(float)
    grad = rbm.gradient_log_psi(v)
    num = numerical_gradient(rbm, v)

    err_a = np.max(np.abs(grad["a"] - num["a"]))
    err_b = np.max(np.abs(grad["b"] - num["b"]))
    err_W = np.max(np.abs(grad["W"] - num["W"]))

    assert err_a < 1e-6, f"N={N}, seed={seed}: a gradient error={err_a:.2e} (saturated)"
    assert err_b < 1e-6, f"N={N}, seed={seed}: b gradient error={err_b:.2e} (saturated)"
    assert err_W < 1e-6, f"N={N}, seed={seed}: W gradient error={err_W:.2e} (saturated)"


@pytest.mark.parametrize("N", [4, 8])
def test_gradient_shapes(N: int):
    """Gradient shapes must match parameter shapes exactly."""
    rng = np.random.default_rng(0)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    v = rng.choice([-1, 1], size=N).astype(float)

    grad = rbm.gradient_log_psi(v)

    assert grad["a"].shape == (N,), f"a shape: {grad['a'].shape} != ({N},)"
    assert grad["b"].shape == (N,), f"b shape: {grad['b'].shape} != ({N},)"
    assert grad["W"].shape == (N, N), f"W shape: {grad['W'].shape} != ({N},{N})"


@pytest.mark.parametrize("N", [4, 8])
def test_gradient_does_not_mutate_weights(N: int):
    """Calling gradient_log_psi must not modify a, b, or W in place."""
    rng = np.random.default_rng(0)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.01, N)
    rbm.b = rng.normal(0, 0.01, N)
    rbm.W = rng.normal(0, 0.01, (N, N))

    a_before = rbm.a.copy()
    b_before = rbm.b.copy()
    W_before = rbm.W.copy()

    v = rng.choice([-1, 1], size=N).astype(float)
    rbm.gradient_log_psi(v)

    assert np.array_equal(rbm.a, a_before), "gradient_log_psi mutated rbm.a"
    assert np.array_equal(rbm.b, b_before), "gradient_log_psi mutated rbm.b"
    assert np.array_equal(rbm.W, W_before), "gradient_log_psi mutated rbm.W"


@pytest.mark.parametrize("N", [4, 8])
def test_gradient_does_not_mutate_input(N: int):
    """Calling gradient_log_psi must not modify the spin configuration v."""
    rng = np.random.default_rng(0)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    v = rng.choice([-1, 1], size=N).astype(float)
    v_before = v.copy()

    rbm.gradient_log_psi(v)

    assert np.array_equal(v, v_before), (
        f"gradient_log_psi mutated v: before={v_before}, after={v}"
    )
