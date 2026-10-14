"""
Mean-field (best product state) reference levels for the transverse-field Ising models.

A product state |alpha> = prod_i (cos(alpha/2)|+1> + sin(alpha/2)|-1>) has
<sz_i> = cos(alpha) and <sx_i> = sin(alpha). For H = -J sum_<ij> sz_i sz_j - h sum_i sx_i
on a lattice with coordination number z (z/2 bonds per site, periodic BC):

    E(alpha)/N = -(z/2) J cos^2(alpha) - h sin(alpha)

minimised at sin(alpha) = h/(zJ) for h <= zJ, giving E_MF/N = -(z/2) J - h^2/(2 z J),
and at alpha = pi/2 (x-polarised) for h >= zJ, giving E_MF/N = -h.
  1D chain  (z=2): E_MF/N = -(1 + h^2/4)
  2D square (z=4): E_MF/N = -(2 + h^2/8)

The RBM ansatz of model.py, Psi(v) = e^(-a.v/2) prod_j [2cosh(b_j + W_j.v)]^(1/2), contains
this state exactly: W = 0 and a_i = -artanh(cos(alpha)), so that <v_i> = tanh(-a_i) = cos(alpha).
Of the two degenerate mean-field states (+-cos(alpha)) this returns the +1-magnetised one.

All energies are totals (not per spin), in the same convention as exact_ground_energy().
"""

import numpy as np

from ising import TransverseFieldIsing1D, TransverseFieldIsing2D

J = 1.0  # both TFIM classes hard-code ferromagnetic J = 1


def coordination(model) -> int:
    if isinstance(model, TransverseFieldIsing2D):
        return 4
    if isinstance(model, TransverseFieldIsing1D):
        return 2
    raise NotImplementedError(
        f"No product-state reference for {model.__class__.__name__} (TFIM 1D/2D only)."
    )


def mean_field_angle(model) -> float:
    """Optimal angle alpha of the uniform product state (alpha = 0: all +1 along z)."""
    z = coordination(model)
    return float(np.arcsin(min(model.h / (z * J), 1.0)))


def product_state_energy(model, alpha: float) -> float:
    """Total energy of the uniform product state at angle alpha."""
    z = coordination(model)
    return model.size * (-(z / 2) * J * np.cos(alpha) ** 2 - model.h * np.sin(alpha))


def mean_field_energy(model) -> float:
    """Total energy of the best uniform product state."""
    return float(product_state_energy(model, mean_field_angle(model)))


def reference_levels(model) -> dict:
    """Exact ground state and the simple product-state levels, as total energies."""
    return {
        "exact": float(model.exact_ground_energy()),
        "mean_field": mean_field_energy(model),
        "all_up": float(product_state_energy(model, 0.0)),  # classical ferromagnet
        "x_polarized": float(product_state_energy(model, np.pi / 2)),  # RBM at a = W = 0
    }


def product_state_rbm_params(model, n_hidden: int):
    """RBMParams representing the mean-field state exactly (W = 0, b = 0)."""
    import jax.numpy as jnp

    from model import RBMParams

    cos_a = np.cos(mean_field_angle(model))
    a0 = -np.arctanh(cos_a) if cos_a < 1.0 else -np.inf  # h = 0 has no finite bias
    N = model.size
    return RBMParams(
        a=jnp.full(N, a0), b=jnp.zeros(n_hidden), W=jnp.zeros((N, n_hidden))
    )


def sample_product_state(model, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Exact i.i.d. samples of |Psi_MF|^2 (independent spins with <v_i> = cos(alpha))."""
    p_up = 0.5 * (1.0 + np.cos(mean_field_angle(model)))
    return np.where(rng.random((n_samples, model.size)) < p_up, 1.0, -1.0)
