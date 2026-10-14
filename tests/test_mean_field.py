"""
Correctness tests for the mean-field (best product state) references in src/mean_field.py.

  closed form   vs  numerical minimisation over the product-state angle
  closed form   vs  <H> of the product-state RBM, evaluated with the training code's own
                    local_energy_batch by exact enumeration (so the RBM contains it)
  mean field    is a variational upper bound on the exact ground energy

Run from the repo root:
    python -m pytest tests/test_mean_field.py -v
"""

import itertools
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

jax.config.update("jax_enable_x64", True)

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from mean_field import (
    mean_field_energy,
    product_state_energy,
    product_state_rbm_params,
    reference_levels,
)
from model import FullyConnectedRBM

CASES = [
    (TransverseFieldIsing1D, 8, h) for h in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
] + [(TransverseFieldIsing2D, 3, h) for h in (0.5, 2.0, 3.044, 5.0)]


@pytest.mark.parametrize("cls,size,h", CASES)
def test_closed_form_is_optimal(cls, size, h):
    model = cls(size, h)
    res = minimize_scalar(
        lambda al: product_state_energy(model, al),
        bounds=(0.0, np.pi / 2),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert mean_field_energy(model) == pytest.approx(res.fun, abs=1e-9)


@pytest.mark.parametrize("h", [0.5, 1.0])
def test_1d_formula(h):
    model = TransverseFieldIsing1D(16, h)
    assert mean_field_energy(model) / 16 == pytest.approx(-(1 + h**2 / 4), abs=1e-14)


@pytest.mark.parametrize("cls,size,h", CASES)
def test_rbm_contains_product_state(cls, size, h):
    model = cls(size, h)
    N = model.size
    rbm = FullyConnectedRBM(N, N, jax.random.PRNGKey(0))
    rbm.params = product_state_rbm_params(model, N)
    V = np.array(list(itertools.product([-1.0, 1.0], repeat=N)))
    a = np.asarray(rbm.a)
    logp = -V @ a  # |Psi|^2 with W = b = 0
    p = np.exp(logp - logp.max())
    p /= p.sum()
    E = float(np.sum(p * np.asarray(model.local_energy_batch(V, rbm))))
    assert E == pytest.approx(mean_field_energy(model), abs=1e-9)


@pytest.mark.parametrize("h", [0.5, 1.0, 1.5])
def test_variational_bound(h):
    lv = reference_levels(TransverseFieldIsing1D(16, h))
    assert lv["exact"] <= lv["mean_field"] <= lv["all_up"] + 1e-12
    assert lv["mean_field"] <= lv["x_polarized"] + 1e-12
