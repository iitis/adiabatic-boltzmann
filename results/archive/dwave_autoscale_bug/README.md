# INVALID — generated with D-Wave `auto_scale=True`

Every result file under this directory was sampled through
`DimodSampler` (`src/sampler.py`) while `auto_scale` defaulted to `True`
(or, for `reverse_annealing()`, was hardcoded `True` with no way to turn
it off) in `dwave()`, `dwave_parallel()`, `fast_anneal()`, and
`reverse_annealing()`.

## Why this invalidates the data

`rbm_to_ising`/`dbm_to_ising` rescale every h/J coefficient by `beta_x`
before submission, intended to fix the effective inverse temperature the
QPU anneals at. D-Wave's SAPI `auto_scale=True` renormalizes h/J again on
the solver side to fill the hardware's coefficient range, silently
undoing that rescale. So every one of these runs sampled at an unknown,
solver-rescaled effective temperature instead of the intended one —
`beta_x` had no real effect.

This is not theoretical: `scripts/output/dtv_autoscale/dtv_autoscale_N8_h1.0.json`
(N=8, h=1.0, exact enumeration) shows D_TV flat at ~63% across
`beta_x` ∈ {1.5, 2, 3, 5, 8} with `auto_scale=True`, vs. a real
beta_x-dependent curve down to ~29.5% with it off.

## What's included

Every `dimod/{pegasus,pegasus_fast,pegasus_mh,pegasus_ra,zephyr,zephyr_fast,zephyr_mh,zephyr_ra}`
result subtree that existed under `results/` prior to the fix — `tfim_1d`,
`tfim_2d`, `lr_tfim_1d`, `heisenberg_j1j2_1d`, `j1j2_1d`, and one
`hparam_search` sweep. `pegasus_mh`/`zephyr_mh` route D-Wave samples
through an additional MH accept/reject filter (`sampler.py`'s
`DWaveMHSampler`) — included here because the underlying QPU proposals
still came from the same buggy `auto_scale=True` calls, though whether
the MH correction itself was also compromised was not separately
investigated.

No result file recorded `auto_scale` or `beta_x` in its stored `config`,
so which exact runs used which settings can't be recovered — every QPU
run in the repo is assumed affected, since no production script ever
passed `auto_scale=False`.

## Fix

`src/sampler.py` now hardcodes `auto_scale=False` in all four D-Wave
sampling paths — no config key, can't regress.

## Status

Archived, not deleted, in case a before/after comparison is useful.
Regeneration with the fix (and a calibrated `beta_x`) is pending.
