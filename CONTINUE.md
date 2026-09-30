# Handoff: Zephyr +CEM error floor at N=32/64

State as of commit `f17b2acef` (main).

## Issue

With the fixed CEM feedback rule, Zephyr (Advantage2) +CEM is good at small N
but stuck at a consistent error floor at N=32/64. Pegasus is fine at every N.

Median relative error, 20 seeds each (h=0.5, lr=0.08, reg=0.05, ns=200, 100 iter):

| N  | Zephyr +CEM | Pegasus +CEM |
|----|-------------|--------------|
| 8  | 0.22%       | 0.09%        |
| 16 | 0.21%       | 0.12%        |
| 32 | 7.14%       | 0.21%        |
| 64 | 4.23%       | 0.19%        |

At N=32, 18/20 seeds sit at about 7% (IQR 7.05-7.40%).

## What is established

- **The error is real.** Unbiased CPU Metropolis on the saved seed-19
  checkpoints (iter 90) gives true <H> = -31.58 at N=32 (7.2%) and -62.47 at
  N=64 (8.2%), matching the QPU estimates
  (`scripts/exper/cem_zephyr_trap_true_energy.py`).
- **At N=32 the samples are faithful in steady state.** Zephyr samples at iter
  91 match true |Psi|^2: 2.90 vs 2.98 domain walls, and <log Psi^2> is equal.
  At N=64, Zephyr is colder than |Psi|^2 (4.3 vs 6.3 walls)
  (`scripts/exper/cem_zephyr_trap_compare_samples.py`).
- **The trained state is a domain-wall local minimum.** It has about 3 walls
  and |m| ~0.4, against |m| ~0.96 in the ground state.
- **It forms during calibration.** Zephyr runs reach about 0% error around
  iter 10 (estimate still biased by sample collapse), then degrade to +7% as
  beta_x climbs from 3 to 6 over about 40 iterations. Pegasus only needs
  beta_x ~2.6 and does not degrade.
- **Zephyr programs small couplings.** At convergence the rms programmed
  coupling |W|/beta_x is about 0.02, against about 0.04 on Pegasus. In the
  N=32 checkpoint, half of the couplings are below 0.01, which is around the
  ICE level. N=16 Zephyr is also at about 0.02 and works fine.
- **A synthetic annealer reproduces it.** This runs the real Trainer and CEM
  rule, sampling at a hidden beta_hw (`scripts/exper/cem_zephyr_trap_synthetic.py`,
  data in `results/cem_zephyr_trap/`). N=32, share of seeds below 1% error:

  | condition                          | good (<1%) |
  |------------------------------------|------------|
  | beta_hw=1 (calibrated)             | 11/20      |
  | beta_hw=6, beta_x_init=1 (Zephyr)  | 3/20 (~7.1%, beta_x ~6.1) |
  | beta_hw=6, beta_x_init=6           | 7/10 (same as calibrated) |
  | beta_hw=6, cem_interval=1          | 3/10       |
  | beta_hw=6, beta_x_init=20 (hot)    | 1/10       |
  | beta_hw=2.6, beta_x_init=1         | 4/10 (real Pegasus does better) |

  So N=32 SR has a domain-wall trap even with perfect sampling, and the long
  too-cold calibration transient (large beta_hw, beta_x starting at 1)
  greatly raises the trap rate. The synthetic model is harsher than real
  Pegasus, so it is not a complete model of the hardware.

## Related data caveats

- Zephyr/Pegasus cem0 runs report energies **below** exact (sample collapse,
  unique-sample ratio about 0.005). Their `error = |E - E_exact|` is not
  comparable to the cem1 numbers.
- `history.beta_eff_cem` stores the post-update beta_x, not the raw beta_hat.
  The raw value can be recovered via beta_hat = (bx[t+5]/bx[t])**(1/0.3).
- Checkpoint filenames lack the seed, and `dwave_samples/` filenames lack the
  cem flag, so both are overwritten across runs. Only seed 19 checkpoints
  exist for N=32/64 Zephyr.
- `git_sha` in results has no dirty-tree marker. Archived runs claim
  `02899b56b` but ran uncommitted code.
- 4/20 Zephyr N=8 runs hit beta_max=20.

## D-Wave budget

The allowance is 60 min total from baseline `2152152.199360213` ms in the root
`time.json`; 21.60 min is used (time_ms 3448080.37). `src/time.json` is an
unused 0. `DimodSampler` opens `time.json` by relative path and creates it at 0
if missing, so always run from the repo root. Use
`scripts/exper/rerun_zephyr_budget_driver.py`, which re-reads the budget
before every run. Keep each invocation to 3-4 min of device time.

Old results are archived in `results/archive/cem_trust_deadlock_bug/` and
`results/archive/cem_beta_x_ema_bug/`.
