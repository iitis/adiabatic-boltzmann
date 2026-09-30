# Handoff: Zephyr +CEM error floor at N=32/64 — explained and fixed

State as of 2026-09-30 evening (on top of `9c9fccc0f`, uncommitted). The D-Wave
project quota is exhausted. The API now rejects problems with "insufficient remaining solver
access time in project julr". This session used 501 s of device time (`time.json`):
474.8 s in the 73 completed runs below (ledger `results/cem_zephyr_protocol_qpu/ledger.jsonl`),
plus ~26 s in 19 N=8/N=16 P4 runs that were cut off by the quota and produced no results
(`*_x*.log`).

## Result

The Zephyr floor was a sampling-temperature problem, not a Zephyr defect.
Protocol **P4** fixes it without hand-picking a temperature.

Relative error of the unbiased CPU-Metropolis <H>_Psi of the final RBM; good = <1%.
Same training as the headline (h=0.5, full RBM, lr=0.08, reg=0.05, ns=200,
100 training iterations, beta_x_init=1):

| Zephyr protocol | N=16 | N=32 | N=64 |
|---|---|---|---|
| headline (CEM every 5 it, log-EMA a=0.3) | 18/20, 0.21% | 2/20, 7.14% | 4/20, 4.23% |
| P2: + 3-draw CEM calibration | – | 11/20, 0.29% | 1/6, 7.23% |
| **P4: CEM calibration + visible-PL tracking** | **7/7, 0.07%** | **19/20, 0.10%** | **19/20, 0.34%** |
| Pegasus headline (for reference) | 20/20, 0.12% | 15/20, 0.21% | 19/20, 0.18% |

For P4 against the headline Zephyr runs, Fisher p is 6e-8 at N=32 and 2e-6 at N=64.
Headline numbers use the reported final energy; the P2/P4 numbers use the true <H>_Psi.
The two agree once sampling is faithful.

## P4 protocol (no pre-chosen temperature)

1. **Calibration, 3 QPU draws.** Parameters are frozen (lr=0). The joint-(v,h) CEM
   runs every draw with a full log step (beta_x <- beta_x * beta_hat).
2. **Training.** At each iteration, fit the visible-marginal temperature of the
   draw by single-spin pseudo-likelihood. The model is p_s(v) ∝ |Psi(v)|^(2s). Update
   beta_x <- beta_x * clip(s, 1/1.5, 1.5)^0.5. The joint CEM is not used after calibration.

The calibration costs +3% device time and the PL fit costs only CPU. PL can't be used for
calibration, because at the near-zero initial couplings s is undetermined (the fit hits its bounds).

## Why the error happened

1. **Too-cold samples trap SR.** On a synthetic annealer with hidden beta_hw=6 (the real
   Trainer), beta_x crawls from 1 to 6 over ~40 iterations with the headline rule.
   The median true trajectory (6 seeds) stays at |m| ≈ 0.2 while the domain walls
   shrink to about 3. Calibrated runs instead magnetise to |m| = 0.97 by iteration ~30.
   With samples from |psi|^(2s), s>1, SR sees an already-ordered distribution. The true
   state never breaks Z2 symmetry and settles in a multi-domain local minimum (the ~7% floor).
   The too-cold samples also bias the estimate: 3% estimated error at iteration 5, against 44% true.
2. **CEM calibration alone (P2) is not enough on hardware**, for two reasons:
   - At the random-init params the CEM reading is noisy (e.g. 4.06 -> 2.56 -> 2.01).
   - The needed beta_x drifts upward during training (for example 2 -> 6 at N=32).
   Every trapped P2 N=32 run started training with beta_x ≈ 1.8-3.7.
3. **The joint CEM mis-calibrates the visible marginal, and more so as N grows.** On the
   final samples of P2 runs, s_PL is 1.05-1.6 at N=32 and 1.5-2.3 at N=64.
   The N=64 samples have half the domain walls of |Psi|^2 (3.05 vs 6.25).
   This is the "N=64 Zephyr is colder" effect noted earlier. Under P4, s_PL = 0.93-1.07 in every run,
   and the sampled walls match |Psi|^2 (1.01 vs 1.04 at N=32, 2.05 vs 2.38 at N=64).
   Steady-state beta_x under P4 is ≈6.4 (N=32) and ≈4.9 (N=64), against ≈5.2 and ≈2.9 under the CEM rule.
   Why the joint fit is off (hidden-unit chains, chain breaks, a non-Boltzmann marginal) was not
   investigated; P4 sidesteps it by controlling the quantity SR actually uses.

## Residual trap (training settings, every device)

Even with calibrated synthetic sampling, N=32 SR at lr=0.08, ns=200 traps 35-65% of seeds.
Examples: 29/60 calibrated Gibbs, 39/60 exact persistent Metropolis, 17/23 with 50 sweeps.
With ns=1000 it is 20/20 exact and 12/12 calibrated synthetic, so the trap is gradient noise.
Better synthetic equilibration hardly helps: a 10x longer anneal gives 25/39 good (`G1`),
and 5x more Metropolis sweeps gives 36/55 (`G2`). Calibrated synthetic sampling with ns=1000 gives 20/20 (`G3`).
Real Zephyr under P4 does better (19/20) than any synthetic control at ns=200. That is
**unexplained**. One idea is that the QPU's small sample-to-sample distortions act as
useful noise, but it is untested.
The P4 failures are N=32 seed 10 (+6.8%) and N=64 seed 13 (+3.9%). Both had faithful
samples (s_PL ≈ 1), i.e. this trap, not sampling.

## Other synthetic results (N=32, `results/cem_zephyr_trap_gpu/`)

These run the real Trainer and CEM with a hidden-temperature annealer or exact sampling;
use `scripts/exper/cem_zephyr_trap_gpu_summary.py`.
- beta_hw=6, headline rule: 6/40 good. Remedies: CEM a=1 36/60; full-step bootstrap 37/60;
  3-draw calibration 30/60. All of these are at the calibrated level (29/60).
- ICE noise on programmed h/J with a calibrated start: σ=0.015 gives 11/20, σ=0.03 gives 7/20.
  So it hurts at σ≈0.03, the order of Zephyr's programmed |J| (~0.02).
- The N=64 synthetic samplers are not trustworthy: exact persistent Metropolis gives 0/4 there,
  while real Pegasus gives 19/20.
- `bhw6_bxi6` is bit-identical to `bhw1_bxi1`, because the synthetic model depends only on
  beta_hw/beta_x. The earlier "beta_x_init=6 -> 7/10" was just the calibrated control.
- Earlier `results/cem_zephyr_trap/` table corrections: the hot start (bxi=20) is 0/10, not 1/10
  (the "good" seed is at -7.1%, i.e. sample collapse); bhw=2.6 is 3/10.

## Files

- `scripts/exper/cem_zephyr_protocol_qpu.py` — budget-guarded Zephyr runner (`--calib 3 --fb cem+pl --ci 1 --alpha 0.5` is P4).
  Data is in `results/cem_zephyr_protocol_qpu/{P2calib3,P4cemPL}_N*.jsonl`: per-iteration E, beta_x, s_PL,
  sample walls/|m|, final params, last-iteration samples, and true-energy scores.
- `scripts/exper/cem_zephyr_protocol_summary.py` — builds the table above.
- `scripts/exper/cem_zephyr_visible_pl.py` — the s_PL estimator, plus a check over saved runs.
- `scripts/exper/cem_zephyr_trap_gpu.py` — synthetic/exact harness (samplers, schedule and feedback knobs, trajectory snapshots),
  with `cem_zephyr_trap_cpu_launch.sh` and `cem_zephyr_trap_gpu_batch1.sh`.
- `results/cem_zephyr_trap_synthetic/` duplicates `results/cem_zephyr_trap/` and is left over from the last commit;
  the empty `.jsonl.tmp` is junk.

## Caveats

- P4 is now also in `src/encoder.Trainer`, opt-in via config `cem_calib_iters=3, beta_feedback="pl"`
  (`pl_interval=1` and `pl_alpha=0.5` by default). On the CLI it is `scripts/main.py --cem --cem-calib 3 --beta-feedback pl`.
  Result filenames get `_calib3_fbpl` appended; the defaults and existing filenames are unchanged.
  Checks: the Newton s_PL estimator (`estimate_beta_visible_pl`) matches the scipy fit on the QPU samples
  to 4 decimals, and the default path reproduces a saved synthetic run.
  Calibration device time is logged in `history["calib_sampling_time_s"]`; it is not included in
  `total_sampling_time_s`. The QPU P4 runs above came from the script wrapper, not this Trainer path.
  The Trainer path was only checked on the synthetic annealer
  (`scripts/exper/cem_zephyr_trainer_p4_synth.py`, `results/cem_zephyr_trap_gpu/T_trainerP4_*`):
  32/60 good at beta_hw=6 with median s_PL 1.000. That equals the calibrated control (29/60) and compares
  with 6/40 for the headline rule.
- The P4 QPU runs don't write the standard `results/tfim_1d/...` files or `dwave_samples/`,
  so the report plotting scripts don't pick them up yet.
- Pegasus was not rerun with P4. Its headline runs were already fine, but the report should use one protocol for both devices.
- 1000 `dwave_samples/.../pegasus/*seed9[0-4]*` files show as deleted in the working tree (they predate this session).
- Earlier caveats still hold: checkpoint and `dwave_samples/` filenames lack seed/cem;
  `beta_eff_cem` stores the post-update beta_x.
