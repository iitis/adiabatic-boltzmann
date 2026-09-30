# Continuing on a different machine

## UPDATE 2026-09-30 (supersedes items 1-2 of "What's actually left to do")

**Zephyr rerun is complete** (commit `b482e20e7`): all 80 Zephyr +CEM runs
(N=8/16/32/64 x seeds 0-19) exist with the bootstrap fix. Device time used
this round: 8.09 min; **total now 21.60 of 60 min** (`time.json` =
3448080.37 ms). Driver: `scripts/exper/rerun_zephyr_budget_driver.py`
(re-reads `time.json` before every run, 3.5 min/batch cap, hard stop at 57).
fig10/10b/10c regenerated.

Median rel. error, Zephyr +CEM (new): N=8 0.22%, N=16 0.21%, **N=32 7.14%,
N=64 4.23%** (Pegasus: 0.09/0.12/0.21/0.19%). N=8 fixed (was 15.8%, 11/20
stuck). N=32/64 unchanged -- the deadlock never touched them.

**Why N=32/64 Zephyr is bad (diagnosed, CPU only, no QPU used):**
- Not an estimator artifact: unbiased CPU Metropolis on the saved seed-19
  checkpoints gives true <H> = -31.58 (N=32, 7.2%) / -62.47 (N=64, 8.2%),
  matching the QPU estimate (`scripts/exper/cem_zephyr_trap_true_energy.py`).
- Not steady-state sampler bias at N=32: real Zephyr samples at iter 91 match
  true |Psi|^2 (2.90 vs 2.98 domain walls, same <log Psi^2>). At N=64 Zephyr
  is somewhat colder than |Psi|^2 (4.3 vs 6.3 walls)
  (`cem_zephyr_trap_compare_samples.py`).
- It is a **variational local minimum** (state with ~3 domain walls, |m|~0.4
  vs ~0.96 in the ground state). Zephyr runs reach ~0% error around iter 10,
  then degrade to +7% while beta_x climbs 3 -> 6.
- Synthetic annealer at hidden beta_hw, running the real Trainer + CEM rule
  (`cem_zephyr_trap_synthetic.py`, data in `results/cem_zephyr_trap/`), N=32:
  calibrated (beta_hw=1) 11/20 seeds good; **beta_hw=6 with beta_x_init=1
  (Zephyr as run) 3/20 good at ~7.1%** -- reproduces the hardware numbers
  (final beta_x ~6.1). beta_hw=6 with beta_x_init=6 behaves exactly like the
  calibrated case (7/10). cem_interval=1 does not help (3/10); a hot start
  (beta_x_init=20) is worse (1/10).
- So: (i) N=32 SR has a domain-wall trap even with perfect sampling;
  (ii) Zephyr's long too-cold calibration transient greatly increases the
  trap rate. Only mitigation found: start beta_x at its calibrated value.
- Caveat: the synthetic model is harsher than real Pegasus (at beta_hw=2.6 it
  traps 6/10, real Pegasus N=32 does not).

**Open decision (user's call):** keep the N=32/64 Zephyr numbers and report
the trap, or rerun N=32/64 Zephyr with `beta_x_init` ~6 / ~3.7 (the
converged values; ~4.2 min device time for 40 runs). Changing the protocol
for one device needs a methods justification in the paper.

**Other issues found:** `_git_sha()` records no dirty-tree marker (archived
runs claim `02899b56b` but ran uncommitted code); `history.beta_eff_cem`
stores post-update beta_x, not raw beta_hat; checkpoint filenames lack seed
and `dwave_samples/` filenames lack the cem flag (both overwrite across runs);
4/20 new Zephyr N=8 runs hit beta_max=20; Zephyr cem0 energies lie *below*
exact (sample collapse) so their `error` values are not comparable.

---

State as of commit `8f37fda54` (main). Read this before doing anything else.

## What happened this session

Found and fixed **two real bugs** in the CEM `beta_x` feedback rule
(`src/encoder.py`), both confirmed on real D-Wave hardware, not just
synthetic tests:

1. **Wrong fixed point.** The old rule blended `beta_x` and the raw CEM
   estimate `beta_hat` with a plain arithmetic mean (linear EMA). `beta_hat`
   estimates a *ratio* (`beta_eff = beta_hw/beta_x`), so any such mean has
   fixed point `beta_x* = beta_hat*`, forcing `beta_eff* = sqrt(beta_hw)`
   instead of 1. Fixed by smoothing in `log(beta_x)` space instead
   (`beta_x <- beta_x * beta_hat**alpha`, same `alpha=0.3`), plus two
   guards: reject a reading pinned at the CEM fit's search bounds, and
   reject a reading implying an implausible (>8x) single-step jump.

2. **Trust-region deadlock**, found only when rerunning the real headline
   experiment at full scale (N=8/16/32/64, both QPUs, n=20 seeds). The
   trust-region guard from bug #1's fix anchors to the *current* `beta_x`,
   which starts at an arbitrary guess (1.0). When the true correction needed
   is large and consistently exceeds the trust ratio from that guess, every
   reading gets rejected forever and `beta_x` can never move — confirmed:
   **11/20 Zephyr N=8 runs and 2/20 N=16 runs were stuck at `beta_x=1.0` for
   the entire 100-iteration run.** Fixed by skipping the trust check only
   for the first-ever accepted correction (`self._cem_bootstrapped` flag in
   `Trainer`), then checking normally after. A "clip to the trust interval"
   alternative was tried and rejected — verified it runs away to `beta_max`
   once `beta_x` has overshot, since the interval is anchored to the
   now-wrong `beta_x` (see git log of `428348b4a` for the full reasoning
   and the synthetic repro).

Both fixes are validated via exact-enumeration math, a second sampler (LSB,
different physics than Gibbs), and real D-Wave hardware (Pegasus + Zephyr),
including a pre-registered protocol (criteria written before running) that
checks the raw CEM reading clusters at `beta_eff=1` in steady state
independent of the unknown hardware `beta_hw`.

## Known open concern (not yet resolved)

After the bootstrap fix, once `beta_x` has climbed to its converged value,
subsequent *correct* steady-state readings (`beta_hat≈1`) can also fail the
trust check — they're naturally far from a large `beta_x` in ratio terms,
even though they're the correct confirmation signal. In the one case tested
so far (Zephyr N=8 seed=1, previously fully deadlocked) it froze at a
good-enough value (final error 0.35% relative) after `beta_x` stopped
updating around iteration 60. This is *better* than the deadlock, but not
rigorously shown to always land somewhere good.

**Before trusting the fix completely**, either:
- Check the real error distribution across the full Zephyr rerun (see
  below) — if errors come out comparable to Pegasus's (~0.1-0.3% median),
  the bootstrap fix is good enough as-is.
- Or investigate a more principled trust check (e.g. compare `beta_hat`
  against a short rolling history of recent readings rather than against
  `beta_x` directly) via **free GPU-only synthetic tests** — do NOT spend
  more real device time chasing this; it's fully reproducible without
  hardware (see `scripts/exper/cem_feedback_rule_test.py` /
  `cem_feedback_gpu_protocol_test.py` for the harness pattern).

## Device-time budget — READ THIS FIRST

**Total allowance: ~60 minutes of real D-Wave device time for this entire
effort.** Tracked via the project's own `time.json` (`time_ms` field,
cumulative across ALL historical usage, not just this work).

- Baseline before this session's D-Wave testing began: `2152152.199360213` ms
- Value at handoff: check `time.json` now; as of this doc being written it
  was `2962770.39` ms (i.e. **13.52 minutes used, ~46.5 minutes remaining**)
- Compute remaining budget with:
  ```python
  import json
  d = json.load(open("time.json"))
  used_min = (d["time_ms"] - 2152152.199360213) / 1000 / 60
  print(f"{used_min:.2f} min used, {60-used_min:.2f} min remaining")
  ```
- **The user explicitly asked for individual test/rerun invocations capped
  at 3-4 minutes of device time each** (not one long blocking campaign).
  Check the budget before AND after every real-hardware invocation.
  Per-100-iteration-run device time observed: Pegasus ~3.3-4.0s, Zephyr
  ~5.9-6.7s (roughly flat across N=8..64 — device time is dominated by
  fixed per-anneal overhead, not qubit count). Wall-clock per run is much
  longer (~70-95s, dominated by network/queue latency, not device time) —
  budget your session's wall-clock time accordingly, this is orthogonal to
  the device-time cap.
- D-Wave token: works (`~/.config/dwave/dwave.conf`), user fixed it
  mid-session after an initial auth failure.

## What's actually left to do

1. **Rerun Zephyr with the bootstrap fix**, in small batches respecting the
   3-4 min/invocation cap. Script: `scripts/exper/rerun_cem_fixed_headline.py`
   (resumable — skips any `(N, device, seed)` whose output file already
   exists). Currently done: only `N=8 seed=1` (the one pilot test). Still
   needed: the other 79 Zephyr seeds across N=8/16/32/64.
   ```
   python scripts/exper/rerun_cem_fixed_headline.py --sizes 8 --devices zephyr --seeds <subset>
   ```
   Pick subsets sized to fit ~3-4 min device time each (roughly 25-35 seeds
   worth at Zephyr's ~6s/run rate — but check the live budget first, and
   remember Pegasus doesn't need rerunning, it's already correct and
   complete at n=20 for all four sizes).
   Old (deadlock-affected) Zephyr results are preserved at
   `results/archive/cem_trust_deadlock_bug/`.
2. Once Zephyr is complete, re-run the same error-distribution health check
   used to originally find the deadlock bug (median/IQR relative error per
   N/device from the result JSONs — see conversation history or just
   recompute: `error` field in each `results/tfim_1d/N/dimod/DEVICE/*cem1*.json.gz`
   is `abs(final_energy - exact_energy)`, divide by `abs(exact_energy)` for
   relative). Confirm Zephyr now looks comparable to Pegasus. If not,
   revisit the "known open concern" above.
3. **Paper writing** (not started):
   - Rewrite Sec. III.C ("Fixed point of the feedback rule") replacing the
     `[TODO – Design a stable feedback rule...]` placeholder with the
     actual derivation, the two-bug fix, and validation evidence.
   - Regenerate Fig. 1 (`scripts/exper/cem_matching_demo.py`) — its own
     open TODO (β̂≈1.94 where β=1 was expected) is likely explained by the
     same class of bug; check if it resolves to ≈1 now.
   - Add a methods footnote on the small-sample DKL estimator bias found
     during validation (naive plug-in KL divergence is severely biased when
     sample count << 2^N state space — see
     `scripts/exper/cem_feedback_exact_dkl_test.py` docstring for the full
     writeup, exact numbers: 0.042 true DKL vs 2.517 naive-estimated DKL at
     N=16 with 200 samples).
   - Consider citing arXiv:2608.04564 ("Quantum annealers as programmable
     thermal machines") as corroborating motivation for Sec III (their
     pseudo-likelihood effective-temperature fitting finds the same
     qualitative phenomenon — sampling temperature is instance/protocol
     dependent, decoupled from physical device temperature — via an
     unrelated method). Frame as "consistent with", NOT "reproduced" — we
     didn't replicate their protocol.
   - Regenerate the actual Fig. 4/5/6 and Table II artifacts once Zephyr
     reruns are done (whatever script in `scripts/viz/` currently produces
     the paper's PDFs/PNGs — not yet identified/updated this session).

## Key files (this session's additions)

- `src/encoder.py` — the actual fix (`is_cem_fit_degenerate`,
  `is_cem_step_untrusted`, `_cem_bootstrapped`, the feedback update block).
  Everything else below is validation/tooling, not shipped code.
- `scripts/exper/rerun_cem_fixed_headline.py` — the real-hardware headline
  rerun harness (in-process, mirrors `scripts/main.py`'s training path,
  writes via `save_results()` so output is plot-script-compatible).
  Resumable, skip-if-exists.
- `scripts/exper/cem_feedback_rule_test.py`,
  `cem_feedback_gpu_protocol_test.py` — GPU-only, exact-`beta_hw`-controlled
  feedback-rule tests (no hardware, free to rerun).
- `scripts/exper/cem_feedback_lsb_validation.py`,
  `cem_feedback_scale_check.py` — LSB-sampler (GPU) validation at N=8 and
  N=16 respectively. Note the DKL-bias caveat docstrings in both.
- `scripts/exper/cem_feedback_dwave_validation.py`,
  `cem_feedback_dwave_protocol_test.py` — earlier, smaller real-hardware
  checks (N=8, then N=16 two-device pre-registered protocol).
- `scripts/exper/cem_feedback_exact_dkl_test.py` — the exact-enumeration
  root-cause script that found the naive-DKL bias.
- `scripts/exper/cem_headline_rerun.py` — GPU-only (LSB) n=20 replication of
  Figs 4/5's statistical design; trajectories saved to
  `plots/cem/cem_headline_rerun.npz`.
- `results/archive/cem_beta_x_ema_bug/` — old results from bug #1 (wrong
  fixed point), preserved.
- `results/archive/cem_trust_deadlock_bug/` — old Zephyr results from bug #2
  (deadlock), preserved.

## Don't re-litigate

- The core estimator (`estimate_beta_eff_cem`) was verified correct against
  exact enumeration early on — the bugs were both in the feedback rule that
  *uses* the estimate, never in the estimator itself.
- Naive small-sample DKL is NOT a reliable metric at N>~10ish (see the
  exact-vs-naive DKL bias finding above) — don't be alarmed by a DKL number
  that looks bad without checking against exact enumeration first, if N is
  small enough to compute it (N<=16 or so).
- LSB was deliberately removed from `src/sampler.py` at commit `0b8ee9f8b`;
  it's reimplemented standalone in the validation scripts on purpose, not
  reintroduced into the shipped sampler.
