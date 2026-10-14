# Handoff: DMRG baseline and the methods-comparison notebook

State as of 2026-10-14 evening. Nothing is committed: every file below is new or untracked, on top of `9bd5ccf8c`.

## Question
Which method best gives the ground state, and ground-state samples, of the models in this repo? Where, if
anywhere, is MCMC-trained VMC (RBM + SR) competitive? The user excluded the free-fermion solution as a
*method*, because it gives no samples. It is still used as the exact reference E0.

## Answer for the 1D TFIM (periodic, N = 16–128, h = 0.5–2)
DMRG beats every Gibbs-trained RBM at every (N, h), in both error and wall time, and it also gives exact
i.i.d. samples. Below, error is per spin and times are wall-clock on the loaded shared machine (±2×).

| h = 1 | product state | Gibbs 300 steps | Gibbs 3000 steps | DMRG χ=8 | DMRG χ=16 | DMRG χ=32 |
|---|---|---|---|---|---|---|
| N=16 | 2.5e-2 | 3.4e-4 (49 s) | 1.1e-4 (366 s) | 5.5e-5 (92 s) | 8.2e-8 (39 s) | 2.1e-11 (35 s) |
| N=32 | 2.4e-2 | 3.2e-3 (41 s) | 2.2e-4 (352 s) | 2.0e-4 (72 s) | 2.6e-6 (52 s) | 1.2e-8 (52 s) |
| N=64 | 2.3e-2 | 2.0e-2 (59 s) | 2.4e-3 (573 s) | 2.1e-4 (122 s) | 1.5e-5 (92 s) | 2.5e-7 (83 s) |
| N=128 | 2.3e-2 | (not evaluated) | (not evaluated) | 9.6e-5 (270 s) | 3.5e-5 (184 s) | 1.6e-6 (178 s) |

- **Energy-equivalent χ.** This is the smallest χ whose DMRG energy is at least as good as the RBM's. It
  compares energies only and is not an entanglement measure. After 3000 steps it is 4–8 at N=16, 2–16 at
  N=32 and 2–4 at N=64. For N ≥ 64 at h ≤ 0.9, Gibbs at 300 steps is *worse* than the product state:
  34× at N=64 and 93× at N=128, both at h=0.5.
- **RBM error budget.** The arms share the ansatz and differ only in how the gradient is estimated (N ≤ 16,
  SR step 300).
  - Monte Carlo noise from 200 i.i.d. samples costs 4–500× over exact SR. The cost is largest deep in
    either phase.
  - Gibbs instead of i.i.d. samples costs only 0.7–2×.
  - So the sampler is not the bottleneck. Samples per step and training length are.
- **Scaling with N.** The product state and fixed-χ DMRG are roughly flat per spin. Gibbs at a fixed step
  budget degrades steeply with N.
- **Where VMC is competitive (literature, not tested here).** VMC wins mainly for frustrated 2D models
  (J1-J2 square near J2/J1 = 0.5), where QMC has a sign problem and DMRG cost grows exponentially with
  width. Even there it takes deep networks (ViT, Commun. Phys. 2024; deep ResNet/MinSR); a shallow RBM with
  Gibbs is not enough. For every model in this repo, VMC is not competitive:
  - all 1D models: DMRG;
  - 2D TFIM at L ≤ 4: exact diagonalisation;
  - sign-free 2D at large L: QMC.

  Triangular/kagome claims are unverified.

## What was done
- **`scripts/validation/dmrg_baseline.py`.** Uses TeNPy 1.1.1, installed into conda env `test` with
  `pip install physics-tenpy`.
  - Two-site DMRG on the periodic TFIM (σ operators, same convention as `src/ising.py`).
  - Takes the best of three starts: all up, all +x, and the previous χ's state. Single-start small-χ
    runs got stuck in start-dependent local minima.
  - The energy recorded is ⟨H⟩ of the returned MPS, not the engine's energy, which is computed before the
    final truncation.
  - Samples are exact and i.i.d., with a random global spin flip because DMRG breaks Z2 for h < 1.
  - χ = 1 is not run: truncated DMRG misses the best product state. The analytic `src/mean_field.py`
    value is used instead.
- **Output files.**
  - Full grid in `results/dmrg_baseline/tfim1d.jsonl` (N 16/32/64/128 × h 0.5/0.9/1/1.1/1.5/2 × χ 2–32,
    120 rows; logs in `results/dmrg_baseline/logs/`).
  - The first single-start run is archived as `tfim1d_v1_singlestart.jsonl` (logs in `logs/v1/`).
    **It is unreliable at small χ, so don't use it.**
  - The current jsonl rows were sampled *before* the Z2 flip was added. Only Z2-even statistics
    (zz, |m|) are stored, so this doesn't matter.
- **`notebooks/mf_vs_h.ipynb`** was rebuilt by `scripts/viz/make_mf_vs_h_notebook.py --execute`, which
  uses an IPC kernel and opens no port or server. It is now organised around the methods:
  - 0: overview table;
  - 1: the methods;
  - 2: comparison — error vs h, accuracy vs wall time (marker size = SR step), summary table at h=1, and
    energy-equivalent χ;
  - 3: impact of each ingredient — RBM error budget, N, χ, training length, variants;
  - 4: validation of the DMRG samples (120 z-scores, 95% within 2);
  - Appendix A: the earlier product-state-vs-RBM analyses for the reviewer question.

  The pre-restructure builder is in the session scratchpad and will not survive. The git-untracked
  original is gone, so the restructure can't be reverted from git.
- **Two independent audits (subagents).**
  - Code: the Hamiltonian matches exact diagonalisation and E0 to 1e-15. The sampler passes a χ² test
    against full |ψ|² at N=6–8.
  - Found and fixed: start-dependent minima, the wrong energy at χ=1, the notebook mixing seeds/steps
    (now a fixed step with seed counts n), the global ">max χ" label, and Z2-broken samples.
  - Conclusions: corrected the earlier quoted numbers (Gibbs N=64 at 300 steps takes 59 s, not 600 s).
    "Orders of magnitude" only holds against χ ≥ 16.
  - DMRG errors of ~1e-14 per spin are the double-precision floor and are shaded in the plots.

## Still to do
1. **Rerun two crashed phase-2 jobs.**
   - `--arm gibbs --N 128 --hs 0.9 --seeds 1-1` (segfault)
   - `--arm gibbs --N 64 --hs 0.9 --seeds 0-2` (abort), both `--steps 100,300,1000,3000`
   - Logs are in `results/mf_vs_h/logs/job__arm_gibbs_N_{128,64}_hs_0.9_*.log`.
   - The runs are resumable, so rerun with the same arguments.
2. **Phase-2 deferred evaluations.** The launcher reported "training done; waiting for evaluators" at
   16:47. Once the `*.eval.jsonl` files are complete, rebuild the notebook. N=128 at h=1 still has no
   evaluated Gibbs checkpoint, so that row of the summary table is empty.
3. **Rebuild the notebook** after (1)–(2):
   `OMP_NUM_THREADS=2 nice /home/phanussek/miniconda3/envs/test/bin/python scripts/viz/make_mf_vs_h_notebook.py --execute`
   It takes about 3 min. Don't start Jupyter servers; the user runs their own on 127.0.0.1:8888.
   - The old `notebooks/mf_vs_h.html` export is stale.
4. **Optional extensions, not started.**
   - **Tighter DMRG timings.** Current timings are single-thread pure-Python TeNPy on a loaded machine.
     Rerun the timings on an idle core, or with ITensor, to tighten the wall-time comparison. VMC wall_s
     includes JIT and comes from mixed hardware.
   - **Stronger sample check.** Compare DMRG samples with exact |ψ0|² at N=16 (total variation distance),
     and with RBM samples via the same statistics.
   - **DMRG for the other repo models** (1D XXZ, 1D J1-J2 XXZ): the script currently does the TFIM only.
   - **A test where VMC could win.** Run 2D J1-J2 (≥ 6×6) against DMRG with growing χ. This needs a new
     model in `src/ising.py` and is the only setting where RBM-VMC could plausibly win.
   - **Fair VMC.** Tune VMC (more samples per step, longer training, M = 2N) before any claim that it is
     "not competitive" at fixed cost. The error budget shows that samples per step dominate.
5. **Commit** when the user asks. Candidates are `scripts/validation/dmrg_baseline.py`,
   `scripts/viz/make_mf_vs_h_notebook.py`, `results/dmrg_baseline/`, `notebooks/mf_vs_h.ipynb`, this
   file, and the other untracked mf_vs_h files from `git status`.
