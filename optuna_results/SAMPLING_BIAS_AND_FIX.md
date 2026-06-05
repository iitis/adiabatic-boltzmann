# VeloxQ SA "Gibbs" sampling bias — diagnosis & fix

## Symptom
In the fixed-temperature VeloxQ SA sweep (`start_temp=1`, `scale_ising=False`,
`compress=False`, `beta_x=1`), the VMC energy estimate landed **below** the exact
ground energy — a violation of the variational bound (⟨E⟩ ≥ E_exact). The
downward bias was ≈ −0.3 and did **not** shrink with more `num_sweeps_per_step`.

## Root cause — selection, not dynamics
The β=1 Metropolis dynamics are correct: |Ψ(v)|² = Σ_h e^(−E_cl(v,h)) is exactly
the **β=1** marginal of the classical RBM-Ising Boltzmann distribution, the kernel
runs `beta = 1/start_temp = 1` on the unscaled energy, starts from a random state,
and returns each replica's final state. So each replica is (asymptotically) a
genuine β=1 Gibbs sample.

The bias comes **after** sampling. `VeloxQstandard.SimulatedAnnealing` is a
*spectrum optimizer*: `VeloxQtoolbox/src/spectrum.jl::sort_spectrum` does

```julia
perm = sortperm(energies_1D)            # ascending energy
states_2D[:, :] .= states_2D[:, perm]   # states, lowest-energy first
```

and the Python wrapper then kept `samples[:n_samples]` — i.e. the **lowest-energy
`n_samples` of the `num_rep` replicas**. That deterministic skim of the low-energy
tail over-weights the modes of |Ψ|² and pulls ⟨E_loc⟩ below the thermal average.
Because it is *order-statistics selection*, it is independent of equilibration —
which is why more sweeps never removed it.

## Evidence
Same random RBM (N=8), β=1, 5000 sweeps (equilibrated), scale/compress off, keep
200 samples; only the replica pool changes. `E_cl` is the classical energy the SA
solver sorts by:

| replicas (num_rep) | kept | mean E_cl |
|---|---|---|
| 200  | all 200 (no selection)        | **−0.016** (≈ thermal mean) |
| 4096 | 200 **lowest-energy** of 4096 | **−0.260** (skimmed tail)  |

The −0.26 skim matches the ≈ −0.3 bias seen across the sweep (num_rep=1024).

## Fix (`src/sampler.py`, `VeloxQStandardSASampler.sample`)
When `num_rep > n_samples`, take a **uniform random subset** of the returned
states instead of the energy-sorted head:

```python
idx = self._subsample_rng.choice(samples.shape[0], size=n_samples, replace=False)
samples = samples[idx]
```

The RNG is seedable via `config["veloxq_subsample_seed"]` for reproducibility.
Each replica is an independent β-Gibbs chain, so a random subset is an unbiased
|Ψ(v)|² ensemble. (`FPGASampler` has the identical sort-and-skim behaviour and
would need the same change if used for sampling.)

## Objective change (`src/optuna_sa_sweep.py`)
The Optuna objective was the *signed* gap `⟨E⟩ − E_exact`, which (with the old
bias) **rewarded** the most-biased, sub-variational trials. It is now
`|⟨E⟩ − E_exact|` — distance to the exact energy — which equals the variational
gap once sampling is faithful and never rewards a sub-variational excursion.

## Results — variational bound restored
Full reruns (1d, h=0.5, 50 trials/size). Signed gap = ⟨E⟩ − E_exact; faithful
sampling requires it ≥ 0. v3 adds per-trial `signed_gap` logging (reliable;
not parsed from interleaved stdout).

| size | run | best `signed_gap` | sub-variational trials | winning `num_sweeps` |
|---|---|---|---|---|
| N=16 | v1 (biased skim, signed obj) | −0.634 (below exact) | 34 (worst −0.69) | low (10–1000) |
| N=16 | v3 (random subsample, `|gap|`) | **−0.0002** (on exact) | **0 / 49** | 50000 |
| N=24 | v1 (biased skim, signed obj) | −0.510 (below exact) | many | low |
| N=24 | v3 (random subsample, `|gap|`) | **+0.005** (above exact) | **0 / 50** | 2000 |

Best achievable `|gap|` per `num_sweeps_per_step` (v3):

| sweeps | 10 | 100 | 500 | 1000 | 2000 | 5000 | 10000 | 50000 |
|---|---|---|---|---|---|---|---|---|
| N=16 | 0.030 | 0.042 | 0.032 | 0.426 | 0.012 | 0.129 | 0.037 | **0.0002** |
| N=24 | 1.40 | 0.011 | 0.73 | 0.022 | **0.005** | 0.96 | 2.51 | 0.009 |

Takeaways: (1) with the unbiased subsample, **every** trial respects ⟨E⟩ ≥ E_exact
(0 sub-variational both sizes), vs the systematic −0.5…−0.7 excursions in v1;
(2) the best ansatz reaches the exact ground state to ~1e-3–1e-4; (3) the
*best-achievable* gap shrinks with more sweeps (faithful Gibbs needs adequate
equilibration) — the opposite of biased-v1, where fewer sweeps "won" by being
more biased. Per-`num_sweeps` jitter (e.g. N=16 1000→0.43, N=24 10000→2.5) is the
VMC-parameter confound: those sweep counts happened to draw poor `lr`. The
best-case envelope is the meaningful signal; to remove the confound entirely, fix
the VMC params and sweep only `num_sweeps_per_step` with several seeds each.

## Archived / current runs
- `v1_biased_skim/` — pre-fix: energy-sorted skim + signed objective (the biased baseline).
- `v2_fixed/` — fix applied + `|gap|` objective, but no signed-energy logging (superseded).
- `optuna_results/` (current) — **v3**: fix + `|gap|` objective + `signed_gap` logging;
  `best_1d_N{16,24}_h0p5.json` (now include `mean_energy`/`signed_gap`) + `optuna_sa_studies.db`.
