# Reproducing the report figures

Every command below was actually run against this checkout during the cleanup that produced
this layout (see git log). "Offline" means CPU-only, no external credentials needed. "QPU/FPGA"
means the from-scratch data-generation step needs D-Wave Ocean SDK credentials, VeloxQ SDK
credentials (`src/velox_api_config.py` + `src/velox_token.txt`, not committed), or the FPGA
itself — but the committed cache/results tree lets you re-plot without any of that via
`--plot-only` (noted per figure).

Scripts are grouped by topic folder; each kept script's own docstring/`--help` has more detail.

## `scripts/dtv/` — D_TV / β_eff validation against exact |Ψ|²

| Figure(s) in report.tex | Command | Mode |
|---|---|---|
| `dtv_classical_N8_M8.pdf`, `dtv_classical_N8_M8_dist.pdf` | `python scripts/dtv/dtv_classical_samplers.py --plot-only` | Offline |
| `dtv_beta_scale_dtv_N8.pdf`, `dtv_beta_scale_beta_N8.pdf` | `python scripts/dtv/dtv_beta_scale.py --plot-only` | Offline |
| `dtv_autoscale_dtv_N8.pdf`, `dtv_autoscale_beta_N8.pdf` | `python scripts/dtv/dtv_autoscale.py --plot-only` | QPU to regenerate from scratch; `--plot-only` replots from the committed `plots/dtv_autoscale/dtv_autoscale_N8_h1.0.json` |

Drop `--plot-only` to retrain/resample from scratch (all offline; `dtv_autoscale.py` additionally
issues live D-Wave calls unless `--plot-only` is given).

## `scripts/viz/` — plotting (figure-critical)

| Figure(s) | Command | Mode |
|---|---|---|
| `embedding/a.pdf`, `b.pdf`, `c.pdf`, `legend.pdf` | `python scripts/viz/plot_embedding_toy.py` | Offline, synthetic toy graph |
| `rbm_abstract.pdf`, `embedding_full_zephyr.pdf` | `python scripts/viz/plot_dwave_embedding.py` | Offline (dwave_networkx + minorminer, no live QPU) |
| `spin_ordering_across_qpt.pdf` | `python scripts/viz/plot_phase_transition_ordering.py --plot-only` | Offline; replots from `plots/phase_transitions/tfim_cache_N16_nh16_it300_ns500_s42.npz`. Drop `--plot-only` to retrain (offline, ~minutes). |
| `sparse_graph_construction/a.pdf`, `b.pdf`, `c.pdf`, `legend.pdf` | `python scripts/viz/plot_sparse_graph_construction.py` | Offline (reads committed `embeddings/*_live.json` hardware-graph snapshots) |
| `sparsity/sparsity_ablation_heatmap.pdf` | `python scripts/viz/plot_sparsity_ablation_heatmap.py` | Offline; reads `plots/sparsity/cache_sparsity_ablation.json` |
| `sparsity/sparsity_ablation_qpu_vs_classical.pdf` | `python scripts/viz/plot_sparsity_ablation_floor.py` | Offline; reads 5 caches under `plots/sparsity/` (see below) |
| `parallel_embedding/parallel_embedding_np4_pegasus.pdf` | `python scripts/viz/plot_parallel_embedding.py --N 8 --n-parallel 4` | Offline, synthetic (dwave_networkx + minorminer) |
| `fig10c_tte_vs_n_self_convergence_*.pdf` | see `scripts/viz/paper_figures.py`'s `fig10c_tte_vs_n_self_convergence()` (called directly, not via `__main__` — see below) | Mixed, see below |

`scripts/viz/plot_style.py` is shared plotting infrastructure imported by nearly every script above
— do not move it, several scripts hardcode `sys.path.insert(..., "scripts/viz")` to find it.

Also kept (validation, not tied to a report figure): `dashboard.py` (Streamlit results browser —
`cd src && streamlit run ../scripts/viz/dashboard.py`), `plot_ite.py` and `plot_ttc.py` (ITE/TTC
scaling plots over `results/`; `plot_ite.py` imports `plot_ttc.py`, keep both together),
`plot_hparam_search.py` (single Optuna-run diagnostics).

### `fig10c_tte_vs_n_self_convergence` in detail

This is the most complex figure: three panels, three independent data pipelines, none of which
is a single "run this to regenerate" script:

- **Classical MCMC / LSB panel**: data from `python scripts/exper/mcmc_matched_sweep.py` (offline,
  CPU-only; results land under `results/tfim_1d/{N}/custom/{method}/`).
- **FPGA/VeloxQ panel**: data from `scripts/fpga/run_fpga_best.py`, driven by
  `scripts/fpga/run_n{64,128,64_128}_sweep.sh`, which read existing Optuna trials from
  `scripts/hparam/hparam_veloxq_tfim_n128.py`. Requires VeloxQ SDK credentials + FPGA access —
  not reproducible from scratch without that hardware; the committed `results/sweeps100/` and
  `results/archive/*veloxq*` trees are the only fallback.
- **D-Wave QPU panel**: data from `python scripts/exper/dwave_matched_sweep.py`. Requires D-Wave
  Ocean SDK credentials + live QPU access + burns metered QPU time; committed
  `results/tfim_1d/*/dimod/` is the fallback.

Once the data exists (from a fresh run or the committed `results/` tree), regenerate just this
figure with:
```python
import sys; sys.path.insert(0, "scripts/viz")
import paper_figures
paper_figures.fig10c_tte_vs_n_self_convergence(cv_threshold=0.05, epsilon=0.1)
```
Do **not** run `python scripts/viz/paper_figures.py` directly for this — its `__main__` block
regenerates ~10 other supervisor-facing figures (fig1-fig9, fig11) that aren't in the report.

`scripts/exper/mcmc_calibration.py` is kept alongside as validation: it justifies the SA
sweep-count choice embedded in `fig10c`'s own code comments (not itself a figure generator).

## `scripts/exper/` — data generation + CEM figures

| Figure(s) | Command | Mode |
|---|---|---|
| `cem_validation_calibration.pdf`, `cem_validation_bias.pdf` | `python scripts/exper/cem_validation_sweep.py --smoke-test` (or without the flag for the full sweep), then `python scripts/viz/plot_cem_validation.py --input plots/cem_validation/<output>.json` | Offline |
| `cem_matching_candidates.pdf`, `cem_matching_objective.pdf` | `python scripts/exper/cem_matching_demo.py` | Offline, self-contained generate+plot |
| `sparsity_ablation_*` data | `python scripts/exper/exact_ansatz_floor.py` (exact-enumeration floor), `python scripts/exper/sparsity_ablation_classical_baselines.py --method {simulated_annealing,gibbs}` (classical baselines; imports `_make_pruned_rbm` from `plot_sparsity_impact.py`) | Offline; both idempotent — skip cells already in their `plots/sparsity/cache_*.json` |
| `parallel_embedding_np_seeds_qpu_time.pdf` | `python scripts/exper/parallel_embedding_experiment.py --plot-only` | QPU to regenerate from scratch; `--plot-only` replots from committed `cache_parallel_embedding_np_seeds.json` |

Also kept (validation, referenced by `paper_figures.py`'s own comments): `embedding_algo_comparison.py`
(`--plot-only` supported), `parallel_embedding_bench.py` (`--plot-only` supported; Wilson-CI helper).

## `scripts/fpga/`, `scripts/hparam/` — FPGA/VeloxQ sweeps (need hardware/credentials)

`run_fpga_best.py` + `run_n{64,128,64_128}_sweep.sh` feed `fig10c`'s FPGA panel (see above).
`hparam_optuna.py` / `hparam_veloxq_tfim_n128.py` / `hparam_veloxq_tfim.py` are the Optuna
hyperparameter search these sweeps read from; `hparam_veloxq_tfim.py` (no `_n128` suffix) is kept
for provenance — it's the one-time producer of the pre-existing N=64 study, not re-run by the
current pipeline.

## `scripts/j1j2/`, `scripts/ite/` — J1-J2 Heisenberg convergence

| Figure | Command | Mode |
|---|---|---|
| `fig_convergence_median.pdf` | `python scripts/j1j2/j1j2_convergence_median.py --plot-only` | Offline; reads the `results/heisenberg_j1j2_1d/`, `results/hparam_search/`, `results/ite/` trees. Drop `--plot-only` to run missing seeds via `scripts/ite/ite_run.py` (offline, CPU). |

## `scripts/validation/` — standalone diagnostics, not tied to a figure

- `fill_reference_energies.py` — populates `src/reference_energies.json`, the exact-energy cache
  every kept script's Ising-model construction reads from internally.
- `measure_dwave_times.py` — QPU access-time characterization diagnostic.

## Not covered by any script

`figures/tta_overview_1.pdf`, `tta_overview_3.pdf`, `anneal_schedule_plot.pdf`, `pegasus.pdf`,
`zephyr.pdf` (in the report repo) have no generating script anywhere — confirmed via full git
history search of both repos. They appear to be hand-made schematic figures committed directly
as finished PDFs.

## `main.py`

`scripts/main.py` is the general single-run training entry point (see repo-root `CLAUDE.md`) —
not tied to any specific report figure, kept as the reference implementation every sweep script
above mirrors.
