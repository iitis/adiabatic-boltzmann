Implementation of [Quantum neural networks to simulate many-body quantum systems](https://arxiv.org/pdf/1805.05462)

Install:
- `pip install -r requirements.txt`
- `pip install -r req_torch.txt`

Run:
- `python scripts/main.py` to train the model and plot the results.

## Reproducing report figures

Commands are offline (CPU-only) unless noted. Full details in each script's `--help`/docstring.

| Figure(s) | Command |
|---|---|
| `embedding/{a,b,c,legend}.pdf` | `python scripts/viz/plot_embedding_toy.py` |
| `rbm_abstract.pdf`, `embedding_full_zephyr.pdf` | `python scripts/viz/plot_dwave_embedding.py` |
| `spin_ordering_across_qpt.pdf` | `python scripts/viz/plot_phase_transition_ordering.py --plot-only` |
| `dtv_classical_*.pdf` | `python scripts/dtv/dtv_classical_samplers.py --plot-only` |
| `dtv_beta_scale_*.pdf` | `python scripts/dtv/dtv_beta_scale.py --plot-only` |
| `dtv_autoscale_*.pdf` | `python scripts/dtv/dtv_autoscale.py --plot-only` (needs QPU to regenerate from scratch) |
| `cem_validation_*.pdf` | `python scripts/exper/cem_validation_sweep.py` → `python scripts/viz/plot_cem_validation.py --input <output>.json` |
| `cem_matching_*.pdf` | `python scripts/exper/cem_matching_demo.py` |
| `fig_convergence_median.pdf` | `python scripts/j1j2/j1j2_convergence_median.py --plot-only` |
| `sparse_graph_construction/*.pdf` | `python scripts/viz/plot_sparse_graph_construction.py` |
| `sparsity_ablation_heatmap.pdf` | `python scripts/viz/plot_sparsity_ablation_heatmap.py` |
| `sparsity_ablation_qpu_vs_classical.pdf` | `python scripts/viz/plot_sparsity_ablation_floor.py` |
| `parallel_embedding_np4_pegasus.pdf` | `python scripts/viz/plot_parallel_embedding.py --N 8 --n-parallel 4` |
| `parallel_embedding_np_seeds_qpu_time.pdf` | `python scripts/exper/parallel_embedding_experiment.py --plot-only` (needs QPU to regenerate) |
| `fig10c_tte_vs_n_self_convergence*.pdf` | see `scripts/viz/paper_figures.py` docstring — mixed classical/FPGA/QPU pipeline |

Not covered by any script (hand-made schematics): `tta_overview_*.pdf`, `anneal_schedule_plot.pdf`, `pegasus.pdf`, `zephyr.pdf`.

`scripts/validation/` and a few `scripts/viz/` files (`dashboard.py`, `plot_ite.py`, `plot_hparam_search.py`, `plot_ttc.py`) are diagnostics, not tied to a figure.
