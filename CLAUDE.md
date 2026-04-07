# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of Variational Monte Carlo (VMC) with Restricted Boltzmann Machines (RBM) to find ground states of the Transverse Field Ising Model (TFIM). Based on [Gardas et al. (arXiv:1805.05462)](https://arxiv.org/pdf/1805.05462). The key idea is to use an RBM as a variational wave function ansatz and optimize it using Stochastic Reconfiguration (SR).

## Commands

All scripts must be run from the `src/` directory (imports are local, not packaged):

```bash
cd src

# Install dependencies
pip install -r ../requirements.txt

# Run training (main entry point)
python main.py --model 1d --size 8 --h 0.5 --rbm full --sampler custom --sampling-method metropolis

# Run tests
python -m pytest test_e2e.py -v

# Run a single test
python -m pytest test_e2e.py::test_psi_ratio_consistent_with_log_psi -v

# Run batch hyperparameter sweep
python performance_run.py

# Run end-to-end diagnostic (no pytest)
python test_e2e.py
```

### Key `main.py` arguments

| Argument | Choices | Default |
|---|---|---|
| `--model` | `1d`, `2d` | `1d` |
| `--size` | int | `16` |
| `--h` | float | `0.5` |
| `--rbm` | `full`, `pegasus`, `zephyr` | `full` |
| `--sampler` | `custom`, `dimod`, `velox` | `dimod` |
| `--sampling-method` | `metropolis`, `simulated_annealing`, `tabu`, `pegasus`, `zephyr`, `velox` | `simulated_annealing` |
| `--iterations` | int | `30` |
| `--learning-rate` | float | `0.1` |

## Architecture

### Core algorithm flow (`src/main.py`)

1. Instantiate Ising model → `ising.py`
2. Instantiate RBM → `model.py`
3. Instantiate sampler → `sampler.py`
4. Create `Trainer` → `encoder.py`
5. `trainer.train()` iterates: sample → compute local energies → SR update
6. `save_results()` writes JSON + convergence plot to `results/{n_hidden}/{sampler}/{method}/`

### Module responsibilities

- **`model.py`** — RBM implementations. `FullyConnectedRBM` (dense W matrix) and `DWaveTopologyRBM` (sparse W masked to a D-Wave QPU subgraph for chain-free embedding). The wave function ansatz is `Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)` (Gardas Eq. 6-7). The `psi_ratio` method computes `Ψ(v_flip)/Ψ(v)` efficiently in log space.

- **`ising.py`** — `TransverseFieldIsing1D` and `TransverseFieldIsing2D`. The `local_energy_batch()` method vectorises local energy computation over all samples simultaneously (no Python loop over samples). Exact reference energies via integral formula (1D) or exact diagonalization using netket (2D, L ≤ 4).

- **`sampler.py`** — Three backends:
  - `ClassicalSampler`: custom Metropolis-Hastings or simulated annealing targeting `|Ψ(v)|²`
  - `DimodSampler`: uses dimod/neal for SA/tabu, or D-Wave QPU for `pegasus`/`zephyr` methods. Caches embeddings per `(n_visible, solver)` key. Logs QPU access time to `time.json`.
  - `VeloxSampler`: uses VeloxQ SDK; requires `velox_api_config.py` and `velox_token.txt` in `src/`.

- **`encoder.py`** — `Trainer` class implements SR optimization matrix-free via conjugate gradient (`SRLinearSystem` + `conjugate_gradient`). Memory cost is `O(n_samples × n_params)` instead of `O(n_params²)`. Includes adaptive `beta_x` scaling and convergence detection.

- **`helpers.py`** — `save_results()` (JSON + plot), checkpoint save/load (pickle), `get_solver_name()` mapping `pegasus`→`Advantage_system6.4`, `zephyr`→`Advantage2_system1.13`.

### Result file naming convention

```
results/{n_hidden}/{sampler}/{method}/result_{model}_h{h}_rbm{rbm}_nh{n_hidden}_lr{lr}_reg{reg}_ns{ns}_seed{seed}_iter{iter}.json
```

### D-Wave budget tracking

`DimodSampler` accumulates QPU access time (microseconds → milliseconds) in `time.json`. `performance_run.py` enforces a 20-minute cumulative budget and aborts D-Wave experiments when exceeded. This file persists across sessions and is never reset automatically.

### Tests (`src/test_e2e.py`)

Tests use N=4 (16 configs, enumerable exactly). Key checks:
- `test_local_energy_consistent_with_hamiltonian`: VMC energy via `local_energy` must match `<Ψ|H|Ψ>/<Ψ|Ψ>` from full matrix
- `test_variational_bound`: `<E>_RBM >= E_exact` (variational principle)
- `test_psi_ratio_consistent_with_log_psi`: fast `psi_ratio` must match `exp(log_psi(v_flip) - log_psi(v))`
- `test_e_loc_sum_equals_matrix_row`: `E_loc(v)` must equal the corresponding Hamiltonian matrix row

## Coding principles

**No silent fallbacks.** Never default to a backup value, a zero, or any substitute when reading shared state (e.g. `time.json`, checkpoints, config files) fails. A silent fallback can corrupt experiment results or silently exceed resource budgets (e.g. QPU time). If something cannot be read or parsed correctly: log the error, skip the current experiment, and continue with the next one. Raise, don't swallow.

**No defensive defaults on critical paths.** `except Exception: return 0` on a QPU budget read is a concrete example of what never to do — it makes the budget check always pass, burning real QPU time silently. The same principle applies to any shared resource or measurement.

### Environment

- Python 3.13, virtualenv at `.venv/`
- Velox SDK credentials: `src/velox_api_config.py` + `src/velox_token.txt` (not committed)
- D-Wave credentials: standard Ocean SDK config (`~/.config/dwave/` or environment variables)
