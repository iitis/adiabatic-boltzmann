# Testing and Benchmarking Framework

This directory contains scripts for systematic testing, evaluation, and analysis of the RBM Ising model implementation.

## Structure

```
experiments/
├── generate_instances.py      # Generate numbered Ising problem instances
├── benchmark.py               # Main benchmark suite runner
├── analyze_results.py         # Aggregate and analyze results
├── visualize_results.py       # Generate visualization code templates
│
├── data/                      # Problem instances (auto-generated)
│   ├── instance_*.json        # Individual Ising instances
│   └── instance_index.json    # Master index of all instances
│
├── results/                   # Benchmark results (auto-generated)
│   ├── summary.json           # High-level summary of all runs
│   ├── statistics.json        # Per-configuration aggregated statistics
│   ├── architecture_comparison.json
│   ├── best_configurations.json
│   └── N{size}/
│       └── h{h_value}/
│           ├── fully_connected/
│           │   ├── run_000.json  # Energy progression, config, metrics
│           │   ├── run_001.json
│           │   └── ...
│           └── dwave_topology/
│               └── run_*.json
│
└── logs/                      # Training logs (optional)
```

## Quick Start

### 1. Generate Problem Instances

Create a set of numbered Ising instances for reproducible benchmarking:

```bash
cd experiments
python generate_instances.py --output-dir data/
```

This creates 36 instances (4 sizes × 3 h-values × 3 runs) with consistent numbering.

Output:
- `data/instance_N_h_ID.json` files with individual problem configurations
- `data/instance_index.json` master index

### 2. Run Benchmark Suite

Execute the full benchmark across different model sizes and parameters:

```bash
python benchmark.py \
  --model 1d \
  --sizes 4 6 8 10 \
  --h-values 0.50 1.00 2.00 \
  --architectures both \
  --runs 3 \
  --iterations 100 \
  --samples 500 \
  --learning-rate 0.1 \
  --results-dir results/
```

**Parameters:**
- `--model`: Ising model type (`1d` or `2d`)
- `--sizes`: System sizes to test
- `--h-values`: Transverse field values
- `--architectures`: `fully_connected`, `dwave_topology`, or `both`
- `--runs`: Number of runs per configuration (for error bars)
- `--iterations`: Training iterations per run
- `--samples`: Samples per iteration (for better statistics)
- `--learning-rate`: Optimization step size
- `--results-dir`: Output directory

**Duration:** ~2-3 minutes for full suite (72 runs × 100 iterations)

### 3. Analyze Results

Aggregate results and compute statistics:

```bash
python analyze_results.py --results-dir results/
```

This generates:
- `results/statistics.json` - Per-configuration mean/std/min/max energies
- `results/architecture_comparison.json` - Fully-connected vs D-Wave comparisons
- `results/best_configurations.json` - Top-performing configurations

Output: Summary table and comparison highlights.

### 4. Visualize Results

Generate visualization scripts:

```bash
python visualize_results.py --results-dir results/
```

This creates template scripts (requires `matplotlib`):

```bash
# Install matplotlib if needed
pip install matplotlib

# Generate plots
python visualize_convergence.py --results-dir results/ --output-dir plots/
python generate_report.py --results-dir results/ --output report.html
```

Plots include:
- Energy convergence curves per configuration
- System size scaling analysis
- Transverse field (h) dependence
- Architecture comparisons

## Results Organization

### Result Files

Each run is saved as a JSON file with structure:

```json
{
  "config": {
    "model_type": "1d",
    "system_size": 4,
    "h": 0.5,
    "architecture": "fully_connected",
    "run_id": 0,
    "learning_rate": 0.1,
    "n_iterations": 100,
    "n_samples": 500
  },
  "metrics": {
    "E_initial": -3.5,
    "E_final": -3.8,
    "E_ground": -3.9,
    "E_improvement": 0.3
  },
  "history": {
    "energy": [E_0, E_1, ..., E_N],
    "error": [err_0, err_1, ..., err_N]
  },
  "timestamp": "2024-03-06T..."
}
```

### Accessing Results Programmatically

```python
import json
from pathlib import Path

results_dir = Path("results")

# Load summary
with open(results_dir / "summary.json") as f:
    summary = json.load(f)

# Load statistics
with open(results_dir / "statistics.json") as f:
    stats = json.load(f)

# Access specific configuration
config_key = "N4_h0.50_fully_connected"
config_stats = stats[config_key]

print(f"Final Energy: {config_stats['final_energy']['mean']:.6f}")
print(f"  ± {config_stats['final_energy']['std']:.6f}")
print(f"Improvement: {config_stats['energy_improvement']['mean']:.6f}")

# Get convergence curve
convergence = config_stats['convergence']['mean']
```

## Batch Testing with Different Configurations

### Quick Test (for development)
```bash
python benchmark.py \
  --sizes 4 6 \
  --h-values 0.50 1.00 \
  --architectures fully_connected \
  --runs 1 \
  --iterations 20 \
  --samples 200
```

### Comprehensive Test (final results)
```bash
python benchmark.py \
  --sizes 4 6 8 10 \
  --h-values 0.50 1.00 2.00 \
  --architectures both \
  --runs 5 \
  --iterations 200 \
  --samples 1000
```

### 1D vs 2D Comparison
```bash
# 1D model
python benchmark.py --model 1d --sizes 4 8 16 32

# 2D model (note: sizes are square numbers)
python benchmark.py --model 2d --sizes 4 9 16 25
```

## Analyzing Specific Comparisons

### Compare Two Architectures

```python
import json
from pathlib import Path

with open("results/architecture_comparison.json") as f:
    comparisons = json.load(f)

# Look at N=8, h=1.0
config = "N8_h1.00"
if config in comparisons:
    comp = comparisons[config]
    print(f"Winner: {comp['winner']}")
    print(f"Energy difference: {comp['energy_diff']:.6f}")
    
    for arch, results in comp['results'].items():
        print(f"{arch}:")
        print(f"  Final Energy: {results['final_energy_mean']:.6f}")
        print(f"  Improvement: {results['energy_improvement_mean']:.6f}")
```

### Extract Best Configuration

```python
import json

with open("results/best_configurations.json") as f:
    best = json.load(f)

print("Overall best:", best['overall_best'])
print("\nBest per size:")
for size, config in best['best_per_size'].items():
    print(f"  {size}: {config['config']}")
```

## Parameter Tuning

The framework is designed to systematically explore:

1. **Model Parameters:**
   - System size: N ∈ [4, 6, 8, 10, ...]
   - Transverse field: h ∈ [0.5, 1.0, 2.0, ...]

2. **Architecture Choices:**
   - Fully connected RBM (dense connectivity)
   - D-Wave topology RBM (sparse, hardware-like)

3. **Training Parameters:**
   - Learning rate: controls optimization speed
   - Number of samples: improves statistics
   - Number of iterations: convergence time
   - Sampling method: Metropolis or simulated annealing

## Expected Results

For the standard 1D transverse field Ising model:

- **Easy regime** (h ~ 1.0, small N): RBM quickly learns ground state
- **Intermediate** (h ~ 2.0, N=6-8): Convergence slower, more variance
- **Hard regime** (large h, large N): May not fully converge in 100 iterations

D-Wave topology may perform:
- **Better:** For hardware-inspired problems
- **Worse:** For fully connected problems
- **Similar:** Often comparable within error bars

## Reproducibility

All experiments use:
- Fixed random seeds (default: 42)
- Deterministic algorithms
- Numbered instances for consistent problem sets

To reproduce results:
```bash
python benchmark.py --results-dir results_v1/
python benchmark.py --results-dir results_v2/
python analyze_results.py --results-dir results_v1/
python analyze_results.py --results-dir results_v2/
# Compare statistics files
```

## Troubleshooting

**Import errors:**
```bash
# Make sure you're in the experiments directory and src is in PYTHONPATH
cd experiments
python benchmark.py --help
```

**Out of memory with large systems:**
- Reduce `--samples` parameter
- Reduce `--iterations`
- Use smaller system `--sizes`

**Slow convergence:**
- Increase `--learning-rate` (e.g., 0.5)
- Increase `--samples` for better gradient estimates

**Matplotlib not found for visualization:**
```bash
pip install matplotlib
python visualize_results.py
```

## Advanced: Custom Configurations

Edit `benchmark.py` directly to add custom test matrices or modify the suite structure for your needs.

## Next Steps

1. ✅ Generate instances: `python generate_instances.py`
2. ✅ Run benchmark: `python benchmark.py`
3. ✅ Analyze: `python analyze_results.py`
4. ✅ Visualize: `python visualize_results.py && python visualize_convergence.py`
5. 📊 Plot custom comparisons using result JSON files
6. 📝 Write up methodology and results

