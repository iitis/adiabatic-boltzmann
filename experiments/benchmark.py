"""
Comprehensive benchmarking framework for variational Boltzmann machine training.

Runs systematic tests across different configurations and saves results
organized by parameters for later analysis and visualization.

Results are saved hierarchically for easy retrieval and analysis:
results/
  ├── summary.json (overall statistics)
  └── N{size}/
      └── h{h_value}/
          ├── fully_connected/
          │   ├── run_000.json  (energy progression, config, metrics)
          │   ├── run_001.json
          │   └── ...
          └── dwave_topology/
              └── run_000.json
              └── ...
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler
from encoder import Trainer


class BenchmarkRunner:
    """Run comprehensive tests across multiple configurations."""
    
    def __init__(self, results_dir: str = "results/", model_type: str = '1d'):
        """
        Initialize benchmark runner.
        
        Args:
            results_dir: Directory to save results
            model_type: '1d' or '2d' Ising model
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_type = model_type
        self.results = []
        
        print(f"Initialized BenchmarkRunner (model_type={model_type}, results_dir={results_dir})")
    
    def run_all_tests(self, system_sizes: List[int] = None, h_values: List[float] = None,
                      architectures: List[str] = None, n_runs: int = 3,
                      n_iterations: int = 100, n_samples: int = 500,
                      learning_rate: float = 0.1, regularization: float = 1e-5) -> Dict[str, Any]:
        """
        Run all test combinations.
        
        Args:
            system_sizes: List of system sizes
            h_values: List of transverse field values
            architectures: List of RBM architectures ('fully_connected', 'dwave_topology')
            n_runs: Number of runs per configuration
            n_iterations: Training iterations per run
            n_samples: Samples per iteration
            learning_rate: Learning rate
            regularization: SR regularization parameter
        
        Returns:
            Dictionary with summary statistics
        """
        # Set defaults
        if system_sizes is None:
            system_sizes = [4, 6, 8, 10]
        if h_values is None:
            h_values = [0.50, 1.00, 2.00]
        if architectures is None:
            architectures = ['fully_connected', 'dwave_topology']
        
        total_tests = len(system_sizes) * len(h_values) * len(architectures) * n_runs
        
        print("=" * 80)
        print("STARTING BENCHMARK SUITE")
        print("=" * 80)
        print(f"Model type: {self.model_type.upper()}")
        print(f"Total tests: {total_tests}")
        print(f"System sizes: {system_sizes}")
        print(f"h values: {h_values}")
        print(f"Architectures: {architectures}")
        print(f"Runs per config: {n_runs}")
        print(f"Iterations per run: {n_iterations}")
        print(f"Samples per iteration: {n_samples}")
        print("=" * 80)
        print()
        
        test_count = 0
        successful = 0
        failed = 0
        
        for size in system_sizes:
            for h in h_values:
                for arch in architectures:
                    for run_id in range(n_runs):
                        test_count += 1
                        try:
                            self.run_single_test(
                                system_size=size,
                                h=h,
                                architecture=arch,
                                run_id=run_id,
                                n_iterations=n_iterations,
                                n_samples=n_samples,
                                learning_rate=learning_rate,
                                regularization=regularization,
                            )
                            successful += 1
                        except Exception as e:
                            print(f"  ✗ FAILED: {e}")
                            failed += 1
                            self.results.append({
                                'system_size': size,
                                'h': h,
                                'architecture': arch,
                                'run_id': run_id,
                                'status': 'failed',
                                'error': str(e)
                            })
        
        summary = self._save_summary(successful, failed, system_sizes, h_values, architectures, n_runs, n_iterations, n_samples, learning_rate)
        
        print("=" * 80)
        print(f"BENCHMARK COMPLETE: {successful} successful, {failed} failed")
        print("=" * 80)
        
        return summary
    
    
    def run_single_test(self, system_size: int, h: float, architecture: str,
                       run_id: int, n_iterations: int = 100, n_samples: int = 500,
                       learning_rate: float = 0.1, regularization: float = 1e-5) -> None:
        """
        Run a single test configuration.
        
        Args:
            system_size: Number of spins
            h: Transverse field strength
            architecture: RBM architecture type ('fully_connected' or 'dwave_topology')
            run_id: Run index
            n_iterations: Training iterations
            n_samples: Samples per iteration
            learning_rate: Learning rate
            regularization: SR regularization
            
        Raises:
            Exception: On training or I/O errors
        """
        print(f"N={system_size}, h={h:.2f}, {architecture}, run={run_id}: ", end="", flush=True)
        
        # Setup result directory
        arch_name = 'fully_connected' if architecture == 'fully_connected' else 'dwave_topology'
        result_dir = (self.results_dir / 
                     f"N{system_size}" /
                     f"h{h:.2f}" /
                     arch_name)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create physics model
        if self.model_type == '1d':
            ising = TransverseFieldIsing1D(size=system_size, h=h)
        elif self.model_type == '2d':
            side = int(np.sqrt(system_size))
            assert side * side == system_size, f"For 2D, system_size must be perfect square"
            ising = TransverseFieldIsing2D(size=side, h=h)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create RBM
        n_hidden = system_size  # Match hidden to visible
        if architecture == 'fully_connected':
            rbm = FullyConnectedRBM(n_visible=system_size, n_hidden=n_hidden)
        else:  # dwave_topology
            rbm = DWaveTopologyRBM(n_visible=system_size, n_hidden=n_hidden)
        
        # Create sampler
        sampler = ClassicalSampler()
        
        # Training config
        config = {
            'learning_rate': learning_rate,
            'n_iterations': n_iterations,
            'n_samples': n_samples,
            'regularization': regularization,
        }
        
        # Train
        trainer = Trainer(rbm, ising, sampler, config)
        history = trainer.train()
        
        # Get ground state energy reference
        try:
            E_ground = ising.exact_ground_energy()
        except:
            E_ground = None
        
        # Compute metrics
        E_final = history['energy'][-1] if history['energy'] else None
        E_initial = history['energy'][0] if history['energy'] else None
        
        # Save run result
        result = {
            'config': {
                'model_type': self.model_type,
                'system_size': system_size,
                'h': h,
                'architecture': architecture,
                'run_id': run_id,
                'learning_rate': learning_rate,
                'n_iterations': n_iterations,
                'n_samples': n_samples,
                'regularization': regularization,
            },
            'metrics': {
                'E_initial': float(E_initial) if E_initial is not None else None,
                'E_final': float(E_final) if E_final is not None else None,
                'E_ground': float(E_ground) if E_ground is not None else None,
                'E_improvement': float(E_initial - E_final) if (E_initial is not None and E_final is not None) else None,
            },
            'history': {
                'energy': [float(e) for e in history['energy']],
                'error': [float(e) for e in history['error']],
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save to JSON
        run_file = result_dir / f"run_{run_id:03d}.json"
        with open(run_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Record result
        self.results.append({
            'system_size': system_size,
            'h': h,
            'architecture': architecture,
            'run_id': run_id,
            'status': 'completed',
            'result_file': str(run_file.relative_to(self.results_dir)),
            'final_energy': float(E_final) if E_final is not None else None,
            'initial_energy': float(E_initial) if E_initial is not None else None,
            'energy_improvement': float(E_initial - E_final) if (E_initial is not None and E_final is not None) else None,
        })
        
        print(f"E={E_final:.6f} (ΔE={result['metrics']['E_improvement']:.6f})")
    
    
    def _save_summary(self, successful: int, failed: int, system_sizes: List[int],
                     h_values: List[float], architectures: List[str], n_runs: int,
                     n_iterations: int, n_samples: int, learning_rate: float) -> Dict[str, Any]:
        """
        Save summary of all results with aggregated statistics.
        
        Args:
            successful: Number of successful runs
            failed: Number of failed runs
            system_sizes: System sizes tested
            h_values: h values tested
            architectures: Architectures tested
            n_runs: Runs per config
            n_iterations: Training iterations
            n_samples: Samples per iteration
            learning_rate: Learning rate used
            
        Returns:
            Summary dictionary
        """
        successful_results = [r for r in self.results if r.get('status') == 'completed']
        failed_results = [r for r in self.results if r.get('status') == 'failed']
        
        # Aggregate by configuration
        by_config = {}
        for result in successful_results:
            key = (result['system_size'], result['h'], result['architecture'])
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(result)
        
        # Compute statistics per config
        config_stats = {}
        for (size, h, arch), results in by_config.items():
            E_finals = [r['final_energy'] for r in results if r['final_energy'] is not None]
            E_improvements = [r['energy_improvement'] for r in results if r['energy_improvement'] is not None]
            
            config_stats[f"N{size}_h{h:.2f}_{arch}"] = {
                'n_spins': size,
                'h': h,
                'architecture': arch,
                'n_runs': len(results),
                'E_final_mean': float(np.mean(E_finals)) if E_finals else None,
                'E_final_std': float(np.std(E_finals)) if E_finals else None,
                'E_improvement_mean': float(np.mean(E_improvements)) if E_improvements else None,
                'E_improvement_std': float(np.std(E_improvements)) if E_improvements else None,
            }
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'test_matrix': {
                'system_sizes': system_sizes,
                'h_values': h_values,
                'architectures': architectures,
                'runs_per_config': n_runs,
            },
            'training_config': {
                'n_iterations': n_iterations,
                'n_samples': n_samples,
                'learning_rate': learning_rate,
            },
            'totals': {
                'total_tests': len(self.results),
                'successful': len(successful_results),
                'failed': len(failed_results),
            },
            'configuration_statistics': config_stats,
            'individual_results': successful_results,
        }
        
        # Save main summary
        summary_file = self.results_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to {summary_file}")
        
        return summary
    


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive RBM benchmark suite for Ising models"
    )
    parser.add_argument(
        '--model', choices=['1d', '2d'], default='1d',
        help='Ising model dimension'
    )
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[4, 6, 8, 10],
        help='System sizes to test'
    )
    parser.add_argument(
        '--h-values', type=float, nargs='+', default=[0.50, 1.00, 2.00],
        help='Transverse field values'
    )
    parser.add_argument(
        '--architectures', choices=['fully_connected', 'dwave_topology', 'both'],
        nargs='+', default=['both'],
        help='RBM architectures to test'
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='Number of runs per configuration'
    )
    parser.add_argument(
        '--iterations', type=int, default=100,
        help='Training iterations per run'
    )
    parser.add_argument(
        '--samples', type=int, default=500,
        help='Samples per iteration'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--results-dir', default='results/',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Process architecture argument
    if 'both' in args.architectures or args.architectures == ['both']:
        architectures = ['fully_connected', 'dwave_topology']
    else:
        architectures = args.architectures
    
    # Run benchmark
    try:
        runner = BenchmarkRunner(results_dir=args.results_dir, model_type=args.model)
        summary = runner.run_all_tests(
            system_sizes=args.sizes,
            h_values=args.h_values,
            architectures=architectures,
            n_runs=args.runs,
            n_iterations=args.iterations,
            n_samples=args.samples,
            learning_rate=args.learning_rate,
        )
        
        return 0
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
