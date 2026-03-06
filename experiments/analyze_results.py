"""
Analyze and aggregate results from benchmark runs.

This script loads all benchmark results, computes statistics,
and generates aggregated views for easy comparison.

Output:
- statistics.json: Aggregated statistics per configuration
- architecture_comparison.json: Architecture comparisons
- best_configurations.json: Top-performing configurations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse


class ResultsAnalyzer:
    """Analyze and aggregate benchmark results."""
    
    def __init__(self, results_dir: str = "results/"):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "summary.json"
        self.results = {}
        self.summary = {}
    
    def load_results(self):
        """Load all results from disk."""
        print(f"Loading results from {self.results_dir}...")
        
        # Load main summary
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.summary = json.load(f)
            print(f"  Loaded summary.json with {self.summary['totals']['successful']} successful runs")
        
        # Load individual run files
        run_count = 0
        for run_file in sorted(self.results_dir.rglob("run_*.json")):
            with open(run_file, 'r') as f:
                run_data = json.load(f)
                config = run_data['config']
                key = (config['system_size'], config['h'], config['architecture'])
                
                if key not in self.results:
                    self.results[key] = []
                self.results[key].append(run_data)
                run_count += 1
        
        print(f"  Loaded {run_count} individual run files")
        return run_count
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics across runs.
        
        Returns:
            Dictionary with statistics per configuration
        """
        print("\nComputing statistics...")
        
        stats = {}
        
        for (size, h, arch), runs in sorted(self.results.items()):
            config_key = f"N{size}_h{h:.2f}_{arch}"
            
            # Extract metrics from all runs
            E_finals = []
            E_improvements = []
            energy_progressions = []
            
            for run in runs:
                if 'metrics' in run and run['metrics']['E_final'] is not None:
                    E_finals.append(run['metrics']['E_final'])
                    if run['metrics']['E_improvement'] is not None:
                        E_improvements.append(run['metrics']['E_improvement'])
                
                if 'history' in run and 'energy' in run['history']:
                    energy_progressions.append(run['history']['energy'])
            
            # Compute statistics
            if E_finals:
                # Final energy stats
                E_final_mean = np.mean(E_finals)
                E_final_std = np.std(E_finals)
                E_final_min = np.min(E_finals)
                E_final_max = np.max(E_finals)
                
                # Improvement stats
                E_improve_mean = np.mean(E_improvements) if E_improvements else None
                E_improve_std = np.std(E_improvements) if E_improvements else None
                
                # Convergence analysis
                # Compute average convergence curve
                if energy_progressions:
                    # Pad to same length
                    max_len = max(len(prog) for prog in energy_progressions)
                    padded = []
                    for prog in energy_progressions:
                        if len(prog) < max_len:
                            # Pad with final value
                            padded.append(prog + [prog[-1]] * (max_len - len(prog)))
                        else:
                            padded.append(prog)
                    
                    energy_array = np.array(padded)
                    E_convergence_mean = energy_array.mean(axis=0).tolist()
                    E_convergence_std = energy_array.std(axis=0).tolist()
                else:
                    E_convergence_mean = None
                    E_convergence_std = None
                
                stats[config_key] = {
                    'n_spins': size,
                    'h': h,
                    'architecture': arch,
                    'n_runs': len(runs),
                    'final_energy': {
                        'mean': float(E_final_mean),
                        'std': float(E_final_std),
                        'min': float(E_final_min),
                        'max': float(E_final_max),
                    },
                    'energy_improvement': {
                        'mean': float(E_improve_mean) if E_improve_mean is not None else None,
                        'std': float(E_improve_std) if E_improve_std is not None else None,
                    },
                    'convergence': {
                        'mean': E_convergence_mean,
                        'std': E_convergence_std,
                    }
                }
        
        print(f"  Computed statistics for {len(stats)} configurations")
        return stats
    
    def compare_architectures(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance of different RBM architectures.
        
        Returns:
            Comparison dictionary
        """
        print("\nComparing architectures...")
        
        comparison = {}
        
        # Group by size and h
        by_config = {}
        for key, stat in stats.items():
            config_key = (stat['n_spins'], stat['h'])
            if config_key not in by_config:
                by_config[config_key] = {}
            by_config[config_key][stat['architecture']] = stat
        
        # Compare
        for (size, h), archs in sorted(by_config.items()):
            if len(archs) > 1:
                comp_key = f"N{size}_h{h:.2f}"
                results = {}
                
                for arch_name, stat in archs.items():
                    results[arch_name] = {
                        'final_energy_mean': stat['final_energy']['mean'],
                        'energy_improvement_mean': stat['energy_improvement']['mean'],
                        'n_runs': stat['n_runs'],
                    }
                
                # Determine winner
                energies = {arch: stat['final_energy']['mean'] for arch, stat in results.items()}
                winner = min(energies, key=energies.get)
                
                energy_list = list(energies.values())
                energy_diff = energy_list[0] - energy_list[1] if len(energy_list) >= 2 else None
                
                comparison[comp_key] = {
                    'results': results,
                    'winner': winner,
                    'energy_diff': energy_diff,
                }
        
        print(f"  Compared {len(comparison)} configuration pairs")
        return comparison
    
    def identify_best_runs(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify best performing configurations.
        
        Returns:
            Dictionary with top configurations
        """
        print("\nIdentifying best configurations...")
        
        best = {
            'overall_best': None,
            'best_per_size': {},
            'best_per_architecture': {},
        }
        
        all_configs = list(stats.items())
        if not all_configs:
            return best
        
        # Overall best (lowest final energy)
        best_config = min(all_configs, 
                         key=lambda x: x[1]['final_energy']['mean'])
        best['overall_best'] = {
            'config': best_config[0],
            'final_energy': best_config[1]['final_energy']['mean'],
            'improvement': best_config[1]['energy_improvement']['mean'],
        }
        
        # Best per size
        by_size = {}
        for key, stat in stats.items():
            size = stat['n_spins']
            if size not in by_size:
                by_size[size] = []
            by_size[size].append((key, stat))
        
        for size, configs in by_size.items():
            best_for_size = min(configs, 
                               key=lambda x: x[1]['final_energy']['mean'])
            best['best_per_size'][f"N{size}"] = {
                'config': best_for_size[0],
                'final_energy': best_for_size[1]['final_energy']['mean'],
            }
        
        # Best per architecture
        by_arch = {}
        for key, stat in stats.items():
            arch = stat['architecture']
            if arch not in by_arch:
                by_arch[arch] = []
            by_arch[arch].append((key, stat))
        
        for arch, configs in by_arch.items():
            best_for_arch = min(configs,
                               key=lambda x: x[1]['final_energy']['mean'])
            best['best_per_architecture'][arch] = {
                'config': best_for_arch[0],
                'final_energy': best_for_arch[1]['final_energy']['mean'],
            }
        
        return best
    
    def save_analysis(self, stats: Dict[str, Any], comparison: Dict[str, Any],
                     best: Dict[str, Any]):
        """Save analysis results to JSON files."""
        print("\nSaving analysis results...")
        
        # Save statistics
        stats_file = self.results_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to {stats_file}")
        
        # Save comparison
        comparison_file = self.results_dir / "architecture_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"  Saved comparison to {comparison_file}")
        
        # Save best configurations
        best_file = self.results_dir / "best_configurations.json"
        with open(best_file, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  Saved best configs to {best_file}")
    
    def print_summary(self, stats: Dict[str, Any], comparison: Dict[str, Any],
                     best: Dict[str, Any]):
        """Print human-readable summary."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nOverall Best Configuration:")
        if best['overall_best']:
            print(f"  {best['overall_best']['config']}")
            print(f"  Final Energy: {best['overall_best']['final_energy']:.6f}")
            print(f"  Improvement: {best['overall_best']['improvement']:.6f}")
        
        print(f"\nBest per System Size:")
        for size, config_info in best['best_per_size'].items():
            print(f"  {size}: {config_info['config']} (E={config_info['final_energy']:.6f})")
        
        print(f"\nBest per Architecture:")
        for arch, config_info in best['best_per_architecture'].items():
            print(f"  {arch}: {config_info['config']} (E={config_info['final_energy']:.6f})")
        
        print(f"\nArchitecture Comparisons (top 5):")
        sorted_comps = sorted([(k, v) for k, v in comparison.items() if v['energy_diff'] is not None],
                             key=lambda x: abs(x[1]['energy_diff']),
                             reverse=True)[:5]
        for config, comp in sorted_comps:
            winner = comp['winner']
            magnitude = abs(comp['energy_diff'])
            print(f"  {config}: {winner} wins by {magnitude:.6f}")
        
        print("\n" + "="*80)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        # Load
        self.load_results()
        
        if not self.results:
            print("No results found to analyze!")
            return
        
        # Analyze
        stats = self.compute_statistics()
        comparison = self.compare_architectures(stats)
        best = self.identify_best_runs(stats)
        
        # Save
        self.save_analysis(stats, comparison, best)
        
        # Print
        self.print_summary(stats, comparison, best)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results"
    )
    parser.add_argument(
        '--results-dir', default='results/',
        help='Results directory to analyze'
    )
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(results_dir=args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()


