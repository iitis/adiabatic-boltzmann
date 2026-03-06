"""
Visualization and plotting utilities for benchmark results.

This script provides templates for visualizing:
1. Energy convergence curves (per configuration and averaged)
2. Architecture comparisons
3. System size scaling analysis
4. Parameter dependency plots

Run after analyze_results.py to generate visualizations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse


class ResultsVisualizer:
    """Visualize benchmark results."""
    
    def __init__(self, results_dir: str = "results/"):
        self.results_dir = Path(results_dir)
        self.stats_file = self.results_dir / "statistics.json"
        self.comparison_file = self.results_dir / "architecture_comparison.json"
        self.stats = {}
        self.comparisons = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load analysis data from disk."""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        
        if self.comparison_file.exists():
            with open(self.comparison_file, 'r') as f:
                self.comparisons = json.load(f)
    
    def generate_convergence_plot_code(self, output_file: str = None):
        """
        Generate Python code for convergence curve plots.
        
        This creates a script that plots energy convergence for:
        - Individual runs
        - Averaged runs per configuration
        
        The generated script uses matplotlib for visualization.
        """
        code = '''"""
Plot energy convergence curves from benchmark results.

Usage:
    python visualize_convergence.py --results-dir results/ --output-dir plots/
"""

import json
import numpy as np
from pathlib import Path
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")


def load_statistics(stats_file):
    """Load statistics from JSON."""
    with open(stats_file, 'r') as f:
        return json.load(f)


def plot_configuration_convergence(stats, config_key, ax, title=None):
    """
    Plot convergence curve for a single configuration.
    
    Args:
        stats: Statistics dictionary
        config_key: Configuration key (e.g., "N4_h0.50_fully_connected")
        ax: Matplotlib axis
        title: Optional title override
    """
    if config_key not in stats:
        print(f"WARNING: Configuration {config_key} not found")
        return
    
    config = stats[config_key]
    convergence = config.get('convergence', {})
    
    if not convergence.get('mean'):
        print(f"No convergence data for {config_key}")
        return
    
    mean = convergence['mean']
    std = convergence['std']
    
    iterations = np.arange(len(mean))
    
    # Plot mean and std
    ax.plot(iterations, mean, 'b-', linewidth=2, label='Mean')
    if std:
        std_arr = np.array(std)
        ax.fill_between(iterations, 
                        np.array(mean) - std_arr, 
                        np.array(mean) + std_arr,
                        alpha=0.3, label='±1 std')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    ax.set_title(title or config_key)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_size_scaling(stats, h_value, architecture, ax):
    """
    Plot energy vs system size for fixed h and architecture.
    
    Shows how performance scales with system size.
    """
    sizes = []
    final_energies = []
    errors = []
    
    for config_key, config in sorted(stats.items()):
        if config['h'] == h_value and config['architecture'] == architecture:
            sizes.append(config['n_spins'])
            final_energies.append(config['final_energy']['mean'])
            errors.append(config['final_energy']['std'])
    
    if not sizes:
        print(f"No data for h={h_value}, {architecture}")
        return
    
    ax.errorbar(sizes, final_energies, yerr=errors, 
               marker='o', linestyle='-', capsize=5, capthick=2)
    ax.set_xlabel('System Size (N)')
    ax.set_ylabel('Final Energy')
    ax.set_title(f'Scaling: h={h_value}, {architecture}')
    ax.grid(True, alpha=0.3)


def plot_h_dependence(stats, size, architecture, ax):
    """
    Plot energy vs transverse field h for fixed size and architecture.
    
    Shows how performance depends on the problem parameter.
    """
    h_vals = []
    final_energies = []
    errors = []
    
    for config_key, config in sorted(stats.items()):
        if config['n_spins'] == size and config['architecture'] == architecture:
            h_vals.append(config['h'])
            final_energies.append(config['final_energy']['mean'])
            errors.append(config['final_energy']['std'])
    
    if not h_vals:
        print(f"No data for N={size}, {architecture}")
        return
    
    ax.errorbar(h_vals, final_energies, yerr=errors,
               marker='s', linestyle='-', capsize=5, capthick=2)
    ax.set_xlabel('Transverse Field (h)')
    ax.set_ylabel('Final Energy')
    ax.set_title(f'h-Dependence: N={size}, {architecture}')
    ax.grid(True, alpha=0.3)


def plot_architecture_comparison(stats, size, h, ax):
    """
    Compare fully_connected vs dwave_topology for fixed N and h.
    """
    fc_config = f"N{size}_h{h:.2f}_fully_connected"
    dw_config = f"N{size}_h{h:.2f}_dwave_topology"
    
    labels = []
    energies = []
    errors = []
    
    for key, label in [(fc_config, 'Fully Connected'), (dw_config, 'D-Wave Topology')]:
        if key in stats:
            labels.append(label)
            energies.append(stats[key]['final_energy']['mean'])
            errors.append(stats[key]['final_energy']['std'])
    
    if len(labels) < 2:
        print(f"Not enough data to compare architectures for N={size}, h={h}")
        return
    
    x = np.arange(len(labels))
    ax.bar(x, energies, yerr=errors, capsize=5, alpha=0.7, color=['blue', 'orange'])
    ax.set_ylabel('Final Energy')
    ax.set_title(f'Architecture Comparison (N={size}, h={h})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument('--results-dir', default='results/',
                       help='Results directory')
    parser.add_argument('--output-dir', default='plots/',
                       help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Plot DPI')
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot generate plots without matplotlib")
        print("Install with: pip install matplotlib")
        return
    
    # Load statistics
    stats_file = Path(args.results_dir) / 'statistics.json'
    if not stats_file.exists():
        print(f"Statistics file not found: {stats_file}")
        print("Run analyze_results.py first")
        return
    
    stats = load_statistics(stats_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots to {output_dir}...")
    
    # 1. Convergence curves for each configuration
    print("  - Generating convergence curves...")
    n_configs = len(stats)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_configs > 1 else [axes]
    
    for idx, config_key in enumerate(sorted(stats.keys())):
        plot_configuration_convergence(stats, config_key, axes[idx])
    
    # Hide unused subplots
    for idx in range(len(stats), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_curves.png', dpi=args.dpi)
    plt.close()
    
    # 2. System size scaling
    print("  - Generating scaling plots...")
    
    # Get unique (h, architecture) combinations
    scaling_configs = set()
    for config in stats.values():
        scaling_configs.add((config['h'], config['architecture']))
    
    fig, axes = plt.subplots(1, len(scaling_configs), 
                            figsize=(6*len(scaling_configs), 5))
    axes = axes if len(scaling_configs) > 1 else [axes]
    
    for ax, (h, arch) in enumerate(sorted(scaling_configs)):
        plot_size_scaling(stats, h, arch, axes[ax])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'size_scaling.png', dpi=args.dpi)
    plt.close()
    
    # 3. h-dependence
    print("  - Generating h-dependence plots...")
    
    h_dep_configs = set()
    for config in stats.values():
        h_dep_configs.add((config['n_spins'], config['architecture']))
    
    fig, axes = plt.subplots(1, len(h_dep_configs),
                            figsize=(6*len(h_dep_configs), 5))
    axes = axes if len(h_dep_configs) > 1 else [axes]
    
    for ax, (size, arch) in enumerate(sorted(h_dep_configs)):
        plot_h_dependence(stats, size, arch, axes[ax])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'h_dependence.png', dpi=args.dpi)
    plt.close()
    
    # 4. Architecture comparisons
    print("  - Generating architecture comparisons...")
    
    arch_comp_configs = set()
    for config in stats.values():
        arch_comp_configs.add((config['n_spins'], config['h']))
    
    n_comps = len(arch_comp_configs)
    n_cols = 3
    n_rows = (n_comps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_comps > 1 else [axes]
    
    for idx, (size, h) in enumerate(sorted(arch_comp_configs)):
        plot_architecture_comparison(stats, size, h, axes[idx])
    
    for idx in range(len(arch_comp_configs), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_comparison.png', dpi=args.dpi)
    plt.close()
    
    print(f"\\nPlots saved to {output_dir}:")
    print("  - convergence_curves.png")
    print("  - size_scaling.png")
    print("  - h_dependence.png")
    print("  - architecture_comparison.png")


if __name__ == "__main__":
    main()
'''
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(code)
            print(f"Generated convergence plot script: {output_file}")
        
        return code
    
    def generate_summary_report_code(self, output_file: str = None):
        """
        Generate code for creating an HTML summary report.
        """
        code = '''"""
Generate HTML summary report of benchmark results.

Usage:
    python generate_report.py --results-dir results/ --output report.html
"""

import json
from pathlib import Path
import argparse
from datetime import datetime


def generate_html_report(results_dir, output_file):
    """
    Generate an HTML report summarizing all results.
    
    Args:
        results_dir: Results directory path
        output_file: Output HTML file path
    """
    results_dir = Path(results_dir)
    
    # Load data
    with open(results_dir / 'summary.json') as f:
        summary = json.load(f)
    
    with open(results_dir / 'statistics.json') as f:
        stats = json.load(f)
    
    with open(results_dir / 'best_configurations.json') as f:
        best = json.load(f)
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RBM Ising Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #0066cc; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .success {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>RBM Ising Model Benchmark Report</h1>
    
    <div class="timestamp">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <p><strong>Total Tests:</strong> {summary['totals']['total_tests']}</p>
        <p><strong>Successful:</strong> <span class="success">{summary['totals']['successful']}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{summary['totals']['failed']}</span></p>
    </div>
    
    <h2>Test Configuration</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Values</th>
        </tr>
        <tr>
            <td>System Sizes</td>
            <td>{', '.join(map(str, summary['test_matrix']['system_sizes']))}</td>
        </tr>
        <tr>
            <td>h Values</td>
            <td>{', '.join(map(str, summary['test_matrix']['h_values']))}</td>
        </tr>
        <tr>
            <td>Architectures</td>
            <td>{', '.join(summary['test_matrix']['architectures'])}</td>
        </tr>
        <tr>
            <td>Runs per Config</td>
            <td>{summary['test_matrix']['runs_per_config']}</td>
        </tr>
    </table>
    
    <h2>Best Configurations</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Configuration</th>
            <th>Final Energy</th>
            <th>Improvement</th>
        </tr>
        <tr>
            <td>1</td>
            <td><strong>{best['overall_best']['config']}</strong></td>
            <td>{best['overall_best']['final_energy']:.6f}</td>
            <td>{best['overall_best']['improvement']:.6f}</td>
        </tr>
    </table>
    
    <h2>Results per Configuration</h2>
    <table>
        <tr>
            <th>Configuration</th>
            <th>N</th>
            <th>h</th>
            <th>Architecture</th>
            <th>Runs</th>
            <th>Final Energy (mean ± std)</th>
            <th>Improvement (mean ± std)</th>
        </tr>
"""
    
    for key in sorted(stats.keys()):
        config = stats[key]
        E_final = config['final_energy']['mean']
        E_final_std = config['final_energy']['std']
        E_imp = config['energy_improvement']['mean']
        E_imp_std = config['energy_improvement']['std']
        
        html += f"""
        <tr>
            <td>{key}</td>
            <td>{config['n_spins']}</td>
            <td>{config['h']:.2f}</td>
            <td>{config['architecture']}</td>
            <td>{config['n_runs']}</td>
            <td>{E_final:.6f} ± {E_final_std:.6f}</td>
            <td>{E_imp:.6f} ± {E_imp_std:.6f}</td>
        </tr>
"""
    
    html += """
    </table>
    
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report")
    parser.add_argument('--results-dir', default='results/',
                       help='Results directory')
    parser.add_argument('--output', default='report.html',
                       help='Output HTML file')
    
    args = parser.parse_args()
    
    generate_html_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
'''
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(code)
            print(f"Generated report script: {output_file}")
        
        return code
    
    def generate_all_templates(self):
        """Generate all visualization template scripts."""
        print("Generating visualization templates...")
        
        # Generate convergence plot script
        self.generate_convergence_plot_code(
            self.results_dir.parent / 'visualize_convergence.py'
        )
        
        # Generate report script
        self.generate_summary_report_code(
            self.results_dir.parent / 'generate_report.py'
        )
        
        print("\nGenerated template scripts:")
        print("  - visualize_convergence.py (requires matplotlib)")
        print("  - generate_report.py (creates HTML report)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization template scripts"
    )
    parser.add_argument(
        '--results-dir', default='results/',
        help='Results directory'
    )
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    visualizer.generate_all_templates()


if __name__ == "__main__":
    main()
