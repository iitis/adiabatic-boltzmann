"""
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
    
    print(f"\nPlots saved to {output_dir}:")
    print("  - convergence_curves.png")
    print("  - size_scaling.png")
    print("  - h_dependence.png")
    print("  - architecture_comparison.png")


if __name__ == "__main__":
    main()
