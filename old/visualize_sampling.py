"""
Visualization script: Compare RBM sampling to ground state

This script:
1. Generates samples from RBM using different samplers
2. Computes energy distribution of samples
3. Compares to exact ground state energy
4. Plots for visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from model_skeleton import FullyConnectedRBM, DWaveTopologyRBM
from sampler_skeleton import ClassicalSampler
from ising_skeleton import TransverseFieldIsing1D


def compute_sample_energies(samples, ising_model, rbm):
    """
    Compute local energy for each sample.
    
    samples: (n_samples, n_visible) array
    ising_model: Ising Hamiltonian
    rbm: RBM instance for psi_ratio
    
    Returns: array of energies
    """
    energies = []
    for v in samples:
        E_loc = ising_model.local_energy(v, lambda v, i: rbm.psi_ratio(v, i))
        energies.append(E_loc)
    return np.array(energies)


def visualize_sampling():
    """Main visualization function."""
    
    # Setup
    print("=" * 70)
    print("VISUALIZATION: RBM Sampling vs Ground State")
    print("=" * 70)
    
    # Parameters
    system_size = 8
    h = 0.5  # Transverse field strength
    n_samples = 500
    
    # Create Ising model (ground truth)
    ising = TransverseFieldIsing1D(size=system_size, h=h)
    exact_E0 = ising.exact_ground_energy()
    print(f"\nSystem: 1D Ising, size={system_size}, h={h}")
    print(f"Exact ground state energy: {exact_E0:.6f}")
    
    # Create RBM
    rbm = FullyConnectedRBM(n_visible=system_size, n_hidden=system_size)
    sampler = ClassicalSampler()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'RBM Sampling: 1D Ising Model (size={system_size}, h={h})', fontsize=14)
    
    # ========== SUBPLOT 1: Metropolis Energy Distribution ==========
    print("\n1. Sampling with Metropolis-Hastings...")
    samples_mh = sampler.sample(rbm, n_samples, config={
        'method': 'metropolis',
        'n_sweeps': 100,
        'n_between': 5
    })
    energies_mh = compute_sample_energies(samples_mh, ising, rbm)
    
    ax = axes[0, 0]
    ax.hist(energies_mh, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(exact_E0, color='red', linestyle='--', linewidth=2, label=f'Exact GS: {exact_E0:.4f}')
    ax.axvline(np.mean(energies_mh), color='green', linestyle='-', linewidth=2, label=f'Sample mean: {np.mean(energies_mh):.4f}')
    ax.set_xlabel('Local Energy', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Metropolis-Hastings Energy Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    print(f"   Mean energy: {np.mean(energies_mh):.6f}")
    print(f"   Std energy:  {np.std(energies_mh):.6f}")
    print(f"   Min energy:  {np.min(energies_mh):.6f}")
    print(f"   Error vs GS: {np.mean(energies_mh) - exact_E0:.6f}")
    
    # ========== SUBPLOT 2: Simulated Annealing Energy Distribution ==========
    print("\n2. Sampling with Simulated Annealing...")
    samples_sa = sampler.sample(rbm, n_samples, config={
        'method': 'simulated_annealing',
        'T_initial': 10.0,
        'T_final': 0.01,
        'n_steps': 5000
    })
    energies_sa = compute_sample_energies(samples_sa, ising, rbm)
    
    ax = axes[0, 1]
    ax.hist(energies_sa, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(exact_E0, color='red', linestyle='--', linewidth=2, label=f'Exact GS: {exact_E0:.4f}')
    ax.axvline(np.mean(energies_sa), color='green', linestyle='-', linewidth=2, label=f'Sample mean: {np.mean(energies_sa):.4f}')
    ax.set_xlabel('Local Energy', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Simulated Annealing Energy Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    print(f"   Mean energy: {np.mean(energies_sa):.6f}")
    print(f"   Std energy:  {np.std(energies_sa):.6f}")
    print(f"   Min energy:  {np.min(energies_sa):.6f}")
    print(f"   Error vs GS: {np.mean(energies_sa) - exact_E0:.6f}")
    
    # ========== SUBPLOT 3: Energy Comparison ==========
    ax = axes[1, 0]
    
    # Create bins
    bins = np.linspace(
        min(np.min(energies_mh), np.min(energies_sa), exact_E0) - 1,
        max(np.max(energies_mh), np.max(energies_sa)) + 1,
        40
    )
    
    ax.hist(energies_mh, bins=bins, alpha=0.5, label='Metropolis-Hastings', color='blue', edgecolor='black')
    ax.hist(energies_sa, bins=bins, alpha=0.5, label='Simulated Annealing', color='orange', edgecolor='black')
    ax.axvline(exact_E0, color='red', linestyle='--', linewidth=2.5, label=f'Exact Ground State')
    ax.set_xlabel('Local Energy', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Comparison: Both Samplers', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # ========== SUBPLOT 4: Statistics Table ==========
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create comparison table
    stats_text = f"""
    SAMPLING STATISTICS
    {'─' * 50}
    
    METROPOLIS-HASTINGS:
      Mean energy:        {np.mean(energies_mh):8.6f}
      Std deviation:      {np.std(energies_mh):8.6f}
      Min energy:         {np.min(energies_mh):8.6f}
      Max energy:         {np.max(energies_mh):8.6f}
      Error vs GS:        {np.mean(energies_mh) - exact_E0:8.6f}
    
    SIMULATED ANNEALING:
      Mean energy:        {np.mean(energies_sa):8.6f}
      Std deviation:      {np.std(energies_sa):8.6f}
      Min energy:         {np.min(energies_sa):8.6f}
      Max energy:         {np.max(energies_sa):8.6f}
      Error vs GS:        {np.mean(energies_sa) - exact_E0:8.6f}
    
    EXACT GROUND STATE:
      Energy:             {exact_E0:8.6f}
    
    {'─' * 50}
    SA better than MH? {np.mean(energies_sa) < np.mean(energies_mh)}
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to: sampling_comparison.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Both samplers should have energies LOWER than random (expected ~-2 to -4)")
    print(f"Both should approach the exact ground state energy: {exact_E0:.6f}")
    print(f"Simulated Annealing may perform better by converging toward minima")
    print("=" * 70)


def compare_rbm_architectures():
    """Compare fully-connected vs D-Wave topology RBM."""
    
    print("\n" + "=" * 70)
    print("COMPARISON: RBM Architectures")
    print("=" * 70)
    
    system_size = 8
    h = 0.5
    n_samples = 300
    
    ising = TransverseFieldIsing1D(size=system_size, h=h)
    exact_E0 = ising.exact_ground_energy()
    sampler = ClassicalSampler()
    
    # Compare architectures
    architectures = [
        ('Fully Connected', FullyConnectedRBM(system_size, system_size)),
        ('D-Wave Topology', DWaveTopologyRBM(system_size, system_size))
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'RBM Architecture Comparison (1D Ising, size={system_size}, h={h})', fontsize=14)
    
    for idx, (name, rbm) in enumerate(architectures):
        print(f"\nTesting {name}...")
        
        samples = sampler.sample(rbm, n_samples, config={
            'method': 'metropolis',
            'n_sweeps': 100,
            'n_between': 3
        })
        energies = compute_sample_energies(samples, ising, rbm)
        
        ax = axes[idx]
        ax.hist(energies, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(exact_E0, color='red', linestyle='--', linewidth=2.5, label=f'Exact GS: {exact_E0:.4f}')
        ax.axvline(np.mean(energies), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(energies):.4f}')
        ax.set_xlabel('Local Energy', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Count parameters
        n_params_a = rbm.n_visible
        n_params_b = rbm.n_hidden
        n_params_w = np.count_nonzero(rbm.get_connectivity_mask() * rbm.W)
        n_params_total = n_params_a + n_params_b + n_params_w
        
        print(f"   Parameters: {n_params_total} (a:{n_params_a}, b:{n_params_b}, W:{n_params_w})")
        print(f"   Mean energy: {np.mean(energies):.6f}")
        print(f"   Error vs GS: {np.mean(energies) - exact_E0:.6f}")
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to: architecture_comparison.png")
    plt.show()


if __name__ == '__main__':
    # Run visualizations
    visualize_sampling()
    compare_rbm_architectures()
    
    print("\n✓ All visualizations complete!")
    print("  - sampling_comparison.png")
    print("  - architecture_comparison.png")
