#!/usr/bin/env python3
"""
Reproduce plots from results JSON files.
Figures 3 & 4: Ground state energy convergence across different system sizes and methods.
Shows convergence to exact ground state energy (from exact diagonalization).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import subprocess
import sys

# Configuration
RESULTS_DIR = Path("results")

def compute_exact_energy(N, h):
    """Compute exact ground state energy using NetKet's exact diagonalization."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/exact_diag_ising.py", "-N", str(N), "-g", str(h)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            # Parse output: "Ground state energy (N=X, h=Y): Z"
            line = result.stdout.strip().split('\n')[-1]
            energy = float(line.split(": ")[-1])
            return energy
        else:
            print(f"Error computing exact energy: {result.stderr}")
            return None
    except Exception as e:
        print(f"Failed to compute exact energy for N={N}, h={h}: {e}")
        return None

def load_results(results_dir=RESULTS_DIR):
    """Load all result files organized by system size and method."""
    results = defaultdict(lambda: defaultdict(list))
    
    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse from config
            config = data["config"]
            size = config["size"]
            sampler = config["sampler"]
            sampling_method = config["sampling_method"]
            model = config["model"]
            seed = config["seed"]
            
            # Organize by size and method
            method_name = f"{sampler}/{sampling_method}"
            results[size][method_name].append({
                "data": data,
                "config": config,
                "seed": seed,
                "model": model,
            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def plot_energy_convergence_by_size(results):
    """
    Plot energy vs iterations for different system sizes.
    Similar to Figure 3: Shows convergence for L=16, 32, 64.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sizes = sorted(results.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        methods_data = results[size]
        
        # Color palette for different methods
        colors = {
            "custom/metropolis": "C0",
            "dimod/pegasus": "C1",
            "dimod/simulated_annealing": "C2",
            "velox/velox": "C3",
        }
        
        for method_name, runs in methods_data.items():
            # Average over seeds - handle different lengths
            all_energies = []
            min_len = float('inf')
            
            for run in runs:
                energy = run["data"]["history"]["energy"]
                all_energies.append(energy)
                min_len = min(min_len, len(energy))
            
            if all_energies and min_len > 0:
                # Truncate to common length
                all_energies = [e[:min_len] for e in all_energies]
                mean_energy = np.mean(all_energies, axis=0)
                std_energy = np.std(all_energies, axis=0)
                iterations = np.arange(len(mean_energy))
                
                color = colors.get(method_name, None)
                ax.plot(iterations, mean_energy, label=method_name, color=color, alpha=0.7)
                ax.fill_between(iterations, 
                               mean_energy - std_energy, 
                               mean_energy + std_energy, 
                               color=color, alpha=0.2)
        
        ax.set_xscale("log")
        ax.set_xlabel("# iteration")
        ax.set_ylabel("E")
        ax.set_title(f"L = {size}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("convergence_by_size.png", dpi=150)
    print("Saved: convergence_by_size.png")
    plt.show()

def plot_energy_convergence_by_method(results):
    """
    Plot energy vs iterations for different methods.
    Shows performance across system sizes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    all_methods = set()
    for size_data in results.values():
        all_methods.update(size_data.keys())
    
    all_methods = sorted(all_methods)
    
    for idx, method_name in enumerate(all_methods):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        sizes = sorted(results.keys())
        
        for size in sizes:
            if method_name in results[size]:
                runs = results[size][method_name]
                all_energies = []
                min_len = float('inf')
                
                for run in runs:
                    energy = run["data"]["history"]["energy"]
                    all_energies.append(energy)
                    min_len = min(min_len, len(energy))
                
                if all_energies and min_len > 0:
                    all_energies = [e[:min_len] for e in all_energies]
                    mean_energy = np.mean(all_energies, axis=0)
                    iterations = np.arange(len(mean_energy))
                    
                    ax.plot(iterations, mean_energy, label=f"L={size}", marker="o", markersize=2)
        
        ax.set_xscale("log")
        ax.set_xlabel("# iteration")
        ax.set_ylabel("E")
        ax.set_title(method_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(all_methods), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig("convergence_by_method.png", dpi=150)
    print("Saved: convergence_by_method.png")
    plt.show()

def plot_error_vs_iterations(results):
    """
    Plot relative error vs iterations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sizes = sorted(results.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        methods_data = results[size]
        exact_energy = EXACT_ENERGIES.get(size, 0.5)
        
        for method_name, runs in methods_data.items():
            all_errors = []
            min_len = float('inf')
            
            for run in runs:
                energy = run["data"]["history"]["energy"]
                error = np.abs(np.array(energy) - exact_energy)
                all_errors.append(error)
                min_len = min(min_len, len(error))
            
            if all_errors and min_len > 0:
                all_errors = [e[:min_len] for e in all_errors]
                mean_error = np.mean(all_errors, axis=0)
                iterations = np.arange(len(mean_error))
                
                ax.semilogy(iterations, mean_error, label=method_name, alpha=0.7)
        
        ax.set_xscale("log")
        ax.set_xlabel("# iteration")
        ax.set_ylabel("|E|")
        ax.set_title(f"L = {size}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("error_vs_iterations.png", dpi=150)
    print("Saved: error_vs_iterations.png")
    plt.show()

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for size in sorted(results.keys()):
        print(f"\nSystem Size L={size}:")
        for method_name in sorted(results[size].keys()):
            runs = results[size][method_name]
            final_errors = [run["data"]["error"] for run in runs]
            
            print(f"  {method_name}:")
            print(f"    Num runs: {len(runs)}")
            print(f"    Final error: {np.mean(final_errors):.6f} ± {np.std(final_errors):.6f}")

if __name__ == "__main__":
    print("Loading results...")
    results = load_results()
    
    print_summary(results)
    
    print("\nGenerating plots...")
    plot_energy_convergence_by_size(results)
    plot_energy_convergence_by_method(results)
    plot_error_vs_iterations(results)
    
    print("\nDone! Check the generated PNG files.")
