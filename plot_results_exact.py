#!/usr/bin/env python3
"""
Plot convergence to exact ground state energy.
One plot per (N, h) combination showing how all methods converge to the exact ground state.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import subprocess
import sys

RESULTS_DIR = Path("results")

def compute_exact_energy(N, h):
    """Compute exact ground state energy using Bethe ansatz analytical solution."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/exact_diag_ising_analytical.py", "-N", str(N), "-g", str(h)],
            capture_output=True,
            text=True,
            timeout=10
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
    """Load all result files organized by (N, h) and method."""
    results = defaultdict(lambda: defaultdict(list))
    
    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            config = data["config"]
            N = config["size"]
            h = config["h"]
            sampler = config["sampler"]
            sampling_method = config["sampling_method"]
            seed = config["seed"]
            
            method_name = f"{sampler}/{sampling_method}"
            results[(N, h)][method_name].append({
                "data": data,
                "config": config,
                "seed": seed,
            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def plot_convergence_to_exact(results):
    """
    Create one plot per (N, h) combo.
    Each plot shows all sampling methods converging to the exact ground state.
    """
    
    # Get unique (N, h) combinations
    combos = sorted(results.keys())
    
    # Color palette for different methods
    colors = {
        "custom/metropolis": "#1f77b4",
        "dimod/pegasus": "#ff7f0e",
        "dimod/simulated_annealing": "#2ca02c",
        "dimod/zephyr": "#d62728",
        "velox/velox": "#9467bd",
    }
    
    figs = []
    
    for N, h in combos:
        # Compute exact energy per spin
        print(f"Computing exact ground state energy for N={N}, h={h}...")
        exact_E_per_spin = compute_exact_energy(N, h)
        
        if exact_E_per_spin is None:
            print(f"  Skipping N={N}, h={h} - could not compute exact energy")
            continue
        
        print(f"  Exact energy per spin: {exact_E_per_spin:.6f}")
        
        methods_data = results[(N, h)]
        
        # Create figure with 2 subplots: energy and delta
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for method_name in sorted(methods_data.keys()):
            runs = methods_data[method_name]
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
                
                # Energy per spin
                mean_energy_per_spin = mean_energy / N
                std_energy_per_spin = std_energy / N
                
                # Delta (error) per spin
                delta = np.abs(mean_energy_per_spin - exact_E_per_spin)
                delta_std = std_energy_per_spin  # std of energy translates to std of delta
                
                color = colors.get(method_name, None)
                
                # Plot 1: Energy convergence
                ax1.plot(iterations, mean_energy_per_spin, label=method_name, color=color, 
                       linewidth=2, alpha=0.8)
                ax1.fill_between(iterations, 
                               mean_energy_per_spin - std_energy_per_spin, 
                               mean_energy_per_spin + std_energy_per_spin, 
                               color=color, alpha=0.15)
                
                # Plot 2: Error/Delta convergence
                ax2.semilogy(iterations, delta, label=method_name, color=color,
                           linewidth=2, alpha=0.8)
                ax2.fill_between(iterations, 
                               delta - delta_std, 
                               delta + delta_std, 
                               color=color, alpha=0.15)
        
        # Add exact ground state energy line to plot 1
        ax1.axhline(y=exact_E_per_spin, color='black', linestyle='--', linewidth=2.5, 
                  label=f'Exact: {exact_E_per_spin:.6f}', zorder=10)
        
        # Configure plot 1
        ax1.set_xscale("log")
        ax1.set_xlabel("# iterations", fontsize=12)
        ax1.set_ylabel("Energy per spin", fontsize=12)
        ax1.set_title(f"Convergence to Ground State (N={N}, h={h})", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(fontsize=10, loc='best')
        
        # Configure plot 2 (delta)
        ax2.set_xscale("log")
        ax2.set_xlabel("# iterations", fontsize=12)
        ax2.set_ylabel("Δ E per spin = |E - E_exact|", fontsize=12)
        ax2.set_title(f"Error to Ground State (N={N}, h={h})", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        
        filename = f"convergence_N{N}_h{h}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        figs.append((filename, fig))
        plt.close(fig)
    
    return figs

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY (PER SPIN)")
    print("="*80)
    
    for N, h in sorted(results.keys()):
        print(f"\n(N={N}, h={h}):")
        
        exact_E_per_spin = compute_exact_energy(N, h)
        if exact_E_per_spin:
            print(f"  Exact ground state energy per spin: {exact_E_per_spin:.6f}")
        
        for method_name in sorted(results[(N, h)].keys()):
            runs = results[(N, h)][method_name]
            final_energies = [run["data"]["history"]["energy"][-1] / N for run in runs]
            final_errors = [abs(e - exact_E_per_spin) for e in final_energies] if exact_E_per_spin else []
            
            print(f"  {method_name}:")
            print(f"    Num runs: {len(runs)}")
            if final_errors:
                print(f"    Final Δ E per spin: {np.mean(final_errors):.6f} ± {np.std(final_errors):.6f}")
            print(f"    Final energy per spin: {np.mean(final_energies):.6f} ± {np.std(final_energies):.6f}")

if __name__ == "__main__":
    print("Loading results...")
    results = load_results()
    
    print_summary(results)
    
    print("\n" + "="*80)
    print("Generating convergence plots...")
    print("="*80)
    figs = plot_convergence_to_exact(results)
    
    print("\n" + "="*80)
    print(f"Done! Generated {len(figs)} plots.")
    print("="*80)
