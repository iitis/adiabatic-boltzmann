#!/usr/bin/env python3
"""
Plot convergence to exact ground state energy.
One plot per (model, N, h, RBM) combination showing how all methods converge to the exact ground state.
Additional comparison plot showing different RBMs for a given (model, N, h).

Plots are saved to plots/energy_convergence/{model}/{rbm}/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "energy_convergence"

METHOD_COLORS = {
    "custom/metropolis":        "#1f77b4",
    "custom/sbm":               "#e377c2",
    "dimod/pegasus":            "#ff7f0e",
    "dimod/simulated_annealing":"#2ca02c",
    "dimod/zephyr":             "#d62728",
    "velox/velox":              "#9467bd",
    "fpga/fpga":                "#17becf",
}

# Colour gradient for FPGA runs at different learning rates (coolwarm, blue→red)
_FPGA_LR_COLORS = [
    "#08519c",  # lr=1e-4
    "#3182bd",  # lr=3e-4
    "#17becf",  # lr=1e-3
    "#e6550d",  # lr=3e-3
    "#a50f15",  # lr=1e-2
]


def _method_color(method_name: str):
    """Return a colour for a method name, with special handling for FPGA LR variants."""
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    if method_name.startswith("fpga/fpga lr="):
        try:
            lr = float(method_name.split("lr=")[1])
            known = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
            if lr in known:
                return _FPGA_LR_COLORS[known.index(lr)]
        except ValueError:
            pass
    return None  # let matplotlib use its default colour cycle

# Reference energies per spin for 2D TFIM (thermodynamic limit)
# Source: Blöte & Deng (2002), Albuquerque et al. (2010)
EXACT_ENERGY_2D_PER_SPIN = {
    0.5: -2.0555,
    1.0: -2.1276,
    2.0: -2.4549,
    3.044: -3.0440,  # critical point
}


EXACT_ENERGY_CACHE_FILE = ROOT / "scripts" / "exact_energy_cache.json"


def _load_cache() -> dict:
    if EXACT_ENERGY_CACHE_FILE.exists():
        with open(EXACT_ENERGY_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(EXACT_ENERGY_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def compute_exact_energy(model, N, h):
    """Return exact ground state energy per spin.

    For 1D: uses Bethe ansatz analytical solution (calls exact_diag_ising_analytical.py).
      Results are cached in scripts/exact_energy_cache.json to avoid repeated subprocess calls.
    For 2D: uses literature reference values (thermodynamic limit).
    """
    if model == "2d":
        if h not in EXACT_ENERGY_2D_PER_SPIN:
            print(f"No 2D reference energy for h={h}. Known values: {list(EXACT_ENERGY_2D_PER_SPIN.keys())}")
            return None
        return EXACT_ENERGY_2D_PER_SPIN[h]

    # 1D: check cache first
    cache_key = f"1d_N{N}_h{h}"
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    # Not cached — run the Bethe ansatz script
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "exact_diag_ising_analytical.py"), "-N", str(N), "-g", str(h)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse output: "Ground state energy (N=X, h=Y): Z"
            line = result.stdout.strip().split('\n')[-1]
            energy = float(line.split(": ")[-1])
            cache[cache_key] = energy
            _save_cache(cache)
            return energy
        else:
            print(f"Error computing exact energy: {result.stderr}")
            return None
    except Exception as e:
        print(f"Failed to compute exact energy for N={N}, h={h}: {e}")
        return None

def load_results(results_dir=RESULTS_DIR):
    """Load all result files organized by (model, N, h, RBM) and method.

    Filters:
    - learning_rate == 0.1  (non-FPGA only; FPGA includes all LRs)
    - n_hidden == n_visible  (n_visible = N for 1d, N*N for 2d)
    - cem == False
    """
    results = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            config = data["config"]
            model = config["model"]
            N = config["size"]
            h = config["h"]
            rbm = config["rbm"]
            sampler = config["sampler"]
            sampling_method = config["sampling_method"]
            seed = config["seed"]
            lr = config["learning_rate"]
            n_hidden = config["n_hidden"]

            n_visible = N if model == "1d" else N * N

            is_fpga = sampler == "fpga"

            if not is_fpga and lr != 0.1:
                continue
            if n_hidden != n_visible:
                continue
            if config.get("cem", False):
                continue

            if is_fpga:
                method_name = f"fpga/fpga lr={lr:.4g}"
            else:
                method_name = f"{sampler}/{sampling_method}"
            # Key includes model, N, h, and RBM type
            results[(model, N, h, rbm)][method_name].append({
                "data": data,
                "config": config,
                "seed": seed,
                "n_visible": n_visible,
            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results

def plot_convergence_to_exact(results):
    """
    Create one plot per (N, h, rbm) combo.
    Each plot shows all sampling methods converging to the exact ground state.
    Saves plots in plots/{rbm}/ directory.
    """
    
    # Get unique (model, N, h, rbm) combinations
    combos = sorted(results.keys())

    figs = []

    for model, N, h, rbm in combos:
        # Compute exact energy per spin
        print(f"Computing exact ground state energy for model={model}, N={N}, h={h}...")
        exact_E_per_spin = compute_exact_energy(model, N, h)

        if exact_E_per_spin is None:
            print(f"  Skipping model={model}, N={N}, h={h}, RBM={rbm} - could not compute exact energy")
            continue

        print(f"  Exact energy per spin: {exact_E_per_spin:.6f} [model={model}, RBM={rbm}]")

        methods_data = results[(model, N, h, rbm)]
        
        # Create figure with 2 subplots: energy and delta
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for method_name in sorted(methods_data.keys()):
            runs = methods_data[method_name]
            all_energies = []
            min_len = float('inf')
            n_visible = runs[0]["n_visible"]

            for run in runs:
                energy = run["data"]["history"]["energy"]
                all_energies.append(energy)
                min_len = min(min_len, len(energy))

            if all_energies and min_len > 0:
                # Truncate to common length
                all_energies = [e[:min_len] for e in all_energies]
                mean_energy = np.mean(all_energies, axis=0)
                iterations = np.arange(len(mean_energy))

                mean_energy_per_spin = mean_energy / n_visible
                delta = np.abs(mean_energy_per_spin - exact_E_per_spin)

                color = _method_color(method_name)
                
                # Plot 1: Energy convergence
                ax1.plot(iterations, mean_energy_per_spin, label=method_name, color=color,
                         linewidth=2, alpha=0.8)

                # Plot 2: Error/Delta convergence
                ax2.semilogy(iterations, delta, label=method_name, color=color,
                             linewidth=2, alpha=0.8)
        
        # Add exact ground state energy line to plot 1
        ax1.axhline(y=exact_E_per_spin, color='black', linestyle='--', linewidth=2.5, 
                  label=f'Exact: {exact_E_per_spin:.6f}', zorder=10)
        
        # Configure plot 1
        ax1.set_xscale("log")
        ax1.set_xlabel("# iterations", fontsize=12)
        ax1.set_ylabel("Energy per spin", fontsize=12)
        ax1.set_title(f"Convergence (model={model}, N={N}, h={h}, RBM={rbm})", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(fontsize=10, loc='best')
        
        # Configure plot 2 (delta)
        ax2.set_xscale("log")
        ax2.set_xlabel("# iterations", fontsize=12)
        ax2.set_ylabel("Δ E per spin = |E - E_exact|", fontsize=12)
        ax2.set_title(f"Error to Ground State (model={model}, N={N}, h={h}, RBM={rbm})", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        
        # Create directory for this model/RBM type if it doesn't exist
        rbm_dir = PLOTS_DIR / model / rbm
        rbm_dir.mkdir(parents=True, exist_ok=True)

        filename = rbm_dir / f"convergence_N{N}_h{h}_rbm{rbm}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        figs.append((str(filename), fig))
        plt.close(fig)
    
    return figs

def plot_rbm_comparison(results):
    """
    Create comparison plots for different RBMs at the same (N, h).
    Shows how different RBMs perform relative to each other.
    Saves in plots/comparisons/ directory.
    """
    # Group by (model, N, h) to see which RBMs are available
    nh_groups = defaultdict(list)
    for model, N, h, rbm in results.keys():
        nh_groups[(model, N, h)].append(rbm)
    
    figs = []

    for (model, N, h), rbm_list in nh_groups.items():
        if len(rbm_list) < 2:
            continue  # Skip if only one RBM type

        exact_E_per_spin = compute_exact_energy(model, N, h)
        if exact_E_per_spin is None:
            continue

        print(f"\nCreating RBM comparison plot for model={model}, N={N}, h={h}")
        print(f"  Available RBMs: {rbm_list}")

        # Create comparisons directory per model
        comp_dir = PLOTS_DIR / model / "comparisons"
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots for each RBM
        fig, axes = plt.subplots(len(rbm_list), 2, figsize=(16, 5*len(rbm_list)))
        if len(rbm_list) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, rbm in enumerate(sorted(rbm_list)):
            ax1, ax2 = axes[idx, 0], axes[idx, 1]
            methods_data = results[(model, N, h, rbm)]
            
            for method_name in sorted(methods_data.keys()):
                runs = methods_data[method_name]
                all_energies = []
                min_len = float('inf')
                n_visible = runs[0]["n_visible"]

                for run in runs:
                    energy = run["data"]["history"]["energy"]
                    all_energies.append(energy)
                    min_len = min(min_len, len(energy))

                if all_energies and min_len > 0:
                    all_energies = [e[:min_len] for e in all_energies]
                    mean_energy = np.mean(all_energies, axis=0)
                    iterations = np.arange(len(mean_energy))

                    mean_energy_per_spin = mean_energy / n_visible
                    delta = np.abs(mean_energy_per_spin - exact_E_per_spin)
                    
                    color = _method_color(method_name)
                    
                    # Left: Energy
                    ax1.plot(iterations, mean_energy_per_spin, label=method_name,
                             color=color, linewidth=2, alpha=0.8)

                    # Right: Delta
                    ax2.semilogy(iterations, delta, label=method_name,
                                 color=color, linewidth=2, alpha=0.8)
            
            # Add exact energy line
            ax1.axhline(y=exact_E_per_spin, color='black', linestyle='--', 
                       linewidth=2.5, label=f'Exact: {exact_E_per_spin:.6f}', zorder=10)
            
            # Configure subplots
            ax1.set_xscale("log")
            ax1.set_xlabel("# iterations", fontsize=11)
            ax1.set_ylabel("Energy per spin", fontsize=11)
            ax1.set_title(f"model={model}, RBM={rbm} - Convergence", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, which='both')
            ax1.legend(fontsize=9, loc='best')
            
            ax2.set_xscale("log")
            ax2.set_xlabel("# iterations", fontsize=11)
            ax2.set_ylabel("Δ E per spin", fontsize=11)
            ax2.set_title(f"model={model}, RBM={rbm} - Error", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, which='both')
            ax2.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        filename = comp_dir / f"rbm_comparison_N{N}_h{h}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        figs.append((str(filename), fig))
        plt.close(fig)
    
    return figs

def plot_summary_pages(results):
    """
    One summary figure per (model, rbm).
    All (N, h) combinations are laid out in a grid (ncols = min(3, n_combos)),
    each cell showing energy convergence and error side by side.
    Saved as plots/{model}/{rbm}/summary.png.
    """
    # Group (N, h) combos by (model, rbm)
    model_rbm_groups = defaultdict(list)
    for model, N, h, rbm in sorted(results.keys()):
        model_rbm_groups[(model, rbm)].append((N, h))

    figs = []

    for (model, rbm), nh_list in sorted(model_rbm_groups.items()):
        n = len(nh_list)
        ncombos_per_row = min(3, n)
        nrows = (n + ncombos_per_row - 1) // ncombos_per_row

        # Outer figure: grid of subfigures, one per (N, h) combo.
        # Each subfigure gets a light background + border to visually group
        # its two panels (energy + delta) as a unit.
        fig = plt.figure(figsize=(10 * ncombos_per_row, 5 * nrows))
        fig.suptitle(f"Summary — model={model}, RBM={rbm}", fontsize=14, fontweight='bold')

        subfigs = fig.subfigures(nrows, ncombos_per_row, wspace=0.04, hspace=0.08)
        # Normalise to 2D array
        subfigs = np.array(subfigs).reshape(nrows, ncombos_per_row)

        for idx, (N, h) in enumerate(sorted(nh_list)):
            row = idx // ncombos_per_row
            col = idx % ncombos_per_row
            subfig = subfigs[row, col]

            # Light background + rounded border to visually group the two panels
            subfig.patch.set_facecolor('#f5f5f5')
            subfig.patch.set_edgecolor('#aaaaaa')
            subfig.patch.set_linewidth(1.2)

            exact_E_per_spin = compute_exact_energy(model, N, h)
            if exact_E_per_spin is None:
                subfig.patch.set_visible(False)
                continue

            ax1, ax2 = subfig.subplots(1, 2)

            methods_data = results[(model, N, h, rbm)]

            for method_name in sorted(methods_data.keys()):
                runs = methods_data[method_name]
                all_energies = []
                min_len = float('inf')
                n_visible = runs[0]["n_visible"]

                for run in runs:
                    energy = run["data"]["history"]["energy"]
                    all_energies.append(energy)
                    min_len = min(min_len, len(energy))

                if not all_energies or min_len == 0:
                    continue

                all_energies = [e[:min_len] for e in all_energies]
                mean_energy = np.mean(all_energies, axis=0)
                iterations = np.arange(len(mean_energy))

                mean_energy_per_spin = mean_energy / n_visible
                delta = np.abs(mean_energy_per_spin - exact_E_per_spin)

                color = _method_color(method_name)
                ax1.plot(iterations, mean_energy_per_spin, label=method_name,
                         color=color, linewidth=1.5, alpha=0.8)
                ax2.semilogy(iterations, delta, label=method_name,
                             color=color, linewidth=1.5, alpha=0.8)

            ax1.axhline(y=exact_E_per_spin, color='black', linestyle='--',
                        linewidth=1.5, label=f'Exact: {exact_E_per_spin:.4f}', zorder=10)

            subfig.suptitle(f"N={N}, h={h}", fontsize=11, fontweight='bold')

            for ax, ylabel, title_suffix in [
                (ax1, "Energy per spin", "Convergence"),
                (ax2, "Δ E per spin", "Error"),
            ]:
                ax.set_xscale("log")
                ax.set_xlabel("# iterations", fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(title_suffix, fontsize=9)
                ax.grid(True, alpha=0.3, which='both')
                ax.legend(fontsize=7, loc='best')

        # Hide unused subfigures in the last row
        for idx in range(n, nrows * ncombos_per_row):
            row = idx // ncombos_per_row
            col = idx % ncombos_per_row
            subfigs[row, col].set_visible(False)

        rbm_dir = PLOTS_DIR / model / rbm
        rbm_dir.mkdir(parents=True, exist_ok=True)
        filename = rbm_dir / "summary.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved summary: {filename}")
        figs.append(str(filename))
        plt.close(fig)

    return figs


def plot_beta_overview(results):
    """
    Summary grid (same layout as plot_summary_pages) showing beta_x dynamics.

    One figure per (model, rbm).  Each cell = one (N, h) combo.
    Each cell shows:
      - beta_x trajectory per sampler  (solid line, mean over seeds)
      - beta_eff_cem scatter points     (open markers, same colour)
      - dashed reference line at β = 1
    Saved as plots/{model}/{rbm}/beta_overview.png.
    """
    model_rbm_groups = defaultdict(list)
    for model, N, h, rbm in sorted(results.keys()):
        model_rbm_groups[(model, rbm)].append((N, h))

    figs = []

    for (model, rbm), nh_list in sorted(model_rbm_groups.items()):
        n = len(nh_list)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig = plt.figure(figsize=(7 * ncols, 4 * nrows))
        fig.suptitle(
            f"β_x Dynamics — model={model}, RBM={rbm}",
            fontsize=14, fontweight="bold",
        )

        subfigs = np.array(
            fig.subfigures(nrows, ncols, wspace=0.04, hspace=0.10)
        ).reshape(nrows, ncols)

        for idx, (N, h) in enumerate(sorted(nh_list)):
            row, col = divmod(idx, ncols)
            subfig = subfigs[row, col]
            subfig.patch.set_facecolor("#f5f5f5")
            subfig.patch.set_edgecolor("#aaaaaa")
            subfig.patch.set_linewidth(1.2)

            ax = subfig.subplots(1, 1)
            methods_data = results[(model, N, h, rbm)]

            for method_name in sorted(methods_data.keys()):
                runs = methods_data[method_name]
                c = _method_color(method_name)

                # ── beta_x mean trajectory ────────────────────────────────
                bx_arrays = [
                    r["data"]["history"].get("beta_x", [])
                    for r in runs
                    if r["data"]["history"].get("beta_x")
                ]
                if bx_arrays:
                    max_len = max(len(a) for a in bx_arrays)
                    mat = np.full((len(bx_arrays), max_len), np.nan)
                    for i, a in enumerate(bx_arrays):
                        mat[i, : len(a)] = a
                    mean_bx = np.nanmean(mat, axis=0)
                    ax.plot(
                        np.arange(len(mean_bx)), mean_bx,
                        color=c, linewidth=1.8, alpha=0.85,
                        label=method_name,
                    )

                # ── beta_eff_cem scatter (sparse — only non-None entries) ──
                for r in runs:
                    raw = r["data"]["history"].get("beta_eff_cem", [])
                    iters = [i for i, v in enumerate(raw) if v is not None]
                    vals  = [v for v in raw if v is not None]
                    if iters:
                        ax.scatter(
                            iters, vals,
                            color=c, s=20, marker="o",
                            edgecolors="black", linewidths=0.4,
                            zorder=5, alpha=0.8,
                        )

            ax.axhline(1.0, color="grey", lw=1.2, ls="--", alpha=0.6)
            subfig.suptitle(f"N={N}, h={h}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Iteration", fontsize=8)
            ax.set_ylabel("β_x", fontsize=8)
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)

        # Hide unused cells
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            subfigs[row, col].set_visible(False)

        rbm_dir = PLOTS_DIR / model / rbm
        rbm_dir.mkdir(parents=True, exist_ok=True)
        filename = rbm_dir / "beta_overview.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
        figs.append(str(filename))
        plt.close(fig)

    return figs


def print_summary(results):
    """Print summary statistics grouped by (N, h, RBM)."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY (PER SPIN)")
    print("="*80)
    
    for model, N, h, rbm in sorted(results.keys()):
        print(f"\n(model={model}, N={N}, h={h}, RBM={rbm}):")

        exact_E_per_spin = compute_exact_energy(model, N, h)
        if exact_E_per_spin:
            print(f"  Exact ground state energy per spin: {exact_E_per_spin:.6f}")

        for method_name in sorted(results[(model, N, h, rbm)].keys()):
            runs = results[(model, N, h, rbm)][method_name]
            final_energies = [run["data"]["history"]["energy"][-1] / run["n_visible"] for run in runs]
            final_errors = [abs(e - exact_E_per_spin) for e in final_energies] if exact_E_per_spin else []
            
            print(f"  {method_name}:")
            print(f"    Num runs: {len(runs)}")
            if final_errors:
                print(f"    Final Δ E per spin: {np.mean(final_errors):.6f} ± {np.std(final_errors):.6f}")
            print(f"    Final energy per spin: {np.mean(final_energies):.6f} ± {np.std(final_energies):.6f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot VMC convergence results")
    parser.add_argument(
        "--rbm",
        choices=["full", "pegasus", "zephyr"],
        default=None,
        help="Only plot results for this RBM type (default: all)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR,
        help="Path to results directory (default: results/)",
    )
    args = parser.parse_args()

    print("Loading results...")
    results = load_results(args.results)

    if args.rbm is not None:
        results = {k: v for k, v in results.items() if k[3] == args.rbm}
        print(f"Filtered to RBM={args.rbm}: {len(results)} combinations")

    print_summary(results)

    print("\n" + "="*80)
    print("Generating convergence plots (one per N, h, RBM)...")
    print("="*80)
    figs1 = plot_convergence_to_exact(results)

    print("\n" + "="*80)
    print("Generating RBM comparison plots (comparing RBMs for same N, h)...")
    print("="*80)
    figs2 = plot_rbm_comparison(results)

    print("\n" + "="*80)
    print("Generating summary pages (one per model/RBM)...")
    print("="*80)
    figs3 = plot_summary_pages(results)

    print("\n" + "="*80)
    print("Generating beta_x overview (one per model/RBM)...")
    print("="*80)
    figs4 = plot_beta_overview(results)

    print("\n" + "="*80)
    print(f"Done! Generated {len(figs1)} convergence plots, {len(figs2)} RBM comparison plots, "
          f"{len(figs3)} summary pages, {len(figs4)} beta overview plots.")
    print("="*80)
