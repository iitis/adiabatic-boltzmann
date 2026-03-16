import json
from pathlib import Path
import numpy as np


def save_results(args, history, ising):
    output_dir = Path(
        f"{args.output_dir}/{args.size}/{args.sampler}/{args.sampling_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": vars(args),
        "history": {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in history.items()
        },
        "final_energy": history["energy"][-1],
        "exact_energy": ising.exact_ground_energy(),
    }

    output_file = output_dir / f"result_{args.model}_h{args.h}_rbm{args.rbm}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Final energy: {results['final_energy']:.6f}")
    print(f"Exact energy: {results['exact_energy']:.6f}")
    print(f"Error: {abs(results['final_energy'] - results['exact_energy']):.6f}")

    # For plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 7. Plot if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history["energy"])
            plt.axhline(
                results["exact_energy"], color="r", linestyle="--", label="Exact"
            )
            plt.xlabel("Iteration")
            plt.ylabel("Energy")
            plt.title("Convergence")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history["error"])
            plt.xlabel("Iteration")
            plt.ylabel("Standard Error")
            plt.title("Energy Variance")

            plt.tight_layout()
            plot_file = plot_dir / f"plot_{args.model}_h{args.h}_rbm{args.rbm}.png"
            plt.savefig(plot_file, dpi=150)
            plt.show()
            print(f"Plot saved to {plot_file}")

        except ImportError:
            print("Matplotlib not available, skipping visualization")
