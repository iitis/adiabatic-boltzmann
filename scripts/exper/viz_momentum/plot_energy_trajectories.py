"""
Diagnostic: plot per-seed energy trajectories for the TFIM momentum sweep
(lr=0.08, damping=0.05, momentum=0.5, nh=N), to classify the failure mode:

  - All seeds flat by iter ~50 with energy >> E_exact   → variational floor
  - Some seeds still descending at iter ~90              → need more epochs
  - Seeds oscillating around a flat plateau              → optimizer overshooting

Reads from results/sweeps{100,2000}/tfim_1d/{N}/velox/simulated_annealing/
filtered to the lr=0.08 / reg=0.05 sweep.

Usage:
    python scripts/exper/viz_momentum/plot_energy_trajectories.py
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path("/home/lpawela/side-projects/qmz/adiabatic-boltzmann/results")
SIZES = [8, 12, 16, 24, 32, 64, 128]
SWEEP_COUNTS = [100, 2000]
LR_TAG = "lr0.08"
REG_TAG = "reg0.05"
OUT_PATH = Path(__file__).parent / "energy_trajectories.png"


def load_run(path: Path) -> dict | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def main() -> int:
    fig, axes = plt.subplots(
        len(SWEEP_COUNTS), len(SIZES),
        figsize=(3.0 * len(SIZES), 3.4 * len(SWEEP_COUNTS)),
        sharey=False,
        squeeze=False,
    )

    n_loaded = 0
    for row, sweeps in enumerate(SWEEP_COUNTS):
        for col, n in enumerate(SIZES):
            ax = axes[row][col]
            d = BASE / f"sweeps{sweeps}" / "tfim_1d" / str(n) / "velox" / "simulated_annealing"
            pattern = str(d / f"result_1d_h0.5_rbmfull_nh{n}_{LR_TAG}_{REG_TAG}_*.json")
            files = sorted(glob.glob(pattern))
            if not files:
                ax.set_title(f"N={n}  sweeps={sweeps}\n(no data)")
                ax.axis("off")
                continue
            exact = None
            for f in files:
                js = load_run(Path(f))
                if js is None:
                    continue
                energies = js["history"]["energy"]
                exact = js.get("exact_energy", exact)
                # Per-iter rel-error from exact (so the y-axis comparison is clean)
                # Plot raw energy for fidelity; horizontal line shows exact.
                ax.plot(energies, color="C0", alpha=0.35, linewidth=0.8)
                n_loaded += 1
            if exact is not None:
                ax.axhline(exact, color="C3", linestyle="--", linewidth=1.0, label="exact")
            ax.set_title(f"N={n}  sweeps={sweeps}  (n={len(files)})", fontsize=10)
            ax.set_xlabel("SR iteration")
            if col == 0:
                ax.set_ylabel("E")
            ax.grid(True, alpha=0.3)
            if col == 0 and row == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "TFIM momentum sweep: per-seed energy trajectory\n"
        f"(lr=0.08, damping=0.05, momentum=0.5, n_hidden=N, 100 SR iters)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Loaded {n_loaded} trajectories. Saved → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
