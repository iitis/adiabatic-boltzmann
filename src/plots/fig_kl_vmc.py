"""
KL distance vs VMC energy accuracy across samplers and TFIM instances.

For each of 5 TFIM instances (2×1D + 3×2D) and each sampler, trains a
fresh RBM for 300 iterations using the existing Trainer infrastructure.
Records the final KL distance (sampler vs |Ψ|², from Trainer.history)
and the final relative energy error vs the exact ground state.

Two-panel comparison plot (same grouped-bar style as fig_kl_samplers):
  Panel (a): KL distance per instance, one bar per sampler
  Panel (b): Relative energy error per instance, one bar per sampler

Saves results to results_kl/vmc_<instance_slug>.json for offline re-plotting.

Run from repo root:
    python src/plots/fig_kl_vmc.py [--plot-only]
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt

from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from encoder import Trainer
from sampler import ClassicalSampler, DimodSampler

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots", "results_kl")

# ---------------------------------------------------------------------------
# TFIM instances  (2×1D + 3×2D)
# ---------------------------------------------------------------------------

TFIM_INSTANCES = [
    {"label": "1D N=8 h=0.5",  "model": "1d", "size": 8, "h": 0.5},
    {"label": "1D N=8 h=1.0",  "model": "1d", "size": 8, "h": 1.0},
    {"label": "2D L=2 h=0.5",  "model": "2d", "size": 2, "h": 0.5},
    {"label": "2D L=2 h=1.0",  "model": "2d", "size": 2, "h": 1.0},
    {"label": "2D L=3 h=0.5",  "model": "2d", "size": 3, "h": 0.5},
]

# ---------------------------------------------------------------------------
# Sampler registry  (same structure as fig_kl_samplers)
# ---------------------------------------------------------------------------

SAMPLER_REGISTRY = [
    {
        "label":        "Metropolis",
        "make_sampler": lambda: ClassicalSampler(method="metropolis"),
    },
    {
        "label":        "SA (custom)",
        "make_sampler": lambda: ClassicalSampler(method="simulated_annealing"),
    },
    {
        "label":        "Gibbs",
        "make_sampler": lambda: ClassicalSampler(method="gibbs"),
    },
    {
        "label":        "LSB",
        "make_sampler": lambda: ClassicalSampler(method="lsb"),
    },
    {
        "label":        "SA (dimod)",
        "make_sampler": lambda: DimodSampler(method="simulated_annealing"),
    },
    {
        "label":        "Tabu",
        "make_sampler": lambda: DimodSampler(method="tabu"),
    },
    {
        "label":        "Pegasus (D-Wave)",
        "make_sampler": lambda: DimodSampler(method="pegasus"),
    },
    {
        "label":        "Zephyr (D-Wave)",
        "make_sampler": lambda: DimodSampler(method="zephyr"),
    },
    {
        "label":        "VeloxQ",
        "make_sampler": lambda: __import__("sampler", fromlist=["VeloxSampler"]).VeloxSampler(method="velox"),
    },
    {
        "label":        "VeloxQ SBM",
        "make_sampler": lambda: __import__("sampler", fromlist=["VeloxSampler"]).VeloxSampler(method="sbm"),
    },
]

SAMPLER_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    "tab:cyan", "tab:olive",
]

TRAIN_CONFIG = {
    "n_iterations":   300,
    "n_samples":      100,
    "learning_rate":  0.1,
    "regularization": 1e-3,
    "use_cem":        False,
    "save_checkpoints": False,
    "stop_at_convergence": False,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ising(inst: dict):
    if inst["model"] == "1d":
        return TransverseFieldIsing1D(inst["size"], inst["h"])
    else:
        return TransverseFieldIsing2D(inst["size"], inst["h"])


def make_rbm(ising) -> FullyConnectedRBM:
    n = ising.size if hasattr(ising, "size") else ising.size ** 2
    # For 2D, n_visible = L*L
    n_visible = getattr(ising, "n_visible", None) or (ising.size ** 2 if hasattr(ising, "size") else ising.size)
    # Normalise: TransverseFieldIsing1D.size = N spins,
    #            TransverseFieldIsing2D.size = L (grid side)
    if isinstance(ising, TransverseFieldIsing2D):
        n_visible = ising.size ** 2
    else:
        n_visible = ising.size
    n_hidden = n_visible
    rbm = FullyConnectedRBM(n_visible, n_hidden)
    return rbm


def instance_slug(inst: dict) -> str:
    return inst["label"].lower().replace(" ", "_").replace("=", "").replace(".", "p")


def save_vmc_results(inst: dict, all_sampler_results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"vmc_{instance_slug(inst)}.json")
    payload = {
        "label":        inst["label"],
        "model":        inst["model"],
        "size":         inst["size"],
        "h":            inst["h"],
        "samplers":     all_sampler_results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_vmc_results(results_dir: str = RESULTS_DIR) -> list:
    """Load all vmc_*.json files. Returns list of payload dicts, sorted by label."""
    out = []
    if not os.path.isdir(results_dir):
        return out
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("vmc_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                out.append(json.load(f))
    return out

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_vmc_experiments():
    for inst in TFIM_INSTANCES:
        print(f"\n{'='*60}")
        print(f"  Instance: {inst['label']}")
        print(f"{'='*60}")

        ising = make_ising(inst)
        try:
            exact_energy = ising.exact_ground_energy()
        except Exception as e:
            print(f"  WARNING: could not compute exact energy: {e}")
            exact_energy = None

        all_sampler_results = {}

        for entry in SAMPLER_REGISTRY:
            slabel = entry["label"]
            print(f"\n  --- Sampler: {slabel} ---")
            try:
                sampler  = entry["make_sampler"]()
                rbm      = make_rbm(ising)
                trainer  = Trainer(rbm, ising, sampler, config=TRAIN_CONFIG, args=None)
                history  = trainer.train()

                final_energy = history["energy"][-1]
                # kl_exact may be None for large N; take last non-None value
                kl_values = [k for k in history["kl_exact"] if k is not None]
                final_kl  = kl_values[-1] if kl_values else None

                energy_error = None
                if exact_energy is not None and exact_energy != 0:
                    energy_error = abs(final_energy - exact_energy) / abs(exact_energy)

                print(f"    final E = {final_energy:.4f}  exact = {exact_energy}  "
                      f"err = {energy_error}  KL = {final_kl}")

                all_sampler_results[slabel] = {
                    "final_energy":  final_energy,
                    "exact_energy":  exact_energy,
                    "energy_error":  energy_error,
                    "final_kl":      final_kl,
                    "energy_history": history["energy"],
                    "kl_history":    [k for k in history["kl_exact"]],
                }

            except Exception as e:
                print(f"    SKIPPED: {e}")
                all_sampler_results[slabel] = None

        path = save_vmc_results(inst, all_sampler_results)
        print(f"\n  Saved: {path}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_kl(sr: dict | None, use_mean: bool) -> float | None:
    """Extract KL from a sampler result dict: final value or mean over all iterations."""
    if sr is None:
        return None
    if use_mean:
        hist = [k for k in (sr.get("kl_history") or []) if k is not None]
        return float(np.mean(hist)) if hist else None
    val = sr.get("final_kl")
    return float(val) if val is not None else None


def plot_vmc_comparison(
    vmc_data: list,
    sampler_labels: list | None = None,
    save_path: str | None = None,
    use_mean_kl: bool = False,
):
    """
    2×2 grid of subplots, one per TFIM instance (up to 4).
    Each subplot has:
      - Left y-axis  (log scale): KL distance bars, coloured by sampler
      - Right y-axis (linear):   relative energy error bars, same colour + hatched
    Samplers on the x-axis.

    use_mean_kl: if True, KL is averaged over all training iterations;
                 if False (default), uses the final iteration's KL.
    """
    from matplotlib.patches import Patch  # noqa: PLC0415

    if sampler_labels is None:
        seen: list[str] = []
        for inst_data in vmc_data:
            for sl in inst_data["samplers"]:
                if sl not in seen:
                    seen.append(sl)
        sampler_labels = seen

    n_inst     = len(vmc_data)
    n_samplers = len(sampler_labels)
    bar_w      = 0.35
    xs         = np.arange(n_samplers)

    kl_label = "mean over all iterations" if use_mean_kl else "final iteration"
    fig, axes = plt.subplots(2, 2, figsize=(max(8, n_samplers * 0.9) * 2, 9),
                             squeeze=False)
    fig.suptitle(
        f"VMC: KL distance ({kl_label}) and energy error per sampler and instance",
        fontsize=13,
    )

    for inst_idx, inst_data in enumerate(vmc_data[:4]):
        row, col = divmod(inst_idx, 2)
        ax_kl  = axes[row][col]
        ax_err = ax_kl.twinx()

        ax_kl.set_title(inst_data["label"], fontsize=10)

        for s_idx, (slabel, color) in enumerate(zip(sampler_labels, SAMPLER_COLORS)):
            sr  = inst_data["samplers"].get(slabel)
            kl  = _get_kl(sr, use_mean_kl)
            err = (sr.get("energy_error") if sr else None)

            if kl is not None and kl > 0:
                ax_kl.bar(xs[s_idx] - bar_w / 2, kl, width=bar_w,
                          color=color, alpha=0.85, zorder=2,
                          label=slabel)
            if err is not None:
                ax_err.bar(xs[s_idx] + bar_w / 2, err, width=bar_w,
                           color=color, alpha=0.45, hatch="///",
                           edgecolor=color, zorder=2)

        ax_kl.set_yscale("log")
        ax_kl.set_ylabel(r"$D_{\mathrm{KL}}(q \| |\Psi|^2)$  [log]", fontsize=8)
        ax_err.set_ylabel(r"$|E-E_{\mathrm{exact}}|/|E_{\mathrm{exact}}|$", fontsize=8)
        ax_kl.set_xticks(xs)
        ax_kl.set_xticklabels(sampler_labels, rotation=25, ha="right", fontsize=7)
        ax_kl.set_xlim(-0.6, n_samplers - 0.4)

    # Hide unused subplots (if fewer than 4 instances)
    for empty_idx in range(n_inst, 4):
        row, col = divmod(empty_idx, 2)
        axes[row][col].set_visible(False)

    # Shared figure legend: metric type (solid = KL, hatched = energy error)
    legend_handles = [
        Patch(facecolor="gray", alpha=0.85,
              label=r"$D_{\mathrm{KL}}$ (left axis, log)"),
        Patch(facecolor="gray", alpha=0.45, hatch="///", edgecolor="gray",
              label=r"Energy error (right axis)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    return fig

def plot_convergence(
    vmc_data: list,
    sampler_labels: list | None = None,
    save_dir: str | None = None,
):
    """
    One figure per TFIM instance, two rows:
      Row 1: energy vs iteration, one line per sampler
      Row 2: KL vs iteration, one line per sampler
    """
    if sampler_labels is None:
        seen = []
        for d in vmc_data:
            for sl in d["samplers"]:
                if sl not in seen:
                    seen.append(sl)
        sampler_labels = seen

    plots_dir = save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots"
    )

    for inst_data in vmc_data:
        label = inst_data["label"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"VMC convergence — {label}", fontsize=12)

        for slabel, color in zip(sampler_labels, SAMPLER_COLORS):
            sr = inst_data["samplers"].get(slabel)
            if not sr:
                continue

            energy_hist = sr.get("energy_history") or []
            kl_hist     = sr.get("kl_history") or []

            if energy_hist:
                iters = np.arange(1, len(energy_hist) + 1)
                axes[0].plot(iters, energy_hist, color=color, lw=1.2, label=slabel)

            # KL history may contain None (large N iterations); plot only non-None
            if kl_hist:
                kl_iters = [i + 1 for i, k in enumerate(kl_hist) if k is not None]
                kl_vals  = [k      for k in kl_hist if k is not None]
                if kl_iters:
                    axes[1].plot(kl_iters, kl_vals, color=color, lw=1.2, label=slabel)

        # Mark exact energy
        exact = inst_data["samplers"]
        exact_e = next(
            (v["exact_energy"] for v in exact.values() if v and v.get("exact_energy") is not None),
            None,
        )
        if exact_e is not None:
            axes[0].axhline(exact_e, color="black", lw=1.0, ls="--", label="Exact")

        axes[0].set_ylabel("Energy", fontsize=10)
        axes[0].set_title("(a) Energy convergence", fontsize=10)
        axes[0].legend(fontsize=7, loc="upper right", ncol=3)

        axes[1].set_ylabel(r"$D_{\mathrm{KL}}(q \| |\Psi|^2)$", fontsize=10)
        axes[1].set_title("(b) Sampling KL during training", fontsize=10)
        axes[1].set_xlabel("Iteration", fontsize=10)
        axes[1].legend(fontsize=7, loc="upper right", ncol=3)

        fig.tight_layout()
        slug = label.lower().replace(" ", "_").replace("=", "").replace(".", "p")
        path = os.path.join(plots_dir, f"kl_vmc_convergence_{slug}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)


def plot_scatter(
    vmc_data: list,
    sampler_labels: list | None = None,
    save_path: str | None = None,
):
    """
    Scatter: final KL (x) vs relative energy error (y).
    One point per (instance × sampler), coloured by sampler, marked by instance.
    """
    if sampler_labels is None:
        seen = []
        for d in vmc_data:
            for sl in d["samplers"]:
                if sl not in seen:
                    seen.append(sl)
        sampler_labels = seen

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("KL distance vs energy error (each point = one instance × sampler)", fontsize=11)

    for slabel, color in zip(sampler_labels, SAMPLER_COLORS):
        for i_idx, inst_data in enumerate(vmc_data):
            sr = inst_data["samplers"].get(slabel)
            if not sr:
                continue
            kl  = sr.get("final_kl")
            err = sr.get("energy_error")
            if kl is None or err is None:
                continue
            marker = markers[i_idx % len(markers)]
            ax.scatter(kl, err, color=color, marker=marker, s=60, zorder=3,
                       label=slabel if i_idx == 0 else "_nolegend_")

    # Legend: samplers by colour
    from matplotlib.lines import Line2D  # noqa: PLC0415
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, label=sl)
        for sl, c in zip(sampler_labels, SAMPLER_COLORS)
    ]
    # Instance legend by marker
    inst_handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], color="gray",
               markersize=8, label=d["label"], linestyle="None")
        for i, d in enumerate(vmc_data)
    ]
    leg1 = ax.legend(handles=handles,       title="Sampler",  fontsize=7, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=inst_handles, title="Instance", fontsize=7, loc="upper right")

    ax.set_xlabel(r"Final $D_{\mathrm{KL}}(q \| |\Psi|^2)$", fontsize=10)
    ax.set_ylabel(r"$|E_{\mathrm{VMC}} - E_{\mathrm{exact}}| / |E_{\mathrm{exact}}|$", fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved JSON results and plot.")
    args = parser.parse_args()

    plots_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(plots_root, exist_ok=True)

    if not args.plot_only:
        run_vmc_experiments()

    vmc_data = load_vmc_results()
    if not vmc_data:
        print(f"No VMC results found in {RESULTS_DIR}. Run without --plot-only first.")
    else:
        plot_vmc_comparison(
            vmc_data,
            save_path=os.path.join(plots_root, "kl_vmc_comparison_final.png"),
            use_mean_kl=False,
        )
        plot_vmc_comparison(
            vmc_data,
            save_path=os.path.join(plots_root, "kl_vmc_comparison_mean.png"),
            use_mean_kl=True,
        )
        plot_convergence(
            vmc_data,
            save_dir=plots_root,
        )
        plot_scatter(
            vmc_data,
            save_path=os.path.join(plots_root, "kl_vmc_scatter.png"),
        )
