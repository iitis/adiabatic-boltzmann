"""
KL divergence comparison across all samplers on a FullyConnectedRBM.

For each sampler method, produces a 2-panel figure (same layout as fig2):
  - Panel (a): D_KL(P_S ∥ B_{β_eff}) per instance + mean ± SE (bar plot)
  - Panel (b): β_eff per instance — KL-minimisation (all) + CEM (joint samplers)

KL is computed over the visible marginal:
  energy(v) = -2 * log_psi(v)
  configs   = all ±1 configurations of n_visible spins (2^n_visible states)

LSB sigma is optimised per instance over sigma_inv2_candidates (same as fig2).
Hardware samplers (pegasus, zephyr, velox) are skipped gracefully if unavailable.

Run from src/:
    python plots/fig_kl_samplers.py
"""

import sys, os, json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent.parent          # repo root
sys.path.insert(0, str(_HERE.parent))       # make src/ importable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler
from kl_utils import (
    all_configs,
    log_boltzmann,
    empirical_dist,
    kl_divergence,
    estimate_beta_kl,
)

RESULTS_DIR = str(ROOT / "plots" / "kl_data")
FIGURES_DIR = ROOT / "figures" / "fig_kl_samplers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_rbm(
    n_visible: int, n_hidden: int, rng: np.random.Generator
) -> FullyConnectedRBM:
    """RBM with weights drawn from N(0, 2/√N), matching the paper's weight scale."""
    rbm = FullyConnectedRBM(n_visible, n_hidden)
    N = n_visible + n_hidden
    scale = 2.0 / np.sqrt(N)
    rbm.a = rng.normal(0, scale, n_visible)
    rbm.b = rng.normal(0, scale, n_hidden)
    rbm.W = rng.normal(0, scale, (n_visible, n_hidden))
    return rbm


def cem_joint(v_samples: np.ndarray, h_samples: np.ndarray, rbm) -> float:
    """Joint-samples CEM: fit β minimising Σ_{l,j} (h_j^(l) - tanh(β·a_j^(l)))²."""
    activation = v_samples @ rbm.W + rbm.b[None, :]

    def objective(beta):
        return float(np.sum((h_samples - np.tanh(beta * activation)) ** 2))

    result = minimize_scalar(objective, bounds=(1e-2, 10.0), method="bounded")
    return float(result.x)


def _kl_for_samples(v_samples, energy_fn, configs):
    """Helper: estimate β_eff by KL min, then return (kl, beta_eff_kl)."""
    v_samples = np.asarray(v_samples, dtype=np.float64)
    beta_eff_kl = estimate_beta_kl(energy_fn, configs, v_samples)
    kl = kl_divergence(
        empirical_dist(v_samples, configs),
        log_boltzmann(energy_fn, configs, beta_eff_kl),
    )
    return kl, beta_eff_kl


# ---------------------------------------------------------------------------
# Sampler registry
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   label          : display name
#   make_sampler   : zero-arg callable → sampler instance (deferred so failures
#                    are caught per-sampler, not at import time)
#   config         : dict passed to sampler.sample()
#   has_joint_h    : True → h samples are from the joint distribution → CEM meaningful
#   optimize_sigma : True → run LSB sigma grid per instance (same as fig2)

SIGMA_INV2_CANDIDATES = [round(0.5 + 0.1 * k, 1) for k in range(16)]  # 0.5 … 2.0

SAMPLER_REGISTRY = [
    {
        "label": "Metropolis",
        "make_sampler": lambda: ClassicalSampler(method="metropolis"),
        "config": {"n_warmup": 200, "n_sweeps": 1},
        "has_joint_h": False,
        "optimize_sigma": False,
    },
    {
        "label": "SA (custom)",
        "make_sampler": lambda: ClassicalSampler(method="simulated_annealing"),
        "config": {"n_warmup": 200},
        "has_joint_h": False,
        "optimize_sigma": False,
    },
    {
        "label": "Gibbs",
        "make_sampler": lambda: ClassicalSampler(method="gibbs"),
        "config": {"n_sweeps": 10, "n_warmup": 200},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
    {
        "label": "LSB",
        "make_sampler": lambda: ClassicalSampler(method="lsb"),
        "config": {"lsb_steps": 100, "lsb_delta": 1.0, "beta_x": 1.0},
        "has_joint_h": True,
        "optimize_sigma": True,
    },
    {
        "label": "SA (dimod)",
        "make_sampler": lambda: DimodSampler(method="simulated_annealing"),
        "config": {},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
    {
        "label": "Tabu",
        "make_sampler": lambda: DimodSampler(method="tabu"),
        "config": {},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
    {
        "label": "Pegasus (D-Wave)",
        "make_sampler": lambda: DimodSampler(method="pegasus"),
        "config": {},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
    {
        "label": "Zephyr (D-Wave)",
        "make_sampler": lambda: DimodSampler(method="zephyr"),
        "config": {},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
    {
        "label": "VeloxQ",
        "make_sampler": lambda: __import__(
            "sampler", fromlist=["VeloxSampler"]
        ).VeloxSampler(method="velox"),
        "config": {},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
    {
        "label": "VeloxQ SBM",
        "make_sampler": lambda: __import__(
            "sampler", fromlist=["VeloxSampler"]
        ).VeloxSampler(method="sbm"),
        "config": {},
        "has_joint_h": True,
        "optimize_sigma": False,
    },
]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_sampler_experiment(
    entry: dict,
    n_instances: int = 10,
    n_visible: int = 10,
    n_hidden: int = 5,
    n_samples: int = 100,
    seed: int = 0,
) -> list:
    label = entry["label"]
    sampler = entry["make_sampler"]()  # instantiate here so exceptions are caught
    base_config = entry["config"]
    has_joint_h = entry["has_joint_h"]
    optimize_sigma = entry["optimize_sigma"]

    rng = np.random.default_rng(seed)
    configs = all_configs(n_visible)
    results = []

    for inst in range(n_instances):
        rbm = random_rbm(n_visible, n_hidden, rng)
        energy_fn = lambda v, rbm=rbm: -2.0 * rbm.log_psi(v)

        if optimize_sigma:
            # --- LSB: grid-search σ per instance, keep lowest KL ---
            best_kl, best_beta_kl, best_beta_cem = float("inf"), None, None

            for sigma_inv2 in SIGMA_INV2_CANDIDATES:
                sigma = 1.0 / np.sqrt(sigma_inv2)
                config = {**base_config, "lsb_sigma": sigma}
                v_s, h_s = sampler.sample(
                    rbm, n_samples, config=config, return_hidden=True
                )
                v_s = np.asarray(v_s, dtype=np.float64)
                h_s = np.asarray(h_s, dtype=np.float64)

                kl, beta_kl = _kl_for_samples(v_s, energy_fn, configs)
                if kl < best_kl:
                    best_kl = kl
                    best_beta_kl = beta_kl
                    best_beta_cem = cem_joint(v_s, h_s, rbm)

            kl, beta_eff_kl, beta_eff_cem = best_kl, best_beta_kl, best_beta_cem

        else:
            v_s, h_s = sampler.sample(
                rbm, n_samples, config=base_config, return_hidden=True
            )
            v_s = np.asarray(v_s, dtype=np.float64)
            h_s = np.asarray(h_s, dtype=np.float64)

            kl, beta_eff_kl = _kl_for_samples(v_s, energy_fn, configs)
            beta_eff_cem = cem_joint(v_s, h_s, rbm) if has_joint_h else None

        print(
            f"  [{label}] inst={inst + 1:2d}  β_eff(KL)={beta_eff_kl:.3f}"
            + (f"  β_eff(CEM)={beta_eff_cem:.3f}" if beta_eff_cem is not None else "")
            + f"  KL={kl:.4f}"
        )

        results.append(
            {"kl": kl, "beta_eff_kl": beta_eff_kl, "beta_eff_cem": beta_eff_cem}
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sampler_results(
    results: list,
    label: str,
    has_joint_h: bool,
    beff_ylim: tuple = (0.0, 4.0),
    save_path: str = None,
):
    n = len(results)
    xs = np.arange(1, n + 1)

    kls = np.array([r["kl"] for r in results])
    beff_kl = np.array([r["beta_eff_kl"] for r in results])
    beff_cem = np.array(
        [
            r["beta_eff_cem"] if r["beta_eff_cem"] is not None else np.nan
            for r in results
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(label, fontsize=13)

    # Panel (a): KL divergence
    ax = axes[0]
    ax.bar(xs, kls, width=0.5, color="tab:blue", alpha=0.85)
    mean, se = kls.mean(), kls.std() / np.sqrt(n)
    ax.axhline(mean, color="tab:blue", lw=1.5, ls="--")
    ax.axhspan(mean - se, mean + se, color="tab:blue", alpha=0.2)
    ax.text(
        n + 0.3, mean, f"{mean:.2f}±{se:.2f}", va="center", color="tab:blue", fontsize=8
    )
    ax.set_xlabel("Instance")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P_S \| B_{\beta_{\mathrm{eff}}})$")
    ax.set_title("(a) Sampling accuracy")
    ax.set_xticks(xs)
    ax.set_xlim(0.5, n + 0.5)

    # Panel (b): β_eff
    ax = axes[1]
    if has_joint_h:
        bar_w = 0.35
        ax.bar(
            xs - bar_w / 2,
            beff_kl,
            width=bar_w,
            color="tab:orange",
            label="KL min",
            alpha=0.85,
        )
        ax.bar(
            xs + bar_w / 2,
            beff_cem,
            width=bar_w,
            color="tab:green",
            label="CEM",
            alpha=0.85,
        )
        ax.legend(fontsize=8)
    else:
        ax.bar(xs, beff_kl, width=0.5, color="tab:orange", alpha=0.85)

    ax.axhline(1.0, color="black", lw=1.2, ls=":")
    ax.set_ylim(*beff_ylim)
    ax.set_xlabel("Instance")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}$")
    ax.set_title(r"(b) Effective inverse temperature $\beta_{\mathrm{eff}}$")
    ax.set_xticks(xs)
    ax.set_xlim(0.5, n + 0.5)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# D-Wave size scaling  (n_visible 11–15, both Pegasus and Zephyr)
# ---------------------------------------------------------------------------
# n_hidden scales with n_visible to keep the ratio ~1:2 and the embedding
# manageable.  2^n_visible is the KL enumeration cost: max 2^15 = 32768.

DWAVE_SIZE_CONFIGS = [
    # (n_visible, n_hidden)
    (11, 6),
    (12, 6),
    (13, 7),
    (14, 7),
    (15, 8),
]

DWAVE_DEVICES = [
    ("Pegasus (D-Wave)", lambda: DimodSampler(method="pegasus")),
    ("Zephyr (D-Wave)", lambda: DimodSampler(method="zephyr")),
]


def run_dwave_size_scaling(
    n_instances: int = 10,
    n_samples: int = 100,
    seed: int = 0,
) -> dict:
    """
    For each (n_visible, n_hidden) in DWAVE_SIZE_CONFIGS and each D-Wave device,
    run n_instances random RBMs and collect KL + β_eff results.

    Returns:
        {device_label: {(nv, nh): [result_dicts]}}
    """
    all_results = {}

    for device_label, make_sampler in DWAVE_DEVICES:
        print(f"\n=== {device_label} — size scaling ===")
        try:
            sampler = make_sampler()
        except Exception as e:
            print(f"  SKIPPED (init failed): {e}")
            continue

        device_results = {}
        for n_visible, n_hidden in DWAVE_SIZE_CONFIGS:
            print(
                f"  n_visible={n_visible}, n_hidden={n_hidden}  (2^{n_visible}={2**n_visible} configs)"
            )
            rng = np.random.default_rng(seed)
            configs = all_configs(n_visible)
            size_results = []

            for inst in range(n_instances):
                rbm = random_rbm(n_visible, n_hidden, rng)
                energy_fn = lambda v, rbm=rbm: -2.0 * rbm.log_psi(v)

                try:
                    v_s, h_s = sampler.sample(
                        rbm, n_samples, config={}, return_hidden=True
                    )
                    v_s = np.asarray(v_s, dtype=np.float64)
                    h_s = np.asarray(h_s, dtype=np.float64)

                    kl, beta_eff_kl = _kl_for_samples(v_s, energy_fn, configs)
                    beta_eff_cem = cem_joint(v_s, h_s, rbm)

                    print(
                        f"    inst={inst + 1:2d}  β_eff(KL)={beta_eff_kl:.3f}  β_eff(CEM)={beta_eff_cem:.3f}  KL={kl:.4f}"
                    )
                    size_results.append(
                        {
                            "kl": kl,
                            "beta_eff_kl": beta_eff_kl,
                            "beta_eff_cem": beta_eff_cem,
                        }
                    )

                except Exception as e:
                    print(f"    inst={inst + 1:2d}  FAILED: {e}")

            device_results[(n_visible, n_hidden)] = size_results

        all_results[device_label] = device_results

    return all_results


def plot_dwave_size_scaling(
    all_results: dict,
    beff_ylim: tuple = (0.0, 4.0),
    save_path: str = None,
):
    """
    2-panel figure comparing Pegasus and Zephyr across sizes.
      Panel (a): mean KL ± SE vs n_visible, one line per device
      Panel (b): mean β_eff (KL-min and CEM) vs n_visible, one line per device
    """
    sizes = [nv for nv, _ in DWAVE_SIZE_CONFIGS]
    colors = {"Pegasus (D-Wave)": "tab:blue", "Zephyr (D-Wave)": "tab:orange"}
    markers = {"Pegasus (D-Wave)": "o", "Zephyr (D-Wave)": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("D-Wave size scaling (RBM, KL over visible marginal)", fontsize=12)

    for device_label, device_results in all_results.items():
        color = colors.get(device_label, "tab:gray")
        marker = markers.get(device_label, "o")

        kl_means, kl_ses = [], []
        bkl_means, bcem_means = [], []

        for nv, nh in DWAVE_SIZE_CONFIGS:
            rs = device_results.get((nv, nh), [])
            if not rs:
                kl_means.append(np.nan)
                kl_ses.append(np.nan)
                bkl_means.append(np.nan)
                bcem_means.append(np.nan)
                continue
            kls = np.array([r["kl"] for r in rs])
            bkls = np.array([r["beta_eff_kl"] for r in rs])
            bcem = np.array([r["beta_eff_cem"] for r in rs])
            kl_means.append(kls.mean())
            kl_ses.append(kls.std() / np.sqrt(len(kls)))
            bkl_means.append(bkls.mean())
            bcem_means.append(bcem.mean())

        kl_means = np.array(kl_means)
        kl_ses = np.array(kl_ses)
        bkl_means = np.array(bkl_means)
        bcem_means = np.array(bcem_means)

        # Panel (a)
        ax = axes[0]
        ax.plot(sizes, kl_means, marker=marker, color=color, label=device_label)
        ax.fill_between(
            sizes, kl_means - kl_ses, kl_means + kl_ses, color=color, alpha=0.2
        )

        # Panel (b)
        ax = axes[1]
        ax.plot(
            sizes,
            bkl_means,
            marker=marker,
            color=color,
            label=f"{device_label} (KL)",
            ls="-",
        )
        ax.plot(
            sizes,
            bcem_means,
            marker=marker,
            color=color,
            label=f"{device_label} (CEM)",
            ls="--",
        )

    axes[0].set_xlabel("n_visible")
    axes[0].set_ylabel(r"$D_{\mathrm{KL}}(P_S \| B_{\beta_{\mathrm{eff}}})$")
    axes[0].set_title("(a) Sampling accuracy vs size")
    axes[0].set_xticks(sizes)
    axes[0].legend(fontsize=8)

    axes[1].axhline(1.0, color="black", lw=1.2, ls=":")
    axes[1].set_ylim(*beff_ylim)
    axes[1].set_xlabel("n_visible")
    axes[1].set_ylabel(r"$\beta_{\mathrm{eff}}$")
    axes[1].set_title(r"(b) $\beta_{\mathrm{eff}}$ vs size")
    axes[1].set_xticks(sizes)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_results(results: list, label: str, results_dir: str = RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    slug = (
        label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
    )
    path = os.path.join(results_dir, f"{slug}.json")
    with open(path, "w") as f:
        json.dump({"label": label, "results": results}, f, indent=2)
    return path


def load_results(path: str) -> tuple[str, list]:
    with open(path) as f:
        data = json.load(f)
    return data["label"], data["results"]


def load_all_results(results_dir: str = RESULTS_DIR) -> dict:
    """Load all saved result JSONs from results_dir. Returns {label: results}."""
    out = {}
    if not os.path.isdir(results_dir):
        return out
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        try:
            label, results = load_results(path)
            out[label] = results
        except KeyError:
            pass  # skip files with a different schema (e.g. vmc_*.json)
    return out


# ---------------------------------------------------------------------------
# Overview plot  (all samplers on one page, KL only)
# ---------------------------------------------------------------------------


def plot_overview(
    all_results: dict,
    n_cols: int = 3,
    beff_ylim: tuple = (0.0, 4.0),
    save_path: str = None,
):
    """
    Grid of subplots — one per sampler — each showing:
      top:    KL per instance (bars) + mean ± SE
      bottom: β_eff (KL-min) and CEM where available

    all_results: {label: results_list}  — as returned by load_all_results()
    """
    labels = list(all_results.keys())
    n = len(labels)
    n_rows_per = 2  # KL row + β_eff row per sampler
    n_cols = min(n_cols, n)
    n_grid_rows = int(np.ceil(n / n_cols)) * n_rows_per

    fig = plt.figure(figsize=(5 * n_cols, 3 * n_grid_rows))
    fig.suptitle(
        "KL divergence across samplers (RBM, visible marginal)", fontsize=13, y=1.01
    )

    for idx, label in enumerate(labels):
        results = all_results[label]
        n_inst = len(results)
        xs = np.arange(1, n_inst + 1)

        kls = np.array([r["kl"] for r in results])
        beff_kl = np.array([r["beta_eff_kl"] for r in results])
        beff_cem = np.array(
            [
                r["beta_eff_cem"] if r.get("beta_eff_cem") is not None else np.nan
                for r in results
            ]
        )
        has_cem = not np.all(np.isnan(beff_cem))

        grid_col = idx % n_cols
        grid_block = idx // n_cols  # which block of rows
        row_kl = grid_block * n_rows_per + 1
        row_beff = row_kl + 1
        total_rows = int(np.ceil(n / n_cols)) * n_rows_per

        ax_kl = fig.add_subplot(
            total_rows, n_cols, row_kl * n_cols + grid_col + 1 - n_cols
        )
        ax_beff = fig.add_subplot(
            total_rows, n_cols, row_beff * n_cols + grid_col + 1 - n_cols
        )

        # --- KL bars ---
        ax_kl.bar(xs, kls, width=0.5, color="tab:blue", alpha=0.85)
        mean, se = kls.mean(), kls.std() / np.sqrt(n_inst)
        ax_kl.axhline(mean, color="tab:blue", lw=1.2, ls="--")
        ax_kl.axhspan(mean - se, mean + se, color="tab:blue", alpha=0.2)
        ax_kl.set_title(label, fontsize=9, fontweight="bold")
        ax_kl.set_ylabel(r"$D_{\mathrm{KL}}$", fontsize=8)
        ax_kl.set_xticks(xs)
        ax_kl.tick_params(labelsize=7)
        ax_kl.text(
            0.98,
            0.95,
            f"{mean:.2f}±{se:.2f}",
            transform=ax_kl.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="tab:blue",
        )

        # --- β_eff bars ---
        bar_w = 0.35
        if has_cem:
            ax_beff.bar(
                xs - bar_w / 2,
                beff_kl,
                width=bar_w,
                color="tab:orange",
                alpha=0.85,
                label="KL",
            )
            ax_beff.bar(
                xs + bar_w / 2,
                beff_cem,
                width=bar_w,
                color="tab:green",
                alpha=0.85,
                label="CEM",
            )
            ax_beff.legend(fontsize=6, loc="upper right")
        else:
            ax_beff.bar(xs, beff_kl, width=0.5, color="tab:orange", alpha=0.85)
        ax_beff.axhline(1.0, color="black", lw=1.0, ls=":")
        ax_beff.set_ylim(*beff_ylim)
        ax_beff.set_ylabel(r"$\beta_{\mathrm{eff}}$", fontsize=8)
        ax_beff.set_xticks(xs)
        ax_beff.set_xlabel("Instance", fontsize=7)
        ax_beff.tick_params(labelsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Overview saved: {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Comparison plot  (selected samplers, fig2-style, single page)
# ---------------------------------------------------------------------------

COMPARISON_LABELS = [
    "Pegasus (D-Wave)",
    "Zephyr (D-Wave)",
    "VeloxQ",
    "VeloxQ SBM",
    "Metropolis",
    "Gibbs",
    "LSB",
]

COMPARISON_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]


def plot_comparison(
    results_dir: str = RESULTS_DIR,
    labels: list = COMPARISON_LABELS,
    save_path: str = None,
):
    """
    Single fig2-style grouped bar plot: one bar per sampler per instance.

    X-axis: instances (1–N).
    Within each instance group: one bar per sampler, coloured by sampler.
    Mean ± SE horizontal lines shown per sampler (across instances).
    """
    all_results = load_all_results(results_dir)

    present = [l for l in labels if l in all_results]
    missing = [l for l in labels if l not in all_results]
    if missing:
        print(f"  Missing results for: {missing} — skipped.")
    if not present:
        print("No results to plot.")
        return

    n_samplers = len(present)
    n_instances = len(all_results[present[0]])
    bar_w = 0.8 / n_samplers
    xs = np.arange(1, n_instances + 1)

    fig, ax = plt.subplots(figsize=(max(10, n_instances * 1.2), 5))

    for i, (label, color) in enumerate(zip(present, COMPARISON_COLORS)):
        kls = np.array([r["kl"] for r in all_results[label]])
        offsets = (i - (n_samplers - 1) / 2) * bar_w
        ax.bar(xs + offsets, kls, width=bar_w, color=color, alpha=0.85, label=label)

        mean, se = kls.mean(), kls.std() / np.sqrt(n_instances)
        ax.axhline(mean, color=color, lw=1.2, ls="--")
        ax.axhspan(mean - se, mean + se, color=color, alpha=0.12)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}" for i in xs], fontsize=9)
    ax.set_xlabel("Instance", fontsize=11)
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P_S \| B_{\beta_{\mathrm{eff}}})$", fontsize=11)
    ax.set_title("Sampling accuracy across methods (RBM, visible marginal)", fontsize=11)
    ax.set_xlim(0.5, n_instances + 0.5)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved: {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-samplers",
        action="store_true",
        help="Run all SAMPLER_REGISTRY experiments and save results to JSON.",
    )
    parser.add_argument(
        "--dwave-scaling",
        action="store_true",
        help="Run D-Wave size-scaling experiment (n_visible 11-15).",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Load saved JSON results and produce the overview grid plot.",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Single fig2-style bar plot comparing selected samplers.",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_samplers:
        for entry in SAMPLER_REGISTRY:
            label = entry["label"]
            print(f"\n=== {label} ===")
            try:
                results = run_sampler_experiment(entry)
                save_results(results, label)
                slug = (
                    label.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("-", "")
                )
                plot_sampler_results(
                    results,
                    label=label,
                    has_joint_h=entry["has_joint_h"],
                    save_path=str(FIGURES_DIR / f"kl_{slug}.png"),
                )
            except Exception as e:
                print(f"  SKIPPED: {e}")

    if args.dwave_scaling:
        scaling_results = run_dwave_size_scaling()
        if scaling_results:
            plot_dwave_size_scaling(
                scaling_results,
                save_path=str(FIGURES_DIR / "kl_dwave_size_scaling.png"),
            )

    if args.overview:
        all_results = load_all_results()
        if not all_results:
            print(f"No saved results found in {RESULTS_DIR}. Run --run-samplers first.")
        else:
            plot_overview(all_results, save_path=str(FIGURES_DIR / "kl_overview.png"))

    if args.comparison:
        plot_comparison(save_path=str(FIGURES_DIR / "kl_comparison.png"))

    if not any(vars(args).values()):
        parser.print_help()
