#!/usr/bin/env python3
"""
Sampler sanity-check: can each backend recover the exact RBM distribution?

For a small RBM (N_visible=8, N_hidden=8, 2^8=256 configs) we:
  1. Enumerate p_exact(v) = |Ψ(v)|² / Z  over all 2^N configs.
  2. Run each sampler for n_samples draws.
  3. Compute KL(q_emp ‖ p_exact) and ESS/n as quality metrics.
  4. Plot empirical vs exact probability for each config.

D-Wave (pegasus/zephyr) samplers are included and gated by the QPU budget in
time.json (same 20-minute limit as performance_run.py).  Access time is logged
to time.json after every successful D-Wave call.

Run from src/:
    python ../scripts/test_samplers.py
    python ../scripts/test_samplers.py --skip-dwave     # classical only
"""

from itertools import product
import sys
import os
import json
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── make src/ importable ──────────────────────────────────────────────────
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))
os.chdir(SRC)

from model import SRBM
from sampler import ClassicalSampler, DimodSampler

# ── QPU budget (mirrors performance_run.py) ───────────────────────────────
DWAVE_BUDGET_MS = 20 * 60 * 1000  # 20 minutes
TIME_FILE = Path("time.json")


def read_qpu_ms() -> float:
    """Read accumulated QPU access time from time.json (ms)."""
    if not TIME_FILE.exists():
        return 0.0
    with TIME_FILE.open("r") as f:
        data = json.load(f)
    return float(data["time_ms"])


def check_qpu_budget():
    """Raise RuntimeError if the QPU budget is already exhausted."""
    used = read_qpu_ms()
    if used >= DWAVE_BUDGET_MS:
        raise RuntimeError(
            f"QPU budget exhausted: {used / 60_000:.2f} min used "
            f"(limit {DWAVE_BUDGET_MS / 60_000:.0f} min). "
            "Skipping D-Wave sampler."
        )


# ---------------------------------------------------------------------------
# Exact distribution helpers
# ---------------------------------------------------------------------------


def enumerate_all(rbm) -> tuple:
    """Return (all_v, p_exact) for all 2^N_visible configs."""
    N = rbm.n_visible
    indices = np.arange(2**N, dtype=np.int32)
    all_v = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(
        np.float64
    ) * 2 - 1
    Theta = all_v @ rbm.W + rbm.b[None, :]
    log_psi2 = -(all_v @ rbm.a) + np.sum(np.logaddexp(Theta, -Theta), axis=1)
    lw = log_psi2 - log_psi2.max()
    p = np.exp(lw)
    p /= p.sum()
    return all_v, p


def exact_distribution(rbm, beta=1.0) -> tuple:
    """
    Return (all_v, p_exact) for all 2^Nv visible configurations.
    This includes visible-visible couplings (rbm.U) and biases.
    Hidden units are summed out analytically.
    """
    Nv = rbm.n_visible
    Nh = rbm.n_hidden

    # enumerate all visible configs
    indices = np.arange(2**Nv, dtype=np.int32)
    all_v = ((indices[:, None] >> np.arange(Nv - 1, -1, -1)) & 1).astype(
        np.float64
    ) * 2 - 1

    # --- visible-visible interaction ---
    vv_term = -0.5 * np.einsum("bi,ij,bj->b", all_v, rbm.U, all_v)

    # --- visible bias ---
    v_bias_term = -all_v @ rbm.a

    # --- hidden units summed out analytically ---
    theta = all_v @ rbm.W + rbm.b[None, :]
    hidden_term = np.sum(np.logaddexp(theta, -theta), axis=1)

    # --- log probability ---
    log_p = beta * (vv_term + v_bias_term) + hidden_term

    # --- normalize ---
    lw = log_p - log_p.max()
    p = np.exp(lw)
    p /= p.sum()

    return all_v, p


def empirical_dist(V: np.ndarray, all_v: np.ndarray) -> np.ndarray:
    """Map samples V (ns, N) to a probability vector over all 2^N configs."""
    config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
    counts = np.zeros(len(all_v))
    for row in V.astype(int).tolist():
        idx = config_idx.get(tuple(row))
        if idx is not None:
            counts[idx] += 1
    return counts / counts.sum() if counts.sum() > 0 else counts


def kl(q: np.ndarray, p: np.ndarray) -> float:
    mask = q > 0
    return float(np.sum(q[mask] * (np.log(q[mask]) - np.log(p[mask] + 1e-300))))


def ess(V: np.ndarray, rbm) -> float:
    Theta = V @ rbm.W + rbm.b[None, :]
    log_psi2 = -(V @ rbm.a) + np.sum(np.logaddexp(Theta, -Theta), axis=1)
    lw = log_psi2 - log_psi2.max()
    w = np.exp(lw)
    w /= w.sum()
    return float(1.0 / np.sum(w**2)) / len(V)


# ---------------------------------------------------------------------------
# Sampler wrappers
# ---------------------------------------------------------------------------


def run_sampler(
    sampler,
    rbm,
    n_samples: int,
    n_runs: int,
    beta_x: float = 2.0,
    is_dwave: bool = False,
):
    """Collect n_samples per run, concatenate across n_runs.

    For D-Wave samplers the QPU budget is checked before every run and the
    access time delta is printed after each successful call.
    """
    all_v = []
    for run_i in range(n_runs):
        if is_dwave:
            check_qpu_budget()
            before_ms = read_qpu_ms()

        result = sampler.sample(rbm, n_samples, config={"beta_x": beta_x})
        v = result[0] if isinstance(result, tuple) else result
        all_v.append(np.asarray(v, dtype=np.float64))

        if is_dwave:
            after_ms = read_qpu_ms()
            delta_ms = after_ms - before_ms
            print(
                f"   run {run_i + 1}/{n_runs}: QPU access {delta_ms:.1f} ms  "
                f"(total used {after_ms / 60_000:.3f} / "
                f"{DWAVE_BUDGET_MS / 60_000:.0f} min)"
            )

    return np.vstack(all_v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-visible", type=int, default=10)
    parser.add_argument("--n-hidden", type=int, default=5)
    parser.add_argument(
        "--n-samples", type=int, default=2000, help="Samples per sampler run"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Independent runs to accumulate (total = n_samples * n_runs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=Path, default=_root / "plots" / "distribution_recovery")
    parser.add_argument(
        "--skip-dwave",
        action="store_true",
        help="Skip D-Wave (pegasus/zephyr) samplers",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    Nv, Nh = args.n_visible, args.n_hidden
    total = args.n_samples * args.n_runs

    assert Nv <= 16, "N_visible must be ≤ 16 for exact enumeration"

    # ── Build a random RBM with moderate coupling strength ──────────────────
    rbm = SRBM(Nv, Nh)
    rbm.a[:] = rng.normal(0, 0.5, Nv)
    rbm.b[:] = rng.normal(0, 0.5, Nh)
    rbm.W[:] = rng.normal(0, 0.5 / np.sqrt(Nv), (Nv, Nh))

    all_v, p_exact = exact_distribution(rbm)
    entropy = float(-np.sum(p_exact * np.log(p_exact + 1e-300)))
    print(
        f"RBM: N_vis={Nv}, N_hid={Nh},  H(p)={entropy:.3f} nats,  "
        f"total samples per sampler={total}"
    )

    # ── QPU budget summary ───────────────────────────────────────────────────
    if not args.skip_dwave:
        used_ms = read_qpu_ms()
        remaining_ms = DWAVE_BUDGET_MS - used_ms
        print(
            f"\nQPU budget: {used_ms / 60_000:.2f} min used, "
            f"{remaining_ms / 60_000:.2f} min remaining "
            f"(limit {DWAVE_BUDGET_MS / 60_000:.0f} min)"
        )

    # ── Define samplers to test ──────────────────────────────────────────────
    # Each entry: (sampler_object, is_dwave)
    samplers: dict[str, tuple] = {
        "metropolis": (ClassicalSampler("metropolis", n_warmup=200, n_sweeps=1), False),
        "simulated_annealing": (
            ClassicalSampler("simulated_annealing", n_warmup=50),
            False,
        ),
        "gibbs": (ClassicalSampler("gibbs", n_warmup=50, n_sweeps=5), False),
        "lsb": (ClassicalSampler("lsb"), False),
    }
    if not args.skip_dwave:
        samplers["pegasus"] = (DimodSampler("pegasus"), True)
        samplers["zephyr"] = (DimodSampler("zephyr"), True)
    samplers = {"lsb": (ClassicalSampler("lsb"), False)}
    results = {}
    for name, (sampler, is_dwave) in samplers.items():
        print(f"\n── {name} ──")
        try:
            V = run_sampler(
                sampler, rbm, args.n_samples, args.n_runs, is_dwave=is_dwave
            )
            q = empirical_dist(V, all_v)
            kl_val = kl(q, p_exact)
            ess_val = ess(V, rbm)
            unique = int(np.sum(q > 0))
            results[name] = dict(V=V, q=q, kl=kl_val, ess=ess_val, unique=unique)
            print(
                f"   KL={kl_val:.4f}  ESS/n={ess_val:.3f}  "
                f"unique configs={unique}/{len(all_v)}"
            )
        except Exception as e:
            print(f"   FAILED: {e}")
            results[name] = None

    # ── Final QPU usage summary ──────────────────────────────────────────────
    if not args.skip_dwave:
        print(f"\nQPU total used after test: {read_qpu_ms() / 60_000:.3f} min")

    # ── Plots ────────────────────────────────────────────────────────────────
    n_samplers = len(samplers)
    fig, axes = plt.subplots(n_samplers + 1, 1, figsize=(14, 3.5 * (n_samplers + 1)))

    # Sort configs by exact probability for nicer plots
    order = np.argsort(p_exact)[::-1]
    x_idx = np.arange(len(all_v))

    # Row 0: exact distribution
    ax = axes[0]
    ax.bar(x_idx, p_exact[order], color="black", alpha=0.7, width=1.0)
    ax.set_title(
        f"Exact  |Ψ(v)|²  (N_vis={Nv}, H={entropy:.2f} nats)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("p(v)")
    ax.set_xlabel("Config (sorted by p)")
    ax.grid(True, axis="y", alpha=0.3)

    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row, (name, res) in enumerate(results.items(), start=1):
        ax = axes[row]
        c = COLORS[row - 1]
        if res is None:
            ax.set_title(f"{name}  —  FAILED", color="red")
            continue

        q_sorted = res["q"][order]
        ax.bar(
            x_idx, p_exact[order], color="black", alpha=0.25, width=1.0, label="exact"
        )
        ax.bar(x_idx, q_sorted, color=c, alpha=0.6, width=1.0, label="empirical")
        title = (
            f"{name}   "
            f"KL={res['kl']:.4f}   "
            f"ESS/n={res['ess']:.3f}   "
            f"unique={res['unique']}/{len(all_v)}"
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("p(v)")
        ax.set_xlabel("Config (sorted by p)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Sampler Distribution Recovery  (N_vis={Nv}, N_hid={Nh}, "
        f"{total} samples each)",
        fontsize=13,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout()
    path = out / "distribution_recovery.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Summary table ────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 2.5 + 0.4 * len(results)))
    ax.axis("off")
    rows_data = []
    for name, res in results.items():
        if res is None:
            rows_data.append([name, "FAIL", "FAIL", "FAIL"])
        else:
            rows_data.append(
                [
                    name,
                    f"{res['kl']:.4f}",
                    f"{res['ess']:.3f}",
                    f"{res['unique']}/{len(all_v)}",
                ]
            )
    tbl = ax.table(
        cellText=rows_data,
        colLabels=["Sampler", "KL(q‖p_exact)", "ESS / n", "Unique configs"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax.set_title(
        f"Distribution Recovery Summary  (N={Nv}, {total} samples)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    path2 = out / "summary_table.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
