#!/usr/bin/env python3
"""
Post-hoc KL divergence evaluation for D-Wave (pegasus/zephyr) and VeloxQ runs.

For each checkpoint with N ≤ KL_MAX_N:
  1. Load final RBM weights
  2. Enumerate all 2^N configs → p_exact
  3. Draw N_SAMPLES reads from the hardware sampler
  4. Compute KL(q_hardware ‖ p_exact) and ESS of hardware samples
  5. Draw the same number from Metropolis → KL(q_metro ‖ p_exact) as baseline
  6. Write results back into the matching result JSON

Time accounting:
  D-Wave QPU time  → time.json["time_ms"]        (via DimodSampler, existing)
  VeloxQ wall time → time.json["velox_time_ms"]  (via log_solver_time_ms)

Run from the repo root:
    python scripts/eval_kl_hardware.py
    python scripts/eval_kl_hardware.py --dry-run
    python scripts/eval_kl_hardware.py --n-samples 2000 --kl-max-n 18
    python scripts/eval_kl_hardware.py --sampler pegasus   # only D-Wave Advantage
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow importing from src/ regardless of working directory
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from helpers import load_rbm_checkpoint, log_solver_time_ms
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler

TIME_PATH = SRC_DIR / "time.json"

KL_MAX_N_DEFAULT = 20  # enumerate 2^N: up to 1M configs, ~80 MB
N_SAMPLES_DEFAULT = 1000

HARDWARE_METHODS = {"pegasus", "zephyr", "velox"}


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------


def compute_p_exact(rbm) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Enumerate all 2^N configurations and compute the exact target distribution.

    Returns:
        all_v        : (2^N, N)  ±1 spin configs
        p_true       : (2^N,)    normalised probabilities |Ψ(v)|²/Z
        config_idx   : dict mapping tuple(v_int) → row index in all_v
    """
    N = rbm.n_visible
    indices = np.arange(2**N, dtype=np.int32)
    all_v = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(
        np.float64
    ) * 2 - 1

    Theta = all_v @ rbm.W + rbm.b[None, :]
    log_psi2 = -(all_v @ rbm.a) + np.sum(np.logaddexp(Theta, -Theta), axis=1)
    lw = log_psi2 - log_psi2.max()
    p_true = np.exp(lw)
    p_true /= p_true.sum()

    config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
    return all_v, p_true, config_idx


def compute_kl_and_ess(
    V: np.ndarray, p_true: np.ndarray, config_idx: dict, rbm
) -> dict:
    """
    Compute KL(q_empirical ‖ p_true) and normalised ESS from samples V.

    KL bias ≈ (K_eff − 1) / (2 * n_samples)  where K_eff = n * ESS.
    """
    ns = len(V)

    # Empirical distribution
    counts = np.zeros(len(p_true))
    for row in V.astype(int).tolist():
        idx = config_idx.get(tuple(row))
        if idx is not None:
            counts[idx] += 1
    q_emp = counts / ns

    # KL(q || p)
    mask = q_emp > 0
    kl = float(np.sum(q_emp[mask] * (np.log(q_emp[mask]) - np.log(p_true[mask]))))

    # KL bias estimate
    Theta_V = V @ rbm.W + rbm.b[None, :]
    log_psi2_V = -(V @ rbm.a) + np.sum(np.logaddexp(Theta_V, -Theta_V), axis=1)
    lw = log_psi2_V - log_psi2_V.max()
    w = np.exp(lw)
    w /= w.sum()
    ess_norm = float(1.0 / np.sum(w**2)) / ns
    k_eff = ess_norm * ns
    bias_est = (k_eff - 1) / (2 * ns)

    n_unique = len(set(map(tuple, V.astype(int).tolist())))

    return {
        "kl": kl,
        "ess_norm": ess_norm,
        "k_eff": k_eff,
        "bias_est": bias_est,
        "n_unique": n_unique,
        "n_samples": ns,
    }


# ---------------------------------------------------------------------------
# Result JSON helpers
# ---------------------------------------------------------------------------


def find_result_json(ckpt_config: dict, results_dir: Path) -> Path | None:
    """Reconstruct the result JSON path from a checkpoint's stored config."""
    cfg = ckpt_config
    use_cem = cfg.get("cem", False)
    out_dir = results_dir / str(cfg["size"]) / cfg["sampler"] / cfg["sampling_method"]
    stem = (
        f"result_{cfg['model']}"
        f"_h{cfg['h']}"
        f"_rbm{cfg['rbm']}"
        f"_nh{cfg['n_hidden']}"
        f"_lr{cfg['learning_rate']}"
        f"_reg{cfg['regularization']}"
        f"_ns{cfg['n_samples']}"
        f"_seed{cfg['seed']}"
        f"_iter{cfg['iterations']}"
    )
    # Try with and without _cem suffix (old files lack it)
    for suffix in [f"_cem{int(use_cem)}", ""]:
        p = out_dir / (stem + suffix + ".json")
        if p.exists():
            return p
    return None


def update_result_json(path: Path, new_fields: dict):
    """Atomically add new_fields to an existing result JSON."""
    with open(path) as f:
        data = json.load(f)
    data.update(new_fields)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Per-checkpoint processing
# ---------------------------------------------------------------------------


def process_checkpoint(ckpt_path: Path, args, results_dir: Path) -> bool:
    """
    Evaluate KL for one checkpoint.  Returns True on success.
    Must be called with CWD = SRC_DIR (VeloxSampler reads relative config files).
    """
    # ── 1. Load checkpoint ────────────────────────────────────────────────
    rbm_state, cfg, iteration = load_rbm_checkpoint(ckpt_path)

    sampler_name = cfg.get("sampler", "")
    method = cfg.get("sampling_method", "")
    N = rbm_state["n_visible"]
    label = f"{sampler_name}/{method}  N={N}  h={cfg.get('h')}  seed={cfg.get('seed')}"

    if method not in HARDWARE_METHODS:
        return False
    if N > args.kl_max_n:
        print(f"  [skip] N={N} > KL_MAX_N={args.kl_max_n}  {label}")
        return False

    # ── 2. Find result JSON; skip if already evaluated ───────────────────
    result_path = find_result_json(cfg, results_dir)
    if result_path is None:
        print(f"  [skip] No result JSON found for {label}")
        return False

    with open(result_path) as f:
        existing = json.load(f)
    if "kl_hardware" in existing and not args.force:
        print(f"  [skip] Already evaluated: {result_path.name}")
        return True

    print(f"\n[eval] {label}")
    print(f"       checkpoint: {ckpt_path.name}")
    print(f"       result:     {result_path.name}")

    # ── 3. Reconstruct RBM ───────────────────────────────────────────────
    rbm = FullyConnectedRBM(N, rbm_state["n_hidden"])
    rbm.a = np.array(rbm_state["a"])
    rbm.b = np.array(rbm_state["b"])
    rbm.W = np.array(rbm_state["W"])

    # ── 4. Exact distribution ────────────────────────────────────────────
    print(f"  Computing p_exact over {2**N} configs ...")
    all_v, p_true, config_idx = compute_p_exact(rbm)

    if args.dry_run:
        print("  [dry-run] skipping sampler calls")
        return True

    # ── 5. Hardware samples ──────────────────────────────────────────────
    hw_result = None
    try:
        if method in ("pegasus", "zephyr"):
            sampler = DimodSampler(method=method)
            # QPU time logged inside DimodSampler.dwave() → time.json["time_ms"]
            V_hw = sampler.sample(
                rbm,
                args.n_samples,
                config={
                    "beta_x": 1.0,
                    "solver": __import__("helpers").get_solver_name(method),
                },
            )

        elif method == "velox":
            from sampler import VeloxSampler

            sampler = VeloxSampler(method="velox")
            t0 = time.perf_counter()
            V_hw = sampler.sample(rbm, args.n_samples, config={"beta_x": 1.0})
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            print(f"  [VeloxQ] wall time: {elapsed_ms:.0f} ms")

        else:
            raise ValueError(f"Unknown hardware method: {method}")

        hw_result = compute_kl_and_ess(V_hw, p_true, config_idx, rbm)
        print(
            f"  [hw]  KL={hw_result['kl']:.4f}  ESS={hw_result['ess_norm']:.3f}"
            f"  K_eff={hw_result['k_eff']:.1f}  bias≈{hw_result['bias_est']:.3f}"
            f"  unique={hw_result['n_unique']}/{args.n_samples}"
        )

    except Exception as e:
        print(f"  [ERROR] Hardware sampling failed: {e}")
        return False

    # ── 6. Metropolis baseline ───────────────────────────────────────────
    metro = ClassicalSampler(method="metropolis", n_warmup=200)
    V_metro = metro.sample(rbm, args.n_samples)
    metro_result = compute_kl_and_ess(V_metro, p_true, config_idx, rbm)
    print(
        f"  [metro] KL={metro_result['kl']:.4f}  ESS={metro_result['ess_norm']:.3f}"
        f"  unique={metro_result['n_unique']}/{args.n_samples}"
    )

    # ── 7. Write back to result JSON ─────────────────────────────────────
    update_result_json(
        result_path,
        {
            "kl_hardware": hw_result["kl"],
            "ess_hardware": hw_result["ess_norm"],
            "kl_metro_baseline": metro_result["kl"],
            "ess_metro": metro_result["ess_norm"],
            "kl_eval": {
                "n_samples": args.n_samples,
                "beta_x": 1.0,
                "kl_max_n": args.kl_max_n,
                "bias_est": hw_result["bias_est"],
                "k_eff": hw_result["k_eff"],
                "n_unique_hw": hw_result["n_unique"],
                "n_unique_metro": metro_result["n_unique"],
                "eval_date": datetime.now().isoformat(timespec="seconds"),
            },
        },
    )
    print(f"  ✓ Updated {result_path.name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc KL evaluation for hardware samplers"
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Root of checkpoints directory",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root of results directory",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES_DEFAULT,
        help="Samples per hardware call (default 1000)",
    )
    parser.add_argument(
        "--kl-max-n",
        type=int,
        default=KL_MAX_N_DEFAULT,
        help="Skip checkpoints with N > this (default 20)",
    )
    parser.add_argument(
        "--sampler",
        choices=["pegasus", "zephyr", "velox", "all"],
        default="all",
        help="Restrict to one hardware type",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find checkpoints and result JSONs but don't call samplers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if kl_hardware already present",
    )
    args = parser.parse_args()

    if args.sampler != "all":
        global HARDWARE_METHODS
        HARDWARE_METHODS = {args.sampler}

    # Collect all candidate checkpoints
    ckpt_paths = sorted(args.checkpoints.rglob("checkpoint_*.pkl"))
    print(f"Found {len(ckpt_paths)} checkpoints under {args.checkpoints}")

    # VeloxSampler reads velox_api_config.py and velox_token.txt from CWD
    orig_dir = Path.cwd()
    os.chdir(SRC_DIR)

    n_ok = n_skip = n_err = 0
    try:
        for ckpt in ckpt_paths:
            try:
                ok = process_checkpoint(ckpt, args, args.results)
                if ok:
                    n_ok += 1
                else:
                    n_skip += 1
            except Exception as e:
                print(f"  [FATAL] {ckpt.name}: {e}")
                n_err += 1
    finally:
        os.chdir(orig_dir)

    print(f"\n{'=' * 50}")
    print(f"Done — evaluated: {n_ok}  skipped: {n_skip}  errors: {n_err}")

    # Print updated time.json summary
    if TIME_PATH.exists():
        t = json.loads(TIME_PATH.read_text())
        print(f"QPU time total  : {t.get('time_ms', 0) / 60000:.2f} min")
        if "velox_time_ms" in t:
            print(f"VeloxQ time total: {t['velox_time_ms'] / 60000:.2f} min")


if __name__ == "__main__":
    main()
