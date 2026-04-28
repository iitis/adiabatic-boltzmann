#!/usr/bin/env python3
"""
Probe the maximum 1D and 2D Ising model sizes that fit on the Zephyr QPU
with a trivial (chain-free) embedding.

The DWaveTopologyRBM uses one physical qubit per logical variable, so the
hardware constraint is simply:

    n_visible + n_hidden  ≤  number of functional qubits

With α = 1 (n_hidden = n_visible) that is:

    2 * n_visible  ≤  #qubits

However, the useful limit is tighter: the mask quality (cross-edges between
the "visible" and "hidden" halves of the selected subgraph) can degrade at
large sizes.  This script measures connectivity for every candidate size and
reports:

  - "viable"        : every visible unit has ≥ 1 hidden neighbour
  - "well-connected": mean visible degree ≥ MIN_MEAN_DEGREE

No QPU annealing is performed — only `to_networkx_graph()` is called.

Usage (from project root, venv active):
    cd src && python ../probe_capacity.py
"""

import csv
import sys
from pathlib import Path

# Run from src/ so local imports work
SCRIPT_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import networkx as nx

from helpers import get_solver_name
from model import DWaveTopologyRBM, _dense_subgraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOLVER_KEY = "zephyr"          # maps to Advantage2_system1.13 via get_solver_name
SEED = 42                       # fixed seed → reproducible subgraph selection
ALPHA = 1                       # n_hidden / n_visible

# Minimum mean visible-unit degree to call a configuration "well-connected"
MIN_MEAN_DEGREE = 3.0

SIZES_1D = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
L_VALUES_2D = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40]

OUTPUT_DIR = Path("probe_results")
OUTPUT_CSV = OUTPUT_DIR / "zephyr_capacity.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_hardware_graph(solver_name: str):
    from dwave.system import DWaveSampler
    print(f"Connecting to D-Wave solver '{solver_name}' to fetch hardware graph …")
    sampler = DWaveSampler(solver=solver_name)
    g = sampler.to_networkx_graph()
    print(
        f"  Hardware graph: {g.number_of_nodes()} qubits, "
        f"{g.number_of_edges()} couplers\n"
    )
    return g


def probe_one(hw_graph, n_visible: int, n_hidden: int) -> dict:
    """
    Try to build a DWaveTopologyRBM mask for (n_visible, n_hidden) and return
    a summary dict.  Returns {"error": str} on failure.
    """
    n_nodes = n_visible + n_hidden
    if hw_graph.number_of_nodes() < n_nodes:
        return {"error": f"Not enough qubits ({hw_graph.number_of_nodes()} < {n_nodes})"}

    try:
        selected = _dense_subgraph(hw_graph, n_nodes, seed=SEED)
        subgraph = hw_graph.subgraph(selected)
        sorted_nodes = sorted(subgraph.nodes())
        qubit_mapping = {phys: idx for idx, phys in enumerate(sorted_nodes)}
        mapped = nx.relabel_nodes(subgraph, qubit_mapping)
        mask = DWaveTopologyRBM._mask_from_graph(mapped, n_visible, n_hidden)
    except Exception as exc:
        return {"error": str(exc)}

    deg_vis = mask.sum(axis=1)   # connections per visible unit
    deg_hid = mask.sum(axis=0)   # connections per hidden unit
    n_conn  = int(mask.sum())

    return {
        "n_qubits":       n_nodes,
        "n_connections":  n_conn,
        "max_connections": n_visible * n_hidden,
        "sparsity":       1.0 - n_conn / (n_visible * n_hidden),
        "deg_vis_mean":   float(deg_vis.mean()),
        "deg_vis_min":    float(deg_vis.min()),
        "deg_vis_max":    float(deg_vis.max()),
        "deg_hid_mean":   float(deg_hid.mean()),
        "deg_hid_min":    float(deg_hid.min()),
        "deg_hid_max":    float(deg_hid.max()),
        "viable":         bool(deg_vis.min() > 0),
        "well_connected": bool(deg_vis.mean() >= MIN_MEAN_DEGREE),
    }


def print_table(rows: list[dict]):
    header = (
        f"{'model':>5} {'size':>5} {'n_vis':>6} {'n_hid':>6} "
        f"{'qubits':>7} {'conns':>7} "
        f"{'deg_v_mean':>10} {'deg_v_min':>9} "
        f"{'viable':>7} {'well_conn':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        if "error" in r:
            print(
                f"{r['model']:>5} {r['size']:>5} {r['n_visible']:>6} {r['n_hidden']:>6} "
                f"  {'ERROR':>7}  {r['error']}"
            )
        else:
            print(
                f"{r['model']:>5} {r['size']:>5} {r['n_visible']:>6} {r['n_hidden']:>6} "
                f"{r['n_qubits']:>7} {r['n_connections']:>7} "
                f"{r['deg_vis_mean']:>10.2f} {r['deg_vis_min']:>9.0f} "
                f"{'YES':>7}" if r["viable"] else
                f"{r['model']:>5} {r['size']:>5} {r['n_visible']:>6} {r['n_hidden']:>6} "
                f"{r['n_qubits']:>7} {r['n_connections']:>7} "
                f"{r['deg_vis_mean']:>10.2f} {r['deg_vis_min']:>9.0f} "
                f"{'NO':>7}",
                end=""
            )
            print(f" {'YES':>9}" if r.get("well_connected") else f" {'NO':>9}")
    print(sep)


def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "size", "n_visible", "n_hidden", "alpha",
        "n_qubits", "n_connections", "max_connections", "sparsity",
        "deg_vis_mean", "deg_vis_min", "deg_vis_max",
        "deg_hid_mean", "deg_hid_min", "deg_hid_max",
        "viable", "well_connected", "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nCSV saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    solver_name: str = get_solver_name(SOLVER_KEY) or ""
    if not solver_name:
        raise ValueError(f"Unknown solver key: {SOLVER_KEY}")
    hw_graph = fetch_hardware_graph(solver_name)
    n_hw_qubits = hw_graph.number_of_nodes()

    rows = []

    # ── 1D models ──────────────────────────────────────────────────────────
    print("=== 1D models ===\n")
    for size in SIZES_1D:
        n_visible = size
        n_hidden  = int(np.ceil(ALPHA * n_visible))
        n_nodes   = n_visible + n_hidden
        if n_nodes > n_hw_qubits:
            print(f"  1D N={size}: would require {n_nodes} qubits — exceeds hardware ({n_hw_qubits}), stopping.")
            break

        stats = probe_one(hw_graph, n_visible, n_hidden)
        row = {
            "model": "1d", "size": size,
            "n_visible": n_visible, "n_hidden": n_hidden,
            "alpha": ALPHA, **stats,
        }
        rows.append(row)
        if "error" in stats:
            print(f"  1D N={size:>5}  ERROR: {stats['error']}")
        else:
            flag = " ✓ well-connected" if stats["well_connected"] else (" ✓ viable" if stats["viable"] else " ✗ DISCONNECTED")
            print(
                f"  1D N={size:>5}  qubits={n_nodes:>5}  "
                f"conns={stats['n_connections']:>6}  "
                f"deg_vis_mean={stats['deg_vis_mean']:>5.2f}  "
                f"min={stats['deg_vis_min']:>3.0f}{flag}"
            )

    # ── 2D models ──────────────────────────────────────────────────────────
    print("\n=== 2D models ===\n")
    for L in L_VALUES_2D:
        n_visible = L * L
        n_hidden  = int(np.ceil(ALPHA * n_visible))
        n_nodes   = n_visible + n_hidden
        if n_nodes > n_hw_qubits:
            print(f"  2D L={L} (N={n_visible}): would require {n_nodes} qubits — exceeds hardware ({n_hw_qubits}), stopping.")
            break

        stats = probe_one(hw_graph, n_visible, n_hidden)
        row = {
            "model": "2d", "size": L,
            "n_visible": n_visible, "n_hidden": n_hidden,
            "alpha": ALPHA, **stats,
        }
        rows.append(row)
        if "error" in stats:
            print(f"  2D L={L:>3} (N={n_visible:>5})  ERROR: {stats['error']}")
        else:
            flag = " ✓ well-connected" if stats["well_connected"] else (" ✓ viable" if stats["viable"] else " ✗ DISCONNECTED")
            print(
                f"  2D L={L:>3} N={n_visible:>5}  qubits={n_nodes:>5}  "
                f"conns={stats['n_connections']:>6}  "
                f"deg_vis_mean={stats['deg_vis_mean']:>5.2f}  "
                f"min={stats['deg_vis_min']:>3.0f}{flag}"
            )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n=== Summary ===\n")

    viable_1d = [r for r in rows if r["model"] == "1d" and r.get("viable")]
    wc_1d     = [r for r in rows if r["model"] == "1d" and r.get("well_connected")]
    viable_2d = [r for r in rows if r["model"] == "2d" and r.get("viable")]
    wc_2d     = [r for r in rows if r["model"] == "2d" and r.get("well_connected")]

    def _max_size(lst, key="size"):
        return max(r[key] for r in lst) if lst else "N/A"

    print(f"  Hardware qubits          : {n_hw_qubits}")
    print(f"  1D max viable            : N = {_max_size(viable_1d)}")
    print(f"  1D max well-connected    : N = {_max_size(wc_1d)}  (mean degree ≥ {MIN_MEAN_DEGREE})")
    print(f"  2D max viable            : L = {_max_size(viable_2d)}  (N = {_max_size(viable_2d, 'n_visible')})")
    print(f"  2D max well-connected    : L = {_max_size(wc_2d)}  (N = {_max_size(wc_2d, 'n_visible')})")

    save_csv(rows, OUTPUT_CSV)
    print_table(rows)


if __name__ == "__main__":
    main()
