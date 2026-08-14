#!/usr/bin/env python3
"""
Four publication figures for RBM / D-Wave embedding.

  rbm_abstract.pdf              — logical bipartite RBM graph
  embedding_topology_pegasus.pdf — DWaveTopologyRBM on Pegasus (identity)
  embedding_topology_zephyr.pdf  — DWaveTopologyRBM on Zephyr  (identity)
  embedding_full_pegasus.pdf     — FullyConnectedRBM on Pegasus (busclique biclique)
  embedding_full_zephyr.pdf      — FullyConnectedRBM on Zephyr  (busclique biclique)

All offline — synthetic hardware graphs, no QPU access needed.
"""
import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import dwave_networkx as dnx
import minorminer.busclique as bc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.dirname(__file__))
from model import DWaveTopologyRBM
from plot_style import setup_style


def _shore_bipartite_selection(hw_graph, index_key, n_visible, n_hidden, seed):
    """Pick a chain-free (Nv, Nh) identity embedding using the same
    shore-aware, dead-unit-free construction as DWaveTopologyRBM."""
    bip_edges = DWaveTopologyRBM._bipartite_edges(hw_graph, index_key)
    visible, hidden = DWaveTopologyRBM._grow_shore_balanced_subgraph(
        hw_graph, bip_edges, index_key, n_visible, n_hidden, seed
    )
    nodes = sorted(visible) + sorted(hidden)
    edges = [(u, v) for u, v in bip_edges if u in visible | hidden and v in visible | hidden]
    return nodes, edges

VIS_COLOR = "#2196F3"
HID_COLOR = "#F44336"
CHAIN_COLOR = "#FF9800"
UNUSED = (0.88, 0.88, 0.88, 1.0)


def _chain_colors(emb, n_visible):
    return {node: VIS_COLOR if node < n_visible else HID_COLOR for node in emb}


def _legend(ax, show_chain=False):
    handles = [
        mpatches.Patch(color=VIS_COLOR, label="visible"),
        mpatches.Patch(color=HID_COLOR, label="hidden"),
    ]
    if show_chain:
        handles.append(mpatches.Patch(color=CHAIN_COLOR, label="chain qubit"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)


def plot_abstract_rbm(n_visible, n_hidden, out):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")

    vis_x = np.linspace(0.05, 0.95, n_visible)
    hid_x = np.linspace(0.05, 0.95, n_hidden)

    for vx in vis_x:
        for hx in hid_x:
            ax.plot([vx, hx], [1, 0], color="k", lw=0.8, alpha=0.3, zorder=0)

    # scatter markers are always screen-round regardless of axes aspect ratio
    s = max(800 // n_visible, 80)
    ax.scatter(vis_x, np.ones(n_visible), s=s, c=VIS_COLOR, zorder=2, linewidths=0)
    ax.scatter(hid_x, np.zeros(n_hidden), s=s, c=HID_COLOR, zorder=2, linewidths=0)

    ax.text(0.5, 1.18, r"\textbf{visible}", ha="center", va="bottom")
    ax.text(0.5, -0.18, r"\textbf{hidden}", ha="center", va="top")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.35, 1.35)

    _legend(ax)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_hardware(G, emb, draw_fn, layout_fn, n_visible, out, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(
        G,
        emb,
        chain_color=_chain_colors(emb, n_visible),
        unused_color=UNUSED,
        node_size=8,
        width=0.4,
        ax=ax,
    )
    # extra chain qubits (index > 0), drawn in a distinct color
    extra = [phys for chain in emb.values() for phys in list(chain)[1:]]
    if extra:
        pos = layout_fn(G)
        nx.draw_networkx_nodes(G, pos, nodelist=extra, node_color=CHAIN_COLOR,
                               node_size=8, ax=ax)
    ax.axis("off")
    _legend(ax, show_chain=bool(extra))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    setup_style(fontsize=12, scale=2.0, grid=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--out-dir", default="plots/embedding")
    args = ap.parse_args()

    Nv = Nh = args.N
    out_dir = os.path.join(args.out_dir, str(args.N))
    os.makedirs(out_dir, exist_ok=True)

    def out(name):
        return os.path.join(out_dir, f"{name}.pdf")

    source_edges = [(i, Nv + j) for i in range(Nv) for j in range(Nh)]

    # 1. abstract RBM
    plot_abstract_rbm(Nv, Nh, out("rbm_abstract"))

    # 2. DWaveTopologyRBM on Pegasus (identity embedding)
    G_peg = dnx.pegasus_graph(16, data=True)
    peg_nodes, peg_edges = _shore_bipartite_selection(G_peg, "pegasus_index", Nv, Nh, 42)
    emb_peg_id = {i: [phys] for i, phys in enumerate(peg_nodes)}
    plot_hardware(G_peg, emb_peg_id, dnx.draw_pegasus_embedding, dnx.pegasus_layout, Nv,
                  out("embedding_topology_pegasus"))

    # 3. DWaveTopologyRBM on Zephyr (identity embedding)
    G_zep = dnx.zephyr_graph(6, data=True)
    zep_nodes, zep_edges = _shore_bipartite_selection(G_zep, "zephyr_index", Nv, Nh, 42)
    emb_zep_id = {i: [phys] for i, phys in enumerate(zep_nodes)}
    plot_hardware(G_zep, emb_zep_id, dnx.draw_zephyr_embedding, dnx.zephyr_layout, Nv,
                  out("embedding_topology_zephyr"), figsize=(8, 8))

    # 4. FullyConnectedRBM on Pegasus (busclique biclique)
    emb_full_peg = bc.busgraph_cache(G_peg).find_biclique_embedding(Nv, Nh)
    if not emb_full_peg:
        raise RuntimeError(f"busclique failed on Pegasus for K_{{{Nv},{Nh}}}")
    plot_hardware(G_peg, emb_full_peg, dnx.draw_pegasus_embedding, dnx.pegasus_layout, Nv,
                  out("embedding_full_pegasus"))

    # 5. FullyConnectedRBM on Zephyr (busclique biclique)
    emb_full_zep = bc.busgraph_cache(G_zep).find_biclique_embedding(Nv, Nh)
    if not emb_full_zep:
        raise RuntimeError(f"busclique failed on Zephyr for K_{{{Nv},{Nh}}}")
    plot_hardware(G_zep, emb_full_zep, dnx.draw_zephyr_embedding, dnx.zephyr_layout, Nv,
                  out("embedding_full_zephyr"), figsize=(8, 8))

    # 6. Omitted-connections summary
    def count_vis_hid_edges(nodes, edges):
        visible, hidden = sorted(nodes[:Nv]), sorted(nodes[Nv:])
        return int(DWaveTopologyRBM._mask_from_qubit_sets(visible, hidden, edges).sum())

    full = Nv * Nh
    peg_kept = count_vis_hid_edges(peg_nodes, peg_edges)
    zep_kept = count_vis_hid_edges(zep_nodes, zep_edges)
    summary_path = os.path.join(out_dir, "topology_omissions.txt")
    with open(summary_path, "w") as f:
        f.write(f"DWaveTopologyRBM omitted connections  (N={args.N}, Nv={Nv}, Nh={Nh})\n")
        f.write(f"Full K_{{Nv,Nh}} connections: {full}\n\n")
        f.write(f"Pegasus (pegasus_graph(16), shore-bipartite, seed=42):\n")
        f.write(f"  kept:    {peg_kept} / {full}  ({100*peg_kept/full:.1f}%)\n")
        f.write(f"  omitted: {full - peg_kept} / {full}  ({100*(full-peg_kept)/full:.1f}%)\n\n")
        f.write(f"Zephyr (zephyr_graph(6), shore-bipartite, seed=42):\n")
        f.write(f"  kept:    {zep_kept} / {full}  ({100*zep_kept/full:.1f}%)\n")
        f.write(f"  omitted: {full - zep_kept} / {full}  ({100*(full-zep_kept)/full:.1f}%)\n")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
