#!/usr/bin/env python3
"""
plot_sparse_graph_construction.py

Three separate single-panel figures (for LaTeX \\subfigure/\\subcaption use),
saved under plots/sparsity/sparse_graph_construction/, showing how
DWaveTopologyRBM's chain-free sparse connectivity is built:

  a.pdf — raw hardware graph, every real coupler, no roles assigned yet
  b.pdf — shore-filtered subgraph: visible (shore 0) vs. hidden-unit
          candidates (shore 1), cross-shore couplers only
  c.pdf — final RBM connectivity: candidates actually selected as hidden
          units, and the edges kept between them

Plus a standalone legend.pdf to place separately in the LaTeX layout. Panels
carry no in-image labels or legend — \\subfigure supplies the (a)/(b)/(c)
labels, and color explanation belongs in the caption text.

Usage (from repo root, no QPU/JAX access needed):
    python scripts/viz/plot_sparse_graph_construction.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import dwave_networkx as dnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import setup_style

VIS_COLOR = "#2196F3"
HID_COLOR = "#F44336"
UNUSED_COLOR = "#D9D9D9"
RAW_EDGE_COLOR = "#8E24AA"
BIP_EDGE_COLOR = "#00897B"
FINAL_EDGE_COLOR = "#424242"

PANEL_FIGSIZE = (4.2, 4.2)
NODE_SIZE = 55


def load_graph_and_embedding(embedding_path: Path):
    with open(embedding_path) as f:
        emb = json.load(f)

    topology = emb["topology"]
    gen = dnx.pegasus_graph if topology == "pegasus" else dnx.zephyr_graph
    index_key = "pegasus_index" if topology == "pegasus" else "zephyr_index"
    shape = (16,) if topology == "pegasus" else (12, 4)

    hw_path = Path(__file__).resolve().parent.parent.parent / "embeddings" / (
        f"_hwgraph_{emb['solver'].replace('/', '_').replace('.', '_')}_live.json"
    )
    with open(hw_path) as f:
        hw_data = json.load(f)
    node_list = [n for n, _ in hw_data["nodes"]]
    edge_list = [tuple(e) for e in hw_data["edges"]]
    # Rebuild via the dwave_networkx generator (not a plain nx.Graph) so the
    # graph carries the family metadata zephyr_layout/pegasus_layout require.
    G = gen(*shape, node_list=node_list, edge_list=edge_list, data=True)
    return G, emb, index_key


def build_candidate_pool(G, emb, index_key):
    """Visible qubits plus every shore-1 qubit directly reachable from them
    (cross-shore only) — i.e. the full hidden-unit candidate pool the
    selection algorithm actually chooses from, whether or not it ended up
    picking a given candidate."""
    visible = set(emb["visible_qubits"])
    hidden = set(emb["hidden_qubits"])

    candidates = set()
    for v in visible:
        candidates |= {
            n for n in G.neighbors(v) if G.nodes[n][index_key][0] != G.nodes[v][index_key][0]
        }
    assert hidden <= candidates, "cached hidden set should be a subset of visible's cross-shore neighbours"

    neighborhood = visible | candidates
    sub = G.subgraph(neighborhood).copy()
    return sub, visible, hidden, candidates


def _new_panel_fig():
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


def _save(fig, out_dir: Path, name: str):
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"saved {p}")
    plt.close(fig)


def main():
    embedding_path = (
        Path(__file__).resolve().parent.parent.parent
        / "embeddings" / "Advantage2_system1_8v_8h_seed42_live.json"
    )
    G, emb, index_key = load_graph_and_embedding(embedding_path)
    sub, visible, hidden, candidates = build_candidate_pool(G, emb, index_key)
    pos = dnx.zephyr_layout(sub) if emb["topology"] == "zephyr" else dnx.pegasus_layout(sub)

    rejected = candidates - hidden
    final_edges = [tuple(e) for e in emb["edges"]]
    bip_edges = [
        (u, v) for u, v in sub.edges()
        if sub.nodes[u][index_key][0] != sub.nodes[v][index_key][0]
    ]

    setup_style(fontsize=10, scale=1.6, grid=False)
    out_dir = Path(__file__).resolve().parent.parent.parent / "plots" / "sparsity" / "sparse_graph_construction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) raw hardware graph: every real coupler, no roles assigned yet, but
    # cross-shore couplers (kept in (b)) already colored distinctly from
    # same-shore ones (dropped in (b)) to preview the filtering step.
    bip_edge_set = set(bip_edges) | {(v, u) for u, v in bip_edges}
    same_shore_edges = [e for e in sub.edges() if e not in bip_edge_set]
    fig, ax = _new_panel_fig()
    nx.draw_networkx_edges(sub, pos, edgelist=same_shore_edges, edge_color=RAW_EDGE_COLOR, width=0.6, ax=ax)
    nx.draw_networkx_edges(sub, pos, edgelist=bip_edges, edge_color=BIP_EDGE_COLOR, width=0.7, ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_color=UNUSED_COLOR, node_size=NODE_SIZE, linewidths=0.4,
                            edgecolors="black", ax=ax)
    _save(fig, out_dir, "a")

    # (b) shore-filtered subgraph: visible (shore 0) vs. hidden candidates (shore 1)
    fig, ax = _new_panel_fig()
    nx.draw_networkx_edges(sub, pos, edgelist=bip_edges, edge_color=BIP_EDGE_COLOR, width=0.7, ax=ax)
    node_colors_b = [VIS_COLOR if n in visible else HID_COLOR for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors_b, node_size=NODE_SIZE, linewidths=0.4,
                            edgecolors="black", ax=ax)
    _save(fig, out_dir, "b")

    # (c) final RBM connectivity: only the selected hidden candidates survive
    fig, ax = _new_panel_fig()
    nx.draw_networkx_edges(sub, pos, edgelist=final_edges, edge_color=FINAL_EDGE_COLOR, width=0.9, ax=ax)
    node_colors_c = [
        VIS_COLOR if n in visible else HID_COLOR if n in hidden else UNUSED_COLOR
        for n in sub.nodes()
    ]
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors_c, node_size=NODE_SIZE, linewidths=0.4,
                            edgecolors="black", ax=ax)
    _save(fig, out_dir, "c")

    # standalone legend, to place separately in the LaTeX layout
    legend_handles = [
        mpatches.Patch(color=VIS_COLOR, label="visible qubit"),
        mpatches.Patch(color=HID_COLOR, label="hidden qubit (selected)"),
        mpatches.Patch(color=UNUSED_COLOR, label="candidate / unused (not selected)"),
    ]
    fig = plt.figure(figsize=(6, 0.4))
    fig.legend(handles=legend_handles, loc="center", ncol=3, frameon=False, fontsize=9)
    _save(fig, out_dir, "legend")


if __name__ == "__main__":
    main()
