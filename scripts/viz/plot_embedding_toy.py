#!/usr/bin/env python3
"""
plot_embedding_toy.py

Minimal textbook example of minor embedding: a triangle (K3, needs all 3
pairwise couplers) cannot be placed directly on a 4-cycle hardware graph
(bipartite, triangle-free) with one physical qubit per logical node. Chaining
two physical qubits into one logical node fixes it.

Three separate single-panel figures (for LaTeX \\subfigure/\\subcaption use),
saved under plots/embedding/toy_triangle_chain/:

  a.pdf — source graph: logical triangle, 3 fully-connected nodes
  b.pdf — target graph: bare 4-cycle hardware qubits, no roles assigned
  c.pdf — embedded result: one logical node realized as a 2-qubit chain

Plus a standalone legend.pdf. Panels carry no in-image labels/legend —
\\subfigure supplies the (a)/(b)/(c) labels, and color explanation belongs in
the caption text.

Usage (from repo root, no QPU access needed):
    python scripts/viz/plot_embedding_toy.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import setup_style

NODE_A_COLOR = "#2196F3"
NODE_B_COLOR = "#4CAF50"
NODE_C_COLOR = "#F44336"
UNUSED_COLOR = "#D9D9D9"
RAW_EDGE_COLOR = "#8E24AA"
FINAL_EDGE_COLOR = "#424242"
CHAIN_EDGE_COLOR = "#FF9800"

PANEL_FIGSIZE = (3.5, 3.5)
NODE_SIZE = 500


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
    setup_style(fontsize=10, scale=1.6, grid=False)
    out_dir = Path(__file__).resolve().parent.parent.parent / "plots" / "embedding" / "toy_triangle_chain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) source graph: logical triangle A-B-C, all 3 edges required ---
    src = nx.Graph()
    src.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    src_pos = {"A": (0.0, 1.0), "B": (-0.87, -0.5), "C": (0.87, -0.5)}
    src_colors = {"A": NODE_A_COLOR, "B": NODE_B_COLOR, "C": NODE_C_COLOR}

    fig, ax = _new_panel_fig()
    nx.draw_networkx_edges(src, src_pos, edge_color=FINAL_EDGE_COLOR, width=1.5, ax=ax)
    nx.draw_networkx_nodes(src, src_pos, node_color=[src_colors[n] for n in src.nodes()],
                            node_size=NODE_SIZE, linewidths=0.8, edgecolors="black", ax=ax)
    _save(fig, out_dir, "a")

    # --- (b) target graph: bare 4-cycle hardware qubits, no roles yet ---
    # u-v-w-x-u cycle; bipartite (no triangle), so a direct 1-to-1 mapping
    # of the source triangle onto it is impossible.
    hw = nx.Graph()
    hw.add_edges_from([("u", "v"), ("v", "w"), ("w", "x"), ("x", "u")])
    hw_pos = {"u": (-0.7, 0.7), "v": (0.7, 0.7), "w": (0.7, -0.7), "x": (-0.7, -0.7)}

    fig, ax = _new_panel_fig()
    nx.draw_networkx_edges(hw, hw_pos, edge_color=RAW_EDGE_COLOR, width=1.2, ax=ax)
    nx.draw_networkx_nodes(hw, hw_pos, node_color=UNUSED_COLOR, node_size=NODE_SIZE,
                            linewidths=0.8, edgecolors="black", ax=ax)
    _save(fig, out_dir, "b")

    # --- (c) embedded result: A -> chain {u, x}, B -> v, C -> w ---
    # u-v realizes A-B, v-w realizes B-C, w-x realizes C-A; x-u is the
    # intra-chain (ferromagnetic) coupling forcing u == x.
    node_colors_c = {"u": NODE_A_COLOR, "x": NODE_A_COLOR, "v": NODE_B_COLOR, "w": NODE_C_COLOR}
    logical_edges = [("u", "v"), ("v", "w"), ("w", "x")]
    chain_edge = [("x", "u")]

    fig, ax = _new_panel_fig()
    nx.draw_networkx_edges(hw, hw_pos, edgelist=logical_edges, edge_color=FINAL_EDGE_COLOR,
                            width=1.5, ax=ax)
    nx.draw_networkx_edges(hw, hw_pos, edgelist=chain_edge, edge_color=CHAIN_EDGE_COLOR,
                            width=2.5, style="dashed", ax=ax)
    nx.draw_networkx_nodes(hw, hw_pos, node_color=[node_colors_c[n] for n in hw.nodes()],
                            node_size=NODE_SIZE, linewidths=0.8, edgecolors="black", ax=ax)
    _save(fig, out_dir, "c")

    # --- standalone legend ---
    legend_handles = [
        mpatches.Patch(color=UNUSED_COLOR, label="hardware qubit (unassigned)"),
        mlines.Line2D([], [], color=RAW_EDGE_COLOR, lw=1.2, label="hardware coupler"),
        mlines.Line2D([], [], color=FINAL_EDGE_COLOR, lw=1.5, label="logical edge realized"),
        mlines.Line2D([], [], color=CHAIN_EDGE_COLOR, lw=2.5, ls="dashed",
                      label="chain coupling (same logical node)"),
    ]
    fig = plt.figure(figsize=(6, 0.8))
    fig.legend(handles=legend_handles, loc="center", ncol=2, frameon=False, fontsize=9)
    _save(fig, out_dir, "legend")


if __name__ == "__main__":
    main()
