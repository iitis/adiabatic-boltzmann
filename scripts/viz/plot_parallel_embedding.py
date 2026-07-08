#!/usr/bin/env python3
"""
plot_parallel_embedding.py

Visualizes n_parallel disjoint K_{Nv,Nh} biclique embeddings sitting
simultaneously on one QPU chip -- the physical picture behind
DimodSampler.sample_parallel / ParallelEmbeddingComposite (src/sampler.py's
_get_parallel_composite): each copy is a separate, non-overlapping, real
busclique embedding of the same logical RBM.

Follows the project's existing low-level drawing convention (see
plot_sparse_graph_construction.py) rather than dwave_networkx's
draw_*_embedding: compute dnx.pegasus_layout on a subgraph restricted to just
the qubits actually used (a tight, roughly-square crop instead of the whole
~5600-qubit chip), draw with nx.draw_networkx_nodes/edges directly, and color
by role -- VIS_COLOR/HID_COLOR (the same two colors used everywhere else in
this project's embedding figures), shared across all copies, so a single
two-entry legend ("visible"/"hidden") covers every copy instead of one
legend entry per copy.

To keep n_parallel copies close together with a roughly square bounding box
(rather than busclique's default sequential search, which tends to lay
successive copies out in a thin row along one edge of the chip -- see the
`--row` flag below), find_close_grid_embeddings confines the search to a
small local window around one anchor embedding, split into a
ceil(sqrt(n))-by-ceil(sqrt(n)) grid of sub-regions, and finds one real
embedding per sub-region.

Offline -- uses the idealized dnx.pegasus_graph(16) / zephyr_graph(12, 4),
no QPU access needed, matching plot_dwave_embedding.py's convention.

Usage (from repo root):
    python scripts/viz/plot_parallel_embedding.py --N 8 --n-parallel 4
    python scripts/viz/plot_parallel_embedding.py --N 8 --n-parallel 5 --row
"""
import argparse
import os
import sys

import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import minorminer.busclique as bc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.dirname(__file__))
from plot_style import setup_style

# Fixed, colorblind-distinguishable hue per copy index -- same hue used
# consistently across figures for a given copy slot (copy identity, not
# rank, drives color). Visible/hidden role is a second, color-independent
# encoding (hatched vs. solid fill) so both copy identity and role are
# readable off one set of markers without a color x role product.
COPY_COLORS = ["#2166ac", "#d62728", "#2ca02c", "#f5a623", "#9467bd", "#17becf", "#8c564b", "#e377c2"]
NODE_SIZE = 70
VISIBLE_HATCH = "////"
# Same "unused chip" grey used elsewhere in the project (plot_dwave_embedding.py,
# plot_sparse_graph_construction.py).
UNUSED_NODE_COLOR = "#D9D9D9"
UNUSED_EDGE_COLOR = "#E5E5E5"


def find_disjoint_embeddings(hw_graph, n_visible, n_hidden, n_parallel):
    """Same sequential remove-and-reembed search as
    DimodSampler._get_parallel_composite (src/sampler.py) actually uses at
    runtime -- copies end up adjacent along a thin strip of the chip."""
    remaining = hw_graph.copy()
    embeddings = []
    for k in range(n_parallel):
        emb = bc.busgraph_cache(remaining).find_biclique_embedding(n_visible, n_hidden)
        if not emb:
            raise RuntimeError(
                f"busclique found only {k} disjoint embeddings for "
                f"K_{{{n_visible},{n_hidden}}}, but {n_parallel} were requested."
            )
        embeddings.append(emb)
        used = {q for chain in emb.values() for q in chain}
        remaining = remaining.copy()
        remaining.remove_nodes_from(used)
    return embeddings


def find_close_grid_embeddings(hw_graph, layout, n_visible, n_hidden, n_parallel, start_frac=0.15):
    """n_parallel real busclique embeddings, spatially confined to a small
    square window (so the resulting figure crops to a compact, roughly
    square region) and spread across a grid of sub-regions within that
    window so copies visibly separate instead of touching/crowding. Each
    embedding is still a genuine K_{Nv,Nh} biclique embedding on the real
    hardware graph -- only the *search region* is constrained, nothing is
    repositioned after the fact.

    The window is centered on the whole chip's own centroid (not on a
    single anchor embedding's position): Pegasus/Zephyr's native layout is
    diamond-shaped, not a filled square, so a window centered off-center
    (e.g. on wherever busclique's cache happens to place one embedding
    first) can have one quadrant sitting almost entirely outside the chip's
    real node range -- stuck at a handful of nodes no matter how much the
    window grows. Centering on the chip's own centroid keeps all four
    quadrants balanced from the start.

    Disjointness: adjacent grid cells share a boundary coordinate (both
    computed from the same np.linspace), so a qubit sitting exactly on that
    line is a valid candidate for both neighboring cells -- and since each
    cell used to be searched independently, busclique could (and did) pick
    that same physical qubit for two different copies. Fixed the same way
    _get_parallel_composite (src/sampler.py) guarantees disjointness for the
    real sampler: track every qubit claimed so far and exclude it from every
    later cell's candidate pool, so no two cells can ever share a qubit
    regardless of boundary geometry.
    """
    ncols = int(np.ceil(np.sqrt(n_parallel)))
    nrows = int(np.ceil(n_parallel / ncols))

    pts_all = np.array(list(layout.values()))
    lo_full, hi_full = pts_all.min(axis=0), pts_all.max(axis=0)
    center = (lo_full + hi_full) / 2
    extent = hi_full - lo_full

    for attempt in range(4):
        frac = start_frac * (1.5 ** attempt)
        half = extent * min(frac, 1.0) / 2
        lo, hi = center - half, center + half
        window_nodes = [n for n, (x, y) in layout.items() if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]]

        x_edges = np.linspace(lo[0], hi[0], ncols + 1)
        y_edges = np.linspace(lo[1], hi[1], nrows + 1)
        embeddings = []
        used_qubits = set()
        ok = True
        for k in range(n_parallel):
            row, col = divmod(k, ncols)
            x_lo, x_hi = x_edges[col], x_edges[col + 1]
            y_lo, y_hi = y_edges[row], y_edges[row + 1]
            cell_nodes = [
                n for n in window_nodes
                if n not in used_qubits and x_lo <= layout[n][0] <= x_hi and y_lo <= layout[n][1] <= y_hi
            ]
            emb = bc.busgraph_cache(hw_graph.subgraph(cell_nodes)).find_biclique_embedding(n_visible, n_hidden)
            if not emb:
                ok = False
                break
            embeddings.append(emb)
            used_qubits.update(q for chain in emb.values() for q in chain)
        if ok:
            return embeddings

    raise RuntimeError(
        f"could not fit {n_parallel} disjoint K_{{{n_visible},{n_hidden}}} embeddings "
        f"into a close-together grid after 4 window-size attempts."
    )


def plot_embeddings(hw_graph, embeddings, n_visible, layout_fn, out, figsize=(5, 5), context_pad=0.4):
    """Draws each copy's real chain structure in its own color (copy
    identity) and role (visible/hidden) as hatched-vs-solid fill on top of
    that color. Since each copy is processed (and its edges restricted to)
    its own qubit set independently, there is no risk of drawing a spurious
    edge between two different copies -- unlike calling
    dwave_networkx.draw_*_embedding on all copies merged into one dict
    without an explicit embedded_graph, which draws a "real" edge for *any*
    hardware coupler between *any* two chains, including ones in different,
    logically disconnected copies.

    context_pad: fraction of the embedded qubits' bounding-box size added as
    margin around it -- that expanded region's real, otherwise-idle chip
    neighborhood (every qubit/coupler there, used or not) is drawn first in
    light grey as background context, purely for visual purposes (it plays
    no role in the actual embedding), then the real embedded structure is
    drawn on top in full color so it stays the visually dominant layer.
    """
    all_qubits = {q for emb in embeddings for chain in emb.values() for q in chain}
    layout_full = layout_fn(hw_graph)
    pts = np.array([layout_full[q] for q in all_qubits])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = (hi - lo) * context_pad + 0.01
    lo_c, hi_c = lo - pad, hi + pad
    context_nodes = [n for n, (x, y) in layout_full.items() if lo_c[0] <= x <= hi_c[0] and lo_c[1] <= y <= hi_c[1]]
    sub = hw_graph.subgraph(context_nodes)
    pos = layout_fn(sub)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background: the real, otherwise-idle chip neighborhood around the
    # embeddings -- every qubit/coupler in view, whether used or not.
    nx.draw_networkx_edges(sub, pos, edge_color=UNUSED_EDGE_COLOR, width=0.4, ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_color=UNUSED_NODE_COLOR, node_size=NODE_SIZE * 0.6,
                            linewidths=0.3, edgecolors="#BFBFBF", ax=ax)

    for k, emb in enumerate(embeddings):
        color = COPY_COLORS[k % len(COPY_COLORS)]
        qlabel = {q: v for v, chain in emb.items() for q in chain}
        copy_sub = sub.subgraph(qlabel.keys())

        # Within one copy: same chain (u == v) or a genuine visible-hidden
        # pair (opposite sides of n_visible) is the only real edge a plain
        # FullyConnectedRBM ever has -- no visible-visible / hidden-hidden
        # couplings, so "opposite sides" already exactly means "connected".
        edges = [
            (p, q) for p, q in copy_sub.edges()
            if qlabel[p] == qlabel[q] or (qlabel[p] < n_visible) != (qlabel[q] < n_visible)
        ]
        nx.draw_networkx_edges(sub, pos, edgelist=edges, edge_color=color, width=0.6, alpha=0.8, ax=ax)

        hidden_nodes = [q for q in qlabel if qlabel[q] >= n_visible]
        visible_nodes = [q for q in qlabel if qlabel[q] < n_visible]

        nx.draw_networkx_nodes(sub, pos, nodelist=hidden_nodes, node_color=color,
                                node_size=NODE_SIZE, linewidths=0.4, edgecolors="black", ax=ax)
        vis_coll = nx.draw_networkx_nodes(sub, pos, nodelist=visible_nodes, node_color=color,
                                           node_size=NODE_SIZE, linewidths=0.4, edgecolors="black", ax=ax)
        if vis_coll is not None:
            vis_coll.set_hatch(VISIBLE_HATCH)

    role_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=VISIBLE_HATCH, label="visible"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="hidden"),
    ]
    ax.legend(handles=role_handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=9, frameon=False)

    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    setup_style(fontsize=12, scale=2.0, grid=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n-parallel", type=int, default=4)
    ap.add_argument("--topology", choices=["pegasus", "zephyr"], default="pegasus")
    ap.add_argument("--out-dir", default="plots/embedding/parallel_embedding")
    ap.add_argument("--row", action="store_true",
                     help="Use the real sequential search (copies end up in a thin "
                          "row) instead of the close-together square-grid search.")
    args = ap.parse_args()

    Nv = Nh = args.N
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.topology == "pegasus":
        hw_graph = dnx.pegasus_graph(16, data=True)
        layout_fn = dnx.pegasus_layout
    else:
        hw_graph = dnx.zephyr_graph(12, 4, data=True)
        layout_fn = dnx.zephyr_layout
    layout = layout_fn(hw_graph)

    if args.row:
        embeddings = find_disjoint_embeddings(hw_graph, Nv, Nh, args.n_parallel)
        suffix = "_row"
    else:
        embeddings = find_close_grid_embeddings(hw_graph, layout, Nv, Nh, args.n_parallel)
        suffix = ""

    chains = [len(v) for emb in embeddings for v in emb.values()]
    print(f"{args.n_parallel} K_{{{Nv},{Nh}}} embeddings on {args.topology}: "
          f"max_chain={max(chains)}, mean_chain={sum(chains) / len(chains):.2f}")

    fname = f"parallel_embedding_np{args.n_parallel}_{args.topology}{suffix}.pdf"
    plot_embeddings(hw_graph, embeddings, Nv, layout_fn, os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()
