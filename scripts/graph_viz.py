#!/usr/bin/env python3
"""
QPU Graph Explorer
==================
Fetches a D-Wave solver's hardware graph and opens an interactive browser
UI for exploring nodes, their connections, and searching by node ID.

Usage
-----
    python explore_qpu_graph.py --solver Advantage_system6.4
    python explore_qpu_graph.py --solver Advantage_system6.4 --port 8765
    python explore_qpu_graph.py --mock                        # offline demo
"""

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path


# ---------------------------------------------------------------------------
# Graph data fetching
# ---------------------------------------------------------------------------

def fetch_qpu_graph(solver_name: str) -> dict:
    """
    Connect to a D-Wave solver and return node/edge data as plain dicts.
    """
    try:
        from dwave.system import DWaveSampler
    except ImportError:
        raise ImportError(
            "dwave-system is required. Install with: pip install dwave-system"
        )

    print(f"  Connecting to solver '{solver_name}' ...")
    sampler   = DWaveSampler(solver=solver_name)
    hw_graph  = sampler.to_networkx_graph()
    props     = sampler.properties

    nodes = [{"id": int(n), "degree": hw_graph.degree(n)} for n in hw_graph.nodes()]
    edges = [{"source": int(u), "target": int(v)} for u, v in hw_graph.edges()]

    print(f"  Nodes : {len(nodes)}")
    print(f"  Edges : {len(edges)}")

    return {
        "solver":   solver_name,
        "topology": props.get("topology", {}).get("type", "unknown"),
        "nodes":    nodes,
        "edges":    edges,
        "h_range":  props.get("h_range",  [-2.0, 2.0]),
        "j_range":  props.get("j_range",  [-1.0, 1.0]),
        "n_qubits": props.get("num_qubits", len(nodes)),
    }


def mock_qpu_graph() -> dict:
    """
    Generate a small Chimera-like graph for offline testing (no QPU token needed).
    """
    import math

    # 4x4 Chimera unit cell (K_{4,4} repeated)
    nodes_set = set()
    edges_set  = set()
    m = 4   # grid size
    k = 4   # shore size

    def chimera_idx(i, j, u, k_idx):
        return i * m * 2 * k + j * 2 * k + u * k + k_idx

    for i in range(m):
        for j in range(m):
            # internal bipartite edges within cell
            for left in range(k):
                for right in range(k):
                    u = chimera_idx(i, j, 0, left)
                    v = chimera_idx(i, j, 1, right)
                    nodes_set.add(u); nodes_set.add(v)
                    edges_set.add((min(u, v), max(u, v)))
            # horizontal chain (left shore)
            if j + 1 < m:
                for ki in range(k):
                    u = chimera_idx(i, j,     0, ki)
                    v = chimera_idx(i, j + 1, 0, ki)
                    nodes_set.add(u); nodes_set.add(v)
                    edges_set.add((min(u, v), max(u, v)))
            # vertical chain (right shore)
            if i + 1 < m:
                for ki in range(k):
                    u = chimera_idx(i,     j, 1, ki)
                    v = chimera_idx(i + 1, j, 1, ki)
                    nodes_set.add(u); nodes_set.add(v)
                    edges_set.add((min(u, v), max(u, v)))

    from collections import Counter
    degree = Counter()
    for u, v in edges_set:
        degree[u] += 1; degree[v] += 1

    nodes = [{"id": n, "degree": degree[n]} for n in sorted(nodes_set)]
    edges = [{"source": u, "target": v} for u, v in sorted(edges_set)]

    print(f"  [mock] Chimera C_{m}  Nodes: {len(nodes)}  Edges: {len(edges)}")
    return {
        "solver":   "mock_chimera_C4",
        "topology": "chimera",
        "nodes":    nodes,
        "edges":    edges,
        "h_range":  [-2.0, 2.0],
        "j_range":  [-1.0, 1.0],
        "n_qubits": len(nodes),
    }


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QPU Graph Explorer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');

  :root {
    --bg:      #07090f;
    --bg2:     #0d1117;
    --bg3:     #151b26;
    --border:  #1e2836;
    --border2: #253044;
    --accent:  #3b82f6;
    --green:   #10b981;
    --yellow:  #f59e0b;
    --red:     #ef4444;
    --text:    #e2e8f0;
    --text2:   #94a3b8;
    --text3:   #4b5f78;
    --mono:    'JetBrains Mono', monospace;
    --display: 'Syne', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono); font-size: 13px; display: flex; flex-direction: column; }

  /* ── header ── */
  header {
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    padding: 12px 24px; border-bottom: 1px solid var(--border);
    background: var(--bg2); flex-shrink: 0;
  }
  .logo { font-family: var(--display); font-size: 17px; font-weight: 800; white-space: nowrap; }
  .logo span { color: var(--accent); }
  .meta { font-size: 11px; color: var(--text2); }
  .meta strong { color: var(--text); }

  /* ── search ── */
  .search-wrap {
    display: flex; align-items: center; gap: 8px; margin-left: auto; flex-wrap: wrap;
  }
  .search-input {
    background: var(--bg3); border: 1px solid var(--border2); color: var(--text);
    font-family: var(--mono); font-size: 12px; padding: 6px 12px; border-radius: 5px;
    width: 200px; outline: none; transition: border-color .15s;
  }
  .search-input:focus { border-color: var(--accent); }
  .search-input::placeholder { color: var(--text3); }
  .btn {
    background: none; border: 1px solid var(--border2); color: var(--text2);
    font-family: var(--mono); font-size: 11px; padding: 6px 12px; border-radius: 5px;
    cursor: pointer; transition: all .15s; white-space: nowrap;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn.primary:hover { background: #2563eb; }

  /* ── layout ── */
  .layout { display: flex; flex: 1; overflow: hidden; }

  /* ── sidebar ── */
  aside {
    width: 300px; min-width: 260px; border-right: 1px solid var(--border);
    background: var(--bg2); display: flex; flex-direction: column; overflow: hidden;
  }
  .sidebar-title {
    font-size: 9px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    color: var(--text3); padding: 12px 16px 8px; border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .node-list { flex: 1; overflow-y: auto; }
  .node-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 7px 16px; border-bottom: 1px solid var(--border); cursor: pointer;
    transition: background .1s; gap: 8px;
  }
  .node-item:hover { background: var(--bg3); }
  .node-item.selected { background: rgba(59,130,246,.12); border-left: 2px solid var(--accent); }
  .node-item.highlighted { background: rgba(245,158,11,.08); }
  .node-id { font-weight: 600; color: var(--text); font-size: 12px; }
  .node-deg { font-size: 10px; color: var(--text3); }
  .deg-badge {
    font-size: 9px; padding: 1px 6px; border-radius: 3px; font-weight: 600;
    background: var(--bg3); color: var(--text2);
  }

  /* ── canvas area ── */
  .canvas-wrap { flex: 1; position: relative; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.grabbing { cursor: grabbing; }

  /* ── info panel ── */
  .info-panel {
    position: absolute; top: 16px; right: 16px; width: 240px;
    background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px; display: none;
  }
  .info-panel.visible { display: block; }
  .info-title { font-family: var(--display); font-size: 14px; font-weight: 800; margin-bottom: 10px; }
  .info-row { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 5px; }
  .info-label { color: var(--text3); }
  .info-val { color: var(--text); font-weight: 600; }
  .info-neighbors { margin-top: 10px; }
  .info-neighbors-title { font-size: 9px; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .neighbor-chips { display: flex; flex-wrap: wrap; gap: 4px; max-height: 120px; overflow-y: auto; }
  .chip {
    font-size: 10px; padding: 2px 7px; border-radius: 3px; cursor: pointer;
    background: var(--bg3); color: var(--text2); border: 1px solid var(--border);
    transition: all .1s;
  }
  .chip:hover { border-color: var(--accent); color: var(--accent); }

  /* ── status bar ── */
  .statusbar {
    font-size: 10px; color: var(--text3); padding: 5px 16px;
    border-top: 1px solid var(--border); background: var(--bg2);
    flex-shrink: 0; display: flex; gap: 20px;
  }
  .statusbar span { white-space: nowrap; }
  .statusbar .accent { color: var(--accent); }

  /* ── legend ── */
  .legend {
    position: absolute; bottom: 16px; left: 16px;
    background: var(--bg2); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 14px; font-size: 10px; color: var(--text2);
  }
  .legend-row { display: flex; align-items: center; gap: 7px; margin-bottom: 4px; }
  .legend-row:last-child { margin-bottom: 0; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="logo">QPU <span>Explorer</span></div>
  <div class="meta">
    Solver: <strong id="h-solver">—</strong> &nbsp;·&nbsp;
    Topology: <strong id="h-topo">—</strong> &nbsp;·&nbsp;
    <strong id="h-nodes">—</strong> nodes &nbsp;·&nbsp;
    <strong id="h-edges">—</strong> edges
  </div>
  <div class="search-wrap">
    <input class="search-input" id="search" type="text" placeholder="Search node ID…" autocomplete="off">
    <button class="btn primary" onclick="doSearch()">Find</button>
    <button class="btn" onclick="resetView()">Reset view</button>
    <button class="btn" onclick="clearSelection()">Clear</button>
  </div>
</header>

<div class="layout">
  <aside>
    <div class="sidebar-title">Nodes — <span id="sidebar-count">0</span></div>
    <div class="node-list" id="node-list"></div>
  </aside>

  <div class="canvas-wrap">
    <canvas id="canvas"></canvas>

    <div class="info-panel" id="info-panel">
      <div class="info-title">Node <span id="ip-id">—</span></div>
      <div class="info-row"><span class="info-label">Degree</span><span class="info-val" id="ip-deg">—</span></div>
      <div class="info-row"><span class="info-label">Qubit index</span><span class="info-val" id="ip-idx">—</span></div>
      <div class="info-neighbors">
        <div class="info-neighbors-title">Connected to (<span id="ip-conn-count">0</span>)</div>
        <div class="neighbor-chips" id="ip-neighbors"></div>
      </div>
    </div>

    <div class="legend">
      <div class="legend-row"><div class="legend-dot" style="background:#3b82f6"></div> Default node</div>
      <div class="legend-row"><div class="legend-dot" style="background:#f59e0b"></div> Selected node</div>
      <div class="legend-row"><div class="legend-dot" style="background:#10b981"></div> Connected neighbour</div>
      <div class="legend-row"><div class="legend-dot" style="background:#ef4444"></div> Search match</div>
    </div>
  </div>
</div>

<div class="statusbar">
  <span>Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Click node to inspect</span>
  <span>Zoom: <span class="accent" id="st-zoom">100%</span></span>
  <span>Pan: <span class="accent" id="st-pan">0, 0</span></span>
  <span id="st-hover"></span>
</div>

<script>
// ── data injected by server ────────────────────────────────────────────────
const GRAPH_DATA = __GRAPH_JSON__;

// ── state ──────────────────────────────────────────────────────────────────
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');

let nodes     = [];   // {id, x, y, degree}
let adjMap    = {};   // id -> Set of neighbour ids
let nodeById  = {};   // id -> node

let selected  = null;   // selected node id
let searchHit = new Set();

// viewport
let scale   = 1;
let offsetX = 0;
let offsetY = 0;

// interaction
let dragging   = false;
let dragStartX = 0;
let dragStartY = 0;
let dragOX     = 0;
let dragOY     = 0;
let hoveredId  = null;

// layout config
const NODE_R    = 4;
const NODE_R_HL = 7;

// ── init ───────────────────────────────────────────────────────────────────
function init() {
  const data = GRAPH_DATA;

  document.getElementById('h-solver').textContent  = data.solver;
  document.getElementById('h-topo').textContent    = data.topology;
  document.getElementById('h-nodes').textContent   = data.nodes.length;
  document.getElementById('h-edges').textContent   = data.edges.length;
  document.getElementById('sidebar-count').textContent = data.nodes.length;

  // Build adjacency
  data.nodes.forEach(n => {
    adjMap[n.id]  = new Set();
    nodeById[n.id] = n;
  });
  data.edges.forEach(e => {
    adjMap[e.source].add(e.target);
    adjMap[e.target].add(e.source);
  });

  // Layout — use a grid/Fruchterman-Reingold approximation
  // For large QPU graphs (5000+ nodes) we use a structured grid
  layoutNodes(data.nodes, data.edges);

  buildSidebar(data.nodes);
  resizeCanvas();
  resetView();
  draw();
}

function layoutNodes(rawNodes, rawEdges) {
  const N = rawNodes.length;

  // For Chimera/Pegasus the node ID encodes position — use it directly
  // Chimera: id = i*m*2*k + j*2*k + u*k + ki  (we extract i,j,u)
  // If max ID is plausible for structured layout, use coordinate extraction;
  // otherwise fall back to force-directed on a random seed grid.

  const maxId = Math.max(...rawNodes.map(n => n.id));
  const cols  = Math.ceil(Math.sqrt(N * 1.6));
  const rows  = Math.ceil(N / cols);
  const padX  = 80;
  const padY  = 80;
  const W     = Math.max(1200, cols * 22);
  const H     = Math.max(800,  rows * 22);

  // Sort nodes by ID so similar IDs cluster
  const sorted = rawNodes.slice().sort((a, b) => a.id - b.id);
  sorted.forEach((n, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    nodeById[n.id].x = padX + col * ((W - 2 * padX) / Math.max(cols - 1, 1));
    nodeById[n.id].y = padY + row * ((H - 2 * padY) / Math.max(rows - 1, 1));
  });

  nodes = rawNodes.map(n => nodeById[n.id]);
}

// ── sidebar ────────────────────────────────────────────────────────────────
function buildSidebar(rawNodes) {
  const list   = document.getElementById('node-list');
  const sorted = rawNodes.slice().sort((a, b) => a.id - b.id);
  const frag   = document.createDocumentFragment();

  sorted.forEach(n => {
    const div = document.createElement('div');
    div.className = 'node-item';
    div.id = `ni-${n.id}`;
    div.innerHTML = `
      <span class="node-id">${n.id}</span>
      <span class="node-deg">deg <span class="deg-badge">${n.degree}</span></span>
    `;
    div.onclick = () => selectNode(n.id, true);
    frag.appendChild(div);
  });
  list.appendChild(frag);
}

function updateSidebarSelection(id) {
  document.querySelectorAll('.node-item.selected').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('.node-item.highlighted').forEach(el => el.classList.remove('highlighted'));
  if (id === null) return;
  const el = document.getElementById(`ni-${id}`);
  if (el) { el.classList.add('selected'); el.scrollIntoView({ block: 'nearest' }); }
  (adjMap[id] || new Set()).forEach(nb => {
    const nbEl = document.getElementById(`ni-${nb}`);
    if (nbEl) nbEl.classList.add('highlighted');
  });
}

// ── selection & info panel ─────────────────────────────────────────────────
function selectNode(id, panTo = false) {
  selected = id;
  updateSidebarSelection(id);

  const n = nodeById[id];
  if (!n) return;

  document.getElementById('ip-id').textContent        = id;
  document.getElementById('ip-deg').textContent       = n.degree;
  document.getElementById('ip-idx').textContent       = id;
  const neighbours = [...(adjMap[id] || [])].sort((a, b) => a - b);
  document.getElementById('ip-conn-count').textContent = neighbours.length;

  const chipsEl = document.getElementById('ip-neighbors');
  chipsEl.innerHTML = '';
  neighbours.forEach(nb => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = nb;
    chip.onclick = () => selectNode(nb, true);
    chipsEl.appendChild(chip);
  });

  document.getElementById('info-panel').classList.add('visible');

  if (panTo) panToNode(n);
  draw();
}

function clearSelection() {
  selected = null;
  searchHit.clear();
  updateSidebarSelection(null);
  document.getElementById('info-panel').classList.remove('visible');
  draw();
}

function panToNode(n) {
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  offsetX = cw / 2 - n.x * scale;
  offsetY = ch / 2 - n.y * scale;
  updateStatus();
}

// ── search ─────────────────────────────────────────────────────────────────
function doSearch() {
  const val = document.getElementById('search').value.trim();
  searchHit.clear();

  if (val === '') { draw(); return; }

  // Try exact match first, then prefix match
  const query = parseInt(val, 10);
  if (!isNaN(query) && nodeById[query]) {
    searchHit.add(query);
    selectNode(query, true);
  } else {
    // Partial string match
    Object.keys(nodeById).forEach(id => {
      if (String(id).includes(val)) searchHit.add(parseInt(id));
    });
    if (searchHit.size === 1) {
      selectNode([...searchHit][0], true);
    } else {
      draw();
    }
  }
}

document.getElementById('search').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

// ── drawing ────────────────────────────────────────────────────────────────
function resizeCanvas() {
  const wrap = canvas.parentElement;
  canvas.width  = wrap.clientWidth  * devicePixelRatio;
  canvas.height = wrap.clientHeight * devicePixelRatio;
  canvas.style.width  = wrap.clientWidth  + 'px';
  canvas.style.height = wrap.clientHeight + 'px';
  ctx.scale(devicePixelRatio, devicePixelRatio);
}

window.addEventListener('resize', () => { resizeCanvas(); draw(); });

function nodeColor(id) {
  if (searchHit.has(id))                                return '#ef4444';
  if (id === selected)                                  return '#f59e0b';
  if (selected !== null && adjMap[selected]?.has(id))   return '#10b981';
  return '#3b82f6';
}

function nodeRadius(id) {
  if (id === selected)                                  return NODE_R_HL;
  if (selected !== null && adjMap[selected]?.has(id))   return NODE_R + 2;
  if (searchHit.has(id))                                return NODE_R + 2;
  return NODE_R;
}

function draw() {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#07090f';
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  // Edges — only draw if there aren't too many, or if a node is selected
  const drawAllEdges = nodes.length < 500;
  ctx.lineWidth = 0.6 / scale;

  if (selected !== null) {
    // Draw edges for selected node
    const sel = nodeById[selected];
    ctx.strokeStyle = '#10b98155';
    ctx.lineWidth = 1.2 / scale;
    ctx.beginPath();
    (adjMap[selected] || new Set()).forEach(nb => {
      const t = nodeById[nb];
      if (!t) return;
      ctx.moveTo(sel.x, sel.y);
      ctx.lineTo(t.x, t.y);
    });
    ctx.stroke();

    // Dim edges for non-selected (subtle background)
    if (drawAllEdges) {
      ctx.strokeStyle = '#1e283644';
      ctx.lineWidth = 0.4 / scale;
      GRAPH_DATA.edges.forEach(e => {
        if (e.source === selected || e.target === selected) return;
        const u = nodeById[e.source], v = nodeById[e.target];
        if (!u || !v) return;
        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);
        ctx.stroke();
      });
    }
  } else if (drawAllEdges) {
    ctx.strokeStyle = '#1e2836';
    ctx.beginPath();
    GRAPH_DATA.edges.forEach(e => {
      const u = nodeById[e.source], v = nodeById[e.target];
      if (!u || !v) return;
      ctx.moveTo(u.x, u.y);
      ctx.lineTo(v.x, v.y);
    });
    ctx.stroke();
  }

  // Nodes
  nodes.forEach(n => {
    const r     = nodeRadius(n.id);
    const color = nodeColor(n.id);
    ctx.beginPath();
    ctx.arc(n.x, n.y, r / scale, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Label for highlighted/selected nodes at sufficient zoom
    if ((n.id === selected || (selected !== null && adjMap[selected]?.has(n.id)) || searchHit.has(n.id)) && scale > 1.5) {
      ctx.fillStyle = '#e2e8f0';
      ctx.font = `${10 / scale}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(String(n.id), n.x, n.y - (r + 3) / scale);
    }
  });

  ctx.restore();
}

// ── viewport ───────────────────────────────────────────────────────────────
function resetView() {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (nodes.length === 0) return;

  const xs = nodes.map(n => n.x);
  const ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const gw   = maxX - minX || 1;
  const gh   = maxY - minY || 1;

  scale   = Math.min((W - 80) / gw, (H - 80) / gh) * 0.9;
  offsetX = W / 2 - (minX + gw / 2) * scale;
  offsetY = H / 2 - (minY + gh / 2) * scale;
  updateStatus();
  draw();
}

function updateStatus() {
  document.getElementById('st-zoom').textContent = `${Math.round(scale * 100)}%`;
  document.getElementById('st-pan').textContent  = `${Math.round(offsetX)}, ${Math.round(offsetY)}`;
}

// ── mouse / touch ──────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  dragging  = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  dragOX    = offsetX;
  dragOY    = offsetY;
  canvas.classList.add('grabbing');
});

canvas.addEventListener('mousemove', e => {
  if (dragging) {
    offsetX = dragOX + (e.clientX - dragStartX);
    offsetY = dragOY + (e.clientY - dragStartY);
    updateStatus();
    draw();
    return;
  }

  // Hover detection — only check nearest nodes
  const mx = (e.clientX - offsetX) / scale;
  const my = (e.clientY - offsetY) / scale;
  let closest = null, closestD = Infinity;

  nodes.forEach(n => {
    const d = (n.x - mx) ** 2 + (n.y - my) ** 2;
    if (d < closestD) { closestD = d; closest = n; }
  });

  const threshold = (NODE_R_HL / scale) ** 2 * 4;
  if (closest && closestD < threshold) {
    hoveredId = closest.id;
    document.getElementById('st-hover').textContent = `Hover: node ${closest.id}  deg ${closest.degree}`;
    canvas.style.cursor = 'pointer';
  } else {
    hoveredId = null;
    document.getElementById('st-hover').textContent = '';
    canvas.style.cursor = dragging ? 'grabbing' : 'grab';
  }
});

canvas.addEventListener('mouseup', e => {
  const moved = Math.abs(e.clientX - dragStartX) + Math.abs(e.clientY - dragStartY);
  dragging = false;
  canvas.classList.remove('grabbing');
  if (moved < 4 && hoveredId !== null) {
    selectNode(hoveredId, false);
  }
});

canvas.addEventListener('mouseleave', () => { dragging = false; canvas.classList.remove('grabbing'); });

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const rect  = canvas.getBoundingClientRect();
  const mx    = e.clientX - rect.left;
  const my    = e.clientY - rect.top;
  const delta = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  offsetX = mx - (mx - offsetX) * delta;
  offsetY = my - (my - offsetY) * delta;
  scale  *= delta;
  scale   = Math.max(0.05, Math.min(scale, 40));
  updateStatus();
  draw();
}, { passive: false });

// ── bootstrap ──────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    graph_json: str = ""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = HTML.replace("__GRAPH_JSON__", self.graph_json)
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress access log noise


def find_free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("", preferred))
            return preferred
        except OSError:
            s.bind(("", 0))
            return s.getsockname()[1]


def serve(graph_data: dict, port: int, open_browser: bool):
    Handler.graph_json = json.dumps(graph_data, separators=(",", ":"))
    port = find_free_port(port)
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Explorer running at {url}")
    print(f"  Press Ctrl-C to stop.\n")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactively explore a D-Wave QPU hardware graph in the browser."
    )
    p.add_argument(
        "--solver", type=str, default=None,
        help="D-Wave solver name, e.g. 'Advantage_system6.4'"
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use a synthetic Chimera graph (no QPU token required)"
    )
    p.add_argument("--port",       type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mock and args.solver is None:
        print("Error: provide --solver <name> or --mock for offline demo.")
        print("       Example: python explore_qpu_graph.py --solver Advantage_system6.4")
        sys.exit(1)

    print("\nQPU Graph Explorer")
    print("──────────────────")

    if args.mock:
        graph_data = mock_qpu_graph()
    else:
        graph_data = fetch_qpu_graph(args.solver)

    serve(graph_data, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
