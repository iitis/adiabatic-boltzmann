"""
RBM Graph Visualizer
====================
Generates a self-contained interactive HTML visualization of an RBM.

Usage:
    python visualize_rbm.py                          # default: FullyConnected N=8, M=8
    python visualize_rbm.py --rbm dwave              # DWaveTopologyRBM
    python visualize_rbm.py --n-visible 12 --n-hidden 12 --output rbm.html
    python visualize_rbm.py --weights path/to/weights.npz

Encoding:
    Nodes   — visible (bottom row) and hidden (top row)
    Node color    — bias value (a for visible, b for hidden): blue=negative, red=positive
    Node size     — |bias| magnitude
    Edge presence — W[i,j] != 0 (sparse for DWave, all for FullyConnected)
    Edge color    — weight sign: blue=negative, red=positive
    Edge width    — |W[i,j]| magnitude
    Hover         — shows exact parameter value
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_rbm(args):
    """Build or load an RBM and return it."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from model import FullyConnectedRBM, DWaveTopologyRBM

    nv = args.n_visible
    nh = args.n_hidden

    if args.weights:
        data = np.load(args.weights)
        if args.rbm == "dwave":
            rbm = DWaveTopologyRBM(nv, nh)
        else:
            rbm = FullyConnectedRBM(nv, nh)
        rbm.a = data["a"]
        rbm.b = data["b"]
        rbm.W = data["W"]
        rbm.W *= rbm.get_connectivity_mask()
        return rbm

    np.random.seed(args.seed)
    if args.rbm == "dwave":
        return DWaveTopologyRBM(nv, nh)
    return FullyConnectedRBM(nv, nh)


def rbm_to_graph_data(rbm) -> dict:
    """Convert RBM parameters to node/edge lists for the visualizer."""
    nv, nh = rbm.n_visible, rbm.n_hidden
    mask = rbm.get_connectivity_mask()

    nodes = []
    # Visible units
    for i in range(nv):
        nodes.append(
            {
                "id": f"v{i}",
                "label": f"v{i}",
                "group": "visible",
                "bias": float(rbm.a[i]),
                "index": i,
            }
        )
    # Hidden units
    for j in range(nh):
        nodes.append(
            {
                "id": f"h{j}",
                "label": f"h{j}",
                "group": "hidden",
                "bias": float(rbm.b[j]),
                "index": j,
            }
        )

    edges = []
    for i in range(nv):
        for j in range(nh):
            if mask[i, j] != 0:
                w = float(rbm.W[i, j])
                edges.append(
                    {
                        "source": f"v{i}",
                        "target": f"h{j}",
                        "weight": w,
                    }
                )

    # Summary stats for legend scaling
    all_weights = [e["weight"] for e in edges]
    all_biases = [n["bias"] for n in nodes]

    return {
        "nodes": nodes,
        "edges": edges,
        "n_visible": nv,
        "n_hidden": nh,
        "n_edges": len(edges),
        "max_w": float(max(abs(w) for w in all_weights)) if all_weights else 1.0,
        "max_bias": float(max(abs(b) for b in all_biases)) if all_biases else 1.0,
        "sparsity": float(1 - mask.sum() / mask.size),
        "rbm_type": type(rbm).__name__,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RBM Graph — {{RBM_TYPE}}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap');
  :root {
    --bg:      #07080d;
    --panel:   #0d0f18;
    --border:  #1a1f2e;
    --accent:  #00d4ff;
    --red:     #ff4757;
    --blue:    #3d8bff;
    --gold:    #ffd32a;
    --text:    #c8d0e0;
    --dim:     #4a5568;
    --mono:    'Space Mono', monospace;
    --display: 'Bebas Neue', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
    overflow: hidden;
  }

  /* ── header ── */
  header {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    flex-wrap: wrap;
  }
  header h1 {
    font-family: var(--display);
    font-size: 26px;
    letter-spacing: 3px;
    color: var(--accent);
    text-shadow: 0 0 18px rgba(0,212,255,.35);
  }
  .meta-pill {
    background: rgba(0,212,255,.06);
    border: 1px solid rgba(0,212,255,.2);
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 10px;
    color: var(--accent);
    letter-spacing: 1px;
  }
  .meta-pill.red { background:rgba(255,71,87,.06); border-color:rgba(255,71,87,.25); color:var(--red) }

  /* ── layout ── */
  .workspace {
    display: grid;
    grid-template-columns: 220px 1fr;
    overflow: hidden;
  }

  /* ── sidebar ── */
  aside {
    border-right: 1px solid var(--border);
    background: var(--panel);
    padding: 20px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .sidebar-section { display: flex; flex-direction: column; gap: 8px }
  .sidebar-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--dim);
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .stat-row { display:flex; justify-content:space-between; align-items:center }
  .stat-key { color: var(--dim); font-size: 10px }
  .stat-val { color: var(--text); font-weight: 700 }
  .stat-val.accent { color: var(--accent) }

  /* legend swatches */
  .legend-item { display:flex; align-items:center; gap:8px; font-size:10px; color:var(--dim) }
  .swatch { width:28px; height:4px; border-radius:2px; flex-shrink:0 }
  .swatch.pos { background: linear-gradient(90deg, #ff475750, var(--red)) }
  .swatch.neg { background: linear-gradient(90deg, #3d8bff50, var(--blue)) }
  .swatch.zero { background: var(--border) }
  .node-swatch { width:12px; height:12px; border-radius:50%; flex-shrink:0 }
  .node-swatch.vis { background: var(--gold); box-shadow: 0 0 6px var(--gold) }
  .node-swatch.hid { background: var(--accent); box-shadow: 0 0 6px var(--accent) }

  /* controls */
  .ctrl-row { display:flex; flex-direction:column; gap:4px }
  .ctrl-row label { font-size:9px; letter-spacing:1px; color:var(--dim) }
  input[type=range] { width:100%; accent-color: var(--accent) }
  .toggle-row { display:flex; align-items:center; gap:8px; font-size:10px; color:var(--dim) }
  input[type=checkbox] { accent-color: var(--accent) }

  /* ── canvas area ── */
  .canvas-wrap {
    position: relative;
    overflow: hidden;
  }
  canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: crosshair;
  }

  /* ── tooltip ── */
  #tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(13,15,24,.96);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 11px;
    color: var(--text);
    line-height: 1.7;
    white-space: nowrap;
    display: none;
    z-index: 10;
    box-shadow: 0 4px 24px rgba(0,212,255,.12);
  }
  #tooltip .tt-title { color:var(--accent); font-weight:700; margin-bottom:2px }
  #tooltip .tt-pos { color:var(--red) }
  #tooltip .tt-neg { color:var(--blue) }
</style>
</head>
<body>

<header>
  <h1>RBM GRAPH</h1>
  <span class="meta-pill">{{RBM_TYPE}}</span>
  <span class="meta-pill">N_VIS = {{N_VIS}}</span>
  <span class="meta-pill">N_HID = {{N_HID}}</span>
  <span class="meta-pill">EDGES = {{N_EDGES}}</span>
  <span class="meta-pill red">SPARSITY = {{SPARSITY}}%</span>
</header>

<div class="workspace">
  <aside>
    <div class="sidebar-section">
      <div class="sidebar-label">Parameters</div>
      <div class="stat-row"><span class="stat-key">visible units</span><span class="stat-val accent">{{N_VIS}}</span></div>
      <div class="stat-row"><span class="stat-key">hidden units</span><span class="stat-val accent">{{N_HID}}</span></div>
      <div class="stat-row"><span class="stat-key">connections</span><span class="stat-val">{{N_EDGES}}</span></div>
      <div class="stat-row"><span class="stat-key">max |W|</span><span class="stat-val" id="s-maxw">—</span></div>
      <div class="stat-row"><span class="stat-key">max |bias|</span><span class="stat-val" id="s-maxb">—</span></div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Legend</div>
      <div class="legend-item"><div class="node-swatch vis"></div>visible unit (σ)</div>
      <div class="legend-item"><div class="node-swatch hid"></div>hidden unit (h)</div>
      <div class="legend-item"><div class="swatch pos"></div>positive weight</div>
      <div class="legend-item"><div class="swatch neg"></div>negative weight</div>
      <div class="legend-item"><div class="swatch zero"></div>zero / masked</div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Display</div>
      <div class="ctrl-row">
        <label>EDGE OPACITY</label>
        <input type="range" id="ctrl-edge-opacity" min="5" max="100" value="60">
      </div>
      <div class="ctrl-row">
        <label>EDGE WIDTH SCALE</label>
        <input type="range" id="ctrl-edge-width" min="1" max="20" value="8">
      </div>
      <div class="ctrl-row">
        <label>NODE SIZE SCALE</label>
        <input type="range" id="ctrl-node-size" min="3" max="20" value="9">
      </div>
      <div class="toggle-row">
        <input type="checkbox" id="ctrl-show-zero" checked>
        <label for="ctrl-show-zero">Show zero edges</label>
      </div>
      <div class="toggle-row">
        <input type="checkbox" id="ctrl-labels" checked>
        <label for="ctrl-labels">Node labels</label>
      </div>
      <div class="toggle-row">
        <input type="checkbox" id="ctrl-animate">
        <label for="ctrl-animate">Pulse animation</label>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Hover Info</div>
      <div style="font-size:10px;color:var(--dim);line-height:1.6">
        Hover over any node or edge to inspect its parameter value.
      </div>
    </div>
  </aside>

  <div class="canvas-wrap" id="canvas-wrap">
    <canvas id="rbm-canvas"></canvas>
    <div id="tooltip"></div>
  </div>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────────────
const DATA = __GRAPH_DATA__;

// ── controls ──────────────────────────────────────────────────────────────
const canvas  = document.getElementById('rbm-canvas');
const ctx     = canvas.getContext('2d');
const wrap    = document.getElementById('canvas-wrap');
const tooltip = document.getElementById('tooltip');

let edgeOpacity  = 0.60;
let edgeWScale   = 8;
let nodeSzScale  = 9;
let showZero     = true;
let showLabels   = true;
let doAnimate    = false;
let animFrame    = 0;
let hoveredNode  = null;
let hoveredEdge  = null;

document.getElementById('ctrl-edge-opacity').addEventListener('input', e => { edgeOpacity = e.target.value/100; draw() });
document.getElementById('ctrl-edge-width').addEventListener('input',   e => { edgeWScale  = +e.target.value;   draw() });
document.getElementById('ctrl-node-size').addEventListener('input',    e => { nodeSzScale = +e.target.value;   draw() });
document.getElementById('ctrl-show-zero').addEventListener('change',   e => { showZero    = e.target.checked;  draw() });
document.getElementById('ctrl-labels').addEventListener('change',      e => { showLabels  = e.target.checked;  draw() });
document.getElementById('ctrl-animate').addEventListener('change',     e => {
  doAnimate = e.target.checked;
  if (doAnimate) requestAnimationFrame(loop);
});

document.getElementById('s-maxw').textContent = DATA.max_w.toFixed(5);
document.getElementById('s-maxb').textContent = DATA.max_bias.toFixed(5);

// ── layout ────────────────────────────────────────────────────────────────
function computeLayout(W, H) {
  const nv = DATA.n_visible, nh = DATA.n_hidden;
  const positions = {};

  const padX = 80;
  const visY = H * 0.75;
  const hidY = H * 0.25;

  // Visible units — bottom row
  for (let i = 0; i < nv; i++) {
    const x = padX + (i / Math.max(nv - 1, 1)) * (W - 2 * padX);
    positions[`v${i}`] = { x, y: visY };
  }
  // Hidden units — top row
  for (let j = 0; j < nh; j++) {
    const x = padX + (j / Math.max(nh - 1, 1)) * (W - 2 * padX);
    positions[`h${j}`] = { x, y: hidY };
  }
  return positions;
}

// ── color helpers ─────────────────────────────────────────────────────────
function weightColor(w, alpha) {
  if (Math.abs(w) < 1e-10) return `rgba(26,31,46,${alpha})`;
  const t = Math.min(1, Math.abs(w) / DATA.max_w);
  if (w > 0) {
    const r = Math.round(61  + t * (255 - 61));
    const g = Math.round(139 + t * (71  - 139));
    const b = Math.round(255 + t * (87  - 255));
    return `rgba(${r},${g},${b},${alpha})`;
  } else {
    const r = Math.round(61  + t * (255 - 61));
    const g = Math.round(139 + t * (71  - 139));
    const b = Math.round(255 + t * (87  - 255));
    // blue for negative
    return `rgba(${Math.round(61 + t*(-61))},${Math.round(100+t*11)},${Math.round(200+t*55)},${alpha})`;
  }
}

function nodeColor(bias, group) {
  const t = DATA.max_bias > 0 ? Math.min(1, Math.abs(bias) / DATA.max_bias) : 0;
  const base = group === 'visible' ? [255, 211, 42] : [0, 212, 255];  // gold vs cyan

  // Darken toward background for zero bias, intensify for large bias
  const brightness = 0.3 + 0.7 * t;
  return `rgba(${Math.round(base[0]*brightness)},${Math.round(base[1]*brightness)},${Math.round(base[2]*brightness)},0.9)`;
}

function nodeGlow(bias, group) {
  const t = DATA.max_bias > 0 ? Math.min(1, Math.abs(bias) / DATA.max_bias) : 0;
  const col = group === 'visible' ? `255,211,42` : `0,212,255`;
  return `rgba(${col},${0.15 + 0.5 * t})`;
}

// ── draw ──────────────────────────────────────────────────────────────────
function draw() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // subtle grid background
  ctx.strokeStyle = 'rgba(26,31,46,0.5)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke() }
  for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke() }

  const pos = computeLayout(W, H);
  const pulse = doAnimate ? Math.sin(animFrame * 0.04) * 0.5 + 0.5 : 1;

  // Build lookup for hover highlighting
  const hoveredNodeEdges = new Set();
  if (hoveredNode) {
    DATA.edges.forEach((e, idx) => {
      if (e.source === hoveredNode || e.target === hoveredNode) hoveredNodeEdges.add(idx);
    });
  }

  // ── draw edges ──
  DATA.edges.forEach((edge, idx) => {
    const isZero = Math.abs(edge.weight) < 1e-10;
    if (isZero && !showZero) return;

    const src = pos[edge.source], tgt = pos[edge.target];
    const isHovered = hoveredEdge === idx;
    const isNodeHov = hoveredNodeEdges.has(idx);
    const dimmed = (hoveredNode || hoveredEdge !== null) && !isHovered && !isNodeHov;

    const wAbs = Math.abs(edge.weight);
    const wNorm = DATA.max_w > 0 ? wAbs / DATA.max_w : 0;
    const lineW = isZero ? 0.4 : Math.max(0.3, wNorm * edgeWScale);
    let alpha = isZero ? 0.08 : edgeOpacity * (doAnimate ? (0.5 + 0.5 * pulse) : 1);
    if (dimmed) alpha *= 0.08;
    if (isHovered || isNodeHov) alpha = Math.min(1, alpha * 2.5);

    // color
    const w = edge.weight;
    let color;
    if (isZero) {
      color = `rgba(26,31,46,${alpha})`;
    } else if (w > 0) {
      const t = wNorm;
      color = `rgba(${Math.round(100+t*155)},${Math.round(80+t*(-9))},${Math.round(100+t*(87-100))},${alpha})`;
      // positive → red-ish
      color = `rgba(${Math.round(120+t*135)},${Math.round(50+t*21)},${Math.round(80+t*7)},${alpha})`;
    } else {
      // negative → blue
      color = `rgba(${Math.round(30+wNorm*90)},${Math.round(100+wNorm*110)},${Math.round(180+wNorm*75)},${alpha})`;
    }

    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = isHovered ? lineW * 2.5 : lineW;
    ctx.stroke();
  });

  // ── draw nodes ──
  DATA.nodes.forEach(node => {
    const p = pos[node.id];
    const isHov = hoveredNode === node.id;
    const dimmed = hoveredNode && !isHov && !hoveredNodeEdges.size;

    const biasNorm = DATA.max_bias > 0 ? Math.abs(node.bias) / DATA.max_bias : 0;
    const r = Math.max(4, biasNorm * nodeSzScale + (nodeSzScale * 0.4));
    const finalR = isHov ? r * 1.4 : r;

    // glow
    if (!dimmed) {
      const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, finalR * 3);
      gradient.addColorStop(0, nodeGlow(node.bias, node.group));
      gradient.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(p.x, p.y, finalR * 3, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
    }

    // node circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, finalR, 0, Math.PI * 2);
    ctx.fillStyle = nodeColor(node.bias, node.group);
    if (dimmed) ctx.fillStyle = 'rgba(26,31,46,0.4)';
    ctx.fill();

    // ring
    ctx.strokeStyle = node.group === 'visible'
      ? `rgba(255,211,42,${isHov ? 0.9 : 0.5})`
      : `rgba(0,212,255,${isHov ? 0.9 : 0.5})`;
    ctx.lineWidth = isHov ? 2 : 1;
    ctx.stroke();

    // label
    if (showLabels) {
      ctx.font = `${isHov ? 'bold ' : ''}9px 'Space Mono', monospace`;
      ctx.fillStyle = isHov ? '#fff' : 'rgba(200,208,224,0.7)';
      ctx.textAlign = 'center';
      ctx.textBaseline = node.group === 'visible' ? 'top' : 'bottom';
      const labelY = node.group === 'visible' ? p.y + finalR + 4 : p.y - finalR - 4;
      ctx.fillText(node.label, p.x, labelY);
    }
  });

  // row labels
  ctx.font = "10px 'Space Mono'";
  ctx.fillStyle = 'rgba(255,211,42,0.5)';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText('VISIBLE  σ', 8, H * 0.75);
  ctx.fillStyle = 'rgba(0,212,255,0.5)';
  ctx.fillText('HIDDEN   h', 8, H * 0.25);
}

// ── animation loop ────────────────────────────────────────────────────────
function loop() {
  animFrame++;
  draw();
  if (doAnimate) requestAnimationFrame(loop);
}

// ── resize ────────────────────────────────────────────────────────────────
function resize() {
  const rect = wrap.getBoundingClientRect();
  canvas.width  = rect.width  * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  canvas.style.width  = rect.width  + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.scale(devicePixelRatio, devicePixelRatio);
  draw();
}
window.addEventListener('resize', resize);

// ── hover / tooltip ───────────────────────────────────────────────────────
function getCanvasXY(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener('mousemove', e => {
  const { x, y } = getCanvasXY(e);
  const W = canvas.getBoundingClientRect().width;
  const H = canvas.getBoundingClientRect().height;
  const pos = computeLayout(W, H);

  // Check nodes first
  let found = null;
  for (const node of DATA.nodes) {
    const p = pos[node.id];
    const biasNorm = DATA.max_bias > 0 ? Math.abs(node.bias) / DATA.max_bias : 0;
    const r = Math.max(4, biasNorm * nodeSzScale + (nodeSzScale * 0.4)) * 1.5;
    if (Math.hypot(x - p.x, y - p.y) < r) { found = node; break; }
  }

  if (found) {
    hoveredNode = found.id;
    hoveredEdge = null;
    const sign = found.bias >= 0 ? '+' : '';
    const cls  = found.bias >= 0 ? 'tt-pos' : 'tt-neg';
    tooltip.innerHTML = `
      <div class="tt-title">${found.id} (${found.group})</div>
      bias <span class="${cls}">${sign}${found.bias.toFixed(6)}</span><br>
      index = ${found.index}
    `;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 10) + 'px';
    draw(); return;
  }

  // Check edges
  let foundEdge = null;
  let minDist = 6;
  DATA.edges.forEach((edge, idx) => {
    if (Math.abs(edge.weight) < 1e-10 && !showZero) return;
    const src = pos[edge.source], tgt = pos[edge.target];
    // Point-to-segment distance
    const dx = tgt.x - src.x, dy = tgt.y - src.y;
    const len2 = dx*dx + dy*dy;
    if (len2 === 0) return;
    const t = Math.max(0, Math.min(1, ((x-src.x)*dx + (y-src.y)*dy) / len2));
    const px = src.x + t*dx - x, py = src.y + t*dy - y;
    const dist = Math.hypot(px, py);
    if (dist < minDist) { minDist = dist; foundEdge = { edge, idx }; }
  });

  if (foundEdge) {
    hoveredEdge = foundEdge.idx;
    hoveredNode = null;
    const w = foundEdge.edge.weight;
    const sign = w >= 0 ? '+' : '';
    const cls  = w >= 0 ? 'tt-pos' : 'tt-neg';
    tooltip.innerHTML = `
      <div class="tt-title">${foundEdge.edge.source} → ${foundEdge.edge.target}</div>
      W[i,j] <span class="${cls}">${sign}${w.toFixed(6)}</span><br>
      |W| = ${Math.abs(w).toFixed(6)}
    `;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 10) + 'px';
    draw(); return;
  }

  // Nothing hovered
  if (hoveredNode !== null || hoveredEdge !== null) {
    hoveredNode = null; hoveredEdge = null;
    tooltip.style.display = 'none';
    draw();
  } else {
    tooltip.style.display = 'none';
  }
});

canvas.addEventListener('mouseleave', () => {
  hoveredNode = null; hoveredEdge = null;
  tooltip.style.display = 'none';
  draw();
});

// ── init ──────────────────────────────────────────────────────────────────
resize();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_html(graph_data: dict) -> str:
    nv = graph_data["n_visible"]
    nh = graph_data["n_hidden"]
    sp = f"{graph_data['sparsity'] * 100:.1f}"
    html = HTML
    html = html.replace("{{RBM_TYPE}}", graph_data["rbm_type"])
    html = html.replace("{{N_VIS}}", str(nv))
    html = html.replace("{{N_HID}}", str(nh))
    html = html.replace("{{N_EDGES}}", str(graph_data["n_edges"]))
    html = html.replace("{{SPARSITY}}", sp)
    html = html.replace("__GRAPH_DATA__", json.dumps(graph_data, separators=(",", ":")))
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Visualize an RBM as an interactive graph")
    p.add_argument("--rbm", choices=["full", "dwave"], default="full")
    p.add_argument("--n-visible", type=int, default=8)
    p.add_argument("--n-hidden", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--weights", type=str, default=None, help="Path to .npz file with keys a, b, W"
    )
    p.add_argument("--output", type=str, default="rbm_graph.html")
    return p.parse_args()


def main():
    args = parse_args()
    rbm = load_rbm(args)
    data = rbm_to_graph_data(rbm)
    html = generate_html(data)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Saved → {out}  ({out.stat().st_size // 1024} KB)")
    print(f"Open:   open {out}")
    print(
        f"RBM:    {data['rbm_type']}  "
        f"N_vis={data['n_visible']}  N_hid={data['n_hidden']}  "
        f"edges={data['n_edges']}  sparsity={data['sparsity']:.1%}"
    )


if __name__ == "__main__":
    main()
