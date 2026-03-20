"""
Generate a self-contained HTML analysis report from benchmark results.

Usage:
    python generate_report.py                          # reads results/, writes report.html
    python generate_report.py --results path/to/results --output report.html
"""

import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(root: Path) -> list[dict]:
    records = []
    for file in root.rglob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
        except Exception:
            print(f"  [warn] skipping: {file}")
            continue

        config = data["config"]
        history = data["history"]

        records.append(
            {
                "file": str(file),
                "size": int(config["size"]),
                "h": float(config["h"]),
                "rbm": config["rbm"],
                "n_hidden": int(config["n_hidden"] or config["size"]),
                "sampler": config["sampler"],
                "sampling_method": config["sampling_method"],
                "lr": float(config["learning_rate"]),
                "reg": float(config["regularization"]),
                "n_samples": int(config["n_samples"]),
                "seed": int(config["seed"]),
                "final_energy": float(data["final_energy"]),
                "exact_energy": float(data["exact_energy"]),
                "abs_error": float(abs(data["final_energy"] - data["exact_energy"])),
                "rel_error": float(
                    abs(data["final_energy"] - data["exact_energy"])
                    / abs(data["exact_energy"])
                    * 100
                ),
                "energy_curve": [float(x) for x in history.get("energy", [])],
                "error_curve": [float(x) for x in history.get("error", [])],
                "energy_error_curve": [
                    float(x) for x in history.get("energy_error", [])
                ],
                "grad_norm_curve": [float(x) for x in history.get("grad_norm", [])],
                "cond_curve": [float(x) for x in history.get("s_condition_number", [])],
                "weight_norm_curve": [float(x) for x in history.get("weight_norm", [])],
            }
        )

    return sorted(records, key=lambda r: (r["size"], r["h"], r["rel_error"]))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VMC Benchmark Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');
  :root {
    --bg:#0a0c10; --bg2:#0f1218; --bg3:#161b24;
    --border:#1e2633; --border2:#2a3444;
    --accent:#3b82f6; --accent2:#60a5fa;
    --green:#10b981; --yellow:#f59e0b; --red:#ef4444;
    --text:#e2e8f0; --text2:#94a3b8; --text3:#64748b;
    --mono:'JetBrains Mono',monospace; --display:'Syne',sans-serif;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.6;min-height:100vh}

  header{border-bottom:1px solid var(--border);padding:16px 32px;display:flex;align-items:center;gap:16px;background:var(--bg2);position:sticky;top:0;z-index:200;flex-wrap:wrap}
  header h1{font-family:var(--display);font-size:19px;font-weight:800;letter-spacing:-.5px;white-space:nowrap}
  header h1 span{color:var(--accent)}
  .run-count{font-size:11px;color:var(--text3);white-space:nowrap}

  .filter-bar{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;margin-left:auto}
  .fg{display:flex;flex-direction:column;gap:2px}
  .fg label{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--text3)}
  .fg select{background:var(--bg3);border:1px solid var(--border2);color:var(--text);font-family:var(--mono);font-size:11px;padding:4px 8px;border-radius:4px;cursor:pointer;min-width:100px}
  .fg select:focus{outline:1px solid var(--accent)}
  .filter-badge{display:none;background:rgba(59,130,246,.15);color:var(--accent2);font-size:10px;padding:3px 8px;border-radius:3px;white-space:nowrap}
  #btn-reset{display:none;background:none;border:1px solid var(--border2);color:var(--text3);font-family:var(--mono);font-size:10px;padding:4px 10px;border-radius:4px;cursor:pointer}
  #btn-reset:hover{border-color:var(--accent);color:var(--accent2)}

  .layout{display:grid;grid-template-columns:242px 1fr;min-height:calc(100vh - 65px)}
  aside{border-right:1px solid var(--border);padding:14px 0;background:var(--bg2);overflow-y:auto;position:sticky;top:65px;height:calc(100vh - 65px)}
  .sidebar-label{font-size:9px;font-weight:700;letter-spacing:2px;color:var(--text3);text-transform:uppercase;padding:12px 14px 5px;display:block}
  .nh-btn{display:block;width:100%;text-align:left;padding:6px 14px;background:none;border:none;color:var(--text2);cursor:pointer;font-family:var(--mono);font-size:11px;border-left:2px solid transparent;transition:all .15s}
  .nh-btn:hover{background:var(--bg3);color:var(--text)}
  .nh-btn.active{color:var(--accent2);border-left-color:var(--accent);background:rgba(59,130,246,.06)}
  .nh-btn .badge{float:right;font-size:9px;color:var(--text3);background:var(--bg3);padding:1px 5px;border-radius:3px}
  .nh-btn .err-badge{float:right;font-size:9px;margin-right:5px;padding:1px 5px;border-radius:3px}
  .nh-btn.dimmed{opacity:.3}

  main{padding:26px 34px;overflow-y:auto;background:var(--bg)}
  .panel{display:none}.panel.active{display:block}

  .section-title{font-family:var(--display);font-size:16px;font-weight:800;margin-bottom:4px;letter-spacing:-.3px}
  .section-sub{color:var(--text3);font-size:11px;margin-bottom:18px}
  .filter-notice{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);border-radius:4px;padding:6px 12px;font-size:11px;color:var(--accent2);margin-bottom:14px;display:none}
  .filter-notice.on{display:block}

  .stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:9px;margin-bottom:20px}
  .stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:5px;padding:11px 13px}
  .stat-label{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px}
  .stat-value{font-size:18px;font-weight:700;font-family:var(--display)}
  .stat-sub{font-size:9px;color:var(--text3);margin-top:2px}
  .good{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}

  .charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px}
  .charts-grid.wide{grid-template-columns:1fr}
  .chart-card{background:var(--bg2);border:1px solid var(--border);border-radius:5px;padding:13px}
  .chart-card.span2{grid-column:span 2}
  .chart-title{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3);margin-bottom:9px}
  .chart-wrap{position:relative;height:200px}
  .chart-wrap.tall{height:270px}

  .run-table-wrap{background:var(--bg2);border:1px solid var(--border);border-radius:5px;overflow:hidden;margin-bottom:18px}
  .run-table-header{padding:9px 13px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
  .run-table-header span{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3)}
  table{width:100%;border-collapse:collapse}
  th{background:var(--bg3);padding:6px 9px;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3);text-align:left;border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
  th:hover{color:var(--text)}
  td{padding:5px 9px;border-bottom:1px solid var(--border);font-size:11px;color:var(--text2);white-space:nowrap}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:var(--bg3)}
  tr.selected td{background:rgba(59,130,246,.08)}
  .pill{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600}
  .pill-green{background:rgba(16,185,129,.15);color:var(--green)}
  .pill-yellow{background:rgba(245,158,11,.15);color:var(--yellow)}
  .pill-red{background:rgba(239,68,68,.15);color:var(--red)}
  .pill-blue{background:rgba(59,130,246,.15);color:var(--accent2)}

  .heatmap-cell{padding:7px 5px;text-align:center;border-radius:4px;font-size:10px;cursor:pointer;transition:transform .1s}
  .heatmap-cell:hover{transform:scale(1.07);z-index:2;position:relative}
  .heatmap-cell .cv{font-size:12px;font-weight:700}
  .heatmap-cell .cs{font-size:9px;color:rgba(255,255,255,.55)}

  .diag-selector{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
  .diag-selector label{font-size:11px;color:var(--text3)}
  .diag-selector select{background:var(--bg2);border:1px solid var(--border2);color:var(--text);font-family:var(--mono);font-size:11px;padding:4px 8px;border-radius:4px;min-width:230px}
  .divider{border:none;border-top:1px solid var(--border);margin:22px 0}
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:var(--bg2)}
  ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
</head>
<body>
<header>
  <h1>VMC <span>Benchmark</span></h1>
  <span class="run-count" id="run-count"></span>
  <div class="filter-bar">
    <div class="fg"><label>Sampler</label>
      <select id="f-sampler" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Method</label>
      <select id="f-method" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>RBM</label>
      <select id="f-rbm" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>n_hidden</label>
      <select id="f-nhidden" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>n_samples</label>
      <select id="f-nsamples" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Reg</label>
      <select id="f-reg" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Learn. Rate</label>
  <select id="f-lr" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Seed</label>
      <select id="f-seed" onchange="applyFilters()"><option value="">All</option></select></div>
    <span class="filter-badge" id="filter-badge"></span>
    <button id="btn-reset" onclick="resetFilters()">✕ Reset</button>
  </div>
</header>

<div class="layout">
  <aside>
    <span class="sidebar-label">Overview</span>
    <button class="nh-btn active" onclick="showPanel('overview')" id="btn-overview">All runs</button>
    <span class="sidebar-label">By (N, h)</span>
    <div id="nh-buttons"></div>
  </aside>
  <main>
    <div class="panel active" id="panel-overview">
      <div class="section-title">Overview</div>
      <div class="section-sub" id="ov-sub"></div>
      <div class="filter-notice" id="ov-notice"></div>
      <div class="stats-row" id="ov-stats"></div>
      <div class="section-title" style="font-size:12px;margin-bottom:10px">Error Heatmap — best rel. error per (N,h)</div>
      <div id="heatmap-container" style="margin-bottom:26px;overflow-x:auto"></div>
      <div class="section-title" style="font-size:12px;margin-bottom:10px">Sampler × Method Comparison</div>
      <div class="charts-grid">
        <div class="chart-card"><div class="chart-title">Best &amp; Mean Error by Sampler/Method</div><div class="chart-wrap"><canvas id="ov-samp-bar"></canvas></div></div>
        <div class="chart-card"><div class="chart-title">Error vs n_samples — by Sampler/Method</div><div class="chart-wrap"><canvas id="ov-err-ns"></canvas></div></div>
        <div class="chart-card"><div class="chart-title">Error vs Regularization — by Sampler/Method</div><div class="chart-wrap"><canvas id="ov-err-reg"></canvas></div></div>
        <div class="chart-card"><div class="chart-title">Error vs Learning Rate — by Sampler/Method</div><div class="chart-wrap"><canvas id="ov-err-lr"></canvas></div></div>
      </div>
    </div>
    <div id="nh-panels"></div>
  </main>
</div>

<script>
const ALL_RUNS = __RUNS_JSON__;

Chart.defaults.color='#64748b'; Chart.defaults.borderColor='#1e2633';
Chart.defaults.font.family="'JetBrains Mono',monospace"; Chart.defaults.font.size=10;
const PAL=['#3b82f6','#10b981','#f59e0b','#ef4444','#a855f7','#06b6d4','#f97316','#ec4899','#84cc16','#14b8a6'];

const $=id=>document.getElementById(id);
const fmtE=v=>typeof v==='number'?v.toExponential(2):'—';
const fmtF=(v,d=4)=>typeof v==='number'?v.toFixed(d):'—';
const fmtP=v=>typeof v==='number'?v.toFixed(3)+'%':'—';
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;
const errCls=p=>p<1?'good':p<5?'warn':'bad';
const pillCls=p=>p<1?'pill-green':p<5?'pill-yellow':'pill-red';

function heatColor(v,mn,mx){
  const t=Math.max(0,Math.min(1,(v-mn)/(mx-mn+1e-9)));
  return `rgb(${Math.round(16+t*223)},${Math.round(185-t*117)},${Math.round(129-t*61)})`;
}
function mkChart(id,type,data,opts={}){
  const ex=Chart.getChart(id);if(ex)ex.destroy();
  const el=$(id);if(!el)return;
  return new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{boxWidth:9,padding:9}}},...opts}});
}
function lineds(label,curve,color,dash=[]){
  return{label,data:curve.map((v,i)=>({x:i,y:v})),borderColor:color,backgroundColor:color+'22',borderWidth:1.5,borderDash:dash,pointRadius:0,tension:.3};
}

// ── keys ──────────────────────────────────────────────────────────────────
const nhKey=r=>`N${r.size}_h${r.h}`;
const nhLabel=r=>`N=${r.size}, h=${r.h}`;
const smKey=r=>`${r.sampler}/${r.sampling_method}`;

const nhKeys=[...new Set(ALL_RUNS.map(nhKey))].sort((a,b)=>{
  const pa=a.match(/N(\d+)_h([\d.]+)/),pb=b.match(/N(\d+)_h([\d.]+)/);
  return(+pa[1]-+pb[1])||(+pa[2]-+pb[2]);
});
const nhLabelMap={};
ALL_RUNS.forEach(r=>{nhLabelMap[nhKey(r)]=nhLabel(r)});

// ── filter state ──────────────────────────────────────────────────────────
let ACTIVE=ALL_RUNS.slice();

function applyFilters(){
  const fs={
    sampler:$('f-sampler').value,
    sampling_method:$('f-method').value,
    rbm:$('f-rbm').value,
    n_hidden:$('f-nhidden').value,
    n_samples:$('f-nsamples').value,
    reg:$('f-reg').value,
    lr:$('f-lr').value,  
    seed:$('f-seed').value,
  };
  ACTIVE=ALL_RUNS.filter(r=>
    (!fs.sampler||r.sampler===fs.sampler)&&
    (!fs.sampling_method||r.sampling_method===fs.sampling_method)&&
    (!fs.rbm||r.rbm===fs.rbm)&&
    (!fs.n_hidden||String(r.n_hidden)===fs.n_hidden)&&
    (!fs.n_samples||String(r.n_samples)===fs.n_samples)&&
    (!fs.reg||String(r.reg)===fs.reg)&&
    (!fs.lr||String(r.lr)=== fs.lr)&& // ← add
    (!fs.seed||String(r.seed)===fs.seed)
  );
  const nActive=Object.values(fs).filter(Boolean).length;
  const badge=$('filter-badge'),reset=$('btn-reset');
  if(nActive){
    badge.textContent=`${nActive} filter${nActive>1?'s':''} · ${ACTIVE.length} runs`;
    badge.style.display='inline-block'; reset.style.display='inline-block';
  } else {
    badge.style.display='none'; reset.style.display='none';
  }
  document.querySelectorAll('.nh-btn[data-k]').forEach(btn=>{
    btn.classList.toggle('dimmed',!ACTIVE.some(r=>nhKey(r)===btn.dataset.k));
  });
  rebuildOverview();
  nhKeys.forEach(k=>rebuildNHCharts(k));
}

function resetFilters(){
  ['f-sampler','f-method','f-rbm','f-nhidden','f-nsamples','f-reg','f-lr','f-seed'].forEach(id=>$(id).value='');
  applyFilters();
}

// populate dropdowns
function popSel(id,vals){
  const el=$(id);
  [...new Set(vals)].sort().forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)});
}
popSel('f-sampler', ALL_RUNS.map(r=>r.sampler));
popSel('f-method',  ALL_RUNS.map(r=>r.sampling_method));
popSel('f-rbm',     ALL_RUNS.map(r=>r.rbm));
popSel('f-nhidden', ALL_RUNS.map(r=>String(r.n_hidden)));
popSel('f-nsamples',ALL_RUNS.map(r=>String(r.n_samples)));
popSel('f-reg',     ALL_RUNS.map(r=>String(r.reg)));
popSel('f-lr', ALL_RUNS.map(r => String(r.lr)));
popSel('f-seed',    ALL_RUNS.map(r=>String(r.seed)));

// run count
$('run-count').textContent=`${ALL_RUNS.length} runs · ${nhKeys.length} (N,h) pairs`;

// sidebar
nhKeys.forEach(key=>{
  const runs=ALL_RUNS.filter(r=>nhKey(r)===key);
  const best=Math.min(...runs.map(r=>r.rel_error));
  const btn=document.createElement('button');
  btn.className='nh-btn'; btn.id='btn-'+key; btn.dataset.k=key;
  btn.onclick=()=>showPanel('nh-'+key);
  btn.innerHTML=`${nhLabelMap[key]}<span class="err-badge ${pillCls(best)}">${best.toFixed(1)}%</span><span class="badge">${runs.length}</span>`;
  $('nh-buttons').appendChild(btn);
});

function showPanel(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nh-btn').forEach(b=>b.classList.remove('active'));
  const p=$('panel-'+id);if(p)p.classList.add('active');
  const b=$('btn-'+id);if(b)b.classList.add('active');
}

// ── overview ──────────────────────────────────────────────────────────────
function rebuildOverview(){
  const runs=ACTIVE;
  if(!runs.length){$('ov-stats').innerHTML=`<div class="stat-card"><div class="stat-value bad">No runs match</div></div>`;return;}

  const errs=runs.map(r=>r.rel_error);
  const best=Math.min(...errs),med=errs.slice().sort((a,b)=>a-b)[Math.floor(errs.length/2)],worst=Math.max(...errs);
  const samplers=[...new Set(runs.map(r=>r.sampler))];
  const methods=[...new Set(runs.map(r=>r.sampling_method))];

  $('ov-sub').textContent=`${runs.length} runs · ${new Set(runs.map(nhKey)).size} (N,h) pairs`;
  const notice=$('ov-notice');
  if(ACTIVE.length<ALL_RUNS.length){notice.textContent=`Filtered: ${ACTIVE.length}/${ALL_RUNS.length} runs`;notice.classList.add('on')}
  else notice.classList.remove('on');

  $('ov-stats').innerHTML=`
    <div class="stat-card"><div class="stat-label">Runs</div><div class="stat-value">${runs.length}</div></div>
    <div class="stat-card"><div class="stat-label">Best Error</div><div class="stat-value ${errCls(best)}">${fmtP(best)}</div></div>
    <div class="stat-card"><div class="stat-label">Median Error</div><div class="stat-value ${errCls(med)}">${fmtP(med)}</div></div>
    <div class="stat-card"><div class="stat-label">Worst Error</div><div class="stat-value ${errCls(worst)}">${fmtP(worst)}</div></div>
    <div class="stat-card"><div class="stat-label">Samplers</div><div class="stat-value" style="font-size:13px">${samplers.join(', ')}</div></div>
    <div class="stat-card"><div class="stat-label">Methods</div><div class="stat-value" style="font-size:13px">${methods.join(', ')}</div></div>
  `;

  // heatmap
  const sizes=[...new Set(runs.map(r=>r.size))].sort((a,b)=>a-b);
  const hs=[...new Set(runs.map(r=>r.h))].sort((a,b)=>a-b);
  const cells=[];
  sizes.forEach(sz=>hs.forEach(h=>{
    const sub=runs.filter(r=>r.size===sz&&r.h===h);
    if(sub.length)cells.push({sz,h,best:Math.min(...sub.map(r=>r.rel_error)),n:sub.length});
  }));
  const allB=cells.map(c=>c.best),minV=Math.min(...allB),maxV=Math.max(...allB);
  const cw=Math.max(72,Math.floor(520/hs.length));
  let html=`<div style="display:grid;grid-template-columns:52px ${hs.map(()=>cw+'px').join(' ')};gap:3px">`;
  html+=`<div></div>`;
  hs.forEach(h=>{html+=`<div style="text-align:center;font-size:9px;color:var(--text3);padding-bottom:3px">h=${h}</div>`});
  sizes.forEach(sz=>{
    html+=`<div style="font-size:9px;color:var(--text3);display:flex;align-items:center">N=${sz}</div>`;
    hs.forEach(h=>{
      const c=cells.find(x=>x.sz===sz&&x.h===h);
      if(!c){html+=`<div style="height:48px;border-radius:4px;background:var(--bg3)"></div>`;return;}
      html+=`<div class="heatmap-cell" style="background:${heatColor(c.best,minV,maxV)}" onclick="showPanel('nh-N${sz}_h${h}')">
        <div class="cv">${c.best.toFixed(2)}%</div><div class="cs">${c.n} runs</div></div>`;
    });
  });
  html+='</div>';
  $('heatmap-container').innerHTML=html;

  // sampler comparison bar
  const smGroups={};
  runs.forEach(r=>{const k=smKey(r);if(!smGroups[k])smGroups[k]=[];smGroups[k].push(r.rel_error)});
  const smKeys=Object.keys(smGroups).sort();
  mkChart('ov-samp-bar','bar',{
    labels:smKeys,
    datasets:[
      {label:'best %',data:smKeys.map(k=>Math.min(...smGroups[k])),backgroundColor:PAL.map(c=>c+'aa'),borderColor:PAL,borderWidth:1},
      {label:'mean %',data:smKeys.map(k=>mean(smGroups[k])),backgroundColor:PAL.map(c=>c+'44'),borderColor:PAL,borderWidth:1},
    ]
  },{scales:{y:{title:{display:true,text:'Rel. error (%)'}}}});

  // per-axis sensitivity — one line per sampler/method
  function ovSensChart(id,field,label){
    const vals=[...new Set(runs.map(r=>r[field]))].sort((a,b)=>a-b);
    const datasets=smKeys.map((sk,i)=>{
      const sub=runs.filter(r=>smKey(r)===sk);
      return{label:sk,data:vals.map(v=>{const s=sub.filter(r=>r[field]===v);return s.length?mean(s.map(r=>r.rel_error)):null}),
        borderColor:PAL[i%PAL.length],backgroundColor:PAL[i%PAL.length]+'22',borderWidth:2,pointRadius:4,spanGaps:true};
    });
    mkChart(id,'line',{labels:vals.map(v=>field==='reg'?Number(v).toExponential(1):String(v)),datasets},
      {scales:{x:{title:{display:true,text:label}},y:{title:{display:true,text:'Mean rel. error (%)'}}}});
  }
  ovSensChart('ov-err-ns', 'n_samples','n_samples');
  ovSensChart('ov-err-reg','reg',       'regularization');
  ovSensChart('ov-err-lr', 'lr',        'learning rate');
}

// ── per (N,h) panel ───────────────────────────────────────────────────────
function buildNHPanel(key){
  const allForKey=ALL_RUNS.filter(r=>nhKey(r)===key).sort((a,b)=>a.rel_error-b.rel_error);
  const exact=allForKey[0].exact_energy;
  const el=document.createElement('div');
  el.className='panel'; el.id='panel-nh-'+key;
  el.innerHTML=`
    <div class="section-title">${nhLabelMap[key]}</div>
    <div class="section-sub">Exact E = ${fmtF(exact,6)} &nbsp;·&nbsp; <span id="nhrc-${key}">${allForKey.length}</span> runs</div>
    <div class="filter-notice" id="nhn-${key}"></div>
    <div class="stats-row" id="nhs-${key}"></div>

    <div class="section-title" style="font-size:12px;margin-bottom:9px">Sampler Comparison</div>
    <div class="charts-grid" style="margin-bottom:18px">
      <div class="chart-card"><div class="chart-title">Best &amp; Mean Error by Sampler/Method</div><div class="chart-wrap"><canvas id="nhsb-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Best Convergence per Sampler/Method</div><div class="chart-wrap tall"><canvas id="nhsc-${key}"></canvas></div></div>
    </div>

    <div class="charts-grid wide">
      <div class="chart-card"><div class="chart-title">Energy Convergence — Top 5 (filtered)</div><div class="chart-wrap tall"><canvas id="conv-${key}"></canvas></div></div>
    </div>
    <div class="charts-grid" style="margin-top:12px">
      <div class="chart-card"><div class="chart-title">Error vs n_samples</div><div class="chart-wrap"><canvas id="sns-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Error vs Regularization</div><div class="chart-wrap"><canvas id="srg-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Error vs Learning Rate</div><div class="chart-wrap"><canvas id="slr-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Error vs n_hidden</div><div class="chart-wrap"><canvas id="snh-${key}"></canvas></div></div>
    </div>

    <hr class="divider">
    <div class="run-table-wrap">
      <div class="run-table-header"><span>All Runs</span><span id="ntc-${key}"></span></div>
      <div style="overflow-x:auto">
        <table id="tbl-${key}"><thead><tr>
          <th>#</th><th>rel err%</th><th>abs err</th><th>final E</th>
          <th>sampler</th><th>method</th><th>lr</th><th>reg</th>
          <th>n_samp</th><th>n_hid</th><th>seed</th>
          <th>‖x‖ last</th><th>κ(S) last</th><th>‖w‖ last</th>
        </tr></thead><tbody id="tb-${key}"></tbody></table>
      </div>
    </div>

    <hr class="divider">
    <div class="section-title" style="font-size:12px;margin-bottom:4px">Run Diagnostics</div>
    <div class="section-sub">Click a row or pick from dropdown</div>
    <div class="diag-selector">
      <label>Run:</label>
      <select id="ds-${key}" onchange="renderDiag('${key}',this.value)"></select>
    </div>
    <div class="charts-grid">
      <div class="chart-card span2"><div class="chart-title">Energy Convergence</div><div class="chart-wrap tall"><canvas id="de-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Gradient Norm ‖x‖</div><div class="chart-wrap"><canvas id="dg-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">S Condition Number κ(S)</div><div class="chart-wrap"><canvas id="dc-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Weight Norm ‖w‖</div><div class="chart-wrap"><canvas id="dw-${key}"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Energy Statistical Error σ/√n</div><div class="chart-wrap"><canvas id="dee-${key}"></canvas></div></div>
    </div>
  `;
  $('nh-panels').appendChild(el);
  setTimeout(()=>rebuildNHCharts(key),0);
}

function rebuildNHCharts(key){
  const allForKey=ALL_RUNS.filter(r=>nhKey(r)===key);
  const runs=ACTIVE.filter(r=>nhKey(r)===key).sort((a,b)=>a.rel_error-b.rel_error);
  const exact=allForKey[0].exact_energy;

  // notice
  const notice=$(`nhn-${key}`);
  if(runs.length<allForKey.length){notice.textContent=`Showing ${runs.length}/${allForKey.length} runs`;notice.classList.add('on')}
  else notice.classList.remove('on');
  $(`nhrc-${key}`).textContent=runs.length;
  $(`ntc-${key}`).textContent=`${runs.length} experiments`;

  if(!runs.length){$(`nhs-${key}`).innerHTML=`<div class="stat-card"><div class="stat-value bad">No runs match</div></div>`;return;}

  const errs=runs.map(r=>r.rel_error);
  const bestE=Math.min(...errs),medE=errs[Math.floor(errs.length/2)],worstE=Math.max(...errs);
  const br=runs[0];
  $(`nhs-${key}`).innerHTML=`
    <div class="stat-card"><div class="stat-label">Exact E</div><div class="stat-value" style="font-size:14px">${fmtF(exact,4)}</div></div>
    <div class="stat-card"><div class="stat-label">Best Error</div><div class="stat-value ${errCls(bestE)}">${fmtP(bestE)}</div><div class="stat-sub">abs: ${fmtF(br.abs_error,5)}</div></div>
    <div class="stat-card"><div class="stat-label">Median Error</div><div class="stat-value ${errCls(medE)}">${fmtP(medE)}</div></div>
    <div class="stat-card"><div class="stat-label">Worst Error</div><div class="stat-value ${errCls(worstE)}">${fmtP(worstE)}</div></div>
    <div class="stat-card"><div class="stat-label">Best Final E</div><div class="stat-value" style="font-size:14px">${fmtF(br.final_energy,4)}</div></div>
  `;

  // sampler comparison bar
  const sg={};
  runs.forEach(r=>{const k=smKey(r);if(!sg[k])sg[k]=[];sg[k].push(r.rel_error)});
  const sKeys=Object.keys(sg).sort();
  mkChart(`nhsb-${key}`,'bar',{
    labels:sKeys,
    datasets:[
      {label:'best %',data:sKeys.map(k=>Math.min(...sg[k])),backgroundColor:PAL.map(c=>c+'aa'),borderColor:PAL,borderWidth:1},
      {label:'mean %',data:sKeys.map(k=>mean(sg[k])),backgroundColor:PAL.map(c=>c+'44'),borderColor:PAL,borderWidth:1},
    ]
  },{scales:{y:{title:{display:true,text:'Rel. error (%)'}}}});

  // best convergence per sampler
  const bestPerSM=sKeys.map(sk=>runs.filter(r=>smKey(r)===sk)[0]).filter(Boolean);
  const maxL=Math.max(1,...bestPerSM.map(r=>r.energy_curve.length));
  mkChart(`nhsc-${key}`,'line',{datasets:[
    ...bestPerSM.map((r,i)=>lineds(`${smKey(r)} (${fmtP(r.rel_error)})`,r.energy_curve,PAL[i%PAL.length])),
    {label:`Exact: ${fmtF(exact,4)}`,data:[{x:0,y:exact},{x:maxL-1,y:exact}],borderColor:'#ef444488',borderDash:[6,3],borderWidth:1.5,pointRadius:0}
  ]},{scales:{x:{type:'linear',title:{display:true,text:'iteration'}},y:{title:{display:true,text:'energy'}}}});

  // top 5 convergence
  const top5=runs.slice(0,5);
  const maxL5=Math.max(1,...top5.map(r=>r.energy_curve.length));
  mkChart(`conv-${key}`,'line',{datasets:[
    ...top5.map((r,i)=>lineds(`#${i+1} ${smKey(r)} ${fmtP(r.rel_error)}`,r.energy_curve,PAL[i])),
    {label:`Exact: ${fmtF(exact,4)}`,data:[{x:0,y:exact},{x:maxL5-1,y:exact}],borderColor:'#ef444488',borderDash:[6,3],borderWidth:1.5,pointRadius:0}
  ]},{scales:{x:{type:'linear',title:{display:true,text:'iteration'}},y:{title:{display:true,text:'energy'}}}});

  // sensitivity — one line per sampler/method
  function sensChart(id,field,label){
    const vals=[...new Set(runs.map(r=>r[field]))].sort((a,b)=>a-b);
    const datasets=sKeys.map((sk,i)=>{
      const sub=runs.filter(r=>smKey(r)===sk);
      return{label:sk,data:vals.map(v=>{const s=sub.filter(r=>r[field]===v);return s.length?mean(s.map(r=>r.rel_error)):null}),
        borderColor:PAL[i%PAL.length],backgroundColor:PAL[i%PAL.length]+'22',borderWidth:2,pointRadius:4,spanGaps:true};
    });
    mkChart(id,'line',{labels:vals.map(v=>field==='reg'?Number(v).toExponential(1):String(v)),datasets},
      {scales:{x:{title:{display:true,text:label}},y:{title:{display:true,text:'Mean rel. error (%)'}}}});
  }
  sensChart(`sns-${key}`,'n_samples','n_samples');
  sensChart(`srg-${key}`,'reg',      'regularization');
  sensChart(`slr-${key}`,'lr',       'learning rate');
  sensChart(`snh-${key}`,'n_hidden', 'n_hidden');

  // table
  const tbody=$(`tb-${key}`),sel=$(`ds-${key}`);
  tbody.innerHTML=''; sel.innerHTML='';
  runs.forEach((r,i)=>{
    const gn=r.grad_norm_curve,cc=r.cond_curve,wn=r.weight_norm_curve;
    const lg=gn&&gn.length?gn[gn.length-1]:null;
    const lc=cc&&cc.length?cc[cc.length-1]:null;
    const lw=wn&&wn.length?wn[wn.length-1]:null;
    const tr=document.createElement('tr');
    tr.onclick=()=>{
      document.querySelectorAll(`#tbl-${key} tr.selected`).forEach(t=>t.classList.remove('selected'));
      tr.classList.add('selected'); sel.value=i; renderDiag(key,i);
    };
    tr.innerHTML=`
      <td>${i+1}</td>
      <td><span class="pill ${pillCls(r.rel_error)}">${fmtP(r.rel_error)}</span></td>
      <td>${fmtF(r.abs_error,5)}</td><td>${fmtF(r.final_energy,4)}</td>
      <td><span class="pill pill-blue">${r.sampler}</span></td>
      <td>${r.sampling_method}</td>
      <td>${r.lr}</td><td>${fmtE(r.reg)}</td>
      <td>${r.n_samples}</td><td>${r.n_hidden}</td><td>${r.seed}</td>
      <td>${lg!==null?fmtF(lg,4):'—'}</td>
      <td>${lc!==null?lc.toExponential(2):'—'}</td>
      <td>${lw!==null?fmtF(lw,4):'—'}</td>
    `;
    tbody.appendChild(tr);
    const o=document.createElement('option');
    o.value=i;
    o.textContent=`#${i+1} ${smKey(r)} err=${fmtP(r.rel_error)} lr=${r.lr} reg=${fmtE(r.reg)} ns=${r.n_samples} seed=${r.seed}`;
    sel.appendChild(o);
  });
  if(tbody.firstElementChild){tbody.firstElementChild.classList.add('selected');renderDiag(key,0);}
}

function renderDiag(key,idx){
  const runs=ACTIVE.filter(r=>nhKey(r)===key).sort((a,b)=>a.rel_error-b.rel_error);
  if(!runs.length)return;
  const r=runs[+idx]; if(!r)return;
  const exact=r.exact_energy,iters=r.energy_curve.length||100;

  function dLine(id,curve,label,color,logy=false){
    const ex=Chart.getChart(id);if(ex)ex.destroy();
    const el=$(id);if(!el||!curve.length)return;
    new Chart(el,{type:'line',data:{datasets:[lineds(label,curve,color)]},options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{boxWidth:8}}},
      scales:{x:{type:'linear',title:{display:true,text:'iteration'}},
              y:{type:logy&&curve.some(v=>v>0)?'logarithmic':'linear',title:{display:true,text:label}}}}});
  }

  const enEx=Chart.getChart(`de-${key}`);if(enEx)enEx.destroy();
  const enEl=$(`de-${key}`);
  if(enEl){new Chart(enEl,{type:'line',data:{datasets:[
    lineds(`VMC [${smKey(r)}]`,r.energy_curve,'#3b82f6'),
    {label:`Exact: ${fmtF(exact,4)}`,data:[{x:0,y:exact},{x:iters-1,y:exact}],borderColor:'#ef4444',borderDash:[6,3],borderWidth:1.5,pointRadius:0}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{boxWidth:8}}},
    scales:{x:{type:'linear',title:{display:true,text:'iteration'}},y:{title:{display:true,text:'energy'}}}}});}

  dLine(`dg-${key}`, r.grad_norm_curve||[],   '‖x‖', '#f59e0b');
  dLine(`dc-${key}`, r.cond_curve||[],         'κ(S)','#ef4444',true);
  dLine(`dw-${key}`, r.weight_norm_curve||[],  '‖w‖', '#a855f7');
  dLine(`dee-${key}`,r.energy_error_curve||[], 'σ/√n','#06b6d4');
}

rebuildOverview();
nhKeys.forEach(k=>buildNHPanel(k));
</script>
</body>
</html>
"""


def generate_report(results_dir: Path, output_path: Path):
    runs = load_results(results_dir)
    if not runs:
        print("No results found.")
        return
    print(f"Loaded {len(runs)} runs.")
    runs_json = json.dumps(runs, indent=None, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__RUNS_JSON__", runs_json)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written → {output_path}  ({output_path.stat().st_size // 1024} KB)")
    print(f"Open with:  open {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate HTML benchmark report")
    p.add_argument("--results", default="results/", help="Results directory")
    p.add_argument("--output", default="report.html", help="Output HTML file")
    args = p.parse_args()
    generate_report(Path(args.results), Path(args.output))


if __name__ == "__main__":
    main()
