"""
VMC / RBM experiment results dashboard.

    cd src
    streamlit run dashboard.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXTEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  New filter         → add one dict to FILTER_AXES
  New scalar metric  → add one tuple to SCALAR_METRICS
  New history series → add one tuple to HISTORY_METRICS
  New solver/run     → drop JSON files in results/{model}/ and hit "Reload data"
                       (auto-discovered, zero code changes needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from helpers import load_result_json

import reference_energies

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIRS = [_ROOT / "results"]

# ── Extension points ───────────────────────────────────────────────────────────
# Add one dict  → new sidebar filter appears automatically
# col must match a key in the flat DataFrame built by load_all_runs()
FILTER_AXES = [
    {"col": "device", "label": "Device (GPU/CPU)"},
    {"col": "model", "label": "Model"},
    {"col": "size", "label": "System size N"},
    {"col": "h", "label": "Field h"},
    {"col": "J", "label": "Coupling J"},
    {"col": "J1", "label": "Coupling J1"},
    {"col": "J2", "label": "Coupling J2"},
    {"col": "delta", "label": "Anisotropy δ"},
    {"col": "sigma", "label": "Sigma"},
    {"col": "alpha", "label": "LR exponent α"},
    {"col": "sampler", "label": "Sampler backend"},
    {"col": "sampling_method", "label": "Sampling method"},
    {"col": "ansatz", "label": "Ansatz (rbm / vit)"},
    {"col": "rbm", "label": "RBM type"},
    {"col": "n_hidden", "label": "Hidden units"},
    {"col": "d_model", "label": "ViT d_model"},
    {"col": "n_layers", "label": "ViT layers"},
    {"col": "n_heads", "label": "ViT heads"},
    {"col": "patch_size", "label": "ViT patch size"},
    {"col": "learning_rate", "label": "Learning rate"},
    {"col": "regularization", "label": "Regularization"},
    {"col": "n_samples", "label": "Samples / iter"},
    {"col": "iterations", "label": "Iterations"},
    {"col": "cem", "label": "CEM"},
    {"col": "mh_warmup", "label": "MH warmup rounds"},
    {"col": "mh_sweeps", "label": "MH sweep rounds"},
    {"col": "num_sweeps", "label": "FPGA/SA sweep count"},
    {"col": "nh_ratio", "label": "Hidden/visible ratio (nh/N)"},
    {"col": "seed", "label": "Seed"},
]

# Add one tuple → metric appears in table and aggregation plots
SCALAR_METRICS = [
    ("error", "Energy error |E_rbm − E_exact|"),
    ("error_per_spin", "Energy error per spin"),
    ("relative_error", "Relative error (%)"),
    ("final_kl_exact", "Final KL divergence"),
    ("final_ess", "Final ESS"),
    ("mean_ess", "Mean ESS"),
    ("sampling_time_s", "Total sampling time (s)"),
    ("mean_time_per_iter", "Mean sampling time / iter (s)"),
    ("final_energy", "Final energy"),
    ("exact_energy", "Exact energy"),
    ("sparsity", "RBM sparsity"),
    ("mean_mh_acceptance_rate", "Mean MH acceptance rate"),
]

# Add one tuple → series appears in convergence curves
HISTORY_METRICS = [
    ("energy", "Energy"),
    ("kl_exact", "KL divergence (exact)"),
    ("ess", "ESS"),
    ("grad_norm", "Gradient norm"),
    ("weight_norm", "Weight norm"),
    ("n_unique_ratio", "Unique sample ratio"),
    ("cg_iterations", "CG iterations"),
    ("cg_residual", "CG residual"),
    ("s_condition_number", "SR condition number"),
    ("sampling_time_s", "Sampling time / iter (s)"),
    ("beta_x", "Beta x"),
    ("mh_acceptance_rate", "MH acceptance rate"),
]

MAX_CURVES = 60  # max lines drawn in convergence tab before a warning


def _filter_summary() -> str:
    """Compact string of active filter selections for use in plot titles."""
    parts = []
    for ax in FILTER_AXES:
        sel = st.session_state.get(f"f_{ax['col']}", [])
        if sel:
            parts.append(f"{ax['label']}: {', '.join(str(s) for s in sel)}")
    return "  |  ".join(parts) if parts else ""


def _titled(main: str) -> str:
    """Return a Plotly title string: main title with active filters as subtitle."""
    fs = _filter_summary()
    return f"{main}<br><sup>{fs}</sup>" if fs else main


# ── Data loading ───────────────────────────────────────────────────────────────


def _n_spins(model, size) -> int:
    """Number of spins for a run: N for 1D, N² for 2D."""
    n = int(size) if pd.notna(size) else 1
    return n if str(model) == "1d" else n * n


@st.cache_data
def load_all_runs(results_dirs: tuple[Path, ...]) -> tuple[pd.DataFrame, dict]:
    """
    Scan all JSON files under every directory in results_dirs.

    Returns
    -------
    df        flat DataFrame — one row per run, columns = config + scalars
    histories {run_id: {metric_key: [values]}}
    """
    records: list[dict] = []
    histories: dict[str, dict] = {}

    # results/archive/** holds superseded/invalid runs (e.g. pre-auto_scale-fix
    # D-Wave data) kept for reference, not live results -- never show them here.
    paths = sorted(
        p for d in results_dirs for p in d.rglob("*.json*")
        if "archive" not in p.relative_to(d).parts
    )
    for path in paths:
        try:
            d = load_result_json(path)
        except Exception:
            continue

        base = next((d for d in results_dirs if path.is_relative_to(d)), path.parent)
        run_id = str(path.relative_to(base))
        cfg = d.get("config", {})

        # Extract num_sweeps from sweepsNNN/ directory component in the path.
        sw_match = next(
            filter(None, (re.fullmatch(r"sweeps(\d+)", part) for part in path.parts)),
            None,
        )
        _path_num_sweeps = int(sw_match.group(1)) if sw_match else None

        row: dict = {"run_id": run_id}

        # Config columns — one per FILTER_AXES entry; missing keys → None
        for ax in FILTER_AXES:
            row[ax["col"]] = cfg.get(ax["col"])

        # Runs without an explicit ansatz field are RBM runs
        if row.get("ansatz") is None:
            row["ansatz"] = "rbm"

        # num_sweeps: prefer config field, fall back to path-derived value.
        if row.get("num_sweeps") is None:
            row["num_sweeps"] = _path_num_sweeps

        # nh_ratio: n_hidden / n_visible (rounded to 1 dp); None if either missing.
        _nh = cfg.get("n_hidden")
        _nv = cfg.get("size")
        row["nh_ratio"] = round(_nh / _nv, 1) if (_nh and _nv) else None

        # Scalar outputs — exact_energy and error come from the master cache,
        # not from the JSON file (those values may be stale or inaccurate).
        for key in (
            "final_energy",
            "final_ess",
            "mean_ess",
            "final_kl_exact",
            "sampling_time_s",
            "gpu_energy_wh",
            "sparsity",
            "mean_mh_acceptance_rate",
        ):
            row[key] = d.get(key)

        # Reference energy: master cache only; None if not yet computed.
        model_str = cfg.get("model")
        size_val = cfg.get("size")
        h_val = cfg.get("h")
        alpha_val = cfg.get("alpha")
        J_val = cfg.get("J")
        J1_val = cfg.get("J1")
        J2_val = cfg.get("J2")
        delta_val = cfg.get("delta")
        row["alpha"] = alpha_val
        row["J"] = J_val
        exact_energy = None
        if model_str and size_val is not None:
            if model_str == "lr1d" and alpha_val is not None and J_val is not None and h_val is not None:
                lr_model_key = f"lr_tfim_1d_alpha{float(alpha_val):.10g}_J{float(J_val):.10g}"
                exact_energy = reference_energies.lookup(
                    lr_model_key, int(size_val), float(h_val)
                )
            elif model_str in ("heisenberg_xxz_1d", "heisenberg_xy_1d") and J_val is not None:
                d_key = 0.0 if model_str == "heisenberg_xy_1d" else float(delta_val) if delta_val is not None else 1.0
                xxz_key = f"heisenberg_xxz_1d_delta{d_key:.10g}"
                exact_energy = reference_energies.lookup(xxz_key, int(size_val), float(J_val))
            elif model_str == "heisenberg_xxz_2d" and J_val is not None and delta_val is not None:
                xxz2d_key = f"heisenberg_xxz_2d_delta{float(delta_val):.10g}"
                exact_energy = reference_energies.lookup(xxz2d_key, int(size_val), float(J_val))
            elif model_str == "j1j2_1d" and J1_val is not None and J2_val is not None and h_val is not None:
                j1j2_key = f"j1j2_1d_J1{float(J1_val):.10g}_J2{float(J2_val):.10g}"
                exact_energy = reference_energies.lookup(j1j2_key, int(size_val), float(h_val))
            elif model_str == "heisenberg_j1j2_1d" and J1_val is not None and J2_val is not None and delta_val is not None:
                heis_j1j2_key = f"heisenberg_j1j2_1d_J1{float(J1_val):.10g}_J2{float(J2_val):.10g}_delta{float(delta_val):.10g}"
                exact_energy = reference_energies.lookup(heis_j1j2_key, int(size_val), float(J1_val))
            elif h_val is not None:
                exact_energy = reference_energies.lookup(
                    str(model_str), int(size_val), float(h_val)
                )
        row["exact_energy"] = exact_energy

        # Recompute error from the authoritative reference.
        final_energy = row.get("final_energy")
        error = (
            abs(final_energy - exact_energy)
            if (final_energy is not None and exact_energy is not None)
            else None
        )
        row["error"] = error

        # Device: read from jax_devices (JAX backend) or fall back to cuda dict
        jax_devices = d.get("jax_devices")
        if jax_devices is not None:
            row["device"] = jax_devices.get("backend", "unknown")
        else:
            cuda = d.get("cuda")
            if cuda is None:
                row["device"] = "unknown"
            elif not cuda.get("torch_cuda_available", False):
                row["device"] = "cpu"
            else:
                row["device"] = cuda.get("torch_device", "gpu")

        # Derived scalars
        n_sp = _n_spins(row.get("model"), row.get("size"))
        row["n_spins"] = n_sp
        row["relative_error"] = (
            abs(error / exact_energy) * 100
            if (error is not None and exact_energy)
            else None
        )
        row["error_per_spin"] = (error / n_sp) if error is not None else None
        ts, iters = row.get("sampling_time_s"), row.get("iterations")
        row["mean_time_per_iter"] = (ts / iters) if (ts is not None and iters) else None

        records.append(row)
        histories[run_id] = d.get("history", {})

    df = pd.DataFrame(records)

    # Numeric coercion so filter options sort correctly
    for col in (
        "size",
        "h",
        "J",
        "J1",
        "J2",
        "delta",
        "sigma",
        "alpha",
        "n_hidden",
        "d_model",
        "n_layers",
        "n_heads",
        "patch_size",
        "learning_rate",
        "regularization",
        "n_samples",
        "iterations",
        "num_sweeps",
        "nh_ratio",
        "seed",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, histories


# ── LR-TFIM reference panel ────────────────────────────────────────────────────


_REF_PANEL_LABEL_COLS = [
    ("model", "{}"),
    ("alpha", "α={:g}"),
    ("J", "J={:g}"),
    ("J1", "J1={:g}"),
    ("J2", "J2={:g}"),
    ("delta", "δ={:g}"),
    ("h", "h={:g}"),
    ("size", "N={:g}"),
]

_REF_PANEL_MAX_SHOWN = 12


def _reference_energy_panel(df: pd.DataFrame) -> None:
    """Show exact reference energy per spin for every distinct physical
    configuration in the current filter selection (any model).

    Reference energies come from the master cache — no computation is
    triggered here, so a config only appears once it's been computed
    elsewhere (e.g. via exact diagonalization or the closed-form TFIM
    solution) and cached in reference_energies.
    """
    key_cols = [c for c, _ in _REF_PANEL_LABEL_COLS if c in df.columns]
    if not key_cols or "exact_energy" not in df.columns:
        return

    ref_rows = (
        df[key_cols + ["exact_energy", "n_spins"]]
        .dropna(subset=["exact_energy"])
        .drop_duplicates(subset=key_cols)
        .sort_values(key_cols)
    )
    if ref_rows.empty:
        return

    st.markdown("### Theoretical Ground State Energy (current filter selection)")

    n_shown = min(len(ref_rows), _REF_PANEL_MAX_SHOWN)
    if len(ref_rows) > _REF_PANEL_MAX_SHOWN:
        st.caption(
            f"Showing {n_shown} of {len(ref_rows)} distinct configurations — "
            "narrow the sidebar filters to see the rest."
        )

    cols = st.columns(min(n_shown, 4))
    for i, (_, rv) in enumerate(ref_rows.head(n_shown).iterrows()):
        n_sp = int(rv["n_spins"]) if pd.notna(rv.get("n_spins")) else int(rv.get("size", 1))
        e_per_spin = rv["exact_energy"] / n_sp
        label_parts = [
            fmt.format(rv[col]) for col, fmt in _REF_PANEL_LABEL_COLS
            if col in rv and pd.notna(rv[col])
        ]
        with cols[i % len(cols)]:
            st.metric(
                label=",  ".join(label_parts),
                value=f"{e_per_spin:.6f}",
                help=f"Exact ground state energy per spin  (E_exact = {rv['exact_energy']:.6f}, N_spins={n_sp})",
            )

    st.markdown("---")


# ── Sidebar ────────────────────────────────────────────────────────────────────


def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filters")

    if st.sidebar.button("Clear all filters"):
        for ax in FILTER_AXES:
            st.session_state[f"f_{ax['col']}"] = []
        st.rerun()

    filtered = df.copy()

    for ax in FILTER_AXES:
        col, label = ax["col"], ax["label"]
        if col not in df.columns:
            continue
        # Use the progressively-narrowed df so options reflect prior selections
        vals = sorted(
            filtered[col].dropna().unique(),
            key=lambda x: (str(type(x).__name__), str(x)),
        )
        if len(vals) <= 1:
            continue
        # Drop stale selections that no longer exist in the available options
        key = f"f_{col}"
        valid_vals = set(vals)
        current = st.session_state.get(key, [])
        cleaned = [v for v in current if v in valid_vals]
        if cleaned != current:
            st.session_state[key] = cleaned
        sel = st.sidebar.multiselect(label, vals, default=[], key=key)
        if sel:
            filtered = filtered[filtered[col].isin(sel)]

    st.sidebar.markdown("---")
    st.sidebar.metric("Runs selected", f"{len(filtered)} / {len(df)}")

    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

    return filtered


# ── Tab helpers ────────────────────────────────────────────────────────────────


def _axis_label(col: str) -> str:
    return next((ax["label"] for ax in FILTER_AXES if ax["col"] == col), col)


def _group_selectbox(
    df: pd.DataFrame, key: str, label: str, prefer: str = "sampling_method"
) -> tuple[str, str]:
    """Return (col, label) for a group-by selectbox."""
    opts = [
        ax
        for ax in FILTER_AXES
        if ax["col"] in df.columns and df[ax["col"]].nunique() > 1
    ]
    default = next((i for i, ax in enumerate(opts) if ax["col"] == prefer), 0)
    chosen = st.selectbox(
        label,
        opts,
        format_func=lambda ax: ax["label"],
        index=default,
        key=key,
    )
    return chosen["col"], chosen["label"]


# ── Tab 1: Run table ───────────────────────────────────────────────────────────


def tab_table(df: pd.DataFrame) -> None:
    preferred = [
        "model",
        "size",
        "h",
        "sampler",
        "sampling_method",
        "rbm",
        "n_hidden",
        "learning_rate",
        "regularization",
        "iterations",
        "seed",
        "error",
        "relative_error",
        "final_kl_exact",
        "final_ess",
        "mean_ess",
        "sampling_time_s",
    ]
    cols = [c for c in preferred if c in df.columns and df[c].notna().any()]

    col_cfg = {
        "error": st.column_config.NumberColumn("Error", format="%.3e"),
        "relative_error": st.column_config.NumberColumn("Rel. err (%)", format="%.2f"),
        "final_kl_exact": st.column_config.NumberColumn("KL div", format="%.4f"),
        "final_ess": st.column_config.NumberColumn("ESS", format="%.3f"),
        "mean_ess": st.column_config.NumberColumn("Mean ESS", format="%.3f"),
        "sampling_time_s": st.column_config.NumberColumn("Time (s)", format="%.2f"),
        "learning_rate": st.column_config.NumberColumn("LR", format="%.0e"),
        "regularization": st.column_config.NumberColumn("Reg", format="%.0e"),
    }

    st.dataframe(df[cols], column_config=col_cfg, use_container_width=True, height=520)


# ── Tab 2: Convergence curves ──────────────────────────────────────────────────


def tab_curves(df: pd.DataFrame, histories: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    metric_idx = c1.selectbox(
        "Y-axis metric",
        range(len(HISTORY_METRICS)),
        format_func=lambda i: HISTORY_METRICS[i][1],
        key="curve_metric",
    )
    metric_key, metric_label = HISTORY_METRICS[metric_idx]

    # Default to learning_rate for easy lr comparison; fall back to sampling_method
    color_col, color_label = _group_selectbox(
        df, "curve_color", "Color by", prefer="learning_rate"
    )

    log_y = c3.checkbox("Log Y", value=False, key="curve_logy")
    show_ref = c4.checkbox(
        "Exact energy ref.", value=(metric_key == "energy"), key="curve_ref"
    )
    per_spin = c5.checkbox(
        "Per spin",
        value=(metric_key == "energy"),
        key="curve_per_spin",
        help="Divide energy by N (1D) or N² (2D). Only meaningful for the energy metric.",
    )
    clip_outliers = c6.checkbox(
        "Clip outliers",
        value=True,
        key="curve_clip",
        help="Restrict Y axis to the 2nd–98th percentile of plotted values, hiding divergent early iterations.",
    )

    runs = df.head(MAX_CURVES)
    if len(df) > MAX_CURVES:
        st.caption(
            f"Showing first {MAX_CURVES} of {len(df)} runs — use filters to narrow."
        )

    # Build long-format table for px.line
    rows = []
    for _, r in runs.iterrows():
        series = histories.get(r["run_id"], {}).get(metric_key, [])
        n_spins = _n_spins(r.get("model"), r.get("size"))
        for i, v in enumerate(series):
            if v is None:
                continue
            val = float(v)
            if per_spin and metric_key == "energy":
                val = val / n_spins
            rows.append(
                {
                    "iteration": i,
                    "value": val,
                    "run_id": r["run_id"],
                    "color_group": str(r.get(color_col, "?")),
                    "sampler": f"{r.get('sampler', '')}/{r.get('sampling_method', '')}",
                    "N": r.get("size"),
                    "h": r.get("h"),
                    "J": r.get("J"),
                    "J1": r.get("J1"),
                    "J2": r.get("J2"),
                    "delta": r.get("delta"),
                    "sigma": r.get("sigma"),
                    "lr": r.get("learning_rate"),
                    "seed": r.get("seed"),
                    "error": f"{r['error']:.3e}" if pd.notna(r.get("error")) else "N/A",
                }
            )

    if not rows:
        st.info(f"No history data for '{metric_label}' in the selected runs.")
        return

    y_label = (
        f"{metric_label} per spin"
        if per_spin and metric_key == "energy"
        else metric_label
    )

    pf = pd.DataFrame(rows)
    fig = px.line(
        pf,
        x="iteration",
        y="value",
        color="color_group",
        line_group="run_id",
        labels={
            "iteration": "Iteration",
            "value": y_label,
            "color_group": color_label,
        },
        hover_data={
            "sampler": True,
            "N": True,
            "h": True,
            "J": True,
            "J1": True,
            "J2": True,
            "delta": True,
            "sigma": True,
            "lr": True,
            "seed": True,
            "error": True,
            "run_id": False,
            "color_group": False,
        },
        height=520,
    )
    fig.update_traces(opacity=0.75, line=dict(width=1.5))

    if show_ref and metric_key == "energy":
        ref_rows = df[["size", "h", "exact_energy", "model"]].dropna().drop_duplicates()
        for _, rv in ref_rows.iterrows():
            y_ref = rv["exact_energy"]
            if per_spin:
                y_ref = y_ref / _n_spins(rv["model"], rv["size"])
            fig.add_hline(
                y=y_ref,
                line_dash="dot",
                line_color="black",
                opacity=0.45,
                annotation_text=f"Exact energy/spin  h={rv['h']}"
                if per_spin
                else f"Exact energy  h={rv['h']}",
                annotation_position="bottom right",
            )

    if log_y:
        fig.update_yaxes(type="log")

    if clip_outliers and not pf.empty:
        import numpy as np

        vals = pf["value"][pf["value"] > 0] if log_y else pf["value"]
        if not vals.empty:
            lo = float(np.nanpercentile(vals, 2))
            hi = float(np.nanpercentile(vals, 98))
            if log_y and lo > 0 and hi > 0:
                lo_l, hi_l = np.log10(lo), np.log10(hi)
                pad = (hi_l - lo_l) * 0.05 or 0.1
                fig.update_yaxes(range=[lo_l - pad, hi_l + pad])
            elif not log_y:
                pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 0.1
                fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(
        hovermode="closest",
        title=_titled(f"Convergence — {y_label}"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Aggregated comparison ───────────────────────────────────────────────


def tab_compare(df: pd.DataFrame) -> None:
    # ── Group comparison ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    avail = [
        (k, l) for k, l in SCALAR_METRICS if k in df.columns and df[k].notna().any()
    ]
    if not avail:
        st.info("No scalar metrics available for the current selection.")
        return

    m_idx = c1.selectbox(
        "Metric",
        range(len(avail)),
        format_func=lambda i: avail[i][1],
        key="cmp_metric",
    )
    metric_key, metric_label = avail[m_idx]

    with c2:
        group_col, group_label = _group_selectbox(
            df, "cmp_group", "Group by", prefer="sampling_method"
        )

    plot_type = c3.selectbox(
        "Plot type", ["Box", "Violin", "Bar (mean ± std)"], key="cmp_type"
    )

    sub = df[[group_col, metric_key]].dropna()
    if sub.empty:
        st.info("No data for this metric / grouping.")
    else:
        kw = dict(
            x=group_col,
            y=metric_key,
            color=group_col,
            labels={group_col: group_label, metric_key: metric_label},
            height=420,
        )
        if plot_type == "Box":
            fig = px.box(sub, **kw)
        elif plot_type == "Violin":
            fig = px.violin(sub, box=True, **kw)
        else:
            agg = sub.groupby(group_col)[metric_key].agg(["mean", "std"]).reset_index()
            fig = px.bar(
                agg,
                x=group_col,
                y="mean",
                error_y="std",
                color=group_col,
                labels={group_col: group_label, "mean": f"Mean — {metric_label}"},
                height=420,
            )
        fig.update_layout(
            showlegend=False,
            title=_titled(f"{metric_label} by {group_label}"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Scaling plot ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Scaling: energy error vs system size N")

    sc1, sc2 = st.columns(2)
    with sc1:
        scale_col, scale_label = _group_selectbox(
            df, "scale_color", "Color by", prefer="sampling_method"
        )
    log_axes = sc2.checkbox("Log–log axes", value=True, key="scale_log")

    _scale_cols = list(
        dict.fromkeys(
            ["size", "error", scale_col, "h", "learning_rate", "seed", "n_hidden"]
        )
    )
    scale_df = df[_scale_cols].dropna(subset=["size", "error", scale_col])
    if scale_df.empty:
        st.info("No size/error data for the current selection.")
        return

    fig2 = px.scatter(
        scale_df,
        x="size",
        y="error",
        color=scale_col,
        labels={
            "size": "System size N",
            "error": "Energy error",
            scale_col: scale_label,
        },
        log_x=log_axes,
        log_y=log_axes,
        height=440,
        hover_data={
            c: True
            for c in ["h", "learning_rate", "seed", "n_hidden"]
            if c in scale_df.columns
        },
    )
    fig2.update_layout(title=_titled("Energy error vs system size N"))
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 4: Correlation ─────────────────────────────────────────────────────────


def tab_correlation(df: pd.DataFrame) -> None:
    st.markdown(
        "Explore whether **KL divergence** and **ESS** are good predictors "
        "of VMC convergence quality (`error`). Each point is one run."
    )

    # Determine which predictor metrics are available
    predictor_opts = [
        (k, l)
        for k, l in SCALAR_METRICS
        if k in df.columns
        and df[k].notna().any()
        and k not in ("error", "final_energy", "exact_energy")
    ]
    target_opts = [
        (k, l) for k, l in SCALAR_METRICS if k in df.columns and df[k].notna().any()
    ]

    if not predictor_opts or not target_opts:
        st.info("Not enough metric data for correlation analysis.")
        return

    c1, c2, c3, c4 = st.columns(4)

    x_idx = c1.selectbox(
        "X axis (predictor)",
        range(len(predictor_opts)),
        format_func=lambda i: predictor_opts[i][1],
        index=next(
            (i for i, (k, _) in enumerate(predictor_opts) if k == "final_kl_exact"), 0
        ),
        key="corr_x",
    )
    x_key, x_label = predictor_opts[x_idx]

    y_idx = c2.selectbox(
        "Y axis (target)",
        range(len(target_opts)),
        format_func=lambda i: target_opts[i][1],
        index=next((i for i, (k, _) in enumerate(target_opts) if k == "error"), 0),
        key="corr_y",
    )
    y_key, y_label = target_opts[y_idx]

    with c3:
        color_col, color_label = _group_selectbox(
            df, "corr_color", "Color by", prefer="sampling_method"
        )

    log_x = c4.checkbox("Log X", value=(x_key == "final_kl_exact"), key="corr_logx")
    log_y = c4.checkbox("Log Y", value=True, key="corr_logy")

    _plot_cols = list(
        dict.fromkeys([x_key, y_key, color_col, "size", "h", "learning_rate", "seed"])
    )
    plot_df = df[_plot_cols].dropna(subset=[x_key, y_key, color_col])

    if plot_df.empty:
        st.info("No runs have both metrics available.")
        return

    fig = px.scatter(
        plot_df,
        x=x_key,
        y=y_key,
        color=color_col,
        labels={x_key: x_label, y_key: y_label, color_col: color_label},
        log_x=log_x,
        log_y=log_y,
        hover_data={
            c: True
            for c in ["size", "h", "learning_rate", "seed"]
            if c in plot_df.columns
        },
        height=520,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(title=_titled(f"{y_label} vs {x_label}"))
    st.plotly_chart(fig, use_container_width=True)

    # Pearson r (log-space when log axes are on)
    import numpy as np

    sub = plot_df[[x_key, y_key]].dropna()
    if len(sub) >= 3:
        xv = np.log(sub[x_key].clip(lower=1e-12)) if log_x else sub[x_key]
        yv = np.log(sub[y_key].clip(lower=1e-12)) if log_y else sub[y_key]
        r = xv.corr(yv)
        space = "log–log" if (log_x and log_y) else ("log–lin" if log_x else "lin")
        st.caption(
            f"Pearson r ({space} space, n={len(sub)}): **{r:.3f}**  "
            f"{'— strong predictor' if abs(r) > 0.7 else '— weak predictor'}"
        )


# ── Tab 5: Timing ─────────────────────────────────────────────────────────────


def tab_timing(df: pd.DataFrame, histories: dict) -> None:
    st.markdown(
        "Sampling time is measured **per iteration** (`history.sampling_time_s`). "
        "The top-level scalar is the cumulative sum. "
        "All plots here use the per-iteration values."
    )

    # ── Section 1: sampler comparison ─────────────────────────────────────────
    st.subheader("Average sampling time per iteration — sampler comparison")

    c1, c2 = st.columns(2)
    with c1:
        grp_col, grp_label = _group_selectbox(
            df, "time_grp", "Group by", prefer="sampling_method"
        )
    plot_type = c2.selectbox(
        "Plot type", ["Box", "Violin", "Bar (mean ± std)"], key="time_plot_type"
    )

    sub = df[["mean_time_per_iter", grp_col]].dropna()
    if sub.empty:
        st.info("No timing data for the current selection.")
    else:
        kw = dict(
            x=grp_col,
            y="mean_time_per_iter",
            color=grp_col,
            labels={grp_col: grp_label, "mean_time_per_iter": "Mean time / iter (s)"},
            height=400,
        )
        if plot_type == "Box":
            fig = px.box(sub, **kw)
        elif plot_type == "Violin":
            fig = px.violin(sub, box=True, **kw)
        else:
            agg = (
                sub.groupby(grp_col)["mean_time_per_iter"]
                .agg(["mean", "std"])
                .reset_index()
            )
            fig = px.bar(
                agg,
                x=grp_col,
                y="mean",
                error_y="std",
                color=grp_col,
                labels={grp_col: grp_label, "mean": "Mean time / iter (s)"},
                height=400,
            )
        fig.update_layout(
            showlegend=False,
            title=_titled(f"Mean sampling time / iter by {grp_label}"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Section 2: time over iterations ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Sampling time per iteration over training")

    tc1, tc2, tc3 = st.columns(3)
    color_col, color_label = _group_selectbox(
        df, "time_color", "Color by", prefer="sampling_method"
    )
    log_t = tc2.checkbox("Log Y", value=False, key="time_logy")
    smooth = tc3.checkbox("Rolling mean (window=10)", value=False, key="time_smooth")

    runs = df.head(MAX_CURVES)
    if len(df) > MAX_CURVES:
        st.caption(
            f"Showing first {MAX_CURVES} of {len(df)} runs — use filters to narrow."
        )

    rows = []
    for _, r in runs.iterrows():
        series = histories.get(r["run_id"], {}).get("sampling_time_s", [])
        if not series:
            continue
        vals = [float(v) for v in series if v is not None]
        if smooth and len(vals) >= 10:
            import pandas as _pd

            vals = list(_pd.Series(vals).rolling(10, min_periods=1).mean())
        for i, v in enumerate(vals):
            rows.append(
                {
                    "iteration": i,
                    "value": v,
                    "run_id": r["run_id"],
                    "color_group": str(r.get(color_col, "?")),
                    "sampler": f"{r.get('sampler', '')}/{r.get('sampling_method', '')}",
                    "N": r.get("size"),
                    "h": r.get("h"),
                    "lr": r.get("learning_rate"),
                    "seed": r.get("seed"),
                }
            )

    if not rows:
        st.info("No per-iteration sampling time data for the selected runs.")
    else:
        pf = pd.DataFrame(rows)
        fig2 = px.line(
            pf,
            x="iteration",
            y="value",
            color="color_group",
            line_group="run_id",
            labels={
                "iteration": "Iteration",
                "value": "Sampling time / iter (s)" + (" [smoothed]" if smooth else ""),
                "color_group": color_label,
            },
            hover_data={
                "sampler": True,
                "N": True,
                "h": True,
                "lr": True,
                "seed": True,
                "run_id": False,
                "color_group": False,
            },
            height=480,
        )
        fig2.update_traces(opacity=0.75, line=dict(width=1.5))
        if log_t:
            fig2.update_yaxes(type="log")
        fig2.update_layout(
            hovermode="closest",
            title=_titled("Sampling time per iteration"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Section 3: cost–quality tradeoff ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Cost–quality tradeoff: sampling time vs energy error")

    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        qa_color_col, qa_color_label = _group_selectbox(
            df, "time_qa_color", "Color by", prefer="sampling_method"
        )
    log_x_qa = qa2.checkbox("Log X (time)", value=True, key="time_qa_logx")
    log_y_qa = qa3.checkbox("Log Y (error)", value=True, key="time_qa_logy")

    _qa_cols = list(
        dict.fromkeys(
            [
                "mean_time_per_iter",
                "error_per_spin",
                qa_color_col,
                "size",
                "h",
                "learning_rate",
                "seed",
            ]
        )
    )
    qa_df = df[_qa_cols].dropna(
        subset=["mean_time_per_iter", "error_per_spin", qa_color_col]
    )

    if qa_df.empty:
        st.info("No data with both timing and error available.")
    else:
        fig3 = px.scatter(
            qa_df,
            x="mean_time_per_iter",
            y="error_per_spin",
            color=qa_color_col,
            labels={
                "mean_time_per_iter": "Mean sampling time / iter (s)",
                "error_per_spin": "Energy error per spin",
                qa_color_col: qa_color_label,
            },
            log_x=log_x_qa,
            log_y=log_y_qa,
            hover_data={
                c: True
                for c in ["size", "h", "learning_rate", "seed"]
                if c in qa_df.columns
            },
            height=460,
            opacity=0.75,
        )
        fig3.update_traces(marker=dict(size=7))
        fig3.update_layout(
            title=_titled("Cost–quality tradeoff: sampling time vs energy error")
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Section 4: time scaling with system size ───────────────────────────────
    st.markdown("---")
    st.subheader("Sampling time scaling with system size")
    st.caption(
        "X axis: number of spins — N for 1D models, N² for 2D models. "
        "A power-law sampler should appear as a straight line on log–log axes."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sc_color_col, sc_color_label = _group_selectbox(
            df, "time_sc_color", "Color by", prefer="sampling_method"
        )
    log_x_sc = sc2.checkbox("Log X (spins)", value=True, key="time_sc_logx")
    log_y_sc = sc3.checkbox("Log Y (time)", value=True, key="time_sc_logy")

    _sc_cols = list(
        dict.fromkeys(
            [
                "n_spins",
                "mean_time_per_iter",
                sc_color_col,
                "model",
                "h",
                "learning_rate",
                "seed",
            ]
        )
    )
    sc_df = df[_sc_cols].dropna(subset=["n_spins", "mean_time_per_iter", sc_color_col])

    if sc_df.empty:
        st.info("No data for this selection.")
    else:
        fig4 = px.scatter(
            sc_df,
            x="n_spins",
            y="mean_time_per_iter",
            color=sc_color_col,
            labels={
                "n_spins": "Number of spins (N or N²)",
                "mean_time_per_iter": "Mean sampling time / iter (s)",
                sc_color_col: sc_color_label,
            },
            log_x=log_x_sc,
            log_y=log_y_sc,
            hover_data={
                c: True
                for c in ["model", "h", "learning_rate", "seed"]
                if c in sc_df.columns
            },
            height=460,
            opacity=0.75,
        )
        fig4.update_traces(marker=dict(size=7))
        fig4.update_layout(title=_titled("Sampling time scaling with system size"))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Section 5: Time to Epsilon (TTE) ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Time to Epsilon (TTE)")
    st.caption(
        "Estimated wall-clock time to find a solution within **ε** of the exact "
        "ground state energy with 99% probability. "
        "Each group's success rate p(ε) is the fraction of runs with error ≤ ε. "
        "**TTE = t_f × ⌈log(0.01) / log(1 − p(ε))⌉**, "
        "where t_f is the mean total sampling time per run."
    )

    import math as _math

    tte_valid = df[["error_per_spin", "sampling_time_s"]].dropna()
    # Diverged runs can record a literal -inf final_energy, which survives
    # dropna() (it isn't NaN) and breaks the epsilon slider's max_value below.
    tte_valid = tte_valid[tte_valid["error_per_spin"].apply(_math.isfinite)]

    if tte_valid.empty:
        st.info("No runs with both error and timing data available.")
        return

    err_max = float(tte_valid["error_per_spin"].max())
    if err_max == 0.0:
        st.info("All selected runs have zero error — TTE equals t_f for every group.")
        return

    epsilon = st.slider(
        "ε — energy error per spin threshold",
        min_value=0.0,
        max_value=float(round(err_max * 1.1, 5)),
        value=float(round(err_max * 0.5, 5)),
        step=float(max(round(err_max / 200, 6), 1e-6)),
        format="%.5f",
        key="tte_epsilon",
        help=(
            "A run 'succeeds' if |E_RBM − E_exact| / N_spins ≤ ε. "
            "Slide right to accept solutions further from the exact ground state."
        ),
    )

    tte_c1, tte_c2 = st.columns(2)
    with tte_c1:
        tte_grp_col, tte_grp_label = _group_selectbox(
            df, "tte_grp", "Group by", prefer="sampling_method"
        )
    log_tte = tte_c2.checkbox("Log Y", value=False, key="tte_logy")

    p_star = 0.99
    tte_rows: list[dict] = []
    skipped: list[str] = []

    for grp_val, grp_df in df.groupby(tte_grp_col):
        valid = grp_df[["error_per_spin", "sampling_time_s"]].dropna()
        if valid.empty:
            continue
        p_eps = float((valid["error_per_spin"] <= epsilon).mean())
        t_f = float(valid["sampling_time_s"].mean())
        if p_eps == 0.0:
            skipped.append(str(grp_val))
            continue
        n_runs = (
            1
            if p_eps >= 1.0
            else _math.ceil(_math.log(1 - p_star) / _math.log(1 - p_eps))
        )
        tte_rows.append(
            {
                tte_grp_col: str(grp_val),
                "TTE (s)": float(t_f * n_runs),
                "p(ε)": p_eps,
                "Runs needed": n_runs,
                "Mean t_f (s)": t_f,
                "n (runs)": len(valid),
            }
        )

    if skipped:
        st.caption(
            f"Groups with p(ε) = 0 at ε = {epsilon:.5f} (excluded): "
            + ", ".join(skipped)
        )

    if not tte_rows:
        st.info(
            f"No group achieved error ≤ {epsilon:.5f}. Increase ε to see TTE values."
        )
    else:
        tte_df = pd.DataFrame(tte_rows)

        fig5 = px.bar(
            tte_df,
            x=tte_grp_col,
            y="TTE (s)",
            color=tte_grp_col,
            text=tte_df["p(ε)"].map(lambda x: f"p={x:.2f}"),
            labels={tte_grp_col: tte_grp_label, "TTE (s)": "TTE (s)"},
            hover_data={
                "p(ε)": True,
                "Runs needed": True,
                "Mean t_f (s)": True,
                "n (runs)": True,
            },
            height=420,
        )
        fig5.update_traces(textposition="outside")
        if log_tte:
            fig5.update_yaxes(type="log")
        fig5.update_layout(
            showlegend=False,
            title=_titled(f"Time to Epsilon  (ε = {epsilon:.5f},  p* = 99%)"),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.dataframe(
            tte_df.rename(columns={tte_grp_col: tte_grp_label}),
            column_config={
                "TTE (s)": st.column_config.NumberColumn("TTE (s)", format="%.2f"),
                "p(ε)": st.column_config.NumberColumn("p(ε)", format="%.3f"),
                "Mean t_f (s)": st.column_config.NumberColumn(
                    "Mean t_f (s)", format="%.2f"
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ── TTE scaling with system size ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("TTE scaling with system size")
    st.caption(
        "X axis: number of spins — N for 1D models, N² for 2D models. "
        "Each point is a (group, size) pair; TTE is computed from all runs at that size. "
        "A power-law relationship appears as a straight line on log–log axes."
    )

    tte_sc_c1, tte_sc_c2, tte_sc_c3 = st.columns(3)
    with tte_sc_c1:
        tte_sc_color_col, tte_sc_color_label = _group_selectbox(
            df, "tte_sc_color", "Color by", prefer="sampling_method"
        )
    log_x_tte_sc = tte_sc_c2.checkbox("Log X (spins)", value=True, key="tte_sc_logx")
    log_y_tte_sc = tte_sc_c3.checkbox("Log Y (TTE)", value=False, key="tte_sc_logy")

    tte_sc_rows: list[dict] = []
    for (grp_val, n_sp), sub_df in df.groupby([tte_sc_color_col, "n_spins"]):
        valid = sub_df[["error_per_spin", "sampling_time_s"]].dropna()
        if valid.empty:
            continue
        p_eps = float((valid["error_per_spin"] <= epsilon).mean())
        t_f = float(valid["sampling_time_s"].mean())
        if p_eps == 0.0:
            continue
        n_runs = (
            1
            if p_eps >= 1.0
            else _math.ceil(_math.log(1 - p_star) / _math.log(1 - p_eps))
        )
        tte_sc_rows.append(
            {
                tte_sc_color_col: str(grp_val),
                "n_spins": int(n_sp),
                "TTE (s)": float(t_f * n_runs),
                "p(ε)": p_eps,
                "n (runs)": len(valid),
            }
        )

    if not tte_sc_rows:
        st.info(f"No group/size combination achieved error ≤ {epsilon:.5f}.")
    else:
        tte_sc_df = pd.DataFrame(tte_sc_rows)
        fig6 = px.scatter(
            tte_sc_df,
            x="n_spins",
            y="TTE (s)",
            color=tte_sc_color_col,
            labels={
                "n_spins": "Number of spins (N or N²)",
                "TTE (s)": "TTE (s)",
                tte_sc_color_col: tte_sc_color_label,
            },
            log_x=log_x_tte_sc,
            log_y=log_y_tte_sc,
            hover_data={"p(ε)": True, "n (runs)": True},
            height=460,
            opacity=0.8,
        )
        fig6.update_traces(marker=dict(size=9))
        fig6.update_layout(
            title=_titled(f"TTE scaling with system size  (ε = {epsilon:.5f})")
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── Section 6: Watt-Hours to Solution (WHS) ─────────────────────────────
    st.markdown("---")
    st.subheader("Watt-Hours to Solution (WHS)")
    st.caption(
        "Estimated GPU energy consumption to find a solution within **ε** of the "
        "exact ground state energy with 99% probability — the energy-consumption "
        "analogue of Time to Epsilon above. "
        "**WHS = Wh_f × ⌈log(0.01) / log(1 − p(ε))⌉**, where Wh_f is each group's "
        "mean GPU energy (watt-hours) per run, measured by polling `nvidia-smi` "
        "power draw during training. Runs without GPU energy data (CPU-only "
        "machines, or nvidia-smi unavailable) are excluded."
    )

    if "gpu_energy_wh" not in df.columns:
        st.info("No GPU energy data recorded for any run in this result set.")
        return

    whs_valid = df[["error_per_spin", "gpu_energy_wh"]].dropna()
    whs_valid = whs_valid[whs_valid["error_per_spin"].apply(_math.isfinite)]
    if whs_valid.empty:
        st.info("No runs with both error and GPU energy data available.")
        return

    whs_c1, whs_c2 = st.columns(2)
    with whs_c1:
        whs_grp_col, whs_grp_label = _group_selectbox(
            df, "whs_grp", "Group by", prefer="sampling_method"
        )
    log_whs = whs_c2.checkbox("Log Y", value=False, key="whs_logy")

    whs_rows: list[dict] = []
    whs_skipped: list[str] = []

    for grp_val, grp_df in df.groupby(whs_grp_col):
        valid = grp_df[["error_per_spin", "gpu_energy_wh"]].dropna()
        if valid.empty:
            continue
        p_eps = float((valid["error_per_spin"] <= epsilon).mean())
        wh_f = float(valid["gpu_energy_wh"].mean())
        if p_eps == 0.0:
            whs_skipped.append(str(grp_val))
            continue
        n_runs = (
            1
            if p_eps >= 1.0
            else _math.ceil(_math.log(1 - p_star) / _math.log(1 - p_eps))
        )
        whs_rows.append(
            {
                whs_grp_col: str(grp_val),
                "WHS (Wh)": float(wh_f * n_runs),
                "p(ε)": p_eps,
                "Runs needed": n_runs,
                "Mean Wh_f": wh_f,
                "n (runs)": len(valid),
            }
        )

    if whs_skipped:
        st.caption(
            f"Groups with p(ε) = 0 at ε = {epsilon:.5f} (excluded): "
            + ", ".join(whs_skipped)
        )

    if not whs_rows:
        st.info(
            f"No group achieved error ≤ {epsilon:.5f}. Increase ε to see WHS values."
        )
        return

    whs_df = pd.DataFrame(whs_rows)

    fig7 = px.bar(
        whs_df,
        x=whs_grp_col,
        y="WHS (Wh)",
        color=whs_grp_col,
        text=whs_df["p(ε)"].map(lambda x: f"p={x:.2f}"),
        labels={whs_grp_col: whs_grp_label, "WHS (Wh)": "WHS (Wh)"},
        hover_data={
            "p(ε)": True,
            "Runs needed": True,
            "Mean Wh_f": True,
            "n (runs)": True,
        },
        height=420,
    )
    fig7.update_traces(textposition="outside")
    if log_whs:
        fig7.update_yaxes(type="log")
    fig7.update_layout(
        showlegend=False,
        title=_titled(f"Watt-Hours to Solution  (ε = {epsilon:.5f},  p* = 99%)"),
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.dataframe(
        whs_df.rename(columns={whs_grp_col: whs_grp_label}),
        column_config={
            "WHS (Wh)": st.column_config.NumberColumn("WHS (Wh)", format="%.4f"),
            "p(ε)": st.column_config.NumberColumn("p(ε)", format="%.3f"),
            "Mean Wh_f": st.column_config.NumberColumn("Mean Wh_f", format="%.4f"),
        },
        use_container_width=True,
        hide_index=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="VMC Results", layout="wide", page_icon="⚛")
    st.title("VMC / RBM Experiment Results")

    with st.spinner("Loading results..."):
        df_all, histories = load_all_runs(tuple(RESULTS_DIRS))

    if df_all.empty:
        st.error(f"No JSON result files found under {RESULTS_DIRS}")
        st.stop()

    df = build_sidebar(df_all)

    if df.empty:
        st.warning("No runs match the current filters.")
        st.stop()

    _reference_energy_panel(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Run table",
            "Convergence curves",
            "Comparison",
            "Correlation",
            "Timing",
        ]
    )

    with tab1:
        tab_table(df)
    with tab2:
        tab_curves(df, histories)
    with tab3:
        tab_compare(df)
    with tab4:
        tab_correlation(df)
    with tab5:
        tab_timing(df, histories)


if __name__ == "__main__":
    main()
