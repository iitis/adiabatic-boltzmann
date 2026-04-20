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
  New solver/run     → drop JSON files in results/ and hit "Reload data"
                       (auto-discovered, zero code changes needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "jax_results"

# ── Extension points ───────────────────────────────────────────────────────────
# Add one dict  → new sidebar filter appears automatically
# col must match a key in the flat DataFrame built by load_all_runs()
FILTER_AXES = [
    {"col": "device", "label": "Device (GPU/CPU)"},
    {"col": "model", "label": "Model"},
    {"col": "size", "label": "System size N"},
    {"col": "h", "label": "Field h"},
    {"col": "sampler", "label": "Sampler backend"},
    {"col": "sampling_method", "label": "Sampling method"},
    {"col": "rbm", "label": "RBM type"},
    {"col": "n_hidden", "label": "Hidden units"},
    {"col": "learning_rate", "label": "Learning rate"},
    {"col": "regularization", "label": "Regularization"},
    {"col": "n_samples", "label": "Samples / iter"},
    {"col": "iterations", "label": "Iterations"},
    {"col": "cem", "label": "CEM"},
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
def load_all_runs(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Scan all JSON files under results_dir.

    Returns
    -------
    df        flat DataFrame — one row per run, columns = config + scalars
    histories {run_id: {metric_key: [values]}}
    """
    records: list[dict] = []
    histories: dict[str, dict] = {}

    for path in sorted(results_dir.rglob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue

        run_id = str(path.relative_to(results_dir))
        cfg = d.get("config", {})

        row: dict = {"run_id": run_id}

        # Config columns — one per FILTER_AXES entry; missing keys → None
        for ax in FILTER_AXES:
            row[ax["col"]] = cfg.get(ax["col"])

        # Scalar outputs
        for key in (
            "final_energy",
            "exact_energy",
            "error",
            "final_ess",
            "mean_ess",
            "final_kl_exact",
            "sampling_time_s",
            "sparsity",
        ):
            row[key] = d.get(key)

        # Device: derive a clean label from the top-level "cuda" dict
        cuda = d.get("cuda")
        if cuda is None:
            row["device"] = "unknown"
        elif not cuda.get("torch_cuda_available", False):
            row["device"] = "cpu"
        else:
            row["device"] = cuda.get("torch_device", "gpu")

        # Derived scalars
        e, ref = row.get("error"), row.get("exact_energy")
        row["relative_error"] = abs(e / ref) * 100 if (e is not None and ref) else None
        n_sp = _n_spins(row.get("model"), row.get("size"))
        row["n_spins"] = n_sp
        row["error_per_spin"] = (e / n_sp) if e is not None else None
        ts, iters = row.get("sampling_time_s"), row.get("iterations")
        row["mean_time_per_iter"] = (ts / iters) if (ts is not None and iters) else None

        records.append(row)
        histories[run_id] = d.get("history", {})

    df = pd.DataFrame(records)

    # Numeric coercion so filter options sort correctly
    for col in (
        "size",
        "h",
        "n_hidden",
        "learning_rate",
        "regularization",
        "n_samples",
        "iterations",
        "seed",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, histories


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
        vals = sorted(
            df[col].dropna().unique(),
            key=lambda x: (str(type(x).__name__), str(x)),
        )
        if len(vals) <= 1:
            continue
        sel = st.sidebar.multiselect(label, vals, default=[], key=f"f_{col}")
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
    c1, c2, c3, c4, c5 = st.columns(5)

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


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="VMC Results", layout="wide", page_icon="⚛")
    st.title("VMC / RBM Experiment Results")

    with st.spinner("Loading results..."):
        df_all, histories = load_all_runs(RESULTS_DIR)

    if df_all.empty:
        st.error(f"No JSON result files found under {RESULTS_DIR}")
        st.stop()

    df = build_sidebar(df_all)

    if df.empty:
        st.warning("No runs match the current filters.")
        st.stop()

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
