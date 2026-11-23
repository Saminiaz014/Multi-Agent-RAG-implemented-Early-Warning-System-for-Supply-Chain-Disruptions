"""Page 2 — Analysis view.

Audience: the thesis author. A scrollable research page exposing every metric
the Phase 9a evaluation suite computes (all 8 metrics from
``evaluation_results.json``), raw per-agent scores, per-day SHAP values,
hand-tuned vs optimized comparisons, and the interactive time-range explorer.
Section 10 is the exception to the synthetic year: it reads the real per-day
series from ``data/live/`` for any region and date range, and ``?view=cases``
opens the page on that section alone.
Every chart has an "Export as JPEG" button that flattens chart + title +
caption into one print-ready image (white background) for the thesis document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.core import (  # noqa: E402
    ACTIONS,
    AGENT_COLORS,
    ALL_AGENTS,
    AVAILABLE_REGIONS,
    DASH_CSS,
    FULL_CONFIG_LABEL,
    RANGE_PRESETS,
    SCENARIOS,
    TIMELINE_AXIS_NOTE,
    analysis_bundle,
    composite_series,
    compute_timeseries,
    end_date_text,
    export_button,
    get_news,
    get_shap_assets,
    live_shap_values,
    load_eval_results,
    load_live_timeseries,
    preset_to_range,
    resolve_mode_params,
    risk_tier,
    split_date_text,
    timeline_dates,
)

_TOTAL_DAYS = 365
_MODE_COLORS = {"hand_tuned": "#1f77b4", "optimized": "#ff7f0e"}
_PROCESSED = _PROJECT_ROOT / "data" / "processed"

#: Held-out real-data disruption checks (Section 7). These are the empirical
#: results of the four historical-event tests in tests/test_scenarios.py, run
#: against the real Shuaiba/FRED CSV record — disruptions never seen during
#: synthetic tuning. Reproduce with: pytest tests/test_scenarios.py -v
_HELD_OUT_CASES = [
    {"Case": "2026 Hormuz shutdown (Mar–May 2026)", "Criterion": "≥25% days HIGH/CRITICAL, peak ≥0.60",
     "Result": "37% escalated (31/83 days), peak composite 0.92", "Pass": "✓"},
    {"Case": "2019 tanker attacks (Jun–Jul 2019)", "Criterion": "≥5 days at MEDIUM+",
     "Result": "20 elevated days, mean composite 0.340", "Pass": "✓"},
    {"Case": "Normal period (Jan–Jun 2023)", "Criterion": "≥50% LOW, ≤5% CRITICAL",
     "Result": "63% LOW (114/181 days), 0% CRITICAL", "Pass": "✓"},
    {"Case": "COVID impact (Mar–Apr 2020)", "Criterion": "≥5 days at MEDIUM+",
     "Result": "25 elevated days, mean composite 0.391", "Pass": "✓"},
]


def _bar_pair(categories, ht_vals, opt_vals, ytitle):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=ht_vals, name="hand-tuned",
                         marker_color=_MODE_COLORS["hand_tuned"]))
    fig.add_trace(go.Bar(x=categories, y=opt_vals, name="optimized",
                         marker_color=_MODE_COLORS["optimized"]))
    fig.update_layout(height=330, template="plotly_dark", barmode="group",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#10151d",
                      margin=dict(l=10, r=10, t=10, b=10), yaxis_title=ytitle,
                      legend=dict(orientation="h", y=1.12))
    return fig


def render() -> None:
    st.markdown(DASH_CSS, unsafe_allow_html=True)
    if st.query_params.get("view") == "cases":
        # Deep link to the real-data explorer alone, skipping the
        # synthetic-year sections and their model fitting.
        st.markdown("## Analysis view — real-data case explorer")
        _section_10_cases()
        return
    st.markdown("## Analysis view — thesis evaluation evidence")
    st.caption(
        "All numbers derive from `data/processed/evaluation_results.json` (Phase 9a suite) "
        "and the same seed-44 held-out test split. Every chart exports as a flattened JPEG "
        "(chart + title + caption, print-ready white background) for the thesis document."
    )

    results = load_eval_results()
    if not results:
        st.error("`data/processed/evaluation_results.json` not found — run "
                 "`python notebooks/evaluation.py` first.")
        return

    _section_1_detection(results)
    _section_2_faithfulness(results)
    _section_3_diversity(results)
    _section_4_baseline(results)
    _section_5_optimization(results)
    _section_6_rag(results)
    _section_7_generalization(results)
    _section_8_decision(results)
    _section_9_explorer()
    _section_10_cases()


# ---------------------------------------------------------------------------
# Sections 1–8 (evaluation_results.json)
# ---------------------------------------------------------------------------


def _section_1_detection(results: dict) -> None:
    st.markdown("### 1 · Detection performance")
    m1 = results.get("metric_1_detection", {})
    ht, opt = m1.get("hand_tuned", {}), m1.get("optimized", {})

    sys_rows = []
    for mode, block in (("hand_tuned", ht), ("optimized", opt)):
        s = block.get("system", {})
        sys_rows.append({"mode": mode, **{k: round(float(s.get(k, 0)), 4)
                        for k in ("precision", "recall", "f1", "fpr", "lead_time_days")}})
    st.dataframe(pd.DataFrame(sys_rows), use_container_width=True, hide_index=True)

    agents = list(ht.get("per_agent", {}).keys())
    fig = _bar_pair(
        [a.replace("_", " ") for a in agents],
        [ht["per_agent"][a]["f1"] for a in agents],
        [opt.get("per_agent", {}).get(a, {}).get("f1", 0) for a in agents],
        "F1",
    )
    st.plotly_chart(fig, use_container_width=True)
    export_button(fig, "m1_per_agent_f1", "Per-agent detection F1 (test split)",
                  "F1 of each agent's 0.50-cutoff alerts vs ground truth; "
                  "hand-tuned vs Optuna-optimized weights. METRIC 1, Phase 9a suite.")


def _section_2_faithfulness(results: dict) -> None:
    st.markdown("### 2 · Explainability faithfulness")
    m2 = results.get("metric_2_faithfulness", {}).get("faithfulness", {})
    fig = _bar_pair(["faithfulness"], [m2.get("hand_tuned", 0)], [m2.get("optimized", 0)],
                    "SHAP top-3 faithfulness")
    fig.add_hline(y=0.8, line=dict(color="#8d99a8", dash="dash"),
                  annotation_text="0.80 target", annotation_font_color="#8d99a8")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, "m2_faithfulness", "SHAP explanation faithfulness",
                      "Fraction of SHAP top-3 features deviating >1.5σ from the "
                      "non-disruption baseline on disruption days. Target >0.8. METRIC 2.")
    with c2:
        # SHAP comparison figures generated by the Phase 4 depth methods.
        shown = False
        for png, cap in (("shap_comparison_waterfall.png",
                          "Hand-tuned vs optimized SHAP waterfall (peak-disruption day; legacy fixture plot)"),
                         ("shap_comparison_importance.png",
                          "Mean |SHAP| feature importance, both weight modes (legacy fixture plot)")):
            p = _PROCESSED / png
            if p.exists():
                st.image(str(p), caption=cap, use_container_width=True)
                shown = True
        if not shown:
            st.caption("SHAP comparison PNGs not found. They were fixture plots from "
                       "`generate_comparison_plot()`; no script regenerates them now.")

    # Per-day SHAP explorer (raw values — belongs here, not on Page 1).
    with st.expander("Per-day SHAP waterfall (interactive)"):
        day = st.slider("Day", 1, _TOTAL_DAYS,
                        int(st.session_state.get("analysis_focus_day", 155)))
        features_df, _, explainer = get_shap_assets()
        res = explainer.explain(features_df.iloc[[min(day - 1, len(features_df) - 1)]])
        ordered = sorted(res["shap_values"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        import plotly.graph_objects as go

        wf = go.Figure(go.Bar(
            x=[v for _, v in ordered][::-1],
            y=[k.replace("_", " ") for k, _ in ordered][::-1],
            orientation="h",
            marker_color=["#d64545" if v > 0 else "#1f77b4" for _, v in ordered][::-1],
        ))
        wf.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="SHAP value")
        st.plotly_chart(wf, use_container_width=True)
        export_button(wf, f"m2_shap_day_{day}", f"SHAP feature attribution — day {day}",
                      "Top-10 SHAP values from the 17-feature RandomForest surrogate "
                      "(R² > 0.99 vs the live pipeline's composite risk).")
        _render_alert_explanation(features_df, res["shap_values"], day)


def _render_alert_explanation(features_df: pd.DataFrame, shap_values: dict, day: int) -> None:
    """Per-alert sentences for ``day``: what each driver measures, whether it is
    high or low, its z-value, and how long and since when it has been so."""
    from src.explainability.alert_explainer import explain_alert

    t = min(day - 1, len(features_df) - 1)
    # Same date axis as the Decision view, so a date here matches the chart there.
    dated = features_df.copy()
    dated.insert(0, "timestamp", timeline_dates(_TOTAL_DAYS)[: len(dated)])
    explanation = explain_alert(dated, shap_values, t, region="dashboard")

    st.markdown("**Per-alert explanation — leading drivers**")
    for sentence in explanation["sentences"]:
        st.markdown(f"- {sentence}")
    st.caption(
        "z compares the day's value with its own 30 preceding days; beyond ±2 it reads "
        "as high or low. Duration counts back at most 15 days. " + TIMELINE_AXIS_NOTE
    )


def _section_3_diversity(results: dict) -> None:
    st.markdown("### 3 · Agent diversity (6 vs 2 vs 1 agents)")
    m3 = results.get("metric_3_agent_diversity", {})
    cfgs = ["6-agent", "2-agent", "1-agent"]      # keys of the stored evaluation results
    rows = []
    for mode in ("hand_tuned", "optimized"):
        for cfg in cfgs:
            d = m3.get(mode, {}).get(cfg, {})
            rows.append({"mode": mode, "config": cfg,
                         "f1": round(float(d.get("f1", 0)), 4),
                         "lead_time_days": round(float(d.get("lead_time_days", 0)), 2),
                         "fpr": round(float(d.get("fpr", 0)), 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    fig = _bar_pair(cfgs,
                    [m3.get("hand_tuned", {}).get(c, {}).get("f1", 0) for c in cfgs],
                    [m3.get("optimized", {}).get(c, {}).get("f1", 0) for c in cfgs], "F1")
    st.plotly_chart(fig, use_container_width=True)
    export_button(fig, "m3_diversity", "Agent-diversity value: 6 vs 2 vs 1 agents",
                  "System F1 on the test split as the agent roster is ablated via the "
                  "inter-agent weight mask. KEY THESIS FINDING — METRIC 3.")


def _section_4_baseline(results: dict) -> None:
    st.markdown("### 4 · Baseline comparison")
    m4 = results.get("metric_4_baseline", {})
    rows, names = [], [("naive_baseline", "naive 2σ"), ("hand_tuned", "hand-tuned"),
                       ("optimized", "optimized")]
    for key, label in names:
        d = m4.get(key, {})
        rows.append({"approach": label, "f1": round(float(d.get("f1", 0)), 4),
                     "lead_time_days": round(float(d.get("lead_time_days", 0)), 2),
                     "fpr": round(float(d.get("fpr", 0)), 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=[r["approach"] for r in rows], y=[r["f1"] for r in rows],
        marker_color=["#5a6b80", _MODE_COLORS["hand_tuned"], _MODE_COLORS["optimized"]],
    ))
    fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="F1")
    st.plotly_chart(fig, use_container_width=True)
    export_button(fig, "m4_baseline", "Multi-agent system vs naive threshold baseline",
                  "F1 of a naive any-feature-2σ flag vs the weighted multi-agent "
                  "composite, test split. METRIC 4.")


def _section_5_optimization(results: dict) -> None:
    st.markdown("### 5 · Weight-optimization impact")
    m5 = results.get("metric_5_optimization_impact", {})
    if not m5.get("available"):
        st.info("optimization_results.json not available.")
        return
    deltas = m5.get("test_deltas", {})
    st.dataframe(pd.DataFrame([
        {"metric": "F1", "hand_tuned": m5["hand_tuned_test"].get("f1"),
         "optimized": m5["optimized_test"].get("f1"), "delta": deltas.get("f1")},
        {"metric": "lead_time_days", "hand_tuned": m5["hand_tuned_test"].get("lead_time_days"),
         "optimized": m5["optimized_test"].get("lead_time_days"),
         "delta": deltas.get("lead_time_days")},
        {"metric": "FPR", "hand_tuned": m5["hand_tuned_test"].get("fpr"),
         "optimized": m5["optimized_test"].get("fpr"), "delta": deltas.get("fpr")},
        {"metric": "objective", "hand_tuned": m5["hand_tuned_test"].get("objective"),
         "optimized": m5["optimized_test"].get("objective"), "delta": deltas.get("objective")},
    ]).round(4), use_container_width=True, hide_index=True)
    st.caption("Optimization deliberately trades ≈0.02 raw F1 for +1.7 days of lead time — "
               "the blended objective (0.5·F1 + 0.3·lead − 0.2·FPR) improves by +0.088.")

    shifts = m5.get("top5_parameter_shifts", [])
    if shifts:
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=[s["abs_delta"] for s in shifts][::-1],
            y=[s["parameter"] for s in shifts][::-1],
            orientation="h", marker_color="#9467bd",
        ))
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="|Δ| hand-tuned → optimized")
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, "m5_param_shifts", "Top-5 parameters shifted by Optuna",
                      "Largest absolute weight/threshold movements between the hand-tuned "
                      "and optimized configurations. KEY THESIS FINDING — METRIC 5.")


def _section_6_rag(results: dict) -> None:
    st.markdown("### 6 · RAG retrieval relevance")
    m6 = results.get("metric_6_rag_relevance", {})
    c1, c2 = st.columns(2)
    c1.metric("Overall relevance", f"{float(m6.get('overall_relevance', 0)):.3f}",
              help="Target > 0.70")
    c2.metric("Mean top-1 similarity", f"{float(m6.get('mean_similarity', 0)):.3f}",
              help="Target > 0.60")
    per = m6.get("per_scenario", [])
    if per:
        st.dataframe(pd.DataFrame(per)[
            ["scenario", "top_match", "similarity", "relevant", "matched_agents", "expected_agents"]
        ], use_container_width=True, hide_index=True)
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=[p["scenario"] for p in per], y=[p["similarity"] for p in per],
            marker_color="#17becf",
        ))
        fig.add_hline(y=0.6, line=dict(color="#8d99a8", dash="dash"),
                      annotation_text="0.60 target", annotation_font_color="#8d99a8")
        fig.update_layout(height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="top-1 cosine similarity")
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, "m6_rag", "RAG retrieval quality per evaluation scenario",
                      "Top-1 similarity of the retrieved historical case per labelled "
                      "signal scenario; relevance = agent-domain overlap. METRIC 6.")


def _section_7_generalization(results: dict) -> None:
    st.markdown("### 7 · Generalization check")
    m7 = results.get("metric_7_generalization", {})
    val, test = m7.get("validation", {}), m7.get("test", {})
    metrics = ["f1", "precision", "recall", "lead_time_days"]
    fig = _bar_pair([m.replace("_", " ") for m in metrics],
                    [float(val.get(m, 0)) for m in metrics],
                    [float(test.get(m, 0)) for m in metrics], "score")
    fig.data[0].name, fig.data[1].name = "validation (seed 43)", "test (seed 44)"
    st.plotly_chart(fig, use_container_width=True)
    export_button(fig, "m7_generalization", "Validation vs held-out test (optimized weights)",
                  "Optimized weights evaluated on the tuning (validation) and held-out "
                  "(test) realisations; a large gap would indicate overfitting. METRIC 7.")
    flag = m7.get("overfit_flag", False)
    st.markdown(("⚠ **Possible overfitting** — test F1 is >0.05 below validation."
                 if flag else
                 "✓ **No overfitting detected** — test performance is consistent with "
                 f"validation (F1 gap {float(m7.get('f1_val_minus_test', 0)):+.3f})."))

    st.markdown("**Held-out real-world disruption checks** — disruptions never seen "
                "during tuning, evaluated on the real Shuaiba/FRED CSV record:")
    st.dataframe(pd.DataFrame(_HELD_OUT_CASES), use_container_width=True, hide_index=True)
    st.caption("Source: the four historical-event tests in `tests/test_scenarios.py` "
               "(reproduce with `pytest tests/test_scenarios.py -v`).")


def _section_8_decision(results: dict) -> None:
    st.markdown("### 8 · Decision effectiveness (SRQ5)")
    st.caption("The same rule set that powers the Decision view's recommendation badge, "
               "evaluated in full here.")
    m8 = results.get("metric_8_decision_effectiveness", {})
    rows = []
    for mode in ("hand_tuned", "optimized"):
        d = m8.get(mode, {})
        rows.append({
            "mode": mode,
            "overall_accuracy": d.get("overall_accuracy"),
            "per_case_accuracy": d.get("per_case_accuracy"),
            **{f"scenario_{k}": v for k, v in d.get("per_scenario_accuracy", {}).items()},
            "baseline_accuracy": d.get("baseline_accuracy"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    cm = m8.get("hand_tuned", {}).get("confusion_matrix", {})
    if cm:
        import plotly.graph_objects as go

        z = [[cm.get(a, {}).get(b, 0) for b in ACTIONS] for a in ACTIONS]
        fig = go.Figure(go.Heatmap(
            z=z, x=ACTIONS, y=ACTIONS, colorscale="Blues",
            text=z, texttemplate="%{text}", showscale=False,
        ))
        fig.update_layout(height=340, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="predicted action", yaxis_title="correct action",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, "m8_confusion", "Decision-effectiveness confusion matrix",
                      "Predicted vs ground-truth actions over the 365-day synthetic year, "
                      "hand-tuned mode. Overall accuracy 0.866 vs naive baseline 0.797. "
                      "KEY THESIS FINDING — METRIC 8 / SRQ5.")


# ---------------------------------------------------------------------------
# Section 9 — interactive time-range explorer
# ---------------------------------------------------------------------------


def _section_9_explorer() -> None:
    import plotly.graph_objects as go

    st.markdown("### 9 · Interactive time-range explorer")
    st.caption(
        "Presets map onto **trailing day-index windows of the 365-day synthetic year** "
        "(e.g. “Last 30 days” = days 336–365) — the data is the seed-44 test realisation, "
        "not calendar time."
    )

    c1, c2, c3 = st.columns([1.4, 1.6, 1])
    preset = c1.selectbox("Time range", RANGE_PRESETS, index=3)
    custom = None
    if preset == "Custom range":
        custom = c2.slider("Custom day range", 1, _TOTAL_DAYS, (100, 200))
    start, end = preset_to_range(preset, custom)
    c3.markdown(f"<br>**days {start + 1} – {end + 1}**", unsafe_allow_html=True)

    if c3.button("Recompute for this range", type="primary"):
        st.session_state["analysis_range"] = (start, end)
    if "analysis_range" not in st.session_state:
        st.session_state["analysis_range"] = (0, _TOTAL_DAYS - 1)

    a_start, a_end = st.session_state["analysis_range"]
    bundle = analysis_bundle(a_start, a_end)
    modes = bundle["modes"]
    eval_json = load_eval_results()
    baseline_acc = (eval_json.get("metric_8_decision_effectiveness", {})
                    .get("hand_tuned", {}).get("baseline_accuracy", 0.0))
    st.markdown(f"**Showing days {a_start + 1} – {a_end + 1}** ({a_end - a_start + 1} days)")

    # 9a — rolling detection
    fig = go.Figure()
    for mode, res in modes.items():
        r = res["rolling"]
        fig.add_trace(go.Scatter(x=r["day"], y=r["f1"], name=f"F1 · {mode}",
                                 line=dict(color=_MODE_COLORS[mode], width=2)))
        fig.add_trace(go.Scatter(x=r["day"], y=r["precision"], name=f"precision · {mode}",
                                 line=dict(color=_MODE_COLORS[mode], width=1, dash="dot"),
                                 opacity=0.55))
        fig.add_trace(go.Scatter(x=r["day"], y=r["recall"], name=f"recall · {mode}",
                                 line=dict(color=_MODE_COLORS[mode], width=1, dash="dash"),
                                 opacity=0.55))
    fig.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="day", yaxis_title="score", hovermode="x unified",
                      legend=dict(orientation="h", y=1.15, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)
    export_button(fig, f"exp_detection_{a_start}_{a_end}",
                  f"Rolling detection performance — days {a_start + 1}–{a_end + 1}",
                  "30-day rolling precision / recall / F1 of HIGH-risk alerts vs ground "
                  "truth, hand-tuned vs optimized.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Lead time per scenario")
        scen = sorted({k for res in modes.values() for k in res["leads"]})
        if scen:
            fig = go.Figure()
            for mode, res in modes.items():
                fig.add_trace(go.Bar(x=[s.split(" — ")[0] for s in scen],
                                     y=[res["leads"].get(s, 0.0) for s in scen],
                                     name=mode, marker_color=_MODE_COLORS[mode]))
            fig.update_layout(height=290, template="plotly_dark", barmode="group",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#10151d",
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="lead (days, cap 5)")
            st.plotly_chart(fig, use_container_width=True)
            export_button(fig, f"exp_lead_{a_start}_{a_end}",
                          f"Early-warning lead time — days {a_start + 1}–{a_end + 1}",
                          "Days between the first MEDIUM alert and each scenario onset.")
        else:
            st.info("No scenario window falls inside the selected range.")

    with col_b:
        st.markdown("##### Decision-effectiveness accuracy")
        fig = go.Figure()
        days_axis = list(range(a_start + 1, a_end + 2))
        for mode, res in modes.items():
            fig.add_trace(go.Scatter(x=days_axis, y=res["decision_rolling"], name=mode,
                                     line=dict(color=_MODE_COLORS[mode], width=2)))
        if baseline_acc:
            fig.add_hline(y=baseline_acc, line=dict(color="#8d99a8", width=1.4, dash="dash"),
                          annotation_text=f"naive baseline {baseline_acc:.3f}",
                          annotation_font_color="#8d99a8")
        fig.update_layout(height=290, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="day", yaxis_title="action accuracy")
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, f"exp_decision_{a_start}_{a_end}",
                      f"Decision-effectiveness accuracy — days {a_start + 1}–{a_end + 1}",
                      "30-day rolling accuracy of predicted vs correct actions with the "
                      "naive-baseline reference. METRIC 8 / SRQ5.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("##### Agent diversity within range")
        cfgs = [FULL_CONFIG_LABEL, "2-agent", "1-agent"]
        fig = go.Figure()
        for mode, res in modes.items():
            fig.add_trace(go.Bar(x=cfgs, y=[res["diversity"][c] for c in cfgs],
                                 name=mode, marker_color=_MODE_COLORS[mode]))
        fig.update_layout(height=290, template="plotly_dark", barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#10151d",
                          margin=dict(l=10, r=10, t=10, b=10), yaxis_title="F1")
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, f"exp_diversity_{a_start}_{a_end}",
                      f"Agent diversity — days {a_start + 1}–{a_end + 1}",
                      f"F1 within the selected range as the roster is ablated {len(ALL_AGENTS)} → 2 → 1.")

    with col_d:
        st.markdown("##### RAG similarity through the range")
        rag = bundle["rag"]
        fig = go.Figure()
        if rag["day"]:
            fig.add_trace(go.Scatter(x=rag["day"], y=rag["similarity"], name="top-1 sim",
                                     line=dict(color="#17becf", width=2)))
            fig.add_hline(y=0.6, line=dict(color="#8d99a8", width=1.2, dash="dash"),
                          annotation_text="0.60 target", annotation_font_color="#8d99a8")
        fig.update_layout(height=290, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="day", yaxis_title="cosine similarity")
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, f"exp_rag_{a_start}_{a_end}",
                      f"RAG top-1 similarity — days {a_start + 1}–{a_end + 1}",
                      f"Sampled every {rag['stride']} day(s) against the static case base.")

    # Raw agent signal timeline (research view of the Page 1 trend).
    with st.expander(f"Raw {len(ALL_AGENTS)}-agent signal timeline (full year)"):
        ts = compute_timeseries("hand_tuned")
        params = resolve_mode_params("hand_tuned")
        risk = composite_series(ts, params, set(ALL_AGENTS))
        fig = go.Figure()
        for name in ALL_AGENTS:
            fig.add_trace(go.Scatter(x=ts["day"], y=ts[name], name=name.replace("_", " "),
                                     line=dict(color=AGENT_COLORS[name], width=1.1),
                                     opacity=0.75))
        fig.add_trace(go.Scatter(x=ts["day"], y=risk, name="composite",
                                 line=dict(color="#dfe6ee", width=3)))
        for label, (s, e) in SCENARIOS.items():
            fig.add_vrect(x0=s + 1, x1=e + 1, fillcolor="#8d99a8", opacity=0.1, line_width=0,
                          annotation_text=label.split(" — ")[0],
                          annotation_font_color="#8d99a8")
        fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=26, b=10),
                          hovermode="x unified",
                          legend=dict(orientation="h", y=1.12, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)
        export_button(fig, "raw_timeline",
                      f"{len(ALL_AGENTS)}-agent anomaly signals and composite risk",
                      "Per-agent anomaly scores and the agreement-amplified composite over "
                      "the 365-day test realisation; shaded bands are injected scenarios.")

    # Summary + CSV export
    st.markdown("##### Range summary")
    rows = []
    for mode, res in modes.items():
        rows.append({
            "mode": mode,
            "rolling_f1_at_range_end": round(res["range_f1"], 4),
            "decision_accuracy": round(res["decision_accuracy"], 4),
            "baseline_decision_accuracy": round(baseline_acc, 4),
            **{f"f1_{k}": round(v, 4) for k, v in res["diversity"].items()},
            **{f"lead_{k.split(' — ')[0]}": v for k, v in res["leads"].items()},
        })
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download summary as CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluation_summary_days_{a_start + 1}_{a_end + 1}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Section 10 — real-data case explorer
# ---------------------------------------------------------------------------

#: Four-tier band colours for the real-data charts (low → critical).
_TIER_BANDS = (("Low", "#1d4a30"), ("Medium", "#4a4410"), ("High", "#4a2f10"),
               ("Critical", "#4a1d1d"))


def _case_query_params(first, last) -> tuple[str, object, object, object]:
    """Initial region, window and snapshot date from the query string.

    ``?region=<key>&start=YYYY-MM-DD&end=YYYY-MM-DD&date=YYYY-MM-DD``. Each
    is optional and clamped to the series; without them the explorer opens on
    Hormuz, the last year and the latest day.
    """
    def _date(name, default):
        raw = st.query_params.get(name)
        if not raw:
            return default
        try:
            value = pd.Timestamp(str(raw)).date()
        except ValueError:
            return default
        return min(max(value, first), last)

    region = str(st.query_params.get("region") or "hormuz").lower()
    end = _date("end", last)
    start = _date("start", max(first, (pd.Timestamp(end) - pd.Timedelta(days=364)).date()))
    return region, start, end, _date("date", end)


def _tier_shapes(fig, thresholds: dict) -> None:
    """Shade the four risk tiers behind a 0–1 score axis."""
    cuts = [0.0, float(thresholds["risk_medium"]), float(thresholds["risk_high"]),
            float(thresholds["risk_critical"]), 1.02]
    for (label, colour), lo, hi in zip(_TIER_BANDS, cuts[:-1], cuts[1:]):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=colour, opacity=0.45, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=10, color="#8d99a8"))


def _label_spans(frame: pd.DataFrame) -> list[tuple]:
    """Start/end dates of the traffic-drop label's runs inside ``frame``."""
    spans, start, prev = [], None, None
    for date, flag in zip(frame["timestamp"], frame["y_true"].astype(bool)):
        if flag and start is None:
            start = date
        if not flag and start is not None:
            spans.append((start, prev))
            start = None
        prev = date
    if start is not None:
        spans.append((start, prev))
    return spans


def _section_10_cases() -> None:
    import plotly.graph_objects as go

    from src.core.regions import get_region
    from src.explainability.alert_explainer import explain_alert

    st.markdown("### 10 · Real-data case explorer")
    st.caption(
        "The per-day series from `scripts/build_live_risk_series.py`: each agent fitted on "
        f"the days before {split_date_text()} (the evaluation's fixed split date), and the "
        "engine's composite risk score. Bands: Low below "
        "risk_medium, Medium to risk_high, High to risk_critical, Critical above. Grey "
        "columns mark the traffic-drop label (30-day mean transits at least 20 % below the "
        "trailing 365-day median for 14 days or more); the dotted line is the snapshot day."
    )
    regions = list(AVAILABLE_REGIONS.values())
    names = {key: name for name, key in AVAILABLE_REGIONS.items()}
    probe = load_live_timeseries("hormuz")
    if probe is None:
        st.info("No real series yet — run `python scripts/build_live_risk_series.py`.")
        return
    first = probe[0]["timestamp"].iloc[0].date()
    last = probe[0]["timestamp"].iloc[-1].date()
    q_region, q_start, q_end, q_snap = _case_query_params(first, last)

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    region = c1.selectbox("Region", regions, format_func=lambda k: names.get(k, k),
                          index=regions.index(q_region) if q_region in regions else 0,
                          key="case_region")
    start = c2.date_input("From", value=q_start, min_value=first, max_value=last,
                          key="case_start")
    end = c3.date_input("To", value=q_end, min_value=first, max_value=last, key="case_end")
    snap = c4.date_input("Snapshot day", value=q_snap, min_value=first, max_value=last,
                         key="case_date")

    live = load_live_timeseries(region)
    if live is None:
        st.info(f"No real series for {names.get(region, region)}.")
        return
    ts, _ = live
    thr = resolve_mode_params("hand_tuned")["thresholds"]
    mask = (ts["timestamp"] >= pd.Timestamp(start)) & (ts["timestamp"] <= pd.Timestamp(end))
    win = ts.loc[mask]
    if win.empty:
        st.info("The selected range holds no scored days.")
        return
    i = min(int(ts["timestamp"].searchsorted(pd.Timestamp(snap))), len(ts) - 1)
    snap_ts = ts["timestamp"].iloc[i]
    snap_risk = float(ts["risk_score"].iloc[i])
    active = [a for a in ALL_AGENTS if a in get_region(region).active_agents()
              and ts[a].notna().any()]
    title = names.get(region, region)
    span = f"{title}, {pd.Timestamp(start):%d %b %Y} – {pd.Timestamp(end):%d %b %Y}"

    def _layout(fig, height, text):
        fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="#0b0e14",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=44, b=10),
                          title=dict(text=text, font=dict(size=13)), hovermode="x unified",
                          legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1,
                                      xanchor="right", font=dict(size=11)))
        fig.update_xaxes(range=[win["timestamp"].iloc[0], win["timestamp"].iloc[-1]])

    def _marks(fig):
        for s0, s1 in _label_spans(win):
            fig.add_vrect(x0=s0, x1=s1, fillcolor="#8d99a8", opacity=0.18, line_width=0)
        if win["timestamp"].iloc[0] <= snap_ts <= win["timestamp"].iloc[-1]:
            fig.add_vline(x=snap_ts, line=dict(color="#f2f2f2", width=1.2, dash="dot"))

    # 10a — composite risk with the four tier bands
    with st.container(key="case_risk"):
        fig = go.Figure()
        _tier_shapes(fig, thr)
        _marks(fig)
        fig.add_trace(go.Scatter(x=win["timestamp"], y=win["risk_score"],
                                 name="composite risk", line=dict(color="#dfe6ee", width=1.8)))
        _layout(fig, 360, f"Composite risk — {span}")
        fig.update_yaxes(range=[0, 1.02], title="risk score")
        st.plotly_chart(fig, use_container_width=True, key="case_risk_chart")
        tiers = win["risk_score"].apply(lambda v: risk_tier(float(v), thr)).value_counts()
        st.caption(
            f"{len(win)} days · mean {win['risk_score'].mean():.2f} · max "
            f"{win['risk_score'].max():.2f} · "
            + " · ".join(f"{t} {int(tiers.get(t, 0))}" for t, _ in _TIER_BANDS)
            + f" · label days {int(win['y_true'].astype(bool).sum())}. Snapshot "
            f"{snap_ts:%d %b %Y}: {snap_risk:.2f} ({risk_tier(snap_risk, thr)})."
        )

    # 10b — per-agent scores
    with st.container(key="case_agents"):
        fig = go.Figure()
        _marks(fig)
        for a in active:
            fig.add_trace(go.Scatter(x=win["timestamp"], y=win[a], name=a.replace("_", " "),
                                     line=dict(color=AGENT_COLORS[a], width=1.3)))
        _layout(fig, 340, f"Agent anomaly scores — {span}")
        fig.update_yaxes(range=[0, 1.02], title="agent score")
        st.plotly_chart(fig, use_container_width=True, key="case_agents_chart")
        on_day = ", ".join(
            f"{a.replace('_', ' ')} {ts[a].iloc[i]:.2f}" if pd.notna(ts[a].iloc[i])
            else f"{a.replace('_', ' ')} no data" for a in active)
        passive = ", ".join(p.replace("_", " ") for p in get_region(region).passive_agents())
        st.caption(f"Snapshot {snap_ts:%d %b %Y}: {on_day}. Gaps are days without source "
                   f"data. Not run in this region: {passive or 'none'}.")

    # 10c — SHAP drivers on the snapshot day
    with st.container(key="case_shap"):
        shap_values, bundle = live_shap_values(region, i)
        ordered = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        fig = go.Figure(go.Bar(
            x=[v for _, v in ordered][::-1],
            y=[k.replace("_", " ") for k, _ in ordered][::-1], orientation="h",
            marker_color=["#d64545" if v > 0 else "#1f77b4" for _, v in ordered][::-1],
        ))
        fig.update_layout(height=330, template="plotly_dark", paper_bgcolor="#0b0e14",
                          plot_bgcolor="#10151d", margin=dict(l=10, r=10, t=44, b=10),
                          title=dict(text=f"SHAP drivers — {title}, {snap_ts:%d %b %Y}",
                                     font=dict(size=13)),
                          xaxis_title="SHAP value (contribution to the risk score)")
        st.plotly_chart(fig, use_container_width=True, key="case_shap_chart")
        explanation = explain_alert(bundle["features"], shap_values, i, region=region)
        for sentence in explanation["sentences"][:3]:
            st.markdown(f"- {sentence}")
        st.caption(f"Random-forest surrogate on the canonical features of the region's "
                   f"enabled agents, fitted on the days before {split_date_text()}; R² on the "
                   f"days from then through {end_date_text()} {bundle['r2_test']:.2f}. Red raises the "
                   "score, blue lowers it.")

    # 10d — news signal
    with st.container(key="case_news"):
        if "news_sentiment" in active:
            fig = go.Figure()
            _marks(fig)
            fig.add_trace(go.Scatter(x=win["timestamp"], y=win["news_sentiment"],
                                     name="news sentiment",
                                     line=dict(color=AGENT_COLORS["news_sentiment"],
                                               width=1.4)))
            _layout(fig, 280, f"News-sentiment agent — {span}")
            fig.update_yaxes(range=[0, 1.02], title="agent score")
            st.plotly_chart(fig, use_container_width=True, key="case_news_chart")
            leads = int((win["top_agent"] == "news_sentiment").sum())
            st.caption(f"Mean {win['news_sentiment'].mean():.2f}, max "
                       f"{win['news_sentiment'].max():.2f}; dominant on {leads} of "
                       f"{len(win)} days.")
        else:
            st.info(f"No news-sentiment series for {title}: the GDELT query was "
                    "rate-limited (HTTP 429) when the series was built, so the composite "
                    "renormalises over the other agents.")
        for item in get_news(region, limit=4):
            st.markdown(f"- {item['title'][:110]} ({item['date']})")
