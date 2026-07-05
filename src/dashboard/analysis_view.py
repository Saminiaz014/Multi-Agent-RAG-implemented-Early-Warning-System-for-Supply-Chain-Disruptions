"""Page 2 — Analysis view.

Audience: the thesis author. A scrollable research page exposing every metric
the Phase 9a evaluation suite computes (all 8 metrics from
``evaluation_results.json``), raw per-agent scores, per-day SHAP values,
hand-tuned vs optimized comparisons, and the interactive time-range explorer.
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
    DASH_CSS,
    RANGE_PRESETS,
    SCENARIOS,
    analysis_bundle,
    composite_series,
    compute_timeseries,
    export_button,
    get_shap_assets,
    load_eval_results,
    preset_to_range,
    resolve_mode_params,
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
                          "Hand-tuned vs optimized SHAP waterfall (peak-disruption day)"),
                         ("shap_comparison_importance.png",
                          "Mean |SHAP| feature importance, both weight modes")):
            p = _PROCESSED / png
            if p.exists():
                st.image(str(p), caption=cap, use_container_width=True)
                shown = True
        if not shown:
            st.caption("SHAP comparison PNGs not found — generated by "
                       "`generate_comparison_plot()` (tests/test_phase4_depth.py).")

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
                      "Top-10 SHAP values from the 20-feature RandomForest surrogate "
                      "(R² > 0.99 vs the live pipeline's composite risk).")


def _section_3_diversity(results: dict) -> None:
    st.markdown("### 3 · Agent diversity (6 vs 2 vs 1 agents)")
    m3 = results.get("metric_3_agent_diversity", {})
    cfgs = ["6-agent", "2-agent", "1-agent"]
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
        cfgs = ["6-agent", "2-agent", "1-agent"]
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
                      "F1 within the selected range as the roster is ablated 6 → 2 → 1.")

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
    with st.expander("Raw 6-agent signal timeline (full year)"):
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
        export_button(fig, "raw_timeline", "Six-agent anomaly signals and composite risk",
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
