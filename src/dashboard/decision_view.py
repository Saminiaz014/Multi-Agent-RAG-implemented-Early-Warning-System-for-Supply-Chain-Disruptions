"""Page 1 — Decision view.

Audience: a mid-level supply-chain manager deciding whether a disruption is
worth acting on. Single viewport, **no scrolling, no raw scores** — risk is
communicated exclusively through the three status words (Critical / High /
Low), a recommended action, and a natural-language explanation.

The assessment day is the most significant risk peak inside the selected
trend window (that peak is what a manager needs to act on); the trend chart,
map, route status list and decision panel all evaluate that same day, and
route selection is shared across the control strip, the status list, and the
map (map clicks sync back via the ``?route=`` query param — see
``core.build_globe_html``).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.core import (  # noqa: E402
    ALL_AGENTS,
    AVAILABLE_REGIONS,
    DASH_CSS,
    STATUS_COLORS,
    STATUS_ICONS,
    build_globe_html,
    classify_level,
    compute_timeseries,
    decide_action,
    generate_risk_narrative,
    get_monitoring_points,
    get_news,
    get_retriever,
    get_routes,
    get_shap_assets,
    get_vessels,
    load_app_config,
    resolve_cesium_token,
    resolve_mode_params,
    route_risk_series,
    select_route,
    status_word,
    sustained_high_flags,
)
from src.evaluation.decision_effectiveness import load_decision_labels  # noqa: E402

_LABELS_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "decision_labels.json"

_WINDOWS = {"30 days": 30, "90 days": 90, "180 days": 180, "Full year": 365}


def _pill(word: str) -> str:
    return (
        f'<span class="status-pill" style="background:{STATUS_COLORS[word]}">'
        f'{STATUS_ICONS[word]} {word}</span>'
    )


def _apply_pending_selection(route_keys: list[str]) -> None:
    """Funnel all selection sources into the route radio *before* it renders.

    Sources: map click (``?route=`` query param) and the status-list row
    buttons (``pending_route`` set on the previous run). The radio widget key
    is then the single source of truth every panel reads.
    """
    pending = st.session_state.pop("pending_route", None)
    qp = st.query_params.get("route")
    for candidate in (pending, qp):
        if candidate in route_keys:
            st.session_state["route_radio"] = candidate
            select_route(candidate)
            break


def render() -> None:
    st.markdown(DASH_CSS, unsafe_allow_html=True)
    config = load_app_config()

    # ---------------- Region selector (always visible, top of page) --------
    head_l, head_r = st.columns([2.2, 1.3])
    with head_l:
        st.markdown("## Supply-chain disruption monitor")
    with head_r:
        # Only Hormuz is populated for the thesis; the underlying fetchers
        # (get_routes / get_news / monitoring points) accept any region key,
        # so post-thesis chokepoints appear here by extending AVAILABLE_REGIONS.
        region_label = st.selectbox("Region", list(AVAILABLE_REGIONS.keys()),
                                    label_visibility="collapsed")
    region = AVAILABLE_REGIONS.get(region_label, "hormuz")

    routes = get_routes(region)
    route_keys = [r["key"] for r in routes]
    _apply_pending_selection(route_keys)
    if "route_radio" not in st.session_state and route_keys:
        st.session_state["route_radio"] = route_keys[0]

    # ---------------- Shared data (default weight mode) --------------------
    params = resolve_mode_params("hand_tuned")
    thresholds = params["thresholds"]
    ts = compute_timeseries("hand_tuned")

    col_left, col_map, col_right = st.columns([1.15, 1.55, 1.15], gap="small")

    # ======================= TOP-LEFT — risk trend =========================
    with col_left:
        w_label = st.radio("Trend window", list(_WINDOWS.keys()), index=3,
                           horizontal=True, label_visibility="collapsed")
        window = _WINDOWS[w_label]

        st.markdown("**Monitored routes**")
        selected_key = st.radio(
            "Monitored routes",
            route_keys,
            format_func=lambda k: next(r["name"] for r in routes if r["key"] == k),
            key="route_radio",
            label_visibility="collapsed",
        ) if route_keys else ""
        select_route(selected_key)
        route = next((r for r in routes if r["key"] == selected_key), None)

        if route is None:
            st.info("No routes are configured for this region yet.")
            return

        series = route_risk_series(ts, params, route)
        lo = max(0, len(series) - window)
        win_days = list(range(lo + 1, len(series) + 1))
        win_vals = series.iloc[lo:].to_numpy()
        peak_pos = int(np.argmax(win_vals))
        peak_day = win_days[peak_pos]                      # 1-based assessment day
        assess_word = status_word(float(series.iloc[peak_day - 1]), thresholds)

        _render_trend_chart(win_days, win_vals, peak_day, thresholds, route["name"])

        # -------- peak explanation (inline, natural language) --------------
        clicked = st.session_state.get("_peak_clicked", False)
        if st.button("Explain the marked peak", key="explain_peak", type="secondary"):
            clicked = True
        if clicked:
            st.session_state["_peak_clicked"] = True
            _render_peak_explanation(ts, peak_day, assess_word, region_label)

    # ======================= assessment context ============================
    day = peak_day
    route_words = {r["key"]: status_word(float(route_risk_series(ts, params, r).iloc[day - 1]),
                                         thresholds) for r in routes}
    features_df, _, explainer = get_shap_assets()
    shap_result = explainer.explain(features_df.iloc[[min(day - 1, len(features_df) - 1)]])
    top_driver = shap_result["top_drivers"][0] if shap_result["top_drivers"] else {"agent": "", "feature": ""}

    # ======================= CENTRE — interactive map ======================
    with col_map:
        points = get_monitoring_points(config)
        focus = points.get(region, {"name": region_label, "lat": 26.56, "lng": 56.25})
        secondary = [{**points[r], "region": r}
                     for r in points if r != region]
        html = build_globe_html(
            token=resolve_cesium_token(),
            focus=focus,
            secondary=secondary,
            risk_level=classify_level(float(series.iloc[day - 1]),
                                      thresholds["risk_high"], thresholds["risk_medium"]),
            top_driver=top_driver.get("feature", "—").replace("_", " "),
            updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            routes=routes,
            vessels=get_vessels(route, day),
            selected_route=selected_key,
            route_status=route_words,
            status=assess_word,
        )
        import streamlit.components.v1 as components

        components.html(html, height=470)
        st.caption(
            f"Routes are colour-coded by status. Click a route on the map to select it; "
            f"vessel markers along **{route['name']}** open detail popups. "
            f"Assessment day: day {day} of the monitored year."
        )

    # ======================= RIGHT column ==================================
    with col_right:
        # ---- Top-right: route status (words only, no numbers) -------------
        st.markdown("**Route status**")
        for r in routes:
            word = route_words[r["key"]]
            sel = " selected" if r["key"] == selected_key else ""
            row_l, row_r = st.columns([2.6, 1.2])
            with row_l:
                st.markdown(
                    f"<div class='route-row{sel}'>{r['name']}<br>{_pill(word)}</div>",
                    unsafe_allow_html=True,
                )
            with row_r:
                if st.button("Select", key=f"sel_{r['key']}",
                             disabled=(r["key"] == selected_key)):
                    st.session_state["pending_route"] = r["key"]
                    st.rerun()

        # ---- Bottom-right: decision support + news -------------------------
        st.markdown("**Recommended action**")
        _render_decision(ts, params, route, day, top_driver)

        st.markdown("**Regional news**")
        scope = st.radio("News scope", ["This route", "All regions"], horizontal=True,
                         label_visibility="collapsed")
        items = get_news(region, all_regions=(scope == "All regions"), limit=4)
        if items:
            for it in items:
                src = f" — {it['source']}" if it["source"] else ""
                st.markdown(
                    f"<div class='news-item'><b>{it['title'][:110]}</b>"
                    f"<br><small>{it['date']}{src}</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No extracted headlines available — run "
                       "`python scripts/populate_knowledge_base.py` to refresh the feed.")
        st.caption("Headlines come from the system's own news-extraction layer, "
                   "scoped to the selected chokepoint.")


# ---------------------------------------------------------------------------
# Panel internals
# ---------------------------------------------------------------------------


def _render_trend_chart(win_days, win_vals, peak_day, thresholds, route_name):
    """Risk trend with hidden numeric axis — status bands + words instead."""
    import plotly.graph_objects as go

    crit = float(thresholds.get("risk_critical", 0.8))
    med = float(thresholds.get("risk_medium", 0.4))
    words = ["Critical" if v >= crit else ("High" if v >= med else "Low") for v in win_vals]

    fig = go.Figure()
    # Status bands instead of a numeric axis (manager-facing view).
    fig.add_hrect(y0=0, y1=med, fillcolor="#1d4a30", opacity=0.35, line_width=0)
    fig.add_hrect(y0=med, y1=crit, fillcolor="#4a3d10", opacity=0.35, line_width=0)
    fig.add_hrect(y0=crit, y1=1.0, fillcolor="#4a1d1d", opacity=0.35, line_width=0)
    for y, lbl in ((med / 2, "Low"), ((med + crit) / 2, "High"), ((crit + 1) / 2, "Critical")):
        fig.add_annotation(x=win_days[0], y=y, text=lbl, showarrow=False,
                           font=dict(size=10, color="#8d99a8"), xanchor="left")
    fig.add_trace(go.Scatter(
        x=win_days, y=win_vals, mode="lines", name="risk",
        line=dict(color="#dfe6ee", width=2.2),
        customdata=words, hovertemplate="day %{x} · %{customdata}<extra></extra>",
    ))
    peak_val = win_vals[win_days.index(peak_day)]
    fig.add_trace(go.Scatter(
        x=[peak_day], y=[peak_val], mode="markers", name="peak",
        marker=dict(size=13, color="#d9a514", symbol="diamond",
                    line=dict(color="#0b0e14", width=1.5)),
        customdata=[words[win_days.index(peak_day)]],
        hovertemplate="most significant peak · day %{x} · %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        height=235, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10151d", margin=dict(l=6, r=6, t=24, b=4), showlegend=False,
        title=dict(text=f"Risk trend — {route_name}", font=dict(size=12)),
        yaxis=dict(range=[0, 1.02], showticklabels=False, showgrid=False),  # no raw scores
        xaxis=dict(title=None, tickfont=dict(size=10)),
    )
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                            key="trend_chart", config={"displayModeBar": False})
    # Clicking the diamond marker (trace 1) opens the inline explanation.
    try:
        pts = event.selection.points if event and event.selection else []
        if any(p.get("curve_number") == 1 for p in pts):
            st.session_state["_peak_clicked"] = True
    except Exception:
        pass


def _render_peak_explanation(ts, peak_day, word, region_label):
    """LLM-generated (or composed) natural-language paragraph — no feature list."""
    features_df, _, explainer = get_shap_assets()
    shap_result = explainer.explain(features_df.iloc[[min(peak_day - 1, len(features_df) - 1)]])
    agreement = int(sum(1 for a in ALL_AGENTS if float(ts[a].iloc[peak_day - 1]) > 0.5))
    text, source = generate_risk_narrative(
        json.dumps(shap_result["top_drivers"]), agreement, word, region_label,
    )
    st.markdown(f"<div class='explain-box'>{text}</div>", unsafe_allow_html=True)
    src_note = "AI-generated summary" if source == "llm" else "generated summary"
    st.caption(f"{src_note} of the explainability model's live output for day {peak_day}.")
    # Deep-link the Analysis page to this day via shared session state.
    st.session_state["analysis_focus_day"] = peak_day
    try:
        st.page_link("pages/2_Analysis_View.py", label="View full breakdown →")
    except Exception:
        st.caption("Full breakdown: open the **Analysis View** page from the sidebar.")


def _render_decision(ts, params, route, day, top_driver):
    """Decision-support badge — rule-based, clearly labelled as such."""
    thresholds = params["thresholds"]
    series = route_risk_series(ts, params, route)
    level = classify_level(float(series.iloc[day - 1]),
                           thresholds["risk_high"], thresholds["risk_medium"])
    sustained = bool(sustained_high_flags(series, thresholds["risk_high"])[day - 1])

    hist = None
    hist_event = ""
    try:
        signals = {a: float(ts[a].iloc[day - 1]) for a in ALL_AGENTS}
        matches = get_retriever().query(signals, top_k=1)
        if matches:
            labels = load_decision_labels(_LABELS_PATH)
            hist = {"similarity": float(matches[0].get("similarity", 0.0)),
                    "action": labels.get(matches[0]["id"], "")}
            if hist["similarity"] > 0.6:
                hist_event = str(matches[0].get("metadata", {}).get("event", ""))
    except Exception:
        pass

    action, rule = decide_action(level, top_driver.get("agent", ""), sustained, hist)
    chip_colors = {"no_action": "#3d4b5e", "monitor": "#d9a514",
                   "reroute": "#d97b14", "escalate": "#d64545"}
    st.markdown(
        f'<span class="action-chip" style="background:{chip_colors.get(action, "#3d4b5e")}">'
        f'{action.replace("_", " ").upper()}</span>',
        unsafe_allow_html=True,
    )
    rationale = f"{status_word(float(series.iloc[day - 1]), thresholds)} status"
    if sustained:
        rationale += ", sustained for more than five days"
    if top_driver.get("agent"):
        rationale += f", driven by {top_driver['agent'].replace('_', ' ')} signals"
    if hist_event:
        rationale += f" — similar to {hist_event}"
    st.markdown(rationale + ".")
    st.markdown(f"<div class='rule-box'>rule fired: {rule}</div>", unsafe_allow_html=True)
    st.caption("Rule-based recommendation, not an automated decision.")
