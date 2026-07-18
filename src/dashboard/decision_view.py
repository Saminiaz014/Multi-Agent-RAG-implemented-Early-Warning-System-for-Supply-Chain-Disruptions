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
``core.build_map_html``).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
    SYNTHETIC_YEAR_START,
    Peak,
    build_map_html,
    build_peak_breakdown,
    classify_level,
    compute_timeseries,
    decide_action,
    detect_high_peaks,
    get_monitoring_points,
    get_news,
    get_retriever,
    get_routes,
    get_shap_assets,
    get_vessels,
    load_app_config,
    quick_jump_range,
    read_last_updated,
    resolve_mode_params,
    route_risk_series,
    select_route,
    status_word,
    sustained_high_flags,
    to_datetime_series,
)
from src.evaluation.decision_effectiveness import load_decision_labels  # noqa: E402

_LABELS_PATH = _PROJECT_ROOT / "data" / "knowledge_base" / "decision_labels.json"

_TREND_PRESETS = ("Last week", "Last month", "Last year", "YTD")


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
        # Freshness badge fed by the daily Airflow run's marker file. Rides
        # inside the header column (shorter than the page title beside it) so
        # the single-viewport guarantee holds; "refreshed" — the pipeline
        # reran — not "updated", which would overclaim changed results.
        refreshed = read_last_updated()
        if refreshed:
            st.caption(f"Data refreshed: {refreshed}")
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
        # Date-range control over the cached year — same dates for every
        # route, so it renders before route selection (layout unchanged).
        dates_index = pd.date_range(SYNTHETIC_YEAR_START, periods=len(ts), freq="D")
        first_date, last_date = dates_index[0].date(), dates_index[-1].date()

        pending_range = st.session_state.pop("pending_trend_range", None)
        if pending_range is not None:
            st.session_state["trend_range"] = pending_range
        if "trend_range" not in st.session_state:
            st.session_state["trend_range"] = (first_date, last_date)
        start_d, end_d = st.slider(
            "Trend window", min_value=first_date, max_value=last_date,
            key="trend_range", format="MMM D", label_visibility="collapsed",
        )
        preset_cols = st.columns(len(_TREND_PRESETS))
        for pcol, preset in zip(preset_cols, _TREND_PRESETS):
            with pcol:
                if st.button(preset, key=f"trend_{preset.replace(' ', '_').lower()}",
                             use_container_width=True):
                    st.session_state["pending_trend_range"] = quick_jump_range(
                        preset, dates_index)
                    st.rerun()

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

        # Re-slice the cached series only — no pipeline recompute.
        series = to_datetime_series(route_risk_series(ts, params, route))
        win = series.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
        if win.empty:
            win = series
        peak_ts = win.idxmax()
        peak_day = int(series.index.get_loc(peak_ts)) + 1  # 1-based assessment day
        assess_peak = Peak(index=peak_day - 1, date=peak_ts,
                           value=float(series.loc[peak_ts]))
        assess_word = status_word(assess_peak.value, thresholds)

        # HIGH-zone local maxima on the full series, filtered to the window.
        all_peaks = detect_high_peaks(series.to_numpy(), series.index,
                                      float(thresholds.get("risk_high", 0.6)))
        visible_peaks = [p for p in all_peaks if start_d <= p.date.date() <= end_d]

        clicked_peak = _render_trend_chart(win, visible_peaks, assess_peak,
                                           thresholds, route["name"])
        if visible_peaks:
            st.caption("Round markers flag high-risk peaks — click one to see why.")
        if clicked_peak is not None:
            _open_peak_dialog(clicked_peak, ts, thresholds)

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
        html = build_map_html(
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


def _render_trend_chart(win, peaks, assess_peak, thresholds, route_name):
    """Risk trend with hidden numeric axis — status bands + words instead.

    The x-axis is a real datetime axis ticked at month boundaries. Returns
    the :class:`Peak` the user clicked (diamond = assessment peak, circles =
    detected HIGH peaks) or ``None``.
    """
    import plotly.graph_objects as go

    crit = float(thresholds.get("risk_critical", 0.8))
    med = float(thresholds.get("risk_medium", 0.4))
    win_vals = win.to_numpy()
    words = ["Critical" if v >= crit else ("High" if v >= med else "Low") for v in win_vals]

    fig = go.Figure()
    # Status bands instead of a numeric axis (manager-facing view).
    fig.add_hrect(y0=0, y1=med, fillcolor="#1d4a30", opacity=0.35, line_width=0)
    fig.add_hrect(y0=med, y1=crit, fillcolor="#4a3d10", opacity=0.35, line_width=0)
    fig.add_hrect(y0=crit, y1=1.0, fillcolor="#4a1d1d", opacity=0.35, line_width=0)
    for y, lbl in ((med / 2, "Low"), ((med + crit) / 2, "High"), ((crit + 1) / 2, "Critical")):
        fig.add_annotation(x=win.index[0], y=y, text=lbl, showarrow=False,
                           font=dict(size=10, color="#8d99a8"), xanchor="left")
    fig.add_trace(go.Scatter(                                   # trace 0 — risk line
        x=win.index, y=win_vals, mode="lines", name="risk",
        line=dict(color="#dfe6ee", width=2.2),
        customdata=words, hovertemplate="%{x|%b %d} · %{customdata}<extra></extra>",
    ))
    assess_word = ("Critical" if assess_peak.value >= crit
                   else ("High" if assess_peak.value >= med else "Low"))
    fig.add_trace(go.Scatter(                                   # trace 1 — assessment day
        x=[assess_peak.date], y=[assess_peak.value], mode="markers", name="peak",
        marker=dict(size=13, color="#d9a514", symbol="diamond",
                    line=dict(color="#0b0e14", width=1.5)),
        customdata=[assess_word],
        hovertemplate="most significant peak · %{x|%b %d} · %{customdata}"
                      "<br>click for the full story<extra></extra>",
    ))
    fig.add_trace(go.Scatter(                                   # trace 2 — HIGH peaks
        x=[p.date for p in peaks], y=[p.value for p in peaks],
        mode="markers", name="high peaks",
        marker=dict(size=9, color="#d97b14", symbol="circle",
                    line=dict(color="#e8edf2", width=1.5)),
        hovertemplate="high-risk peak · %{x|%b %d}<br>click for the full story"
                      "<extra></extra>",
    ))
    # Month-boundary ticks on a true datetime axis; short windows fall back
    # to plotly's automatic date ticks (a 7-day window has no month starts).
    crosses_year = win.index[0].year != win.index[-1].year
    xaxis = dict(title=None, tickfont=dict(size=10), type="date",
                 tickformat="%b %Y" if crosses_year else "%b")
    if len(win) >= 60:
        xaxis["dtick"] = "M1"
    else:
        xaxis["tickformat"] = "%b %d"
    fig.update_layout(
        height=235, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10151d", margin=dict(l=6, r=6, t=24, b=4), showlegend=False,
        title=dict(text=f"Risk trend — {route_name}", font=dict(size=12)),
        yaxis=dict(range=[0, 1.02], showticklabels=False, showgrid=False),  # no raw scores
        xaxis=xaxis,
    )
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                            key="trend_chart", selection_mode="points",
                            config={"displayModeBar": False})
    # Map a clicked marker back to its Peak. Selections persist across
    # reruns, so remember the last-handled one and only react to changes
    # (otherwise the dialog would reopen on every unrelated rerun).
    try:
        pts = event.selection.points if event and event.selection else []
        for p in pts:
            curve, point = p.get("curve_number"), int(p.get("point_index", -1))
            if curve == 2 and 0 <= point < len(peaks):
                clicked = peaks[point]
            elif curve == 1:
                clicked = assess_peak
            else:
                continue
            sig = (curve, point, str(clicked.date))
            if st.session_state.get("_last_peak_sig") != sig:
                st.session_state["_last_peak_sig"] = sig
                return clicked
    except Exception:
        pass
    return None


def _open_peak_dialog(peak, ts, thresholds):
    """Modal drill-down: plain-language 'why this peak is high' (no numbers).

    Uses ``st.dialog`` (Streamlit >= 1.34 modal) so the page never reflows —
    Page 1 keeps its single-viewport guarantee.
    """

    @st.dialog("Why risk peaked here")
    def _dlg():
        features_df, _, explainer = get_shap_assets()
        shap_row = explainer.explain(
            features_df.iloc[[min(peak.index, len(features_df) - 1)]])
        signals = {a: float(ts[a].iloc[peak.index]) for a in ALL_AGENTS}
        # LLM seam: resolves False today (no key); the template path runs.
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        breakdown = build_peak_breakdown(
            peak, shap_row, get_retriever(),
            signals=signals, thresholds=thresholds, use_llm=use_llm,
        )
        st.markdown(f"<div class='explain-box'>{breakdown.narrative}</div>",
                    unsafe_allow_html=True)
        st.markdown("**What stood out that day**")
        for d in breakdown.drivers:
            st.markdown(f"- {d['label']} — {d['level']}")
        if breakdown.precedent_title:
            st.markdown(f"**Historical precedent:** {breakdown.precedent_title}")
        st.caption("Plain-language summary composed from the system's explainability "
                   "and historical case layers.")
        # Deep-link the Analysis page to this day via shared session state.
        st.session_state["analysis_focus_day"] = peak.index + 1
        try:
            st.page_link("pages/2_Analysis_View.py", label="View full breakdown →")
        except Exception:
            st.caption("Full breakdown: open the **Analysis View** page from the sidebar.")

    _dlg()


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
