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

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.regions import get_region  # noqa: E402
from src.dashboard.core import (  # noqa: E402
    AGENT_COLORS,
    get_region_map,
    ALL_AGENTS,
    AVAILABLE_REGIONS,
    DASH_CSS,
    TIMELINE_AXIS_NOTE,
    agent_contributions,
    detect_risk_spikes,
    timeline_dates,
    STATUS_COLORS,
    STATUS_ICONS,
    build_map_html,
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
    resolve_mode_params,
    route_risk_series,
    select_route,
    status_word,
    sustained_high_flags,
)
from src.dashboard.llm_explanations import explain_spike  # noqa: E402
from src.dashboard.vessel_data import get_vessels_for_region  # noqa: E402
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


def _apply_region_query_param() -> None:
    """Let ``?region=<key>`` preselect the region, mirroring ``?route=``.

    Makes a region deep-linkable — useful for sharing a view, and for driving
    the page deterministically in a browser session. Applied before the
    selectbox renders so the widget picks it up; an unknown key is ignored
    rather than raising, since the query string is user-supplied.
    """
    requested = st.query_params.get("region")
    if not requested:
        return
    label = next(
        (name for name, key in AVAILABLE_REGIONS.items()
         if key == str(requested).strip().lower()),
        None,
    )
    if label and st.session_state.get("selected_region") != label:
        st.session_state["selected_region"] = label


def render() -> None:
    st.markdown(DASH_CSS, unsafe_allow_html=True)
    _apply_region_query_param()

    # ---------------- Region selector (always visible, top of page) --------
    head_l, head_r = st.columns([2.2, 1.3])
    with head_l:
        st.markdown("## Supply-chain disruption monitor")
    with head_r:
        # All four Phase 11 chokepoints, sourced from the region registry via
        # AVAILABLE_REGIONS so the selector cannot drift from it.
        region_label = st.selectbox("Region", list(AVAILABLE_REGIONS.keys()),
                                    key="selected_region",
                                    label_visibility="collapsed")
    region = AVAILABLE_REGIONS.get(region_label, "hormuz")

    # The selected region's merged config drives which agents are active for
    # this run. Cached per region by load_app_config, so switching back is free.
    config = load_app_config(region)

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
            # All four registered regions have corridors as of Phase 12.5, so
            # this is now only reached by a region in the registry that has no
            # _ROUTES entry yet. Show its real activation rather than a blank.
            _render_region_without_routes(region)
            return

        series = route_risk_series(ts, params, route, region)
        lo = max(0, len(series) - window)
        win_days = list(range(lo + 1, len(series) + 1))
        win_vals = series.iloc[lo:].to_numpy()
        peak_pos = int(np.argmax(win_vals))
        peak_day = win_days[peak_pos]                      # 1-based assessment day
        assess_word = status_word(float(series.iloc[peak_day - 1]), thresholds)

        # Display date axis (a rolling window ending today — see core).
        all_dates = timeline_dates(len(series))
        win_dates = [all_dates[d - 1] for d in win_days]
        spikes = detect_risk_spikes(series, all_dates, thresholds)

        _render_trend_chart(win_days, win_vals, peak_day, thresholds, route["name"],
                            win_dates=win_dates, spikes=spikes)
        st.caption(TIMELINE_AXIS_NOTE)

        # -------- peak explanation (inline, natural language) --------------
        clicked = st.session_state.get("_peak_clicked", False)
        if st.button("Explain the marked peak", key="explain_peak", type="secondary"):
            clicked = True
        if clicked:
            st.session_state["_peak_clicked"] = True
            _render_peak_explanation(ts, peak_day, assess_word, region_label)

        # -------- spike explanation (click a red ring on the chart) --------
        _render_spike_panel(region, ts, params, spikes, all_dates)

    # ======================= assessment context ============================
    day = peak_day
    route_words = {r["key"]: status_word(float(route_risk_series(ts, params, r, region).iloc[day - 1]),
                                         thresholds) for r in routes}
    features_df, _, explainer = get_shap_assets()
    shap_result = explainer.explain(features_df.iloc[[min(day - 1, len(features_df) - 1)]])
    top_driver = shap_result["top_drivers"][0] if shap_result["top_drivers"] else {"agent": "", "feature": ""}

    # ======================= CENTRE — interactive map ======================
    with col_map:
        points = get_monitoring_points(config)
        # settings.yaml's monitoring_points cover hormuz/red_sea/malacca/suez
        # only, so fall back to the region's own centre rather than a hardcoded
        # Hormuz point — otherwise Panama's marker lands in the Persian Gulf.
        centre_lng, centre_lat = get_region_map(region)["center"]
        focus = points.get(
            region, {"name": region_label, "lat": centre_lat, "lng": centre_lng}
        )
        # Only the region under analysis is marked; the previous cross-region
        # markers offered jumps out of it.
        secondary: list[dict] = []
        html = build_map_html(
            focus=focus,
            secondary=secondary,
            risk_level=classify_level(float(series.iloc[day - 1]),
                                      thresholds["risk_high"], thresholds["risk_medium"]),
            top_driver=top_driver.get("feature", "—").replace("_", " "),
            updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            routes=routes,
            vessels=get_vessels(route, day, region=region),
            selected_route=selected_key,
            route_status=route_words,
            status=assess_word,
            region=region,
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
        _render_decision(ts, params, route, day, top_driver, region)

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

    # ============ FULL WIDTH — agent breakdown + vessel detail =============
    _render_agent_breakdown(region, ts, all_dates, win_dates)
    _render_vessel_panel(region, ts, params, day, routes)


# ---------------------------------------------------------------------------
# Panel internals
# ---------------------------------------------------------------------------


def _render_spike_panel(region, ts, params, spikes, all_dates) -> None:
    """Explain the spike the user clicked on the trend chart.

    Renders nothing until a spike is selected, so the default view is unchanged.

    Args:
        region: Region key.
        ts: Per-agent score frame.
        params: Weight/threshold params.
        spikes: All spikes in the series.
        all_dates: The full display axis.
    """
    day = st.session_state.get("_selected_spike_day")
    if day is None:
        return
    spike = next((s for s in spikes if s["day"] == day), None)
    if spike is None:
        return

    active = set(get_region(region).active_agents())
    row = {a: float(ts[a].iloc[day - 1]) for a in ALL_AGENTS if a in active and a in ts}
    drivers = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
    agreement = sum(1 for v in row.values() if v > 0.5)

    text, source = explain_spike(region, spike, drivers, agreement)

    with st.container(border=True):
        head, close = st.columns([5, 1])
        with head:
            st.markdown(
                f"**Risk analysis — crossed into {spike['level']} on "
                f"{spike['date']:%d %b %Y}**"
            )
        with close:
            if st.button("Close", key="close_spike"):
                st.session_state.pop("_selected_spike_day", None)
                st.rerun()

        st.markdown(text)
        st.markdown("**Contributing agents**")
        for name, score in drivers:
            st.markdown(f"- {name.replace('_', ' ').title()} — {score:.2f}")
        st.caption(
            ("Generated by Claude." if source == "llm"
             else "Composed from the live agent scores (no ANTHROPIC_API_KEY set).")
            + " Explains the detector signals, not a real-world event — the date is a"
            " position on the display axis."
        )


def _render_agent_breakdown(region, ts, all_dates, win_dates=None) -> None:
    """Per-agent score lines on the same axis as the composite trend.

    Only the region's active agents are drawn; a passive agent is omitted
    rather than shown as a flat zero, which would read as "measured and quiet"
    instead of "not run in this region".

    Rendered with plotly rather than ``st.line_chart`` for two reasons: the
    x-range is pinned to the composite chart's window (Altair pads the domain,
    which left the two charts visibly out of step), and plotly's legend is
    click-to-toggle natively.

    Args:
        win_dates: The composite chart's visible window. The breakdown is
            clipped to exactly this span so the two axes line up.
    """
    import plotly.graph_objects as go

    active = set(get_region(region).active_agents())
    frame = agent_contributions(ts, all_dates, active)
    if frame.empty:
        return
    if win_dates is not None and len(win_dates):
        frame = frame.loc[win_dates[0]:win_dates[-1]]

    colors = {a.replace("_", " ").title(): AGENT_COLORS[a] for a in ALL_AGENTS}

    st.markdown("**Agent contributions**")
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode="lines", name=column,
            line=dict(color=colors.get(column, "#8d99a8"), width=1.6),
            hovertemplate=f"{column} · %{{x|%d %b %Y}} · %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10151d", margin=dict(l=6, r=6, t=10, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
        yaxis=dict(range=[0, 1.02], title=None, gridcolor="#1c2431"),
        # Pinned to the composite's window so the two charts stay in step.
        xaxis=dict(title=None, tickfont=dict(size=10),
                   range=[frame.index[0], frame.index[-1]]),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    passive = get_region(region).passive_agents()
    st.caption(
        f"One line per active agent, on the same window as the trend above. "
        f"Click a name in the legend to toggle it. "
        f"Not shown (passive here): {', '.join(passive) or 'none'}. "
        f"{TIMELINE_AXIS_NOTE}"
    )


def _render_vessel_panel(region, ts, params, day, routes) -> None:
    """Vessel markers for the region, colour-coded by their corridor's risk."""
    vessels = get_vessels_for_region(region, day, ts, params, routes=routes)
    if not vessels:
        return

    st.markdown("**Vessels on monitored corridors**")
    labels = [
        f"{v['color_hex'] and ''}{v['name']} · {v['type']} · {v['status'].upper()}"
        for v in vessels
    ]
    choice = st.selectbox(
        "Vessel", range(len(vessels)), format_func=lambda i: labels[i],
        key="selected_vessel", label_visibility="collapsed",
    )
    vessel = vessels[choice]

    cols = st.columns(4)
    cols[0].metric("Status", vessel["status"].upper())
    cols[1].metric("Type", vessel["type"])
    cols[2].metric("Flag", vessel["flag"])
    cols[3].metric("ETA", vessel["eta"])
    st.markdown(
        f"<div class='route-row'>{vessel['id']} — bound for "
        f"<b>{vessel['destination']}</b> on <b>{vessel['route']}</b>, "
        f"{vessel['speed_kn']} kn "
        f"<span style='color:{vessel['color_hex']}'>●</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Vessel records are deterministic synthetic placeholders for this view — "
        "the pipeline tracks daily aggregate arrivals, not per-vessel AIS. A "
        "vessel's status is its corridor's risk for the assessment day, so "
        "vessels on one corridor share it."
    )


def _render_region_without_routes(region: str) -> None:
    """Explain a region that has pipeline config but no drawn route corridors.

    Every region carries real agent activation, AIS bounds, ACLED countries and
    news keywords (Phase 11), but ``core._ROUTES`` only defines corridor
    geometry for Hormuz. Rather than a bare "nothing here", show what the
    region's pipeline would actually run, straight from the registry.

    Args:
        region: Canonical region key.
    """
    from src.core.regions import get_region

    cfg = get_region(region)
    st.info(
        f"**{cfg.display_name}** has no mapped route corridors yet — those are "
        "drawn for the Strait of Hormuz only. Its detection pipeline is fully "
        "configured; the summary below is live from the region registry."
    )
    st.markdown("**Active agents**")
    st.markdown(
        "\n".join(f"- {name.replace('_', ' ').title()}" for name in cfg.active_agents())
        or "- _none_"
    )
    passive = cfg.passive_agents()
    if passive:
        st.markdown("**Passive agents** — excluded from this region's run")
        for name in passive:
            reason = cfg.passive_reasons.get(name, "No reason recorded.")
            st.markdown(f"- **{name.replace('_', ' ').title()}** — {reason}")


def _render_trend_chart(win_days, win_vals, peak_day, thresholds, route_name,
                        win_dates=None, spikes=()):
    """Risk trend with hidden numeric axis — status bands + words instead.

    Phase 12.5: the x-axis carries real dates when ``win_dates`` is supplied,
    and threshold-crossing spikes are drawn as a clickable third trace.

    Args:
        win_dates: Display dates aligned with ``win_days``. See
            :func:`src.dashboard.core.timeline_dates` — these are a rolling
            window ending today, not the series' own timestamps.
        spikes: Spikes inside this window, from ``core.detect_risk_spikes``.
    """
    import plotly.graph_objects as go

    # Plot against dates when available, falling back to day indices so the
    # Analysis view and any other caller keep working unchanged.
    xs = list(win_dates) if win_dates is not None else win_days
    day_to_x = dict(zip(win_days, xs))

    crit = float(thresholds.get("risk_critical", 0.8))
    med = float(thresholds.get("risk_medium", 0.4))
    words = ["Critical" if v >= crit else ("High" if v >= med else "Low") for v in win_vals]

    fig = go.Figure()
    # Status bands instead of a numeric axis (manager-facing view).
    fig.add_hrect(y0=0, y1=med, fillcolor="#1d4a30", opacity=0.35, line_width=0)
    fig.add_hrect(y0=med, y1=crit, fillcolor="#4a3d10", opacity=0.35, line_width=0)
    fig.add_hrect(y0=crit, y1=1.0, fillcolor="#4a1d1d", opacity=0.35, line_width=0)
    for y, lbl in ((med / 2, "Low"), ((med + crit) / 2, "High"), ((crit + 1) / 2, "Critical")):
        fig.add_annotation(x=xs[0], y=y, text=lbl, showarrow=False,
                           font=dict(size=10, color="#8d99a8"), xanchor="left")
    hover_x = "%{x|%d %b %Y}" if win_dates is not None else "day %{x}"
    fig.add_trace(go.Scatter(
        x=xs, y=win_vals, mode="lines", name="risk",
        line=dict(color="#dfe6ee", width=2.2),
        customdata=words, hovertemplate=f"{hover_x} · %{{customdata}}<extra></extra>",
    ))
    peak_val = win_vals[win_days.index(peak_day)]
    fig.add_trace(go.Scatter(
        x=[day_to_x[peak_day]], y=[peak_val], mode="markers", name="peak",
        marker=dict(size=13, color="#d9a514", symbol="diamond",
                    line=dict(color="#0b0e14", width=1.5)),
        customdata=[words[win_days.index(peak_day)]],
        hovertemplate=f"most significant peak · {hover_x} · %{{customdata}}<extra></extra>",
    ))
    # Trace 2 — threshold crossings. Clicking one opens its explanation.
    in_window = [s for s in spikes if s["day"] in day_to_x]
    if in_window:
        fig.add_trace(go.Scatter(
            x=[day_to_x[s["day"]] for s in in_window],
            y=[s["risk"] for s in in_window],
            mode="markers", name="spikes",
            marker=dict(size=9, color="#d64545", symbol="circle-open",
                        line=dict(width=2.2)),
            customdata=[s["level"] for s in in_window],
            hovertemplate=(
                f"crossed into %{{customdata}} · {hover_x}"
                "<br><i>click to explain</i><extra></extra>"
            ),
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
    # Trace 1 (the diamond) opens the peak explanation; trace 2 (a spike ring)
    # selects that spike for its own explanation.
    try:
        pts = event.selection.points if event and event.selection else []
        if any(p.get("curve_number") == 1 for p in pts):
            st.session_state["_peak_clicked"] = True
        for p in pts:
            if p.get("curve_number") == 2:
                idx = p.get("point_index")
                if idx is not None and 0 <= idx < len(in_window):
                    st.session_state["_selected_spike_day"] = in_window[idx]["day"]
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


def _render_decision(ts, params, route, day, top_driver, region=None):
    """Decision-support badge — rule-based, clearly labelled as such."""
    thresholds = params["thresholds"]
    series = route_risk_series(ts, params, route, region)
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
