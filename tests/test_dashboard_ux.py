"""Decision-view UX layer: corridors, date axis, spikes, vessels, explanations.

Phase 12.5. Streamlit pages can't be driven headlessly here, so these target
the data functions the page renders from, plus the render helpers via
monkeypatched ``st`` calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.regions import get_region, list_regions
from src.dashboard.core import (
    TIMELINE_AXIS_NOTE,
    TIMELINE_SOURCE_RANGE,
    TIMELINE_SYNTHETIC_CAPTION,
    agent_contributions,
    detect_risk_spikes,
    get_region_map,
    get_routes,
    get_vessels,
    region_destinations,
    relative_axis_ticks,
    relative_day_label,
    timeline_dates,
)
from src.dashboard.llm_explanations import (
    cache_size,
    clear_explanation_cache,
    explain_spike,
)
from src.dashboard.vessel_data import (
    get_vessel_details,
    get_vessels_for_region,
    risk_to_color,
    risk_to_status,
)


@pytest.fixture(autouse=True)
def _clear_explanations() -> None:
    clear_explanation_cache()
    yield
    clear_explanation_cache()


class TestRegionCorridors:
    """Every region has drawable corridors and map framing (P12.5 step 1)."""

    def test_all_four_regions_have_corridors(self) -> None:
        for region in list_regions():
            routes = get_routes(region)
            assert len(routes) >= 3, f"{region}: too few corridors"
            for route in routes:
                assert {"key", "name", "agents", "coords"} <= set(route), region
                assert len(route["coords"]) >= 3, f"{region}/{route['key']}"

    def test_corridor_risk_excludes_agents_passive_in_the_region(self) -> None:
        """A corridor's trend must not fold in a signal the region doesn't run.

        Hormuz's eastbound corridor lists ``routing``, which Phase 11 muted
        everywhere. The corridor keeps listing it (that is genuinely what the
        corridor is about, and it should come back if routing is re-enabled),
        so ``route_risk_series`` intersects with the region's active set —
        without that, the trend renormalises over a muted agent.
        """
        from src.dashboard.core import (
            composite_series,
            compute_timeseries,
            resolve_mode_params,
            route_risk_series,
        )

        ts = compute_timeseries("hand_tuned")
        params = resolve_mode_params("hand_tuned")

        for region in list_regions():
            active = set(get_region(region).active_agents())
            for route in get_routes(region):
                scoped = route_risk_series(ts, params, route, region)
                expected = composite_series(ts, params, set(route["agents"]) & active)
                assert scoped.equals(expected), f"{region}/{route['key']}"

        # And the intersection actually bites on the known case.
        eastbound = next(
            r for r in get_routes("hormuz") if r["key"] == "eastbound_tss"
        )
        assert "routing" in eastbound["agents"]
        assert not route_risk_series(ts, params, eastbound, "hormuz").equals(
            route_risk_series(ts, params, eastbound)
        )

    def test_map_framing_surrounds_the_regions_corridors(self) -> None:
        """The camera centre must sit near the geometry it frames."""
        for region in list_regions():
            centre_lng, centre_lat = get_region_map(region)["center"]
            points = [pt for r in get_routes(region) for pt in r["coords"]]
            lngs = [p[0] for p in points]
            lats = [p[1] for p in points]

            assert min(lngs) - 1 <= centre_lng <= max(lngs) + 1, region
            assert min(lats) - 1 <= centre_lat <= max(lats) + 1, region

    def test_chokepoint_labels_are_named_and_distinct(self) -> None:
        seen: set[str] = set()
        for region in list_regions():
            for point in get_region_map(region)["chokepoints"]:
                assert point["name"].strip()
                assert point["name"] not in seen, f"duplicate label {point['name']}"
                seen.add(point["name"])

    def test_suez_is_not_a_selectable_region(self) -> None:
        """Suez is out of scope for the dashboard (it stays in settings.yaml)."""
        assert "suez" not in list_regions()
        assert get_routes("suez") == []

    def test_vessels_get_region_appropriate_destinations(self) -> None:
        """A Panama transit should not be bound for Ras Tanura."""
        for region in list_regions():
            allowed = set(region_destinations(region))
            for vessel in get_vessels(get_routes(region)[0], day=155, region=region):
                assert vessel["destination"] in allowed, region
        assert set(region_destinations("panama")).isdisjoint(
            region_destinations("hormuz")
        )


class TestTimelineDates:
    """The display date axis (P12.5 step 3)."""

    def test_axis_is_daily_and_ends_today(self) -> None:
        axis = timeline_dates(365)
        assert len(axis) == 365
        assert axis[-1] == pd.Timestamp.now().normalize()
        assert (axis.to_series().diff().dropna() == pd.Timedelta(days=1)).all()

    def test_end_is_overridable_and_length_configurable(self) -> None:
        axis = timeline_dates(30, end="2026-03-31")
        assert len(axis) == 30
        assert axis[-1] == pd.Timestamp("2026-03-31")
        assert axis[0] == pd.Timestamp("2026-03-02")

    def test_the_shift_is_declared_not_silent(self) -> None:
        """The axis is a display window, so the UI must say so.

        The series' own timestamps are 2025-01-01..2025-12-31; the axis maps
        them onto a window ending today. That is a presentation choice, and a
        reader who was not in the room must be able to see that it was made.
        The long note records the provenance; the short caption is what the
        charts render.
        """
        assert TIMELINE_SOURCE_RANGE == ("2025-01-01", "2025-12-31")
        assert "display window" in TIMELINE_AXIS_NOTE
        for fragment in TIMELINE_SOURCE_RANGE:
            assert fragment in TIMELINE_AXIS_NOTE

        lowered = TIMELINE_SYNTHETIC_CAPTION.lower()
        assert "synthetic" in lowered
        assert "not calendar dates" in lowered


class TestRelativeAxisLabels:
    """Ticks read as distance from today, not as calendar dates (P12.5 polish).

    A spike labelled "18 Jan 2026" invites exactly the misreading the axis
    can't support — that something happened in January. Distance-from-today
    keeps the recency the axis is for and drops the false precision.
    """

    def test_labels_are_relative_not_calendar(self) -> None:
        axis = timeline_dates(365)
        _, ticktext = relative_axis_ticks(axis)

        assert ticktext[-1] == "Today"
        assert ticktext[0] == "364d ago"
        for label in ticktext:
            assert "2025" not in label and "2026" not in label

    def test_ticks_use_the_compact_form(self) -> None:
        """Five long labels rotate diagonally in the narrow trend chart."""
        _, ticktext = relative_axis_ticks(timeline_dates(30))
        assert all(len(label) <= 9 for label in ticktext), ticktext

    def test_single_label_edges(self) -> None:
        today = pd.Timestamp("2026-08-20")
        assert relative_day_label("2026-08-20", today) == "Today"
        assert relative_day_label("2026-08-19", today) == "1 day ago"
        assert relative_day_label("2026-07-21", today) == "30 days ago"
        # Compact form is for ticks; prose keeps the readable wording.
        assert relative_day_label("2026-07-21", today, compact=True) == "30d ago"
        assert relative_day_label("2026-08-20", today, compact=True) == "Today"

    def test_ticks_span_both_ends(self) -> None:
        axis = timeline_dates(90)
        tickvals, ticktext = relative_axis_ticks(axis, n_ticks=4)

        assert tickvals[0] == axis[0]
        assert tickvals[-1] == axis[-1]
        assert len(tickvals) == len(ticktext) >= 4

    def test_empty_axis_is_safe(self) -> None:
        assert relative_axis_ticks([]) == ([], [])

    def test_both_charts_share_one_tick_builder(self) -> None:
        """Composite and agent charts must not drift apart in formatting."""
        import src.dashboard.decision_view as view

        axis = timeline_dates(120)
        config = view._relative_xaxis(axis)

        assert config["tickmode"] == "array"
        assert config["ticktext"][-1] == "Today"
        assert config["tickvals"][0] == axis[0]


class TestRegionSelector:
    """Region picker as buttons (P12.5 polish)."""

    def test_every_region_has_a_button_label(self) -> None:
        import src.dashboard.decision_view as view

        for key in list_regions():
            assert key in view._REGION_BUTTONS, f"{key} has no button label"
            icon, short = view._REGION_BUTTONS[key]
            assert icon and short.strip()

    def test_button_labels_are_distinct(self) -> None:
        """Two identical labels would make two chokepoints indistinguishable."""
        import src.dashboard.decision_view as view

        labels = [f"{i} {s}" for i, s in view._REGION_BUTTONS.values()]
        assert len(set(labels)) == len(labels)


class TestTextCleanup:
    """Captions carry only what the UI can't say for itself (P12.5 polish)."""

    def test_agent_caption_explains_missing_lines_not_the_legend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legend interaction is self-evident; a *missing* agent line is not."""
        import src.dashboard.decision_view as view
        from src.dashboard.core import compute_timeseries

        captions: list[str] = []
        monkeypatch.setattr(view.st, "markdown", lambda *a, **k: None)
        monkeypatch.setattr(view.st, "plotly_chart", lambda *a, **k: None)
        monkeypatch.setattr(view.st, "caption", lambda text, **k: captions.append(text))

        ts = compute_timeseries("hand_tuned")
        axis = timeline_dates(len(ts))
        view._render_agent_breakdown("panama", ts, axis, axis)

        assert captions, "the chart must still declare the synthetic timeline"
        caption = captions[0]
        assert "Synthetic" in caption
        # Panama's passive agents are named, since their lines are absent.
        assert "geopolitical" in caption and "routing" in caption
        # …and the self-evident instruction is gone.
        assert "legend" not in caption.lower()

    def test_decision_panel_drops_the_raw_rule_trace(self) -> None:
        """The plain-language rationale carries the "why"; the rule notation
        was technical clutter. It still lives on the Analysis view, and
        ``decide_action`` still returns it — only the badge box is gone.
        """
        source = (
            __import__("pathlib").Path(__import__("src.dashboard.decision_view",
                                                  fromlist=["__file__"]).__file__)
            .read_text(encoding="utf-8")
        )
        assert "rule-box" not in source
        assert "rule fired" not in source
        # The honesty disclaimer stays.
        assert "not an automated decision" in source


class TestSpikeDetection:
    """Threshold crossings on the trend chart (P12.5 step 4)."""

    _THRESHOLDS = {"risk_medium": 0.4, "risk_high": 0.6, "risk_critical": 0.8}

    def _spikes(self, values: list[float]) -> list[dict]:
        return detect_risk_spikes(values, timeline_dates(len(values)), self._THRESHOLDS)

    def test_reports_only_upward_crossings(self) -> None:
        spikes = self._spikes([0.1, 0.5, 0.1, 0.5])
        assert [s["level"] for s in spikes] == ["Medium", "Medium"]
        assert [s["day"] for s in spikes] == [2, 4]

    def test_one_spike_per_jump_at_the_highest_band(self) -> None:
        """0.1 -> 0.9 is one Critical spike, not Medium + High + Critical."""
        spikes = self._spikes([0.1, 0.9])
        assert len(spikes) == 1
        assert spikes[0]["level"] == "Critical"

    def test_uses_the_engines_own_thresholds(self) -> None:
        """Bands come from config, not a second hardcoded set."""
        spikes = detect_risk_spikes(
            [0.1, 0.5], timeline_dates(2), {"risk_medium": 0.9}
        )
        assert spikes == []

    def test_dates_align_with_the_axis(self) -> None:
        axis = timeline_dates(4)
        spikes = detect_risk_spikes([0.1, 0.5, 0.5, 0.85], axis, self._THRESHOLDS)
        for spike in spikes:
            assert spike["date"] == axis[spike["day"] - 1]

    def test_short_series_is_safe(self) -> None:
        assert self._spikes([]) == []
        assert self._spikes([0.9]) == []


class TestAgentContributions:
    """The multi-agent breakdown graph (P12.5 step 8)."""

    def test_only_active_agents_get_a_line(self) -> None:
        """A passive agent is omitted, not drawn flat at zero.

        A zero line reads as "measured and quiet"; omission reads as "not run
        here", which is what passive actually means.
        """
        from src.dashboard.core import compute_timeseries

        ts = compute_timeseries("hand_tuned")
        axis = timeline_dates(len(ts))

        for region in list_regions():
            active = get_region(region).active_agents()
            frame = agent_contributions(ts, axis, set(active))
            titles = {a.replace("_", " ").title() for a in active}

            assert set(frame.columns) == titles, region
            assert "Routing" not in frame.columns, region

    def test_indexed_on_the_display_axis(self) -> None:
        from src.dashboard.core import compute_timeseries

        ts = compute_timeseries("hand_tuned")
        axis = timeline_dates(len(ts))
        frame = agent_contributions(ts, axis, {"shipping"})

        assert frame.index.name == "date"
        assert frame.index[-1] == axis[len(ts) - 1]


class TestVesselData:
    """Vessel-level risk and detail lookup (P12.5 step 6)."""

    def test_risk_bands_map_to_colours_and_statuses(self) -> None:
        assert (risk_to_color(0.0), risk_to_color(0.29)) == ("green", "green")
        assert (risk_to_color(0.3), risk_to_color(0.59)) == ("yellow", "yellow")
        assert (risk_to_color(0.6), risk_to_color(1.0)) == ("red", "red")
        assert risk_to_status(0.1) == "normal"
        assert risk_to_status(0.45) == "delayed"
        assert risk_to_status(0.95) == "critical"

    def test_vessels_carry_their_corridors_risk(self) -> None:
        """Vessel risk is the corridor's pipeline score, not an invented one."""
        from src.dashboard.core import compute_timeseries, route_risk_series

        ts = compute_timeseries("hand_tuned")
        params = __import__(
            "src.dashboard.core", fromlist=["resolve_mode_params"]
        ).resolve_mode_params("hand_tuned")
        routes = get_routes("hormuz")
        vessels = get_vessels_for_region("hormuz", 155, ts, params, routes=routes)

        assert vessels
        for route in routes:
            expected = round(
                float(route_risk_series(ts, params, route, "hormuz").iloc[154]), 4
            )
            on_route = [v for v in vessels if v["route_key"] == route["key"]]
            assert on_route, route["key"]
            # All vessels on a corridor share it — the pipeline has no
            # per-vessel signal, so spreading it would be invented detail.
            assert {v["risk_score"] for v in on_route} == {expected}

    def test_every_vessel_is_marked_synthetic(self) -> None:
        from src.dashboard.core import compute_timeseries, resolve_mode_params

        ts = compute_timeseries("hand_tuned")
        params = resolve_mode_params("hand_tuned")
        for vessel in get_vessels_for_region("malacca", 100, ts, params):
            assert vessel["synthetic"] is True
            assert vessel["color_hex"].startswith("#")

    def test_details_lookup_by_id(self) -> None:
        from src.dashboard.core import compute_timeseries, resolve_mode_params

        ts = compute_timeseries("hand_tuned")
        params = resolve_mode_params("hand_tuned")
        vessels = get_vessels_for_region("hormuz", 155, ts, params)

        assert get_vessel_details(vessels[0]["id"], vessels) == vessels[0]
        assert get_vessel_details("no-such-vessel", vessels) is None

    def test_region_without_corridors_yields_no_vessels(self) -> None:
        from src.dashboard.core import compute_timeseries, resolve_mode_params

        ts = compute_timeseries("hand_tuned")
        params = resolve_mode_params("hand_tuned")
        assert get_vessels_for_region("atlantis", 1, ts, params) == []


class TestSpikeExplanations:
    """Explanations for a clicked spike (P12.5 step 5)."""

    _SPIKE = {
        "day": 155,
        "date": pd.Timestamp("2026-06-02"),
        "risk": 0.83,
        "level": "Critical",
    }
    _DRIVERS = [("shipping", 0.91), ("geopolitical", 0.78), ("news_sentiment", 0.55)]

    def test_composes_from_live_drivers_without_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        text, source = explain_spike("hormuz", self._SPIKE, self._DRIVERS, 3)

        assert source == "composed"
        assert "Strait of Hormuz" in text
        assert "critical" in text.lower()
        assert "shipping" in text  # the top driver is named

    def test_explanation_reflects_corroboration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One elevated detector must not read like a confirmed event."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        lone, _ = explain_spike("panama", self._SPIKE, [("shipping", 0.7)], 1)
        assert "noise" in lone.lower()

        clear_explanation_cache()
        broad, _ = explain_spike("panama", self._SPIKE, self._DRIVERS, 4)
        assert "agree" in broad.lower()

    def test_no_drivers_is_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        text, _ = explain_spike("malacca", self._SPIKE, [], 0)
        assert "no single agent" in text.lower()

    def test_cached_per_region_day_and_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        first = explain_spike("hormuz", self._SPIKE, self._DRIVERS, 3)
        assert explain_spike("hormuz", self._SPIKE, self._DRIVERS, 3) is first
        assert cache_size() == 1

        explain_spike("panama", self._SPIKE, self._DRIVERS, 3)
        assert cache_size() == 2

    def test_prompt_forbids_inventing_real_events(self) -> None:
        """The axis is relabelled, so a date here is not a calendar event.

        An explanation naming a real incident on an axis date would be
        fabrication presented as analysis — the system prompt has to rule it
        out explicitly, and the composed path can't do it by construction.
        """
        from src.dashboard.llm_explanations import _SYSTEM

        lowered = _SYSTEM.lower()
        assert "never name a real incident" in lowered
        assert "synthetic" in lowered

    def test_refusal_falls_back_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A safety refusal is realistic here — these prompts name sanctions.

        On a refusal ``content`` is empty, so reading it first would raise;
        the caller must check ``stop_reason`` and compose instead.
        """
        import src.dashboard.llm_explanations as module

        class _Refusal:
            stop_reason = "refusal"
            stop_details = type("D", (), {"category": "cyber"})()
            content: list = []

        monkeypatch.setattr(module, "_call_anthropic", lambda prompt: None)
        text, source = explain_spike("hormuz", self._SPIKE, self._DRIVERS, 3)
        assert source == "composed" and text

        # And the real reader must not index empty content.
        assert _Refusal().content == []
