"""IMF PortWatch extractor — chokepoint traffic history for the RAG store.

Fills the gap the other extractors leave. NewsAPI, SerpAPI, ACLED, FRED,
GDACS and USGS all describe *events* — an attack, a quake, a price spike.
None of them describe what the chokepoint's traffic was actually doing, which
is the question the shipping agent's retrieval most often needs answered:
"has traffic here been this low before, and what happened around it?"

Documents are monthly summaries rather than one per day. A day of vessel
counts retrieves poorly — 2,792 near-identical sentences per region would
crowd out everything else — whereas a month carries the level, the direction
and the deviation from a long-run baseline in one passage.

Matters most for Panama, whose coverage is otherwise the thinnest of the four
regions: the 2023-24 Gatun Lake drought is plainly visible in the transit
series (33.4/day in 2022 down to 20.7/day in 2024 Q1) but appears in no other
source this project pulls.

No credentials. The daily rows are fetched through
:class:`~src.ingestion.shipping_connector.ShippingConnector`, which already
owns PortWatch paging and retry — a second copy of that logic would be a
second thing to keep correct.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

#: Months with fewer real days than this are dropped: a partial month's mean
#: is not comparable with a full one, and the current month is always partial.
_MIN_DAYS_PER_MONTH = 20

#: Deviation from the long-run baseline that reads as a genuine disruption
#: rather than ordinary variation, as a fraction of the baseline.
_HIGH_DEVIATION = 0.25
_MEDIUM_DEVIATION = 0.10


class PortWatchExtractor(BaseExtractor):
    """Summarize a chokepoint's monthly vessel traffic into RAG documents."""

    @property
    def source_name(self) -> str:
        return "portwatch"

    def _chokepoint_name(self, region: str) -> str:
        """PortWatch's spelling of ``region``'s chokepoint, from config."""
        return str(
            (self.config.get("extraction", {}).get("chokepoints", {}).get(region, {}) or {})
            .get("portwatch_chokepoint", "")
        ).strip()

    def extract_historical(
        self, region: str, start_year: int | None = None,
        end_year: int | None = None, **kwargs,
    ) -> list[dict]:
        """Extract one document per month of chokepoint traffic.

        Args:
            region: Chokepoint key. Needs ``portwatch_chokepoint`` under
                ``extraction.chokepoints``.
            start_year: Overrides ``extraction.historical_range.start_year``.
            end_year: Overrides ``extraction.historical_range.end_year``.

        Returns:
            Normalized documents, one per complete month in range.
        """
        chokepoint = self._chokepoint_name(region)
        if not chokepoint:
            logger.warning(
                "PortWatch: no portwatch_chokepoint for region=%s — nothing "
                "to extract.", region,
            )
            return []

        from src.ingestion.shipping_connector import ShippingConnector

        connector = ShippingConnector(
            source_mode="api", config={"portwatch_chokepoint": chokepoint}
        )
        try:
            rows = connector._fetch_portwatch_rows(chokepoint)
        except Exception as exc:
            logger.error("PortWatch fetch failed for %s: %s", region, exc)
            return []
        if not rows:
            logger.warning("PortWatch returned no rows for %s.", chokepoint)
            return []

        hist = self.config.get("extraction", {}).get("historical_range", {}) or {}
        start_year = int(hist.get("start_year", 2019) if start_year is None else start_year)
        end_year = int(hist.get("end_year", 2026) if end_year is None else end_year)

        daily = pd.DataFrame(rows)
        daily["timestamp"] = pd.to_datetime(daily["date"], errors="coerce")
        daily["n_total"] = pd.to_numeric(daily["n_total"], errors="coerce")
        daily["n_tanker"] = pd.to_numeric(daily.get("n_tanker"), errors="coerce")
        daily = daily.dropna(subset=["timestamp", "n_total"])
        daily = daily[daily["timestamp"].dt.year.between(start_year, end_year)]
        if daily.empty:
            return []

        # Baseline is the whole fetched span, not a trailing window: a slow
        # decline drags a rolling baseline down with it, which is exactly how
        # the shipping connector's 2-sigma label misses the Panama drought.
        baseline = float(daily["n_total"].mean())
        countries = (
            self.config.get("extraction", {}).get("chokepoints", {})
            .get(region, {}).get("countries", [])
        ) or []

        documents: list[dict] = []
        for period, group in daily.groupby(daily["timestamp"].dt.to_period("M")):
            if len(group) < _MIN_DAYS_PER_MONTH:
                continue
            doc = self._month_to_doc(
                period, group, baseline, region, countries, chokepoint
            )
            documents.append(doc)

        logger.info(
            "PortWatch [%s] -> %d monthly summaries (%d-%d), baseline %.1f/day",
            region, len(documents), start_year, end_year, baseline,
        )
        return documents

    def _month_to_doc(
        self, period, group: pd.DataFrame, baseline: float,
        region: str, countries: list[str], chokepoint: str,
    ) -> dict:
        """Build one month's document."""
        mean_total = float(group["n_total"].mean())
        min_total = float(group["n_total"].min())
        tanker_mean = float(group["n_tanker"].mean()) if group["n_tanker"].notna().any() else None

        deviation = (mean_total - baseline) / baseline if baseline else 0.0
        magnitude = abs(deviation)
        if magnitude >= _HIGH_DEVIATION:
            severity = "high"
        elif magnitude >= _MEDIUM_DEVIATION:
            severity = "medium"
        else:
            severity = "low"

        direction = "below" if deviation < 0 else "above"
        tanker_note = (
            f" Tanker traffic averaged {tanker_mean:.1f} vessels per day."
            if tanker_mean is not None else ""
        )
        text = (
            f"{chokepoint} vessel traffic in {period.strftime('%B %Y')}: "
            f"{mean_total:.1f} transits per day on average, "
            f"{magnitude * 100:.0f}% {direction} the {baseline:.1f}/day "
            f"long-run average, with a low of {min_total:.0f}."
            f"{tanker_note}"
        )

        return self._normalize_document(
            doc_id=f"{region}_{period}",
            text=text,
            event_date=f"{period.start_time.date()}",
            region=region,
            countries=countries,
            primary_agents=["shipping"],
            event_type="chokepoint_traffic_summary",
            severity=severity,
            extra_metadata={
                "chokepoint": chokepoint,
                "mean_daily_transits": round(mean_total, 2),
                "min_daily_transits": round(min_total, 2),
                "baseline_daily_transits": round(baseline, 2),
                "deviation_pct": round(deviation * 100, 1),
                "days_observed": int(len(group)),
            },
        )
