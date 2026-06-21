"""FRED API extractor for market/economic signals.

Covers: market agent. Free API key from https://fred.stlouisfed.org/.
Rate limit: 120 requests/minute.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

FRED_SERIES: dict[str, dict] = {
    "DCOILBRENTEU": {"name": "Brent Crude Oil Price", "description": "Europe Brent Spot Price FOB (Dollars per Barrel)"},
    "DCOILWTICO": {"name": "WTI Crude Oil Price", "description": "Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"},
    "DTWEXBGS": {"name": "Trade Weighted US Dollar Index", "description": "Broad goods and services trade-weighted dollar index"},
    "BAMLH0A0HYM2": {"name": "High Yield Bond Spread", "description": "ICE BofA US High Yield Index Option-Adjusted Spread"},
}

DISRUPTION_PERIODS: list[dict] = [
    {"name": "hormuz_2019_tanker_attacks", "start": "2019-05-01", "end": "2019-08-31",
     "region": "hormuz", "agents": ["market", "geopolitical"]},
    {"name": "ever_given_2021", "start": "2021-03-15", "end": "2021-04-30",
     "region": "suez", "agents": ["market", "routing", "shipping"]},
    {"name": "iran_sanctions_2012", "start": "2011-11-01", "end": "2013-03-31",
     "region": "hormuz", "agents": ["market", "geopolitical"]},
    {"name": "houthi_red_sea_2024", "start": "2023-11-01", "end": "2024-06-30",
     "region": "red_sea", "agents": ["market", "geopolitical"]},
    {"name": "cyclone_gonu_2007", "start": "2007-05-01", "end": "2007-07-31",
     "region": "hormuz", "agents": ["market", "natural_disaster"]},
]


class FREDExtractor(BaseExtractor):
    """Extract economic/market data from the Federal Reserve FRED API."""

    @property
    def source_name(self) -> str:
        return "fred"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.api_key = self._api_key("fred")
        if not self.api_key:
            logger.warning("FRED API key not configured — set api_keys.fred / FRED_API_KEY.")

    def _get_series_observations(self, series_id: str, start_date: str = "", end_date: str = "") -> list[dict]:
        if not self.api_key:
            return []
        self._rate_limit_wait()
        url = f"{FRED_BASE_URL}/series/observations"
        params: dict[str, Any] = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("FRED request failed for %s: %s", series_id, exc)
            return []

        results = []
        for obs in data.get("observations", []):
            val = obs.get("value", ".")
            if val == ".":
                continue
            results.append({"date": obs["date"], "value": float(val)})
        return results

    @staticmethod
    def _compute_disruption_metrics(observations: list[dict], window: int = 30) -> dict:
        if len(observations) < window:
            return {"spike_pct": 0.0, "volatility": 0.0}

        values = [o["value"] for o in observations]
        pre_window = values[:window] if len(values) > window else values
        avg_baseline = sum(pre_window) / len(pre_window)
        max_val, min_val = max(values), min(values)
        spike_pct = ((max_val - avg_baseline) / avg_baseline * 100) if avg_baseline else 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        volatility = (variance ** 0.5) / mean * 100 if mean else 0.0

        return {
            "max_value": round(max_val, 2),
            "min_value": round(min_val, 2),
            "avg_baseline": round(avg_baseline, 2),
            "spike_pct": round(spike_pct, 2),
            "volatility": round(volatility, 2),
        }

    def extract_historical(self, region: str, **kwargs) -> list[dict]:
        documents: list[dict] = []
        region_periods = [p for p in DISRUPTION_PERIODS if p["region"] == region]

        for period in region_periods:
            for series_id, series_info in FRED_SERIES.items():
                observations = self._get_series_observations(series_id, period["start"], period["end"])
                if not observations:
                    continue

                metrics = self._compute_disruption_metrics(observations)
                logger.info(
                    "FRED [%s] %s -> %d obs, spike=%.1f%%",
                    period["name"], series_id, len(observations), metrics["spike_pct"],
                )

                text = (
                    f"During {period['name'].replace('_', ' ')}, "
                    f"{series_info['name']} ({series_info['description']}) "
                    f"showed a price spike of {metrics['spike_pct']:.1f}% above the 30-day baseline "
                    f"average of ${metrics.get('avg_baseline', 0):.2f}. Price ranged from "
                    f"${metrics.get('min_value', 0):.2f} to ${metrics.get('max_value', 0):.2f} "
                    f"with {metrics['volatility']:.1f}% volatility. Period: {period['start']} to {period['end']}."
                )

                if metrics["spike_pct"] > 20:
                    severity = "high"
                elif metrics["spike_pct"] > 8:
                    severity = "medium"
                else:
                    severity = "low"

                documents.append(self._normalize_document(
                    doc_id=f"{period['name']}_{series_id}",
                    text=text,
                    event_date=period["start"],
                    region=region,
                    countries=self.config.get("extraction", {}).get("chokepoints", {}).get(region, {}).get("countries", []),
                    primary_agents=period["agents"],
                    event_type=f"market_{series_id.lower()}",
                    severity=severity,
                    extra_metadata={
                        "series_id": series_id,
                        "series_name": series_info["name"],
                        "spike_pct": metrics["spike_pct"],
                        "volatility": metrics["volatility"],
                        "disruption_period": period["name"],
                    },
                ))

        return documents
