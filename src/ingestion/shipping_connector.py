"""Hybrid AIS-style connector for the Strait of Hormuz corridor.

Supports three ingestion modes selected via ``source_mode``:

- ``"csv"``: Load real daily vessel arrivals from the IMF PortWatch
  Shuaiba (Kuwait) dataset. This is the primary, evidence-grade source
  used for thesis evaluation — it captures the April-May 2026 Strait of
  Hormuz shutdown directly.
- ``"synthetic"``: Generate the legacy daily dataset with three injected
  disruption scenarios. Retained as a fallback so detection agents have
  ground-truth labels to validate against when the real CSV is missing.
- ``"api"``: Live AIS streaming via aisstream.io. Stubbed out — raises
  ``NotImplementedError`` and documents the planned integration.

The connector emits a unified daily-frequency schema that downstream
agents can consume regardless of which mode produced the data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import pandas as pd
from scipy import stats

from src.ingestion.base_connector import BaseConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic-mode constants (preserved from the original implementation).
# ---------------------------------------------------------------------------

_BASELINE_VESSEL_COUNT: float = 70.0
_BASELINE_DELAY_HOURS: float = 5.0
_BASELINE_CONGESTION: float = 0.25
_BASELINE_OIL_PRICE: float = 77.5

_VESSEL_NOISE_SD: float = 4.0
_DELAY_NOISE_SD: float = 1.2
_CONGESTION_NOISE_SD: float = 0.05
_OIL_NOISE_SD: float = 2.0


@dataclass(frozen=True)
class _DisruptionScenario:
    """Configuration for a single injected disruption window (synthetic mode)."""

    name: str
    start_day: int
    end_day: int
    ramp_days: int
    decay_days: int
    vessel_drop_range: tuple[float, float]
    delay_mult_range: tuple[float, float]
    congestion_range: tuple[float, float]
    oil_pct_range: tuple[float, float]


_SCENARIOS: tuple[_DisruptionScenario, ...] = (
    _DisruptionScenario(
        name="Moderate Tension",
        start_day=60,
        end_day=74,
        ramp_days=3,
        decay_days=5,
        vessel_drop_range=(0.20, 0.35),
        delay_mult_range=(2.0, 3.0),
        congestion_range=(0.50, 0.70),
        oil_pct_range=(0.10, 0.15),
    ),
    _DisruptionScenario(
        name="Major Blockage",
        start_day=150,
        end_day=170,
        ramp_days=3,
        decay_days=5,
        vessel_drop_range=(0.50, 0.70),
        delay_mult_range=(4.0, 6.0),
        congestion_range=(0.70, 0.95),
        oil_pct_range=(0.25, 0.40),
    ),
    _DisruptionScenario(
        name="Brief Incident",
        start_day=280,
        end_day=290,
        ramp_days=2,
        decay_days=3,
        vessel_drop_range=(0.10, 0.20),
        delay_mult_range=(1.5, 2.0),
        congestion_range=(0.40, 0.60),
        oil_pct_range=(0.05, 0.10),
    ),
)


# ---------------------------------------------------------------------------
# CSV-mode constants.
# ---------------------------------------------------------------------------

_CSV_DEFAULT_PATH: str = "data/raw/shuaiba_arrivals.csv"
_CSV_VESSEL_TYPE_COLUMNS: tuple[str, ...] = (
    "Container",
    "Dry Bulk",
    "General Cargo",
    "Roll-on/roll-off",
    "Tanker",
)
_CSV_MA_COLUMN: str = "7-day Moving Average"
_CSV_TIMESTAMP_COLUMN: str = "DateTime"

_BASE_DELAY_HOURS: float = 4.0
_DELAY_MIN: float = 1.0
_DELAY_MAX: float = 72.0
_ROLLING_WINDOW: int = 30
_ROLLING_MIN_PERIODS: int = 5
_ANOMALY_SIGMA: float = 2.0
_MIN_RUN_LENGTH: int = 3

# The Strait of Hormuz shutdown window flagged as ground truth even when
# the 2σ persistence rule would also catch it; pinning the dates makes the
# label robust against rolling-window edge effects.
_KNOWN_SHUTDOWN_START: pd.Timestamp = pd.Timestamp("2026-04-01")
_KNOWN_SHUTDOWN_END: pd.Timestamp = pd.Timestamp("2026-05-31")

# --- IMF PortWatch daily chokepoint transits (no credentials required) ------
_PORTWATCH_QUERY_URL: str = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
#: Fields pulled per day. n_total/n_tanker feed the agent-facing columns; the
#: per-type counts travel for provenance and future features.
_PORTWATCH_FIELDS: tuple[str, ...] = (
    "date", "portname", "n_total", "n_tanker", "n_container",
    "n_dry_bulk", "n_general_cargo", "n_roro", "n_cargo", "capacity",
)
#: The layer's maxRecordCount. Truncation is signalled by
#: ``exceededTransferLimit``, so paging follows that flag, not this number.
_PORTWATCH_PAGE_SIZE: int = 2000
#: Only this chokepoint gets the April-May 2026 shutdown label.
_PORTWATCH_HORMUZ: str = "Strait of Hormuz"

_API_TIMEOUT: int = 60
_API_ATTEMPTS: int = 3
_API_BACKOFF_SECONDS: float = 2.0


class ShippingConnector(BaseConnector):
    """Hybrid Strait-of-Hormuz vessel-traffic connector.

    Loads real daily vessel arrivals at Shuaiba port (Kuwait), or falls
    back to a synthetic generator with three injected disruption
    scenarios. A future API mode for live AIS streaming is stubbed out.

    Args:
        source_mode: Ingestion mode — one of ``"csv"``, ``"synthetic"``,
            or ``"api"``. When ``None``, the value is taken from
            ``config["source_mode"]`` and defaults to ``"csv"``.
        config: Connector-specific configuration block — typically
            ``settings["ingestion"]["shipping"]`` from ``settings.yaml``.
    """

    LOCATION: str = "Shuaiba Port, Persian Gulf"
    SOURCE: str = "shipping"

    # Columns emitted by :meth:`load_from_csv`.
    FEATURE_COLUMNS_CSV: tuple[str, ...] = (
        "vessel_count",
        "tanker_count",
        "vessel_count_7dma",
        "avg_delay_hours",
        "congestion_index",
    )
    # Columns emitted by :meth:`generate_synthetic`.
    FEATURE_COLUMNS_SYNTHETIC: tuple[str, ...] = (
        "vessel_count",
        "avg_delay_hours",
        "congestion_index",
        "oil_price_usd",
    )
    # Backwards-compatible alias used by older callers.
    FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS_SYNTHETIC

    def __init__(
        self,
        source_mode: str | None = None,
        config: dict | None = None,
    ) -> None:
        cfg: dict = dict(config) if config is not None else {}
        super().__init__(cfg)
        self.source_mode: str = (source_mode or cfg.get("source_mode") or "csv").lower()
        self.csv_path: str = cfg.get("csv_path", _CSV_DEFAULT_PATH)
        self.vessel_type_columns: list[str] = list(
            cfg.get("vessel_type_columns", _CSV_VESSEL_TYPE_COLUMNS)
        )
        api_cfg: dict = cfg.get("api", {}) or {}
        self.api_endpoint: str | None = api_cfg.get("endpoint")
        self.api_key: str | None = api_cfg.get("key")
        self.api_bounding_box: dict | None = api_cfg.get("bounding_box")
        self.ais_bounds: list | None = self._resolve_ais_bounds(cfg)

    @staticmethod
    def _resolve_ais_bounds(cfg: dict) -> list | None:
        """Resolve the region's AIS bounding box, or ``None`` if unavailable.

        Order of preference:

        1. ``ais_bounds`` — ``[[lat_min, lon_min], [lat_max, lon_max]]``,
           projected from the active region's ``aisstream.bbox`` by
           :func:`src.core.config_manager.load_config_for_region`.
        2. ``api.bounding_box`` — the pre-Phase-11 ``{lat_min, lat_max,
           lon_min, lon_max}`` dict in settings.yaml, converted to the same
           nested-list form.

        Missing bounds are not an error: CSV and synthetic modes never need
        them, and they are the modes the pipeline actually runs in.

        Args:
            cfg: The connector's config block (``ingestion.shipping``).

        Returns:
            Bounds as ``[[lat_min, lon_min], [lat_max, lon_max]]``, or ``None``.
        """
        bounds = cfg.get("ais_bounds")
        if bounds is not None:
            try:
                (lat_min, lon_min), (lat_max, lon_max) = bounds
                normalised = [
                    [float(lat_min), float(lon_min)],
                    [float(lat_max), float(lon_max)],
                ]
            except (TypeError, ValueError) as exc:
                # Malformed bounds degrade to None rather than failing the run;
                # CSV/synthetic modes are unaffected and API mode reports it.
                logger.warning(
                    "[ShippingConnector] ignoring malformed ais_bounds %r "
                    "(expected [[lat_min, lon_min], [lat_max, lon_max]]): %s",
                    bounds,
                    exc,
                )
                return None
            logger.info(
                "[ShippingConnector] region-specific AIS bounds: %s", normalised
            )
            return normalised

        box = cfg.get("api", {}) or {}
        box = box.get("bounding_box") or {}
        if all(k in box for k in ("lat_min", "lon_min", "lat_max", "lon_max")):
            derived = [
                [float(box["lat_min"]), float(box["lon_min"])],
                [float(box["lat_max"]), float(box["lon_max"])],
            ]
            logger.debug(
                "[ShippingConnector] no ais_bounds; derived from "
                "api.bounding_box: %s",
                derived,
            )
            return derived

        logger.debug(
            "[ShippingConnector] no ais_bounds configured — CSV / synthetic "
            "modes do not need them."
        )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """Route to the configured ingestion mode and validate the result.

        Returns:
            Validated, cleaned daily-frequency DataFrame.

        Raises:
            ValueError: If ``self.source_mode`` is not recognised.
            NotImplementedError: If ``source_mode == "api"``.
            FileNotFoundError: If CSV mode is selected but the file is missing.
        """
        mode = self.source_mode
        if mode == "csv":
            df = self.load_from_csv()
        elif mode == "synthetic":
            days = int(self.config.get("days", 365))
            seed = int(self.config.get("seed", 42))
            df = self.generate_synthetic(days=days, seed=seed)
        elif mode == "api":
            df = self.fetch_from_api()
        else:
            raise ValueError(
                f"Unknown shipping source_mode={self.source_mode!r}; "
                "expected 'csv', 'synthetic', or 'api'."
            )
        df = self.validate(df)
        self._log_summary(df, mode)
        return df

    def fetch_and_validate(self) -> pd.DataFrame:
        """Override base implementation — :meth:`fetch` already validates."""
        return self.fetch()

    def load_from_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load and transform the IMF PortWatch Shuaiba arrivals CSV.

        The raw CSV records daily vessel arrivals at Shuaiba port (Kuwait,
        Persian Gulf, downstream of the Strait of Hormuz). Vessel counts
        are summed across types, congestion and delay are derived from a
        30-day rolling baseline, and a ground-truth ``is_disruption``
        label is computed from a 2σ persistence rule plus the known
        April-May 2026 shutdown window.

        Args:
            path: Optional override for the CSV location. When ``None``,
                ``self.csv_path`` from config is used.

        Returns:
            DataFrame with columns ``timestamp``, ``vessel_count``,
            ``tanker_count``, ``vessel_count_7dma``, ``avg_delay_hours``,
            ``congestion_index``, ``oil_price_usd`` (NaN — filled later by
            the market connector), and ``is_disruption``.

        Raises:
            FileNotFoundError: If the CSV file cannot be located.
            ValueError: If expected vessel-type columns are missing.
        """
        csv_path = Path(path) if path is not None else Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Shuaiba arrivals CSV not found at {csv_path}. "
                "Set ingestion.shipping.csv_path in config/settings.yaml or "
                "place the file at the default location."
            )

        raw = pd.read_csv(csv_path)
        missing = [c for c in self.vessel_type_columns if c not in raw.columns]
        if missing:
            raise ValueError(
                f"CSV missing expected vessel-type columns: {missing}"
            )

        df = pd.DataFrame()
        df["timestamp"] = pd.to_datetime(raw[_CSV_TIMESTAMP_COLUMN])
        df["vessel_count"] = (
            raw[self.vessel_type_columns].sum(axis=1).astype(float)
        )
        df["tanker_count"] = raw["Tanker"].astype(float)
        df["vessel_count_7dma"] = pd.to_numeric(
            raw.get(_CSV_MA_COLUMN), errors="coerce"
        )

        df = self._derive_daily_features(df, apply_known_shutdown=True)
        self._report_csv_stats(df)
        return df

    def _derive_daily_features(
        self, df: pd.DataFrame, *, apply_known_shutdown: bool
    ) -> pd.DataFrame:
        """Derive the agent-facing feature columns from daily vessel counts.

        Shared by :meth:`load_from_csv` and :meth:`fetch_from_api` so both
        sources produce identical semantics — ``congestion_index`` and
        ``avg_delay_hours`` are transforms of ``vessel_count`` against its own
        rolling baseline, not independent measurements, and defining them
        twice would let the two paths drift apart.

        Args:
            df: Frame with ``timestamp``, ``vessel_count``, ``tanker_count``
                and ``vessel_count_7dma``, in any row order.
            apply_known_shutdown: Whether to force ``is_disruption`` true
                across the April-May 2026 Hormuz shutdown window. **Only true
                for Hormuz.** That shutdown is a Hormuz event; applying it to
                Panama or Malacca would label unrelated days as disrupted and
                silently corrupt the ground truth.

        Returns:
            The frame with derived columns, ordered by timestamp.
        """
        df = df.sort_values("timestamp").reset_index(drop=True)

        rolling_mean = df["vessel_count"].rolling(
            window=_ROLLING_WINDOW, min_periods=_ROLLING_MIN_PERIODS
        ).mean()
        rolling_std = df["vessel_count"].rolling(
            window=_ROLLING_WINDOW, min_periods=_ROLLING_MIN_PERIODS
        ).std()

        overall_mean = float(df["vessel_count"].mean())
        overall_std = float(df["vessel_count"].std(ddof=0))
        rolling_mean_filled = rolling_mean.fillna(overall_mean)
        rolling_std_filled = rolling_std.fillna(overall_std)

        # congestion_index ∈ [0, 1]: rises when vessel arrivals fall below
        # the local rolling baseline (interpreted as upstream bottleneck).
        denom = rolling_mean_filled.replace(0.0, np.nan)
        congestion = 1.0 - df["vessel_count"] / denom
        df["congestion_index"] = (
            congestion.clip(lower=0.0, upper=1.0).fillna(0.0)
        )

        # avg_delay_hours ∈ [1, 72]: inverse-proportional to vessel count
        # relative to baseline; floor on vessel_count avoids div-by-zero.
        vessel_floor = df["vessel_count"].clip(lower=0.1)
        delay = _BASE_DELAY_HOURS * (rolling_mean_filled / vessel_floor)
        df["avg_delay_hours"] = delay.clip(lower=_DELAY_MIN, upper=_DELAY_MAX)

        # Ground-truth disruption label: 2σ drop persisting ≥ 3 consecutive days.
        threshold = rolling_mean_filled - _ANOMALY_SIGMA * rolling_std_filled
        below = (df["vessel_count"] < threshold).to_numpy()
        flags = self._flag_persistent_runs(below, _MIN_RUN_LENGTH)
        if apply_known_shutdown:
            flags = flags | df["timestamp"].between(
                _KNOWN_SHUTDOWN_START, _KNOWN_SHUTDOWN_END
            ).to_numpy()
        df["is_disruption"] = flags

        # oil_price_usd is provided by the market connector; placeholder here.
        df["oil_price_usd"] = np.nan

        return df[[
            "timestamp",
            "vessel_count",
            "tanker_count",
            "vessel_count_7dma",
            "avg_delay_hours",
            "congestion_index",
            "oil_price_usd",
            "is_disruption",
        ]]

    def generate_synthetic(self, days: int = 365, seed: int = 42) -> pd.DataFrame:
        """Generate the legacy daily synthetic Hormuz dataset.

        This is the testing fallback when real CSV data is unavailable.
        Three disruption scenarios (Moderate Tension, Major Blockage,
        Brief Incident) are injected with gradual onset and recovery so
        detection agents see realistic ground-truth labels.

        Args:
            days: Number of consecutive days to simulate.
            seed: NumPy random seed for reproducibility.

        Returns:
            DataFrame with columns ``timestamp``, ``vessel_count``,
            ``avg_delay_hours``, ``congestion_index``, ``oil_price_usd``,
            and the ground-truth label ``is_disruption``.
        """
        if days <= 0:
            raise ValueError("days must be positive.")

        rng = np.random.default_rng(seed)
        timestamps = pd.date_range(start="2025-01-01", periods=days, freq="D")

        vessel = rng.normal(_BASELINE_VESSEL_COUNT, _VESSEL_NOISE_SD, size=days)
        delay = rng.normal(_BASELINE_DELAY_HOURS, _DELAY_NOISE_SD, size=days)
        delay = np.clip(delay, 1.5, None)
        congestion = rng.normal(_BASELINE_CONGESTION, _CONGESTION_NOISE_SD, size=days)
        oil = rng.normal(_BASELINE_OIL_PRICE, _OIL_NOISE_SD, size=days)
        is_disruption = np.zeros(days, dtype=bool)

        for scenario in _SCENARIOS:
            self._apply_scenario(
                rng=rng,
                scenario=scenario,
                vessel=vessel,
                delay=delay,
                congestion=congestion,
                oil=oil,
                is_disruption=is_disruption,
                total_days=days,
            )

        congestion = np.clip(congestion, 0.0, 1.0)
        vessel_int = np.clip(np.round(vessel).astype(int), 0, None)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "vessel_count": vessel_int,
                "avg_delay_hours": np.round(delay, 2),
                "congestion_index": np.round(congestion, 3),
                "oil_price_usd": np.round(oil, 2),
                "is_disruption": is_disruption,
            }
        )

        self._report_separation(df)
        return df

    def generate_dataset(self, days: int = 365, seed: int = 42) -> pd.DataFrame:
        """Backwards-compatible alias for :meth:`generate_synthetic`."""
        return self.generate_synthetic(days=days, seed=seed)

    def _fetch_portwatch_rows(self, chokepoint: str) -> list[dict]:
        """Page the full daily history for one chokepoint.

        The feature service caps a response at ``maxRecordCount`` (2,000) and
        signals truncation with ``exceededTransferLimit``, so paging is driven
        by that flag rather than by guessing the row count. Requests are
        retried: the service returns intermittent errors under load, and a
        dropped page would silently punch a hole in the series.

        Args:
            chokepoint: Name exactly as spelled in the layer's ``portname``.

        Returns:
            Raw attribute dicts, one per day.

        Raises:
            RuntimeError: If a page cannot be fetched after retrying, rather
                than returning a short series that looks like quiet days.
        """
        rows: list[dict] = []
        offset = 0
        while True:
            payload = None
            for attempt in range(1, _API_ATTEMPTS + 1):
                try:
                    response = requests.get(
                        _PORTWATCH_QUERY_URL,
                        params={
                            "where": f"portname='{chokepoint}'",
                            "outFields": ",".join(_PORTWATCH_FIELDS),
                            "orderByFields": "date",
                            "resultOffset": offset,
                            "resultRecordCount": _PORTWATCH_PAGE_SIZE,
                            "returnGeometry": "false",
                            "f": "json",
                        },
                        timeout=_API_TIMEOUT,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    break
                except Exception as exc:
                    if attempt == _API_ATTEMPTS:
                        raise RuntimeError(
                            f"PortWatch page at offset {offset} for "
                            f"{chokepoint!r} failed after {_API_ATTEMPTS} "
                            f"attempts: {exc}"
                        ) from exc
                    logger.warning(
                        "[ShippingConnector/PortWatch] offset=%d attempt %d/%d "
                        "failed (%s) — retrying",
                        offset, attempt, _API_ATTEMPTS, exc,
                    )
                    time.sleep(_API_BACKOFF_SECONDS * attempt)

            page = [f.get("attributes", {}) for f in payload.get("features", [])]
            rows.extend(page)
            if not payload.get("exceededTransferLimit") or not page:
                return rows
            offset += len(page)

    def fetch_from_api(self) -> pd.DataFrame:
        """Fetch daily chokepoint transits from the IMF PortWatch API.

        Same provider as :meth:`load_from_csv`, which reads a static PortWatch
        export for a single Persian Gulf port. This reads PortWatch's
        ``Daily_Chokepoints_Data`` layer instead: daily transit counts at the
        chokepoint itself, for every region, from 2019-01-01 to the present,
        refreshed daily. No credentials.

        ``n_total`` becomes ``vessel_count`` and ``n_tanker`` becomes
        ``tanker_count``; the remaining columns are derived by
        :meth:`_derive_daily_features`, exactly as for the CSV path.

        Returns:
            Daily-frequency frame in the connector's standard schema.

        Raises:
            ValueError: If the region has no ``portwatch_chokepoint``
                configured, or the API returns no rows for it — an empty
                series here means a name mismatch, not a quiet chokepoint.
        """
        chokepoint = str(self.config.get("portwatch_chokepoint") or "").strip()
        if not chokepoint:
            raise ValueError(
                "No portwatch_chokepoint configured for this region. Set "
                "ingestion.shipping.portwatch_chokepoint in the region "
                "overlay (config/regions/<region>.yaml)."
            )

        rows = self._fetch_portwatch_rows(chokepoint)
        if not rows:
            raise ValueError(
                f"PortWatch returned no rows for portname={chokepoint!r}. "
                "The name must match the layer's spelling exactly, e.g. "
                "'Strait of Hormuz', 'Bab el-Mandeb Strait'."
            )

        raw = pd.DataFrame(rows)
        df = pd.DataFrame()
        df["timestamp"] = pd.to_datetime(raw["date"])
        df["vessel_count"] = pd.to_numeric(raw["n_total"], errors="coerce").astype(float)
        df["tanker_count"] = pd.to_numeric(raw["n_tanker"], errors="coerce").astype(float)
        df = df.dropna(subset=["timestamp", "vessel_count"])
        df["vessel_count_7dma"] = (
            df.sort_values("timestamp")["vessel_count"]
            .rolling(window=7, min_periods=1).mean()
        )

        # The April-May 2026 shutdown is a Hormuz event; labelling another
        # region's days from it would corrupt that region's ground truth.
        is_hormuz = chokepoint == _PORTWATCH_HORMUZ

        df = self._derive_daily_features(df, apply_known_shutdown=is_hormuz)
        logger.info(
            "[ShippingConnector/PortWatch] %s -> %d days [%s .. %s] "
            "mean_vessels=%.1f disruption_days=%d",
            chokepoint, len(df), df["timestamp"].min().date(),
            df["timestamp"].max().date(), df["vessel_count"].mean(),
            int(df["is_disruption"].sum()),
        )
        return df

    # Synthetic-schema CSV columns produced by generate_synthetic / save_raw.
    _SYNTHETIC_SCHEMA: tuple[str, ...] = (
        "timestamp",
        "vessel_count",
        "avg_delay_hours",
        "congestion_index",
        "oil_price_usd",
        "is_disruption",
    )

    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load a *synthetic-schema* shipping CSV (the :meth:`save_raw` format).

        This is the round-trip counterpart to :meth:`save_raw`: it reads back
        a file written by ``generate_synthetic → save_raw`` and returns it as
        a typed DataFrame. It is distinct from :meth:`load_from_csv`, which
        parses the very different real-world IMF PortWatch arrivals layout.

        Args:
            path: CSV location. When ``None``, ``config["csv_path"]`` is used,
                defaulting to ``data/raw/shipping_hormuz.csv``.

        Returns:
            DataFrame with the six synthetic-schema columns, ``timestamp``
            parsed to datetime and ``is_disruption`` cast to ``bool``.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing or ``timestamp`` /
                ``vessel_count`` contain NaN (non-recoverable schema breaks).
        """
        csv_path = Path(path) if path is not None else Path(
            self.config.get("csv_path", "data/raw/shipping_hormuz.csv")
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"Shipping CSV not found at {csv_path}.")

        df = pd.read_csv(csv_path)
        missing = [c for c in self._SYNTHETIC_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(
                f"Shipping CSV {csv_path} missing required columns: {missing}"
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["is_disruption"] = df["is_disruption"].astype(bool)
        for col in ("vessel_count", "avg_delay_hours", "congestion_index", "oil_price_usd"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df["timestamp"].isna().any() or df["vessel_count"].isna().any():
            raise ValueError(
                f"Shipping CSV {csv_path} has NaN in critical columns "
                "('timestamp' / 'vessel_count')."
            )
        return df[list(self._SYNTHETIC_SCHEMA)]

    def fetch_api(self) -> pd.DataFrame:
        """Graceful API fallback — warn and return a synthetic dataset.

        Unlike :meth:`fetch_from_api` (which hard-raises for the planned live
        aisstream.io integration), this convenience hook never fails: it logs
        a warning and returns :meth:`generate_dataset` so callers selecting
        ``api`` mode before keys are configured still get usable data.

        Returns:
            A freshly generated synthetic shipping DataFrame.
        """
        logger.warning("API mode not configured — using synthetic")
        return self.generate_dataset(
            days=int(self.config.get("days", 365)),
            seed=int(self.config.get("seed", 42)),
        )

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schema, domain, and gap checks; returns a cleaned DataFrame.

        Args:
            df: DataFrame returned by :meth:`load_from_csv` or
                :meth:`generate_synthetic`.

        Returns:
            Cleaned DataFrame with rows sorted by timestamp and small
            (≤ 2-day) gaps forward-filled.

        Raises:
            AssertionError: If a non-recoverable schema or domain check fails.
        """
        assert "timestamp" in df.columns, "Missing 'timestamp' column."
        assert "vessel_count" in df.columns, "Missing 'vessel_count' column."

        df = df.sort_values("timestamp").reset_index(drop=True).copy()

        assert df["timestamp"].notna().all(), "NaN values in 'timestamp'."
        assert df["vessel_count"].notna().all(), "NaN values in 'vessel_count'."
        assert (df["vessel_count"] >= 0).all(), "Negative 'vessel_count' values."
        if "congestion_index" in df.columns:
            ci = df["congestion_index"].dropna()
            assert ci.between(0.0, 1.0).all(), (
                "'congestion_index' out of [0, 1] for non-NaN values."
            )
        assert df["timestamp"].is_monotonic_increasing, (
            "Timestamps must be monotonically increasing."
        )

        gaps = df["timestamp"].diff().dt.days.dropna().astype(int)
        large_gaps = gaps[gaps > 2]
        if not large_gaps.empty:
            logger.warning(
                "Shipping data has %d gap(s) > 2 days (max %d days).",
                len(large_gaps),
                int(large_gaps.max()),
            )
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill(limit=2)
        return df

    def save_raw(
        self,
        df: pd.DataFrame | None = None,
        path: str | Path = "data/raw/shipping_processed.csv",
    ) -> Path:
        """Persist a processed dataset as CSV.

        Args:
            df: DataFrame to write. When ``None``, :meth:`fetch` is invoked
                so the connector loads from its configured source.
            path: Destination CSV path. Parent directories are created.

        Returns:
            Resolved absolute path of the written file.
        """
        if df is None:
            df = self.fetch()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Wrote %d rows to %s", len(df), out)
        return out.resolve()

    def to_unified_signals(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert a dataset into the unified signal-record schema.

        Emits one record per (timestamp, feature) pair. Excludes the
        ground-truth ``is_disruption`` label and drops NaN-valued cells so
        the output is always JSON-serialisable.

        Args:
            df: DataFrame returned by :meth:`load_from_csv` or
                :meth:`generate_synthetic`.

        Returns:
            List of dicts with keys ``timestamp``, ``source``, ``feature``,
            ``value``, ``location``.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {"timestamp", "is_disruption"}
        ]
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).isoformat()
            for feature in feature_cols:
                val = row[feature]
                if pd.isna(val):
                    continue
                records.append(
                    {
                        "timestamp": ts,
                        "source": self.SOURCE,
                        "feature": feature,
                        "value": float(val),
                        "location": self.LOCATION,
                    }
                )
        return records

    def to_signal_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Backwards-compatible alias for :meth:`to_unified_signals`."""
        return self.to_unified_signals(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_scenario(
        self,
        rng: np.random.Generator,
        scenario: _DisruptionScenario,
        vessel: np.ndarray,
        delay: np.ndarray,
        congestion: np.ndarray,
        oil: np.ndarray,
        is_disruption: np.ndarray,
        total_days: int,
    ) -> None:
        """Mutate per-feature arrays in place to inject one synthetic disruption."""
        start = scenario.start_day
        end = min(scenario.end_day, total_days - 1)
        if start >= total_days:
            return

        vessel_drop = rng.uniform(*scenario.vessel_drop_range)
        delay_mult = rng.uniform(*scenario.delay_mult_range)
        target_congestion = rng.uniform(*scenario.congestion_range)
        oil_pct = rng.uniform(*scenario.oil_pct_range)

        window_len = end - start + 1
        ramp = min(scenario.ramp_days, max(window_len // 3, 1))
        decay = min(scenario.decay_days, max(window_len - ramp - 1, 1))

        for offset, day_idx in enumerate(range(start, end + 1)):
            if offset < ramp:
                intensity = (offset + 1) / (ramp + 1)
            elif offset >= window_len - decay:
                tail_offset = offset - (window_len - decay)
                intensity = 1.0 - (tail_offset + 1) / (decay + 1)
            else:
                intensity = 1.0

            vessel[day_idx] *= 1.0 - vessel_drop * intensity
            delay[day_idx] *= 1.0 + (delay_mult - 1.0) * intensity
            congestion[day_idx] = (
                congestion[day_idx] * (1.0 - intensity)
                + target_congestion * intensity
            )
            oil[day_idx] *= 1.0 + oil_pct * intensity
            is_disruption[day_idx] = True

        logger.info(
            "Injected scenario '%s' days %d-%d "
            "(vessel -%.0f%%, delay x%.2f, congestion %.2f, oil +%.0f%%)",
            scenario.name,
            start,
            end,
            vessel_drop * 100,
            delay_mult,
            target_congestion,
            oil_pct * 100,
        )

    @staticmethod
    def _flag_persistent_runs(arr: np.ndarray, min_length: int) -> np.ndarray:
        """Flag positions belonging to runs of ``True`` of length ≥ ``min_length``.

        Args:
            arr: Boolean (or 0/1) array.
            min_length: Minimum run length required to flag.

        Returns:
            Boolean array of the same shape; ``True`` at positions whose
            enclosing consecutive-``True`` run has length ≥ ``min_length``.
        """
        n = arr.shape[0]
        out = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if arr[i]:
                j = i
                while j < n and arr[j]:
                    j += 1
                if (j - i) >= min_length:
                    out[i:j] = True
                i = j
            else:
                i += 1
        return out

    def _log_summary(self, df: pd.DataFrame, mode: str) -> None:
        """Emit a one-line summary covering rows, date range, and label count."""
        n = len(df)
        n_dis = (
            int(df["is_disruption"].sum())
            if "is_disruption" in df.columns else 0
        )
        ts_min = pd.Timestamp(df["timestamp"].min()).date()
        ts_max = pd.Timestamp(df["timestamp"].max()).date()
        msg = (
            f"[ShippingConnector] mode='{mode}' rows={n} "
            f"range=[{ts_min} .. {ts_max}] disruption_days={n_dis}"
        )
        logger.info(msg)
        print(msg)

    @staticmethod
    def _report_csv_stats(df: pd.DataFrame) -> None:
        """Print descriptive stats + a Welch t-test for the CSV mode."""
        vc = df["vessel_count"].to_numpy(dtype=float)
        ts_min = pd.Timestamp(df["timestamp"].min()).date()
        ts_max = pd.Timestamp(df["timestamp"].max()).date()
        n_disrupt = int(df["is_disruption"].sum())
        header = (
            f"[ShippingConnector/CSV] rows={len(df)} "
            f"range=[{ts_min} .. {ts_max}] "
            f"vessel_count mean={vc.mean():.2f} std={vc.std():.2f} "
            f"min={vc.min():.0f} max={vc.max():.0f} "
            f"disruption_days={n_disrupt}"
        )
        logger.info(header)
        print(header)

        normal = df.loc[~df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        if len(normal) > 1 and len(disrupted) > 1:
            t_stat, p_val = stats.ttest_ind(normal, disrupted, equal_var=False)
            sep = (
                f"[ShippingConnector/CSV] Welch t-test vessel_count "
                f"(normal n={len(normal)} vs disruption n={len(disrupted)}): "
                f"t={float(t_stat):.2f}, p={float(p_val):.2e}"
            )
            logger.info(sep)
            print(sep)

    @staticmethod
    def _report_separation(df: pd.DataFrame) -> None:
        """Print a Welch-style t-statistic separating disruption vs normal (synthetic)."""
        normal = df.loc[~df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        disrupted = df.loc[df["is_disruption"], "vessel_count"].to_numpy(dtype=float)
        if len(disrupted) == 0 or len(normal) == 0:
            return
        diff = normal.mean() - disrupted.mean()
        pooled_se = np.sqrt(
            normal.var(ddof=1) / len(normal)
            + disrupted.var(ddof=1) / len(disrupted)
        )
        t_stat = diff / pooled_se if pooled_se > 0 else float("inf")
        msg = (
            f"[ShippingConnector] vessel_count separation — "
            f"normal mean={normal.mean():.2f} (n={len(normal)}), "
            f"disruption mean={disrupted.mean():.2f} (n={len(disrupted)}), "
            f"Welch t={t_stat:.2f}"
        )
        logger.info(msg)
        print(msg)
