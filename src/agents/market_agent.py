"""Market anomaly detection agent for Brent / trade volume / freight signals.

This agent is the secondary signal source in the multi-agent DSS pipeline.
Market features (oil price, trade throughput, freight rates) react to
physical-flow disruptions with a 1-2 day lag and exhibit strong temporal
autocorrelation, so the detection strategy here differs from
:class:`~src.agents.shipping_agent.ShippingAgent`:

* No Isolation Forest. Instead, **rolling-window Z-scores** (default 30-day
  trailing window) are computed per feature on every call — this adapts to
  slow drift in the underlying time series and avoids the look-ahead bias
  a global StandardScaler would introduce on autocorrelated data.
* The combined anomaly score is a **weighted mean of absolute z-scores**.
  In the synthetic 3-feature mode the weights are (oil 0.40, trade 0.35,
  freight 0.25). When the optional FRED-mode ``freight_services_pct_change``
  feature is present, the agent redistributes to (oil 0.35, trade 0.30,
  freight 0.20, freight-services 0.15).
* Validation requires the **oil price** z-score plus **at least one** of
  trade volume / freight rate to clear ``z_threshold`` simultaneously —
  a hard-coded asymmetry, since oil is the most direct price-side signal
  of a Hormuz disruption and a move in trade volume or freight without a
  corresponding oil move is far more likely to be ordinary market noise.

The agent supports both real and synthetic data:

- Synthetic mode emits the three base features (no schema change).
- CSV (FRED) mode emits ``freight_services_pct_change`` in addition.
  Because FRED Brent history reaches back to 1987, the agent applies a
  trailing **recent-baseline filter** (default 5 years) before computing
  rolling stats so older price regimes ($20/bbl in the 1990s vs $100+
  post-2022) do not pollute the anomaly baseline.

The agent is deterministic and stateless across :meth:`run` invocations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent, DetectionResult

logger = logging.getLogger(__name__)


_FEATURE_COLUMNS: tuple[str, ...] = (
    "brent_crude_usd",
    "trade_volume_index",
    "freight_rate_index",
)
_OPTIONAL_FREIGHT_SERVICES: str = "freight_services_pct_change"

_DEFAULT_LOCATION: str = "Strait of Hormuz"
_REAL_DATA_LOCATION: str = "Global/Persian Gulf"

_PERSISTENCE_DAYS: int = 2
_DEFAULT_WINDOW: int = 30
_DEFAULT_BASELINE_YEARS: int = 5
_OIL_FEATURE: str = "brent_crude_usd"

# Synthetic-mode weights (three features).
_FEATURE_WEIGHTS_3: dict[str, float] = {
    "brent_crude_usd": 0.40,
    "trade_volume_index": 0.35,
    "freight_rate_index": 0.25,
}
# FRED-mode weights (four features) — redistribute to keep oil dominant but
# leave room for the freight-services momentum signal.
_FEATURE_WEIGHTS_4: dict[str, float] = {
    "brent_crude_usd": 0.35,
    "trade_volume_index": 0.30,
    "freight_rate_index": 0.20,
    "freight_services_pct_change": 0.15,
}

# Map active feature names → emitted z-score column names. The three base
# entries are always present; the fourth appears only when the optional
# FRED feature is detected at fit time.
_ZSCORE_NAME_MAP: dict[str, str] = {
    "brent_crude_usd": "oil_zscore",
    "trade_volume_index": "trade_volume_zscore",
    "freight_rate_index": "freight_zscore",
    _OPTIONAL_FREIGHT_SERVICES: "freight_services_zscore",
}


class MarketAgent(BaseAgent):
    """Rolling Z-score anomaly detector for market signals.

    Each row is scored by computing per-feature z-scores against a trailing
    rolling window, taking a feature-weighted mean of their absolute
    values, and normalising the result by ``z_threshold`` so that the
    combined ``anomaly_score`` lies in ``[0, 1]`` (saturating at 1.0 when
    the weighted |z| equals the configured threshold).

    Validation gates the raw flag with two checks:

    1. **Persistence** — the row must sit in a run of ``>= 2`` flagged days.
    2. **Oil-led multi-feature** — the oil-price z-score must exceed
       ``z_threshold`` AND at least one of the trade-volume / freight-rate
       z-scores must also exceed ``z_threshold``. An oil spike on its own,
       or a freight spike with quiet oil, is treated as ordinary noise.

    Configuration (read from ``config['agents']['market']`` in
    ``settings.yaml``):

    ============== ======= =====================================================
    Key            Default Description
    ============== ======= =====================================================
    z_threshold    2.5     Per-feature absolute z-score that counts as elevated
    threshold      0.55    Min combined score to set ``is_anomaly=True``
    window         30      Trailing rolling-window length, in days
    baseline_years 5       Recent-history horizon (in years) used for rolling
                           stats. Long FRED histories are clipped to the last
                           ``baseline_years`` so old price regimes don't bias
                           the baseline. Set to ``0`` to disable the clip.
    location       auto    Override emitted ``location``; auto-detected from
                           data otherwise (Global/Persian Gulf when the FRED
                           freight-services column is present, Strait of
                           Hormuz for synthetic).
    ============== ======= =====================================================

    Args:
        config: Agent-specific configuration block. The keys above are
            consulted; sensible defaults apply for any missing key.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="market", config=dict(config or {}))
        self._feature_columns: list[str] = list(_FEATURE_COLUMNS)
        self._extra_features: list[str] = []
        self._z_threshold: float = float(self.config.get("z_threshold", 2.5))
        self._threshold: float = float(self.config.get("threshold", 0.55))
        self._window: int = int(self.config.get("window", _DEFAULT_WINDOW))
        self._baseline_years: int = int(
            self.config.get("baseline_years", _DEFAULT_BASELINE_YEARS)
        )
        self._resolved_location: str = str(
            self.config.get("location") or _DEFAULT_LOCATION
        )
        # Optional override of the three base feature weights (oil / trade /
        # freight). When set, it replaces the module-level schedule; the
        # optional freight-services weight keeps its default share.
        self._weight_override: dict[str, float] | None = None
        weights_cfg = self.config.get("weights") or {}
        if {"oil", "trade_volume", "freight"} & set(weights_cfg):
            self.set_weights(
                float(weights_cfg.get("oil", 0.40)),
                float(weights_cfg.get("trade_volume", 0.35)),
                float(weights_cfg.get("freight", 0.25)),
            )

    def set_weights(self, oil: float, trade_volume: float, freight: float) -> None:
        """Override the oil / trade-volume / freight feature weights.

        Values are normalised to sum to 1.0 across the three base features.
        When the optional FRED ``freight_services_pct_change`` feature is
        active, the three base weights are scaled to occupy 0.85 of the mass
        and the freight-services feature keeps a fixed 0.15 share (mirroring
        the hand-tuned 4-feature schedule).

        Args:
            oil: Raw weight on the Brent-crude z-score.
            trade_volume: Raw weight on the trade-volume z-score.
            freight: Raw weight on the freight-rate z-score.
        """
        total = oil + trade_volume + freight
        if total <= 0:
            raise ValueError("MarketAgent.set_weights: weights must sum to > 0.")
        self._weight_override = {
            "brent_crude_usd": oil / total,
            "trade_volume_index": trade_volume / total,
            "freight_rate_index": freight / total,
        }

    def set_threshold(self, threshold: float) -> None:
        """Override the minimum combined score to flag a row anomalous."""
        self._threshold = float(threshold)

    def set_z_threshold(self, z_threshold: float) -> None:
        """Override the per-feature absolute z-score elevation threshold."""
        self._z_threshold = float(z_threshold)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> None:
        """Validate schema, discover optional features, mark the agent fitted.

        Rolling Z-scores are computed inline during :meth:`preprocess`, so
        no global parameters need to be learned. ``fit`` exists to satisfy
        the :class:`BaseAgent` ABC contract, fail fast on schema breaks,
        and pin the active feature set + location for the run.

        Args:
            df: Historical market frame; must contain the three base
                feature columns. ``freight_services_pct_change``,
                ``timestamp``, and ``is_disruption`` are optional.
        """
        self._validate_columns(df)
        self._discover_optional_features(df)
        self._resolve_location(df)
        self._is_fitted = True
        logger.info(
            "[MarketAgent.fit] schema validated | features=%s | window=%d | "
            "z_threshold=%.2f | baseline_years=%d",
            self._active_features(),
            self._window,
            self._z_threshold,
            self._baseline_years,
        )

    # ----------------------------------------------------------- preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the recent-baseline filter and attach rolling stats.

        Long FRED histories (1987-2026) are first clipped to the trailing
        ``baseline_years`` so the rolling baseline reflects the current
        price regime. Then, per active feature, a trailing rolling mean
        and standard deviation (``window`` days, ``min_periods=2``) are
        attached. ``min_periods=2`` means the very first row is dropped
        (single-sample std is undefined) but subsequent early rows still
        get a rolling estimate from whatever history is available.

        Args:
            data: Raw market frame.

        Returns:
            Frame with the original feature columns preserved, plus
            ``<feature>_rolling_mean`` and ``<feature>_rolling_std`` for
            each active feature. ``timestamp`` and ``is_disruption``
            (when present) are carried through.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "MarketAgent.preprocess called before fit() — call fit() first."
            )
        self._validate_columns(data)

        df = data.copy()
        active = self._active_features()
        missing = [c for c in active if c not in df.columns]
        if missing:
            raise ValueError(
                f"MarketAgent: optional feature(s) {missing} present at "
                "fit() time but missing now — fit and inference data must "
                "share the same schema."
            )

        if "timestamp" in df.columns and self._baseline_years > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff = df["timestamp"].max() - pd.DateOffset(years=self._baseline_years)
            n_before = len(df)
            df = df.loc[df["timestamp"] >= cutoff].reset_index(drop=True)
            if n_before != len(df):
                logger.info(
                    "[MarketAgent.preprocess] recent-baseline clip: "
                    "%d → %d rows (last %d years from %s)",
                    n_before,
                    len(df),
                    self._baseline_years,
                    cutoff.date(),
                )

        df[active] = df[active].ffill()
        df = df.dropna(subset=active).reset_index(drop=True)

        for col in active:
            roll = df[col].rolling(window=self._window, min_periods=2)
            df[f"{col}_rolling_mean"] = roll.mean()
            df[f"{col}_rolling_std"] = roll.std(ddof=0)

        before = len(df)
        df = df.dropna(
            subset=[f"{c}_rolling_std" for c in active]
        ).reset_index(drop=True)
        logger.info(
            "[MarketAgent.preprocess] rolling stats (window=%d, features=%d) | "
            "%d → %d rows",
            self._window,
            len(active),
            before,
            len(df),
        )
        return df

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        """Score each row by feature-weighted absolute z-scores.

        Per-feature ``z = (value - rolling_mean) / rolling_std`` is
        computed first; rows where the rolling std collapses below
        ``1e-9`` get ``z = 0`` for that feature (degenerate flat window).
        The combined score is the configured weighted mean of
        ``|z|``, normalised by ``z_threshold`` and clipped at 1.0 so a
        weighted |z| equal to the threshold saturates the score.

        Args:
            data: Output of :meth:`preprocess`.

        Returns:
            ``data`` with ``oil_zscore``, ``trade_volume_zscore``,
            ``freight_zscore``, optional ``freight_services_zscore``,
            ``anomaly_score`` (in ``[0, 1]``), and ``is_anomaly``
            (``anomaly_score >= threshold``) appended.
        """
        if not self._is_fitted:
            raise RuntimeError("MarketAgent.detect called before fit().")

        out = data.copy()
        weighted = np.zeros(len(out), dtype=float)
        weights = self._feature_weights()
        active = self._active_features()

        for col in active:
            mu = out[f"{col}_rolling_mean"].to_numpy()
            sd = out[f"{col}_rolling_std"].to_numpy()
            sd_safe = np.where(sd > 1e-9, sd, np.nan)
            z = (out[col].to_numpy() - mu) / sd_safe
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z_col = _ZSCORE_NAME_MAP[col]
            out[z_col] = z
            weighted += weights[col] * np.abs(z)

        out["anomaly_score"] = np.minimum(weighted / self._z_threshold, 1.0)
        out["is_anomaly"] = out["anomaly_score"] >= self._threshold
        logger.info(
            "[MarketAgent.detect] %d rows scored over %d features | "
            "%d raw anomalies (threshold=%.2f, z_threshold=%.2f)",
            len(out),
            len(active),
            int(out["is_anomaly"].sum()),
            self._threshold,
            self._z_threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Filter raw flags with persistence + oil-led feature gates.

        A row is marked ``validated=True`` iff:

        1. ``is_anomaly`` is part of a run of length
           ``>= _PERSISTENCE_DAYS`` (suppresses single-day spikes).
        2. ``|oil_zscore| > z_threshold`` (oil is the lead indicator).
        3. At least one of ``|trade_volume_zscore|`` or
           ``|freight_zscore|`` exceeds ``z_threshold`` (corroboration).

        The optional freight-services feature contributes to the anomaly
        score but is intentionally excluded from the corroboration OR so
        that the gate stays identical in synthetic and FRED modes.

        Args:
            signals: Output of :meth:`detect`.

        Returns:
            ``signals`` with ``oil_elevated``, ``other_elevated``, and
            the boolean ``validated`` columns appended.
        """
        s = signals.copy().reset_index(drop=True)
        flags = s["is_anomaly"].astype(bool).to_numpy()

        oil_elevated = (s["oil_zscore"].abs() > self._z_threshold).to_numpy()
        other_elevated = (
            (s["trade_volume_zscore"].abs() > self._z_threshold)
            | (s["freight_zscore"].abs() > self._z_threshold)
        ).to_numpy()

        combined = flags & oil_elevated & other_elevated

        validated = np.zeros_like(combined)
        n = len(combined)
        i = 0
        while i < n:
            if combined[i]:
                j = i
                while j < n and combined[j]:
                    j += 1
                if (j - i) >= _PERSISTENCE_DAYS:
                    validated[i:j] = True
                i = j
            else:
                i += 1

        s["oil_elevated"] = oil_elevated
        s["other_elevated"] = other_elevated
        s["validated"] = validated
        logger.info(
            "[MarketAgent.validate] %d/%d raw anomalies survived validation",
            int(s["validated"].sum()),
            int(flags.sum()),
        )
        return s

    # ---------------------------------------------------------------- output
    def output(self, validated_signals: pd.DataFrame) -> list[dict[str, Any]]:
        """Group consecutive validated days into market anomaly windows.

        Confidence per window is the mean fraction of active features
        elevated per day. With 3 active features (synthetic) the
        denominator is 3; with 4 active features (FRED) it is 4.

        Args:
            validated_signals: Output of :meth:`validate`.

        Returns:
            One dictionary per contiguous validated window, conforming to
            the unified anomaly-report schema. The ``signals`` dict always
            contains ``oil_zscore``, ``trade_volume_zscore``,
            ``freight_zscore``; ``freight_services_zscore`` is appended
            when that feature is active.
        """
        s = validated_signals.reset_index(drop=True)
        flags = s["validated"].astype(bool).to_numpy()

        windows: list[tuple[int, int]] = []
        start: int | None = None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                windows.append((start, i - 1))
                start = None
        if start is not None:
            windows.append((start, len(flags) - 1))

        results: list[dict[str, Any]] = []
        zt = self._z_threshold
        active = self._active_features()
        z_cols = [_ZSCORE_NAME_MAP[c] for c in active]
        for start_idx, end_idx in windows:
            w = s.iloc[start_idx : end_idx + 1]
            elevated_per_day = pd.Series(0, index=w.index)
            for c in z_cols:
                if c in w.columns:
                    elevated_per_day = (
                        elevated_per_day + (w[c].abs() > zt).astype(int)
                    )
            confidence = (
                float(elevated_per_day.mean() / len(active))
                if active else 0.0
            )
            start_ts = w["timestamp"].iloc[0] if "timestamp" in w.columns else None
            end_ts = w["timestamp"].iloc[-1] if "timestamp" in w.columns else None
            signals = {
                "oil_zscore": float(w["oil_zscore"].abs().max()),
                "trade_volume_zscore": float(w["trade_volume_zscore"].abs().max()),
                "freight_zscore": float(w["freight_zscore"].abs().max()),
            }
            if "freight_services_zscore" in w.columns:
                signals["freight_services_zscore"] = float(
                    w["freight_services_zscore"].abs().max()
                )

            results.append(
                {
                    "agent": "market",
                    "anomaly_score": float(w["anomaly_score"].max()),
                    "confidence": confidence,
                    "signals": signals,
                    "start_timestamp": (
                        str(pd.Timestamp(start_ts).date())
                        if start_ts is not None
                        else None
                    ),
                    "end_timestamp": (
                        str(pd.Timestamp(end_ts).date())
                        if end_ts is not None
                        else None
                    ),
                    "location": self._resolved_location,
                }
            )

        logger.info(
            "[MarketAgent.output] produced %d anomaly windows (location=%s)",
            len(results),
            self._resolved_location,
        )
        return results

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Run the full preprocess → detect → validate → output pipeline.

        If the agent has not yet been fitted, it auto-fits on ``data``.
        Rolling stats mean ``fit`` is essentially a schema check, so
        auto-fit is safe in production as well as evaluation.

        Args:
            data: Raw market frame.

        Returns:
            List of validated anomaly window reports.
        """
        if not self._is_fitted:
            logger.info("[MarketAgent.run] auto-fitting on input frame")
            self.fit(data)

        logger.info("[MarketAgent.run] step 1/4 preprocess")
        prepped = self.preprocess(data)
        logger.info("[MarketAgent.run] step 2/4 detect")
        scored = self.detect(prepped)
        logger.info("[MarketAgent.run] step 3/4 validate")
        validated = self.validate(scored)
        logger.info("[MarketAgent.run] step 4/4 output")
        reports = self.output(validated)
        logger.info(
            "[MarketAgent.run] complete | %d windows reported", len(reports)
        )
        return reports

    # ----------------------------------------------------- helper utilities
    def run_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the pipeline and return the per-row validated frame.

        Args:
            data: Raw market frame.

        Returns:
            DataFrame from :meth:`validate` — z-scores, anomaly scores,
            and the boolean ``validated`` column.
        """
        if not self._is_fitted:
            self.fit(data)
        prepped = self.preprocess(data)
        scored = self.detect(prepped)
        return self.validate(scored)

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        """Adapt the validated frame to the :class:`DetectionResult` contract.

        Args:
            validated: Output of :meth:`validate`.

        Returns:
            DetectionResult with the combined anomaly score, the validated
            flag, and the active feature list (which may extend beyond the
            three base features when the FRED freight-services column was
            discovered at fit time).
        """
        active = self._active_features()
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=active,
            metadata={
                "z_threshold": self._z_threshold,
                "threshold": self._threshold,
                "window": self._window,
                "baseline_years": self._baseline_years,
                "feature_count": len(active),
                "location": self._resolved_location,
            },
        )

    # ---------------------------------------------------------------- guard
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise if any of the three base feature columns are missing."""
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"MarketAgent: missing required feature columns: {missing}"
            )

    # ------------------------------------------------- internal helpers
    def _active_features(self) -> list[str]:
        """Return the ordered list of features used by the model."""
        return list(self._feature_columns) + list(self._extra_features)

    def _feature_weights(self) -> dict[str, float]:
        """Return the per-feature weights for the current active set."""
        if self._weight_override is not None:
            base = dict(self._weight_override)
            if _OPTIONAL_FREIGHT_SERVICES in self._extra_features:
                # Reserve a fixed 0.15 share for freight-services, scale the
                # three base weights into the remaining 0.85.
                fs_share = _FEATURE_WEIGHTS_4[_OPTIONAL_FREIGHT_SERVICES]
                scale = 1.0 - fs_share
                base = {k: v * scale for k, v in base.items()}
                base[_OPTIONAL_FREIGHT_SERVICES] = fs_share
            return base
        if _OPTIONAL_FREIGHT_SERVICES in self._extra_features:
            return dict(_FEATURE_WEIGHTS_4)
        return dict(_FEATURE_WEIGHTS_3)

    def _discover_optional_features(self, df: pd.DataFrame) -> None:
        """Detect the optional FRED freight-services column."""
        self._extra_features = []
        if _OPTIONAL_FREIGHT_SERVICES in df.columns:
            self._extra_features.append(_OPTIONAL_FREIGHT_SERVICES)
            logger.info(
                "[MarketAgent.fit] discovered optional feature: %s "
                "(switching to 4-feature weight schedule)",
                _OPTIONAL_FREIGHT_SERVICES,
            )

    def _resolve_location(self, df: pd.DataFrame) -> None:
        """Choose ``self._resolved_location`` from config or input columns."""
        if "location" in self.config and self.config["location"]:
            self._resolved_location = str(self.config["location"])
            return
        if _OPTIONAL_FREIGHT_SERVICES in df.columns:
            self._resolved_location = _REAL_DATA_LOCATION
        else:
            self._resolved_location = _DEFAULT_LOCATION
