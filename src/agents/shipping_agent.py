"""Shipping anomaly detection agent for the Strait of Hormuz corridor.

This agent is the primary signal source in the multi-agent DSS pipeline. It
ingests preprocessed shipping data (vessel counts, transit delays, corridor
congestion), produces anomaly scores via an Isolation Forest combined with a
per-feature Z-score fallback, validates raw signals to suppress false
positives, and emits structured anomaly reports for downstream aggregation,
explainability, and RAG context retrieval.

The agent works against both data sources produced by
:class:`~src.ingestion.ShippingConnector`:

- Synthetic mode — three base features (``vessel_count``,
  ``avg_delay_hours``, ``congestion_index``).
- CSV (Shuaiba PortWatch) mode — the same three plus ``tanker_count``
  (the most Hormuz-sensitive vessel class) and a derived
  ``vessel_count_trend`` (= ``vessel_count`` − ``vessel_count_7dma``)
  capturing momentum of arrival rates.

Optional features are auto-discovered at :meth:`fit` time so the agent
remains a drop-in across both modes. The detection logic is deterministic
given the same input and configuration (``random_state=42`` is pinned on
the Isolation Forest).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.agents.base_agent import BaseAgent, DetectionResult

logger = logging.getLogger(__name__)


_FEATURE_COLUMNS: tuple[str, ...] = (
    "vessel_count",
    "avg_delay_hours",
    "congestion_index",
)
_OPTIONAL_TANKER: str = "tanker_count"
_OPTIONAL_MA: str = "vessel_count_7dma"
_TREND_COLUMN: str = "vessel_count_trend"

_DEFAULT_LOCATION: str = "Strait of Hormuz"
_REAL_DATA_LOCATION: str = "Shuaiba Port, Persian Gulf"

#: Duration signal. A sustained shift is measured against a *trailing* long
#: baseline, because a rolling one adapts to the new level and reads a settled
#: disruption as calm.
_LEVEL_WINDOW: int = 30
_LEVEL_BASELINE: int = 365
#: Shortfall scoring 1.0. A 40% sustained drop is full strength.
_LEVEL_FULL_SCALE: float = 0.40
#: Shortfall below which a day does not count toward persistence at all.
#: Without a floor, ordinary variation around the baseline accumulates.
_LEVEL_MIN_SHORTFALL: float = 0.10
#: Days over which persistence is measured. A dip has to hold to score: this
#: signal exists to sustain an alert through a disruption, not to add a second
#: way of reacting to one bad week.
_LEVEL_PERSISTENCE_DAYS: int = 14

_FEATURE_ELEVATION_Z: float = 1.5
_PERSISTENCE_DAYS: int = 2
_MIN_FEATURES_ELEVATED: int = 2

# Map active feature names → emitted z-score column names. The three base
# entries are always present; optional entries appear only when their source
# columns are detected at fit time.
_ZSCORE_NAME_MAP: dict[str, str] = {
    "vessel_count": "vessel_count_zscore",
    "avg_delay_hours": "delay_zscore",
    "congestion_index": "congestion_zscore",
    _OPTIONAL_TANKER: "tanker_zscore",
    _TREND_COLUMN: "trend_zscore",
}


class ShippingAgent(BaseAgent):
    """Multi-feature anomaly detector for Strait of Hormuz shipping signals.

    The agent combines an Isolation Forest (primary, contamination-driven
    multivariate detector) with a per-feature Z-score fallback (secondary,
    catches univariate outliers the forest may smooth over). Raw flags are
    then validated by requiring (a) ``>= 2`` consecutive anomalous days and
    (b) ``>= 2`` of the active features showing elevated absolute z-scores,
    eliminating single-day noise spikes and lone-feature artefacts.

    Configuration (read from ``config['agents']['shipping']`` in
    ``settings.yaml``):

    ============== ======= =====================================================
    Key            Default Description
    ============== ======= =====================================================
    contamination  0.1     Expected anomaly fraction for the Isolation Forest
    threshold      0.65    Minimum combined score to flag a row anomalous
    z_threshold    3.0     Absolute z-score at which a feature is "extreme"
    location       auto    Override the emitted ``location`` string; when
                           absent the location is auto-detected from the data
                           (Shuaiba Port for real CSV mode, Strait of Hormuz
                           for synthetic).
    ============== ======= =====================================================

    Args:
        config: Agent-specific configuration block. The keys above are
            consulted; sensible defaults apply for any missing key.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="shipping", config=dict(config or {}))
        self._scaler: StandardScaler | None = None
        self._iforest: IsolationForest | None = None
        #: Fit-time isolation-forest score range. Normalising against this
        #: rather than the scored batch keeps a score comparable across
        #: windows, which is what lets a sustained shift stay flagged.
        self._iforest_low: float = 0.0
        self._iforest_high: float = 1.0
        self._feature_columns: list[str] = list(_FEATURE_COLUMNS)
        self._extra_features: list[str] = []
        self._contamination: float = float(self.config.get("contamination", 0.1))
        self._threshold: float = float(self.config.get("threshold", 0.65))
        self._z_threshold: float = float(self.config.get("z_threshold", 3.0))
        # Isolation-Forest vs Z-score blend for the combined anomaly score.
        # Hand-tuned default is 0.70 / 0.30; the optimizer can retune it.
        weights_cfg = self.config.get("weights") or {}
        self._if_weight: float = float(weights_cfg.get("isolation_forest", 0.70))
        self._zscore_weight: float = float(weights_cfg.get("zscore", 0.30))
        self._resolved_location: str = str(
            self.config.get("location") or _DEFAULT_LOCATION
        )

    def set_weights(self, if_weight: float, zscore_weight: float) -> None:
        """Override the Isolation-Forest / Z-score blend weights.

        Args:
            if_weight: Weight on the normalised Isolation-Forest score.
            zscore_weight: Weight on the normalised max-|z| fallback score.
                The two are used as given (no internal renormalisation), so
                pass values that sum to 1.0 for a convex blend.
        """
        self._if_weight = float(if_weight)
        self._zscore_weight = float(zscore_weight)

    def set_threshold(self, threshold: float) -> None:
        """Override the minimum combined score to flag a row anomalous."""
        self._threshold = float(threshold)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the StandardScaler and Isolation Forest on historical data.

        Auto-discovers optional Shuaiba PortWatch features (``tanker_count``,
        ``vessel_count_7dma`` → ``vessel_count_trend``) and folds them into
        the active feature set when present. To avoid leakage, when a
        ground-truth ``is_disruption`` column is present the scaler and
        forest are fit only on the rows where it is ``False``.

        Args:
            df: Historical feature frame; must contain the three base
                shipping feature columns and may optionally contain
                ``tanker_count``, ``vessel_count_7dma``, and
                ``is_disruption`` (used for leak-free training-window
                selection).
        """
        self._validate_columns(df)
        self._discover_optional_features(df)
        self._resolve_location(df)

        train = self._derive_features(df.copy())
        if "is_disruption" in train.columns:
            mask = train["is_disruption"].astype(bool)
            train = train.loc[~mask]
            logger.info(
                "[ShippingAgent.fit] using %d non-disruption rows for fit "
                "(filtered out %d disruption rows)",
                len(train),
                int(mask.sum()),
            )
        active = self._active_features()
        features = train[active].ffill().dropna()

        self._scaler = StandardScaler().fit(features.to_numpy())
        scaled = self._scaler.transform(features.to_numpy())
        self._iforest = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=200,
        ).fit(scaled)

        train_scores = -self._iforest.decision_function(scaled)
        # Percentiles, not min/max: one extreme training row would otherwise
        # set the ceiling and compress everything beneath it.
        self._iforest_low = float(np.percentile(train_scores, 5))
        self._iforest_high = float(np.percentile(train_scores, 95))
        if self._iforest_high - self._iforest_low < 1e-9:
            self._iforest_high = self._iforest_low + 1.0

        self._is_fitted = True
        logger.info(
            "[ShippingAgent.fit] fitted scaler+IsolationForest on %d rows | "
            "features=%s | contamination=%.3f",
            len(features),
            active,
            self._contamination,
        )

    # ----------------------------------------------------------- preprocess
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select features, derive optionals, fill gaps, and scale.

        Args:
            data: Raw shipping frame containing at minimum the three base
                feature columns plus any optional columns the agent was
                fitted on. ``timestamp`` and ``is_disruption`` are
                preserved when present.

        Returns:
            Frame with the active feature columns transformed onto the
            scaler's standard space (mean 0, unit variance, in z-score
            units), plus passthrough timestamp / ground-truth labels.
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError(
                "ShippingAgent.preprocess called before fit() — call fit() "
                "with a clean training window first."
            )
        self._validate_columns(data)

        df = self._derive_features(data.copy())
        active = self._active_features()
        missing = [c for c in active if c not in df.columns]
        if missing:
            raise ValueError(
                f"ShippingAgent: optional feature(s) {missing} present at "
                "fit() time but missing now — fit and inference data must "
                "share the same schema."
            )

        df[active] = df[active].ffill()
        df = df.dropna(subset=active).reset_index(drop=True)

        scaled = self._scaler.transform(df[active].to_numpy())
        out = pd.DataFrame(scaled, columns=active)
        if "timestamp" in df.columns:
            out.insert(0, "timestamp", pd.to_datetime(df["timestamp"]).values)
        if "is_disruption" in df.columns:
            out["is_disruption"] = df["is_disruption"].astype(bool).values
        # Unscaled counts travel alongside the standardised features: the
        # duration score needs the raw level, not a z-score against a baseline
        # that has already absorbed the shift.
        if "vessel_count" in df.columns:
            out["_raw_vessel_count"] = df["vessel_count"].to_numpy()
        logger.info(
            "[ShippingAgent.preprocess] scaled %d rows over %d features %s",
            len(out),
            len(active),
            active,
        )
        return out

    @staticmethod
    def level_shift_score(vessel_count: pd.Series) -> pd.Series:
        """Score how far traffic sits below its own trailing annual baseline.

        The shock detectors cannot hold a flag across a sustained disruption.
        Two of the three features they read -- ``congestion_index`` and
        ``avg_delay_hours`` -- are derived from ``vessel_count`` against a
        30-day rolling baseline, so once traffic settles lower that baseline
        follows it down and both return to calm values. Measured on the
        evaluation set, their correlation with a sustained-disruption label is
        -0.06 and -0.05: no signal at all, leaving one useful feature of three.

        **This score is not evidence of detection skill against a
        shipping-derived label.** It is computed from the series such a label
        is built from and will predict one nearly by construction. It earns
        its place operationally -- an alert should persist while the
        disruption does -- not evaluatively.

        Args:
            vessel_count: Daily transit counts, ordered by date.

        Returns:
            Score in [0, 1]; 0 when traffic is at or above baseline.
        """
        rolling = vessel_count.rolling(
            _LEVEL_WINDOW, min_periods=max(_LEVEL_WINDOW // 3, 2)
        ).mean()
        baseline = (
            vessel_count.shift(_LEVEL_WINDOW)
            .rolling(_LEVEL_BASELINE, min_periods=_LEVEL_BASELINE // 4)
            .median()
        )
        shortfall = (baseline - rolling) / baseline.replace(0.0, np.nan)
        magnitude = (shortfall / _LEVEL_FULL_SCALE).clip(0.0, 1.0).fillna(0.0)

        # Persistence gate: the fraction of the trailing fortnight that also
        # sat meaningfully below baseline. A one-off dip scores near zero
        # however deep; a shift that holds reaches full strength.
        below = (shortfall > _LEVEL_MIN_SHORTFALL).fillna(False).astype(float)
        persistence = below.rolling(
            _LEVEL_PERSISTENCE_DAYS, min_periods=_LEVEL_PERSISTENCE_DAYS
        ).mean().fillna(0.0)
        return (magnitude * persistence).clip(0.0, 1.0)

    # --------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        """Score each row with Isolation Forest + Z-score fallback.

        The Isolation Forest score is ``-decision_function`` min-max
        normalised into ``[0, 1]`` (1 = most anomalous). The Z-score
        fallback takes the maximum absolute scaled value across all active
        features and clips it at ``z_threshold`` before normalising. The
        combined score is ``0.7 * iforest + 0.3 * max_abs_z``.

        Args:
            data: Output of :meth:`preprocess` — already standardised.

        Returns:
            Frame with the input columns plus per-feature z-scores,
            ``isolation_score``, ``max_zscore_norm``, ``anomaly_score``,
            and the boolean ``is_anomaly`` flag.
        """
        if not self._is_fitted or self._iforest is None:
            raise RuntimeError("ShippingAgent.detect called before fit().")

        active = self._active_features()
        scaled = data[active].to_numpy()
        iforest_raw = self._iforest.decision_function(scaled)
        iforest_score = -iforest_raw  # higher = more anomalous

        # Normalised against the training range, not this batch's own range.
        # Batch min-max destroys the signal a sustained disruption carries:
        # when every day in a window is anomalous, rescaling by that window's
        # extremes maps its least-anomalous day to 0 and its most to 1, so a
        # uniformly disrupted month and a uniformly calm one both span [0, 1].
        # Anchoring to the fit-time distribution makes scores absolute.
        span = self._iforest_high - self._iforest_low
        iforest_norm = np.clip(
            (iforest_score - self._iforest_low) / span, 0.0, 1.0
        )

        max_abs_z = np.max(np.abs(scaled), axis=1)
        max_z_norm = np.minimum(max_abs_z / self._z_threshold, 1.0)

        shock = self._if_weight * iforest_norm + self._zscore_weight * max_z_norm

        # max(), not a weighted blend. The two answer different questions --
        # "did something just change?" and "are we still below normal?" -- and
        # are meant to be true at different times. Averaging lets a calm shock
        # detector drag down an active duration signal through the middle of a
        # disruption, the same dilution the tier ablation showed when
        # averaging agents together.
        if "_raw_vessel_count" in data.columns:
            duration = self.level_shift_score(
                pd.Series(data["_raw_vessel_count"].to_numpy())
            ).to_numpy()
            combined = np.maximum(shock, duration)
        else:
            duration = np.zeros_like(shock)
            combined = shock

        out = data.copy()
        out["shock_score"] = shock
        out["duration_score"] = duration
        for idx, feat in enumerate(active):
            out[_ZSCORE_NAME_MAP[feat]] = scaled[:, idx]
        out["isolation_score"] = iforest_norm
        out["max_zscore_norm"] = max_z_norm
        out["anomaly_score"] = combined
        out["is_anomaly"] = combined >= self._threshold
        logger.info(
            "[ShippingAgent.detect] %d rows scored over %d features | "
            "%d raw anomalies (threshold=%.3f)",
            len(out),
            len(active),
            int(out["is_anomaly"].sum()),
            self._threshold,
        )
        return out

    # ------------------------------------------------------------- validate
    def validate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Filter raw flags to reduce false positives.

        A row is marked ``validated=True`` only when both:

        1. **Persistence** — the anomaly is part of a run of at least
           ``_PERSISTENCE_DAYS`` consecutive flagged rows (single-day
           spikes are dropped). There is no upper cap, so multi-month
           disruptions (e.g. the 2026 Hormuz shutdown) are flagged in
           full.
        2. **Multi-feature** — at least ``_MIN_FEATURES_ELEVATED`` of the
           active features have ``|z| > _FEATURE_ELEVATION_Z`` on that
           row (lone-feature outliers are dropped).

        Args:
            signals: Output of :meth:`detect`.

        Returns:
            ``signals`` with added ``features_elevated``, ``features_total``,
            and boolean ``validated`` columns.
        """
        s = signals.copy().reset_index(drop=True)
        is_anom = s["is_anomaly"].astype(bool).to_numpy()

        persistent = np.zeros_like(is_anom)
        n = len(is_anom)
        i = 0
        while i < n:
            if is_anom[i]:
                j = i
                while j < n and is_anom[j]:
                    j += 1
                if (j - i) >= _PERSISTENCE_DAYS:
                    persistent[i:j] = True
                i = j
            else:
                i += 1

        z_cols = [
            _ZSCORE_NAME_MAP[f]
            for f in self._active_features()
            if _ZSCORE_NAME_MAP[f] in s.columns
        ]
        elevated = pd.Series(0, index=s.index)
        for col in z_cols:
            elevated = elevated + (s[col].abs() > _FEATURE_ELEVATION_Z).astype(int)
        feature_check = (elevated >= _MIN_FEATURES_ELEVATED).to_numpy()

        s["features_elevated"] = elevated.to_numpy()
        s["features_total"] = len(z_cols)
        s["validated"] = persistent & feature_check
        logger.info(
            "[ShippingAgent.validate] %d/%d raw anomalies survived validation "
            "(features=%d, persistence>=%d, elevated>=%d)",
            int(s["validated"].sum()),
            int(is_anom.sum()),
            len(z_cols),
            _PERSISTENCE_DAYS,
            _MIN_FEATURES_ELEVATED,
        )
        return s

    # ---------------------------------------------------------------- output
    def output(self, validated_signals: pd.DataFrame) -> list[dict[str, Any]]:
        """Group consecutive validated days into anomaly windows.

        Confidence is the mean per-day fraction of features elevated
        (``features_elevated / features_total``) inside the window —
        windows where every active feature fires on every day get
        confidence 1.0.

        Args:
            validated_signals: Output of :meth:`validate`.

        Returns:
            List of dictionaries — one per contiguous validated window —
            conforming to the unified anomaly-report schema. The
            ``signals`` dict always contains the three base z-scores plus
            ``tanker_zscore`` / ``trend_zscore`` when those features were
            active.
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
        for start_idx, end_idx in windows:
            w = s.iloc[start_idx : end_idx + 1]
            n_features = (
                int(w["features_total"].iloc[0])
                if "features_total" in w.columns
                else len(self._active_features())
            )
            confidence = (
                float(w["features_elevated"].mean() / n_features)
                if n_features > 0 else 0.0
            )
            start_ts = w["timestamp"].iloc[0] if "timestamp" in w.columns else None
            end_ts = w["timestamp"].iloc[-1] if "timestamp" in w.columns else None
            signals = {
                "vessel_count_zscore": float(w["vessel_count_zscore"].abs().max()),
                "delay_zscore": float(w["delay_zscore"].abs().max()),
                "congestion_zscore": float(w["congestion_zscore"].abs().max()),
            }
            if "tanker_zscore" in w.columns:
                signals["tanker_zscore"] = float(w["tanker_zscore"].abs().max())
            if "trend_zscore" in w.columns:
                signals["trend_zscore"] = float(w["trend_zscore"].abs().max())

            results.append(
                {
                    "agent": "shipping",
                    "anomaly_score": float(w["anomaly_score"].max()),
                    "confidence": confidence,
                    "signals": signals,
                    "start_timestamp": (
                        str(pd.Timestamp(start_ts).date())
                        if start_ts is not None else None
                    ),
                    "end_timestamp": (
                        str(pd.Timestamp(end_ts).date())
                        if end_ts is not None else None
                    ),
                    "location": self._resolved_location,
                }
            )

        logger.info(
            "[ShippingAgent.output] produced %d anomaly windows (location=%s)",
            len(results),
            self._resolved_location,
        )
        return results

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Run the full preprocess → detect → validate → output pipeline.

        If the agent has not yet been fitted, it auto-fits on ``data`` —
        useful for unsupervised one-shot evaluation. When the input frame
        carries an ``is_disruption`` ground-truth column, a confusion
        matrix + precision / recall / F1 / TPR / FPR line is logged at
        the end (in addition to the structured reports returned).

        Args:
            data: Raw shipping frame.

        Returns:
            List of validated anomaly window reports (see :meth:`output`).
        """
        if not self._is_fitted:
            logger.info("[ShippingAgent.run] auto-fitting on input frame")
            self.fit(data)

        logger.info("[ShippingAgent.run] step 1/4 preprocess")
        scaled = self.preprocess(data)
        logger.info("[ShippingAgent.run] step 2/4 detect")
        scored = self.detect(scaled)
        logger.info("[ShippingAgent.run] step 3/4 validate")
        validated = self.validate(scored)
        logger.info("[ShippingAgent.run] step 4/4 output")
        reports = self.output(validated)
        if "is_disruption" in validated.columns:
            self._log_eval_metrics(validated)
        logger.info(
            "[ShippingAgent.run] complete | %d windows reported", len(reports)
        )
        return reports

    # ----------------------------------------------------- helper utilities
    def run_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the pipeline and return the per-row validated frame.

        Convenience wrapper used by evaluation harnesses that need access
        to per-row predictions for confusion-matrix / TPR / FPR scoring.
        :meth:`run` only emits aggregated windows.

        Args:
            data: Raw shipping frame.

        Returns:
            The DataFrame returned by :meth:`validate` — i.e. detection
            scores plus the boolean ``validated`` column.
        """
        if not self._is_fitted:
            self.fit(data)
        scaled = self.preprocess(data)
        scored = self.detect(scaled)
        return self.validate(scored)

    def to_detection_result(self, validated: pd.DataFrame) -> DetectionResult:
        """Adapt the validated frame to the :class:`DetectionResult` contract.

        Args:
            validated: Output of :meth:`validate`.

        Returns:
            DetectionResult populated with the combined anomaly score, the
            validated flag, and the active feature list (which may extend
            beyond the three base features when optional Shuaiba columns
            were discovered at fit time).
        """
        active = self._active_features()
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=validated["anomaly_score"].to_numpy(dtype=float),
            anomaly_flags=validated["validated"].to_numpy(dtype=bool),
            feature_names=active,
            metadata={
                "contamination": self._contamination,
                "threshold": self._threshold,
                "z_threshold": self._z_threshold,
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
                f"ShippingAgent: missing required feature columns: {missing}"
            )

    # ------------------------------------------------- internal helpers
    def _active_features(self) -> list[str]:
        """Return the ordered list of features used by the model."""
        return list(self._feature_columns) + list(self._extra_features)

    def _discover_optional_features(self, df: pd.DataFrame) -> None:
        """Detect optional Shuaiba PortWatch columns and update active set."""
        self._extra_features = []
        if _OPTIONAL_TANKER in df.columns:
            self._extra_features.append(_OPTIONAL_TANKER)
            logger.info(
                "[ShippingAgent.fit] discovered optional feature: %s",
                _OPTIONAL_TANKER,
            )
        if _OPTIONAL_MA in df.columns:
            self._extra_features.append(_TREND_COLUMN)
            logger.info(
                "[ShippingAgent.fit] deriving %s from %s",
                _TREND_COLUMN, _OPTIONAL_MA,
            )

    def _derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived columns (currently the 7-day-MA trend)."""
        if _OPTIONAL_MA in df.columns and _TREND_COLUMN in self._extra_features:
            df[_TREND_COLUMN] = df["vessel_count"] - df[_OPTIONAL_MA]
        return df

    def _resolve_location(self, df: pd.DataFrame) -> None:
        """Choose ``self._resolved_location`` from config or input columns."""
        if "location" in self.config and self.config["location"]:
            self._resolved_location = str(self.config["location"])
            return
        if _OPTIONAL_TANKER in df.columns or _OPTIONAL_MA in df.columns:
            self._resolved_location = _REAL_DATA_LOCATION
        else:
            self._resolved_location = _DEFAULT_LOCATION

    def _log_eval_metrics(self, validated: pd.DataFrame) -> None:
        """Log confusion matrix + precision / recall / F1 / TPR / FPR."""
        y_true = validated["is_disruption"].astype(bool).to_numpy()
        y_pred = validated["validated"].astype(bool).to_numpy()
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        tpr = recall
        fpr = fp / max(fp + tn, 1)
        msg = (
            f"[ShippingAgent.eval] confusion TN={tn} FP={fp} FN={fn} TP={tp} | "
            f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f} | "
            f"TPR={tpr:.3f} FPR={fpr:.3f}"
        )
        logger.info(msg)
        print(msg)
