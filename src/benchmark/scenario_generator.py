"""Scenario spec loader + materializer for EVAL01.

A scenario is a YAML *spec* (onset/ramp/duration + a per-agent baseline and
optional effect) — never hardcoded Python. :func:`materialize_scenario`
turns that spec into the same daily-frequency signal shape the rest of the
pipeline already consumes (mirroring the ramp/decay convention used by
``src/ingestion/geopolitical_connector.py``), so a benchmark run is just
another synthetic realisation, not a parallel data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.benchmark.regions import KNOWN_DOMAINS, Region

_REQUIRED_TOP_LEVEL = (
    "scenario_id",
    "region",
    "class",
    "seed",
    "days",
    "event",
    "signals",
    "noise",
)
_REQUIRED_EVENT_KEYS = ("onset_day", "ramp_days", "duration_days", "peak_band")


@dataclass
class ScenarioSpec:
    """YAML-backed scenario definition.

    Attributes:
        scenario_id: Unique scenario identifier, e.g. ``"hormuz_P_CRIT"``.
        region: Region key this scenario targets (see :mod:`src.benchmark.regions`).
        scenario_class: Benchmark class — ``P-CRIT`` / ``P-HIGH`` / ``N-QUIET`` /
            ``N-DECOY`` (YAML key is ``class``; renamed here since ``class``
            is a reserved word).
        seed: NumPy RNG seed for reproducibility.
        days: Number of days to materialize.
        event: ``{onset_day, ramp_days, duration_days, peak_band}``.
        signals: Per-domain ``{baseline: {mean, std}, effect: {...} | None}``,
            keyed by agent domain (subset of :data:`~src.benchmark.regions.KNOWN_DOMAINS`).
        noise: ``{autocorrelation, seasonality: {period, amplitude}, missing_data_rate}``.
    """

    scenario_id: str
    region: str
    scenario_class: str
    seed: int
    days: int
    event: dict[str, Any]
    signals: dict[str, Any]
    noise: dict[str, Any]


def load_scenario(yaml_path: str) -> ScenarioSpec:
    """Load and validate a scenario YAML.

    Args:
        yaml_path: Path to a scenario spec, e.g.
            ``config/benchmark/scenarios/hormuz_P_CRIT.yaml``.

    Returns:
        Populated :class:`ScenarioSpec`.

    Raises:
        ValueError: If required top-level or ``event`` keys are missing.
    """
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ValueError(f"Scenario spec {yaml_path} missing keys: {missing}")

    event = dict(raw["event"])
    missing_event = [k for k in _REQUIRED_EVENT_KEYS if k not in event]
    if missing_event:
        raise ValueError(
            f"Scenario spec {yaml_path} 'event' missing keys: {missing_event}"
        )

    return ScenarioSpec(
        scenario_id=str(raw["scenario_id"]),
        region=str(raw["region"]),
        scenario_class=str(raw["class"]),
        seed=int(raw["seed"]),
        days=int(raw["days"]),
        event=event,
        signals=dict(raw["signals"] or {}),
        noise=dict(raw["noise"] or {}),
    )


# --------------------------------------------------------------------- noise
def _ar1_noise(rng: np.random.Generator, n: int, std: float, phi: float) -> np.ndarray:
    """Stationary AR(1) noise with marginal std ``std`` and lag-1 corr ``phi``."""
    if std <= 0:
        return np.zeros(n)
    phi = float(np.clip(phi, -0.99, 0.99))
    z = rng.normal(0.0, 1.0, size=n)
    noise = np.empty(n, dtype=float)
    noise[0] = z[0] * std
    scale = std * np.sqrt(max(1.0 - phi**2, 1e-9))
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + scale * z[t]
    return noise


def _seasonal(days: int, amplitude: float, period: float, std: float) -> np.ndarray:
    """Sinusoidal seasonality scaled to the domain's own baseline std."""
    if period <= 0 or amplitude == 0:
        return np.zeros(days)
    t = np.arange(days)
    return amplitude * std * np.sin(2 * np.pi * t / period)


def _apply_missing(rng: np.random.Generator, values: np.ndarray, rate: float) -> np.ndarray:
    """Mask a random ``rate`` fraction of entries as NaN (missing data)."""
    if rate <= 0:
        return values
    out = values.copy()
    out[rng.random(len(values)) < rate] = np.nan
    return out


# -------------------------------------------------------------------- effect
def _intensity_curve(ramp_days: int, duration_days: int) -> np.ndarray:
    """Per-offset intensity in [0, 1] for a ramp-up / plateau / ramp-down window."""
    window_len = max(duration_days, 0)
    ramp = min(ramp_days, window_len)
    decay = min(ramp_days, max(window_len - ramp, 0))
    plateau = max(window_len - ramp - decay, 0)

    intensity = np.zeros(window_len, dtype=float)
    for offset in range(window_len):
        if offset < ramp:
            intensity[offset] = (offset + 1) / ramp if ramp > 0 else 1.0
        elif offset < ramp + plateau:
            intensity[offset] = 1.0
        else:
            tail = offset - (ramp + plateau)
            intensity[offset] = 1.0 - (tail + 1) / decay if decay > 0 else 0.0
    return intensity


def _apply_effect(
    values: np.ndarray, effect: dict[str, Any], event: dict[str, Any], days: int
) -> np.ndarray:
    """Ramp a signal toward ``effect['target']`` over the event window.

    ``lead_days`` shifts the agent's window relative to ``event.onset_day``:
    positive leads the physical onset (starts earlier, as in
    ``geopolitical_connector.py``), negative lags it (starts later).
    """
    effect_type = effect["type"]
    target = float(effect["target"])
    lead_days = int(effect.get("lead_days", 0))

    onset_day = int(event["onset_day"])
    ramp_days = int(event["ramp_days"])
    duration_days = int(event["duration_days"])

    start = onset_day - lead_days
    intensity = _intensity_curve(ramp_days, duration_days)

    out = values.copy()
    for offset, day in enumerate(range(start, start + len(intensity))):
        if day < 0 or day >= days:
            continue
        strength = intensity[offset]
        if effect_type == "multiplicative_ramp":
            out[day] = out[day] * (1.0 + (target - 1.0) * strength)
        elif effect_type == "additive_ramp":
            out[day] = out[day] + target * strength
        else:
            raise ValueError(f"Unknown effect type: {effect_type!r}")
    return out


# ------------------------------------------------------------------- labels
def _ground_truth_labels(
    spec: ScenarioSpec, days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive y_disruption / y_band / y_action from the scenario class + event window.

    Only ``P-*`` classes (P-CRIT, P-HIGH, ...) carry a positive label; ``N-*``
    classes (N-QUIET, N-DECOY, ...) are negative even when a decoy signal
    ramps, since the point of a decoy is a signal move with no true event.
    """
    y_disruption = np.zeros(days, dtype=int)
    y_band = np.full(days, "quiet", dtype=object)

    if spec.scenario_class.strip().upper().startswith("P"):
        onset_day = int(spec.event["onset_day"])
        duration_days = int(spec.event["duration_days"])
        peak_band = str(spec.event.get("peak_band", "quiet"))
        start = max(onset_day, 0)
        end = min(onset_day + duration_days - 1, days - 1)
        if start <= end:
            y_disruption[start : end + 1] = 1
            y_band[start : end + 1] = peak_band

    y_action = y_disruption.copy()
    return y_disruption, y_band, y_action


# -------------------------------------------------------------- materialize
def materialize_scenario(spec: ScenarioSpec, region: Region) -> pd.DataFrame:
    """Generate synthetic per-domain signals for a scenario.

    Args:
        spec: Loaded scenario spec (see :func:`load_scenario`).
        region: The spec's target region (see
            :func:`~src.benchmark.regions.load_region`).

    Returns:
        DataFrame with columns ``[timestamp, *KNOWN_DOMAINS, y_disruption,
        y_band, y_action]``. Domains outside ``region.active_domains`` are
        *disabled* (all-NaN). Domains in ``active_domains`` whose signal has
        ``effect: null`` are *enabled-but-silent* — baseline + noise only,
        no ramp.
    """
    days = spec.days
    timestamps = pd.date_range("2025-01-01", periods=days, freq="D")
    rng = np.random.default_rng(spec.seed)

    autocorr = float(spec.noise.get("autocorrelation", 0.0))
    seasonality_cfg = spec.noise.get("seasonality") or {}
    period = float(seasonality_cfg.get("period", 0) or 0)
    amplitude = float(seasonality_cfg.get("amplitude", 0.0))
    missing_rate = float(spec.noise.get("missing_data_rate", 0.0))

    columns: dict[str, np.ndarray] = {}
    for domain in KNOWN_DOMAINS:
        sig = spec.signals.get(domain)
        if domain not in region.active_domains or not sig:
            columns[domain] = np.full(days, np.nan)
            continue

        baseline = sig.get("baseline", {}) or {}
        mean = float(baseline.get("mean", 0.0))
        std = float(baseline.get("std", 0.0))

        values = mean + _ar1_noise(rng, days, std, autocorr) + _seasonal(
            days, amplitude, period, std
        )

        effect = sig.get("effect")
        if effect:
            values = _apply_effect(values, effect, spec.event, days)

        columns[domain] = _apply_missing(rng, values, missing_rate)

    y_disruption, y_band, y_action = _ground_truth_labels(spec, days)

    df = pd.DataFrame({"timestamp": timestamps, **columns})
    df["y_disruption"] = y_disruption
    df["y_band"] = y_band
    df["y_action"] = y_action
    return df
