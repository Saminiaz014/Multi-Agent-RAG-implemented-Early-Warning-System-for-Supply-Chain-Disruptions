"""Method comparison and measured agent ablation on the real evaluation set.

Reads the cached frames from ``scripts/build_eval_dataset.py`` — real connector
features, real level-shift labels — and scores every method on the same
held-out window.

Three design decisions carry most of the weight:

**One temporal split, applied to everyone.** Every region trains on the days
before ``evaluation.split_date`` in ``config/settings.yaml`` and is tested
from that date to ``evaluation.end_date``; nothing is fitted on the test
window. A fixed date replaced the old last-30% rule, whose cut moved whenever
the series grew. A supervised model scored
on the rows it was fitted to would post an inflated number that is not
comparable with an unsupervised detector, so both are judged on data neither
has seen. The split is temporal rather than random because shuffling days of a
time series leaks the future into the past.

**Tiers build real agents.** A tier is a set of agent classes; each is fitted
and run, and the composite is the weighted mean of their anomaly scores using
the project's own weights. Approximating a tier by averaging "the first N
feature columns" would not measure the agents at all — the first two columns
of these frames are both shipping features, so a Tier 2 built that way
contains no market signal despite being labelled as containing one. Agents
get the same history as the baselines: fitted on train only, scored on train
plus test, test rows kept. Scoring the test window alone restarted their
rolling windows at the split (eval/diagnostics/split_invariance.py).
An agent its region disables (``agents.<name>.enabled: false``, set in
config/regions/ on evidence grounds, as the orchestrator honours it) is not
run and not fused: a tier fuses its enabled agents, with the weights
renormalised over them, so a tier whose added agent is disabled equals the
previous tier. ``eval/method_comparison_all_agents.csv`` re-scores every
tier with every agent enabled, as an appendix sensitivity only, and
``eval/method_comparison_no_eq1.csv`` re-scores every tier with the shipping
agent's Eq. 1 duration term off (``agents.shipping.level_shift``), paired
against the same tier with it on. ``eval/agent_ablation.csv`` (Eq. 1 on) and
``eval/agent_ablation_no_eq1.csv`` (off) score each enabled agent alone and
the full enabled set with one agent left out, paired against the full set.

**Malacca is a false-positive harness, not a detection test.** It has zero
labelled disruptions across 2019-2026, so AUC is undefined there. What it can
measure is how often each method fires when nothing is happening.

Days whose news GDELT never published (the 18-day June 2025 outage, and two
single days in training) are *masked*: they stay in the frame so rolling
features see real calendar time, but no method is scored on them and they are
left out of every training threshold. Each tier row also carries a paired
block-bootstrap ``delta_auc`` against Tier 1 on the same resampled blocks, and
``eval/split_sensitivity.csv`` re-scores Tier 1 and Tier 4 at the
``evaluation.sensitivity_split_dates``. The oracle is written as family
``ceiling``: a reference row, not a baseline.

A day alerts when its score is ``>=`` the method's training 90th percentile.
The three controls whose meaning is an alert pattern define it directly
instead (:data:`_DIRECT_ALERTS`): M1 alerts every day, M2 on no day, the
ORACLE on the days after a labelled day. Their scores still give AUC; the
``alert_rule`` column records which rule each row used.

Usage::

    python scripts/run_method_comparison.py
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score  # noqa: E402

from src.core.config_manager import load_config_for_region  # noqa: E402
from src.evaluation.block_bootstrap import (block_bootstrap_ci,  # noqa: E402
                                            paired_auc_delta_ci)
from src.evaluation.window import (bootstrap_config, eval_window,  # noqa: E402,F401
                                   sensitivity_split_dates, split_by_date)

logger = logging.getLogger(__name__)

_CACHE = _PROJECT_ROOT / "data" / "eval"
_OUT = _PROJECT_ROOT / "eval"

#: Agents in the order tiers add them, with the feature prefix each reads.
_TIER_ORDER: tuple[str, ...] = (
    "shipping", "market", "geopolitical", "natural_disaster", "news_sentiment",
)


#: How closely a method's own statistic resembles the label's construction.
#:
#: The label is "30-day mean of vessel_count at least 20% below its trailing
#: 365-day median, sustained 14+ days". A method computing the same statistic
#: predicts it largely by construction, so its score measures overlap with the
#: label definition rather than detection skill. This is recorded per method
#: so a ranking cannot silently be topped by the most circular entry.
#:
#:   high   - reads vessel_count against a trailing long baseline: the label's
#:            own statistic, differing only in smoothing or thresholding
#:   medium - reads vessel_count against a short/rolling baseline, or fits on
#:            the label directly
#:   low    - reads other features, or the shape rather than the level
#:   n/a    - controls and oracles, which are reference points not methods
_CIRCULARITY: dict[str, str] = {
    "M0 random": "n/a",
    "M1 always alert": "n/a",
    "M2 never alert": "n/a",
    "ORACLE label t-1": "n/a",
    "M3 persistence": "high",
    "M8 matrix profile": "low",
    "B1 rolling z": "medium",
    "B2 MA crossover": "medium",
    "B3 isolation forest": "low",
    "B4 EWMA deviation": "medium",
    "B5 AR residual": "low",
    "B6 CUSUM": "high",
    "B7 logistic regression": "medium",
    "B8 random forest": "medium",
}
#: Tiers 1+ include the shipping agent's level-shift feature, which is the
#: label's statistic; tiers beyond 1 dilute it with non-shipping agents.
_TIER_CIRCULARITY = "high"
#: The same tiers without that feature (Eq. 1 off): the shipping agent then
#: reads vessel_count and features derived from it against a 30-day rolling
#: baseline, the "medium" definition above.
_NO_EQ1_CIRCULARITY = "medium"


# --------------------------------------------------------------- utilities
def _load(region: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(_CACHE / f"{region}.csv", parse_dates=["timestamp"])
    manifest = json.loads((_CACHE / f"{region}.manifest.json").read_text(encoding="utf-8"))
    return frame, manifest


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train before ``split_date``; test from it through ``end_date``.

    See :func:`src.evaluation.window.split_by_date`.
    """
    return split_by_date(frame)


def _threshold(scores: np.ndarray, train_scores: np.ndarray | None,
               train_keep: np.ndarray | None = None) -> float:
    """The alert threshold: the 90th percentile of the training-window scores.

    A day alerts when its score is ``>=`` this threshold (see
    :func:`_score_metrics`). ``train_keep`` drops masked training days from
    the reference.
    """
    reference = (np.asarray(scores, dtype=float) if train_scores is None
                 else np.asarray(train_scores, dtype=float))
    if train_scores is not None and train_keep is not None:
        reference = reference[np.asarray(train_keep, dtype=bool)]
    return float(np.quantile(np.nan_to_num(reference), 0.90))


def _evaluate(y_true: np.ndarray, scores: np.ndarray,
              train_scores: np.ndarray | None = None, *,
              keep: np.ndarray | None = None, train_keep: np.ndarray | None = None,
              alerts: np.ndarray | None = None, **bootstrap) -> dict:
    """:func:`_score_metrics` plus moving-block bootstrap CIs on the test window.

    The threshold stays the training one in every resample. ``keep`` marks
    the test days that are scored and ``train_keep`` the training days the
    threshold is taken from; masked days (a news outage) count in neither.
    ``alerts`` gives the test days' alerts directly (:data:`_DIRECT_ALERTS`),
    in which case no threshold is taken and AUC still comes from ``scores``.
    Bootstrap settings default to settings.yaml ``evaluation.bootstrap``, and
    keyword overrides (e.g. ``n_resamples``) are for tests.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    mask = np.ones(len(y), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if alerts is None:
        threshold = _threshold(s[mask] if train_scores is None else s, train_scores,
                               train_keep)
        metrics = _score_metrics(y[mask], s[mask], threshold=threshold)
    else:
        threshold, alerts = None, np.asarray(alerts, dtype=bool)
        if len(alerts) != len(y):
            raise ValueError(f"alerts has {len(alerts)} days, y_true {len(y)}")
        metrics = _score_metrics(y[mask], s[mask], alerts=alerts[mask])
    cfg = {**bootstrap_config(), **bootstrap}
    ci = block_bootstrap_ci(y, s, threshold, alerts=alerts, keep=mask, **cfg)
    return {**metrics, **ci, "bootstrap_block": cfg["block"],
            "bootstrap_n": cfg["n_resamples"], "bootstrap_ci": cfg["ci"]}


def _delta(y_true: np.ndarray, scores: np.ndarray, reference: np.ndarray, *,
           keep: np.ndarray | None = None, **bootstrap) -> dict:
    """Paired ``AUC(scores) - AUC(reference)`` with its block-bootstrap CI.

    Same settings and seed as :func:`_evaluate`, so the difference is taken
    over exactly the resampled blocks the per-method intervals use.
    """
    cfg = {**bootstrap_config(), **bootstrap}
    return paired_auc_delta_ci(y_true, scores, reference, keep=keep, **cfg)


def require_complete_acled(manifest: dict, region: str | None = None) -> None:
    """Refuse a frame whose ACLED events may be cut at a page.

    The connector once read one page per country-year, cutting every
    country-year over 5,000 events; a full (cursor) fetch records its row
    counts in the manifest's ``geopolitical_source``. Used by every consumer of
    the frames (this harness, the surrogate and explanation scripts). The round
    counts are recomputed from the recorded country-years, not taken from the
    recorded flag list.

    Raises:
        ValueError: If the frame has geopolitical features but no source block,
            or a country-year count is a positive multiple of the page size.
    """
    if "geopolitical" not in (manifest.get("domains_present") or {}):
        return
    region = region or manifest.get("region")
    acled = manifest.get("geopolitical_source") or {}
    if not acled:
        raise ValueError(
            f"{region}: the manifest records no geopolitical_source, so this frame's "
            "ACLED events may be cut at one page (5,000 per country-year); rebuild "
            "with scripts/build_eval_dataset.py --force."
        )
    size = int(acled.get("page_size") or 5000)
    counted = [f"{c['country']} {c['year']}: {c['events']}"
               for c in acled.get("country_years") or []
               if c["events"] > 0 and c["events"] % size == 0]
    capped = counted or acled.get("round_cap_country_years") or []
    if capped:
        raise ValueError(
            f"{region}: ACLED country-years with exactly a multiple of the page size "
            f"{size} (treated as capped): {capped[:3]}. Rebuild with "
            "scripts/build_eval_dataset.py --force; if the count is genuine, a rebuild "
            "with agents.geopolitical.api.page_size set to another value (say 4999) "
            "gives the same count, while a capped year changes."
        )


def _masked_days(frame: pd.DataFrame, manifest: dict) -> np.ndarray:
    """True on the frame's days whose news GDELT never published.

    Raises:
        ValueError: If the frame was built under the old policy that
            forward-filled those days: their news values would be invented.
    """
    region = manifest.get("region")
    source = manifest.get("news_source") or {}
    if source.get("days_forward_filled"):
        raise ValueError(
            f"{region}: news days were forward-filled when this "
            "frame was built; rebuild it with scripts/build_eval_dataset.py "
            "(masked policy)."
        )
    # --refresh-news swaps only the news block, so a refreshed old frame
    # would pass the check above with its other domains still interpolated
    # and FRED still on reference dates. Only a full build records these.
    gap = (manifest.get("gap_fill") or {}).get("policy") or ""
    if not gap.startswith("forward-fill"):
        raise ValueError(
            f"{region}: the manifest records no forward-fill gap_fill policy, "
            "so this frame predates the no-interpolation build (or was only "
            "news-refreshed); rebuild with scripts/build_eval_dataset.py --force."
        )
    if "market" in (manifest.get("domains_present") or {}):
        dating = (manifest.get("market_source") or {}).get("dating") or ""
        if not dating.startswith("first-release"):
            raise ValueError(
                f"{region}: market values are not on first-release dates "
                "(manifest market_source); rebuild with "
                "scripts/build_eval_dataset.py --force."
            )
    require_complete_acled(manifest, region)
    days = set(source.get("days_masked") or [])
    stamps = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d")
    return stamps.isin(days).to_numpy()


def _family(name: str, kind: str) -> str:
    """Row family: the oracle is a ceiling, not a baseline a tier competes with."""
    return "ceiling" if kind == "oracle" else "baseline"


def _score_metrics(
    y_true: np.ndarray, scores: np.ndarray, train_scores: np.ndarray | None = None,
    *, threshold: float | None = None, alerts: np.ndarray | None = None,
) -> dict:
    """AUC plus threshold metrics at a train-calibrated operating point.

    These methods emit scores on incomparable scales (a z-score, a tree
    probability, an isolation-forest margin), so a shared 0.5 would measure
    calibration rather than discrimination. The threshold is each method's own
    90th percentile *on the training window* — a "top 10% of quiet-period
    scores" operating point, chosen without reference to the test labels.

    Thresholding at the test median instead would force the alert rate to 50%
    for every method by construction, which makes FPR uninformative and the
    false-positive harness meaningless.

    A day alerts when ``score >= threshold`` (issue 9). With ``>``, a score
    tied with its own 90th percentile could never alert: a constant control
    never fired, and a score floored at 0 on at least 90% of training days had a
    threshold of 0 that its zero days could not reach. With ``>=`` a tie
    alerts, so such a method alerts on every tied day instead; that is
    reported, not special-cased (eval/diagnostics/alert_rule.py). The three
    controls whose meaning is an alert pattern pass ``alerts`` instead
    (:data:`_DIRECT_ALERTS`), so M2 never alerts and the ORACLE alerts on
    the days after a labelled day; AUC still comes from ``scores``.

    ``threshold`` passes in one already computed by :func:`_threshold`.
    """
    scores = np.nan_to_num(np.asarray(scores, dtype=float))
    if alerts is not None:
        if threshold is not None or train_scores is not None:
            raise ValueError("give alerts, or a threshold / train_scores, not both")
        predicted = np.asarray(alerts, dtype=bool).astype(int)
        if len(predicted) != len(scores):
            raise ValueError(f"alerts has {len(predicted)} days, scores {len(scores)}")
    else:
        if threshold is None:
            threshold = _threshold(scores, train_scores)
        predicted = (scores >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "f1": np.nan, "fpr": np.nan,
                "alert_rate": float(predicted.mean())}
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "alert_rate": float(predicted.mean()),
    }


class _NotTrainable(RuntimeError):
    """A supervised method has no positives to fit on.

    Distinct from a failure: it is a property of the data. Every labelled
    disruption in this dataset is recent (Houthi 2024, Gatun drought
    2023-24), so a temporal split leaves the training window with none.
    """


# -------------------------------------------------------------- baselines
def _primary(frame: pd.DataFrame) -> pd.Series:
    """The series the univariate baselines watch: chokepoint transits.

    Named explicitly rather than taken as ``columns[0]``, which in these
    frames is the timestamp.
    """
    return frame["shipping__vessel_count"]


def b1_rolling_z(train, test, features):
    """Rolling z-score of transits; disruption is a *drop*, so negate."""
    joined = pd.concat([train, test])
    values = _primary(joined)
    mean = values.rolling(14, min_periods=5).mean()
    std = values.rolling(14, min_periods=5).std()
    z = ((values - mean) / (std + 1e-6)).fillna(0.0)
    return (-z).to_numpy()


def b2_ma_crossover(train, test, features):
    """Fast MA below slow MA — a downtrend in transits."""
    joined = pd.concat([train, test])
    values = _primary(joined)
    fast = values.rolling(7, min_periods=3).mean()
    slow = values.rolling(30, min_periods=10).mean()
    return ((slow - fast) / (slow + 1e-6)).fillna(0.0).to_numpy()


def b3_isolation_forest(train, test, features):
    """Unsupervised multivariate outlier score, fitted on train only."""
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(train[features].fillna(0.0).to_numpy())
    joined = pd.concat([train, test])[features].fillna(0.0).to_numpy()
    return -model.score_samples(joined)


def b4_ewma_deviation(train, test, features):
    """Deviation of transits from an exponentially weighted baseline."""
    joined = pd.concat([train, test])
    values = _primary(joined)
    ewma = values.ewm(alpha=0.1, adjust=False).mean()
    return ((ewma - values) / (ewma + 1e-6)).fillna(0.0).to_numpy()


def b5_ar_residual(train, test, features):
    """Residual of a 2-lag autoregression, coefficients fitted on train."""
    joined = pd.concat([train, test])
    values = _primary(joined).to_numpy(dtype=float)
    lagged = np.column_stack([values[1:-1], values[:-2]])
    target = values[2:]
    cut = len(train) - 2
    coef, *_ = np.linalg.lstsq(lagged[:cut], target[:cut], rcond=None)
    residual = target - lagged @ coef
    padded = np.concatenate([[0.0, 0.0], residual])
    return -padded


def b6_cusum(train, test, features):
    """One-sided CUSUM on transits, calibrated on train statistics."""
    joined = pd.concat([train, test])
    values = _primary(joined).to_numpy(dtype=float)
    mean = float(np.mean(values[: len(train)]))
    std = float(np.std(values[: len(train)])) or 1.0
    cusum, total = np.zeros(len(values)), 0.0
    for i, value in enumerate(values):
        total = max(0.0, total + (mean - value) / std - 0.5)
        cusum[i] = total
    return cusum


def b7_logistic_regression(train, test, features):
    """Supervised: fitted on the *training* labels, scored on held-out data."""
    y = train["y_true"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        # No positives to learn from. Returning zeros would score AUC 0.5 and
        # read as "no better than chance" when the truth is "cannot be trained".
        raise _NotTrainable("no positive examples in the training window")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train[features].fillna(0.0).to_numpy(), y)
    joined = pd.concat([train, test])[features].fillna(0.0).to_numpy()
    return model.predict_proba(joined)[:, 1]


def b8_random_forest(train, test, features):
    """Supervised: fitted on the training labels, scored on held-out data."""
    y = train["y_true"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise _NotTrainable("no positive examples in the training window")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(train[features].fillna(0.0).to_numpy(), y)
    joined = pd.concat([train, test])[features].fillna(0.0).to_numpy()
    return model.predict_proba(joined)[:, 1]


def m0_random(train, test, features):
    """Trivial control: seeded noise. A working harness must score this ~0.5."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, len(train) + len(test))


def m1_always(train, test, features):
    """Trivial control: constant 1. Alerts every day (:data:`_DIRECT_ALERTS`)."""
    return np.ones(len(train) + len(test))


def m2_never(train, test, features):
    """Trivial control: constant 0. Alerts on no day (:data:`_DIRECT_ALERTS`).

    Its alerts are defined directly: a constant equals its own training 90th
    percentile, so under the ``>=`` threshold rule it would alert every day.
    """
    return np.zeros(len(train) + len(test))


def m3_persistence(train, test, features):
    """Yesterday's *observed traffic* shortfall, carried forward one day.

    Persistence on the feature, not on the label. A persistence baseline that
    reads ``y_true.shift(1)`` scores near 1.0 here and means nothing: the label
    marks runs of 14+ sustained days, so yesterday's answer gives away today's,
    and no detector has yesterday's ground truth at inference — that is the
    thing being detected. See :func:`oracle_label_persistence` for the version
    that does use it, reported separately as a ceiling.
    """
    joined = pd.concat([train, test])
    values = _primary(joined)
    baseline = values.rolling(365, min_periods=90).median()
    shortfall = ((baseline - values) / baseline).fillna(0.0)
    return shortfall.shift(1).fillna(0.0).to_numpy()


def m8_matrix_profile(train, test, features, window: int = 14):
    """1-NN distance from each subsequence to its nearest non-trivial match.

    A subsequence with no close match anywhere else in the series is unusual.
    Implemented directly rather than via ``stumpy`` (not installed): the
    z-normalised sliding windows are compared by matrix product, which is
    tractable at this length, and the alternative of quietly substituting an
    IsolationForest would have duplicated B3 under a second name.

    Trivial matches are excluded — adjacent windows overlap and would
    otherwise be every subsequence's nearest neighbour at distance ~0.
    """
    joined = pd.concat([train, test])
    values = _primary(joined).to_numpy(dtype=float)
    n = len(values) - window + 1
    if n < window * 2:
        return np.zeros(len(joined))

    # Sliding windows, z-normalised so shape is compared rather than level.
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    normed = (windows - mean) / np.where(std > 1e-9, std, 1.0)

    # Squared Euclidean distance via the inner-product identity.
    sq = (normed ** 2).sum(axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * (normed @ normed.T)
    np.fill_diagonal(dist, np.inf)

    # Exclusion zone: neighbours within one window overlap the query.
    idx = np.arange(n)
    dist[np.abs(idx[:, None] - idx[None, :]) < window] = np.inf

    profile = np.sqrt(np.clip(dist.min(axis=1), 0.0, None))
    # Windows are anchored at their start; pad the tail to full length.
    return np.concatenate([profile, np.full(len(joined) - n, profile[-1])])


def oracle_label_persistence(train, test, features):
    """Yesterday's ground truth. Not a method — a ceiling.

    Reported to show what perfect one-day-lagged knowledge would score, so the
    real methods can be read against an upper bound rather than only against
    chance. Any method approaching this is either excellent or leaking.

    Its alerts are defined directly, as the days after a labelled day
    (:func:`_yesterday_positive`). With under 10% positive training days its
    training 90th percentile is 0, and the threshold rule would alert every day.
    """
    joined = pd.concat([train, test])
    return joined["y_true"].astype(float).shift(1).fillna(0.0).to_numpy()


def _yesterday_positive(train, test) -> np.ndarray:
    """The ORACLE's alerts: test days whose previous day is labelled positive.

    The first test day reads the last training day, so ``train`` must be the
    full, contiguous training window.
    """
    label = pd.concat([train, test])["y_true"].astype(int).shift(1).fillna(0)
    return label.to_numpy()[-len(test):] == 1


BASELINES = {
    "M0 random": (m0_random, "control"),
    "M1 always alert": (m1_always, "control"),
    "M2 never alert": (m2_never, "control"),
    "M3 persistence": (m3_persistence, "unsupervised"),
    "M8 matrix profile": (m8_matrix_profile, "unsupervised"),
    "ORACLE label t-1": (oracle_label_persistence, "oracle"),
    "B1 rolling z": (b1_rolling_z, "unsupervised"),
    "B2 MA crossover": (b2_ma_crossover, "unsupervised"),
    "B3 isolation forest": (b3_isolation_forest, "unsupervised"),
    "B4 EWMA deviation": (b4_ewma_deviation, "unsupervised"),
    "B5 AR residual": (b5_ar_residual, "unsupervised"),
    "B6 CUSUM": (b6_cusum, "unsupervised"),
    "B7 logistic regression": (b7_logistic_regression, "supervised"),
    "B8 random forest": (b8_random_forest, "supervised"),
}

#: The alert rule of every row not in :data:`_DIRECT_ALERTS`.
_THRESHOLD_RULE = "score >= training p90"

#: Controls whose alerts are defined directly, not by the threshold rule:
#: name -> (the ``alert_rule`` recorded, ``(train, test) -> test-day alerts``).
#: Their meaning is the alert pattern itself, which the rule would distort
#: (a constant ties its own threshold). Their scores still give AUC.
_DIRECT_ALERTS: dict[str, tuple[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]]] = {
    "M1 always alert": ("direct: every day",
                        lambda train, test: np.ones(len(test), dtype=bool)),
    "M2 never alert": ("direct: no day",
                       lambda train, test: np.zeros(len(test), dtype=bool)),
    "ORACLE label t-1": ("direct: yesterday's label is 1", _yesterday_positive),
}


def _alert_rule(name: str) -> str:
    return _DIRECT_ALERTS[name][0] if name in _DIRECT_ALERTS else _THRESHOLD_RULE


# ---------------------------------------------------------------- agents
def _enabled(config: dict, domain: str) -> bool:
    """Whether the region config enables ``domain``'s agent.

    ``agents.<domain>.enabled``, default true: the flag the orchestrator
    honours. Region YAMLs set it false where the evidence review found no
    driver for that domain (config/regions/, src/core/regions.py).
    """
    return bool(((config.get("agents") or {}).get(domain) or {}).get("enabled", True))


def _all_enabled(config: dict) -> dict:
    """A copy of ``config`` with every tier agent enabled (appendix sensitivity)."""
    out = copy.deepcopy(config)
    agents = dict(out.get("agents") or {})
    for domain in _TIER_ORDER:
        agents[domain] = {**(agents.get(domain) or {}), "enabled": True}
    out["agents"] = agents
    return out


def _without_eq1(config: dict) -> dict:
    """A copy of ``config`` with the shipping agent's Eq. 1 duration term off.

    ``agents.shipping.level_shift: false`` restores the pre-f7b87ff
    aggregation (the shock score alone); see :class:`ShippingAgent`.
    """
    out = copy.deepcopy(config)
    agents = dict(out.get("agents") or {})
    agents["shipping"] = {**(agents.get("shipping") or {}), "level_shift": False}
    out["agents"] = agents
    return out


def _agent_frame(frame: pd.DataFrame, domain: str) -> pd.DataFrame | None:
    """Strip the ``domain__`` prefix so an agent sees the columns it expects."""
    columns = [c for c in frame.columns if c.startswith(f"{domain}__")]
    if not columns:
        return None
    out = frame[["timestamp", *columns]].copy()
    return out.rename(columns={c: c.split("__", 1)[1] for c in columns})


def _history(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The days an agent may read when scoring ``test``: train then test.

    The baselines' ``pd.concat([train, test])``, with any day present in both
    kept once (scoring train against itself must not double it).
    """
    joined = pd.concat([train, test])
    stamps = pd.to_datetime(joined["timestamp"])
    return (joined.assign(_ts=stamps).drop_duplicates("_ts", keep="last")
            .sort_values("_ts", kind="stable").drop(columns="_ts"))


def _agent_scores(domain: str, config: dict, train, test,
                  history: pd.DataFrame | None = None) -> np.ndarray | None:
    """Fit one agent on train and return its per-day anomaly score on test.

    The agent is fitted on ``train`` only, then scored on ``history`` (by
    default train followed by test, the same days the baselines read), and
    only the test-window rows are kept. Scoring the test window on its own
    restarted every rolling window and trailing baseline at the split: the
    shipping agent's duration score, a 30-day mean against a lagged 365-day
    median, was blind for its first ~121 test days. Nothing is refitted on
    test data.

    MarketAgent still clips its input to the five years before the last
    history day (``market_agent.py`` recent-baseline filter); with the current
    split dates that leaves well over its 30-day window before every split.

    An agent the region config disables (:func:`_enabled`) is not built or
    run: it returns ``None``, like an unavailable agent, for every caller
    (tiers, single-agent runs, the diagnostics that score through here;
    eval/diagnostics/market_diff.py builds MarketAgent itself on purpose).

    Args:
        history: Overrides the scoring history, e.g. the whole frame when the
            fitting window and the test window are not adjacent (diagnostics).

    Raises:
        ValueError: If an overriding ``history`` lacks a test day or runs past
            the last test day.
    """
    from src.agents.disaster_agent import DisasterAgent
    from src.agents.geopolitical_agent import GeopoliticalAgent
    from src.agents.market_agent import MarketAgent
    from src.agents.news_agent import NewsAgent
    from src.agents.shipping_agent import ShippingAgent

    classes = {
        "shipping": ShippingAgent, "market": MarketAgent,
        "geopolitical": GeopoliticalAgent, "natural_disaster": DisasterAgent,
        "news_sentiment": NewsAgent,
    }
    if not _enabled(config, domain):
        logger.info("  agent %s disabled in the region config", domain)
        return None
    train_frame = _agent_frame(train, domain)
    test_frame = _agent_frame(test, domain)
    if train_frame is None or test_frame is None:
        return None
    if history is not None:
        # Checked before the try below, which would report a bad override as
        # "agent unavailable"; a test day missing from history would score 0.
        days = pd.to_datetime(history["timestamp"])
        wanted = pd.to_datetime(test["timestamp"])
        missing = wanted[~wanted.isin(days)]
        if len(missing):
            raise ValueError(f"history lacks {len(missing)} test days "
                             f"({missing.min().date()} .. {missing.max().date()})")
        if days.max() > wanted.max():
            raise ValueError(f"history runs after the test window "
                             f"({days.max().date()} > {wanted.max().date()})")
    history_frame = _agent_frame(_history(train, test) if history is None else history,
                                 domain)

    try:
        agent = classes[domain](config=dict(config.get("agents", {}).get(domain, {})))
        agent.fit(train_frame)
        scored = agent.detect(agent.preprocess(history_frame))
        # Align on timestamp: some agents drop rows during preprocessing, and
        # positional assembly would silently shift one agent's scores against
        # another's.
        aligned = (
            pd.Series(
                scored["anomaly_score"].to_numpy(dtype=float),
                index=pd.to_datetime(scored["timestamp"]),
            )
            .reindex(pd.to_datetime(test_frame["timestamp"]))
            .to_numpy(dtype=float)
        )
        return np.nan_to_num(aligned, nan=0.0)
    except Exception as exc:
        logger.warning("  agent %s unavailable: %s", domain, str(exc)[:120])
        return None


def _all_agent_scores(config: dict, train, test,
                      domains: tuple[str, ...] = _TIER_ORDER) -> dict[str, np.ndarray | None]:
    """Each agent fitted once on train and scored on test.

    Every tier is composed from these, so Tier k and Tier 1 share the exact
    same shipping-agent scores and their paired difference measures only
    what the added agents change. A disabled agent's entry is ``None``.
    """
    return {domain: _agent_scores(domain, config, train, test) for domain in domains}


def _raw_weights(tier: int, scores: dict[str, np.ndarray | None],
                 config: dict) -> dict[str, float]:
    """The config weight of each agent a tier fuses, in tier order.

    A tier fuses the first ``tier`` agents of :data:`_TIER_ORDER` that the
    region enables and that produced scores. A disabled agent is left out
    even if ``scores`` holds its scores.
    """
    weights = config.get("weights", {}) or {}
    return {domain: float(weights.get(domain, 0.1)) for domain in _TIER_ORDER[:tier]
            if _enabled(config, domain) and scores.get(domain) is not None}


def _mass(raw: dict[str, float]) -> float:
    """The weights' total, added one at a time in tier order.

    Not ``sum()``: since Python 3.12 it is compensated, and for four of these
    weights it gives 0.8819999999999999 instead of 0.882, which would move a
    tier whose agents did not change.
    """
    mass = 0.0
    for weight in raw.values():
        mass += weight
    return mass


def _fusion_weights(tier: int, scores: dict[str, np.ndarray | None],
                    config: dict) -> dict[str, float]:
    """Each fused agent's weight in a tier, renormalised to sum to 1."""
    raw = _raw_weights(tier, scores, config)
    mass = _mass(raw)
    return {domain: w / mass for domain, w in raw.items()} if mass > 0 else {}


def _compose(tier: int, scores: dict[str, np.ndarray | None],
             config: dict) -> tuple[np.ndarray | None, list[str]]:
    """Composite for a tier: the weighted mean of its enabled agents' scores.

    Equal to ``sum(_fusion_weights[d] * scores[d])``; computed as
    ``sum(w * s) / sum(w)``, both added one at a time in tier order, so a tier
    whose agents did not change keeps its exact earlier values.
    """
    raw = _raw_weights(tier, scores, config)
    mass = _mass(raw)
    if mass <= 0:
        return None, []
    total = None
    for domain, weight in raw.items():
        agent = scores[domain]
        total = weight * agent if total is None else total + weight * agent
    return total / mass, list(raw)


def tier_scores(tier: int, config: dict, train, test) -> tuple[np.ndarray | None, list[str]]:
    """Composite score for a tier: the weighted mean of its enabled agents' scores.

    Args:
        tier: The tier's agents are the first ``tier`` of :data:`_TIER_ORDER`,
            less any the region config disables (:func:`_enabled`).

    Returns:
        ``(scores, agents_used)``. ``scores`` is ``None`` if no enabled agent
        in the tier could be built.
    """
    return _compose(tier, _all_agent_scores(config, train, test, _TIER_ORDER[:tier]), config)


#: What :func:`_masked_days` has verified about every frame scored here.
_FRAME_POLICY = ("news masked on days GDELT never published; other gaps "
                 "forward-filled; FRED values on their first-release dates")

#: Baselines fitted row by row (no rolling state), which can simply leave
#: masked training days out of the fit. Rolling baselines keep the calendar.
_ROW_WISE: frozenset[str] = frozenset(
    {"B3 isolation forest", "B7 logistic regression", "B8 random forest"})


def _window(test: pd.DataFrame, y: np.ndarray, keep: np.ndarray) -> dict:
    return {"test_start": str(test["timestamp"].min().date()),
            "test_end": str(test["timestamp"].max().date()),
            "n_test": int(keep.sum()), "n_masked": int((~keep).sum()),
            "test_positives": int(y[keep].sum()), "frame_policy": _FRAME_POLICY}


def _tier_rows(region: str, config: dict, train, test, keep, train_keep,
               tiers: tuple[int, ...], window: dict, *,
               agent_scores: tuple[dict, dict] | None = None, **bootstrap) -> list[dict]:
    """Scored rows for the given tiers, each with its paired delta vs Tier 1.

    ``agent_scores`` passes in ``(on_test, on_train)`` already computed, as
    :func:`_all_agent_scores` returns them; by default they are computed here.
    """
    y = test["y_true"].astype(int).to_numpy()
    domains = _TIER_ORDER[:max(tiers)]
    if agent_scores is None:
        on_test = _all_agent_scores(config, train, test, domains)
        on_train = _all_agent_scores(config, train, train, domains)
    else:
        on_test, on_train = agent_scores
    tier1, _ = _compose(1, on_test, config)
    evaluable = len(np.unique(y[keep])) > 1
    rows = []
    for tier in tiers:
        scores, used = _compose(tier, on_test, config)
        if scores is None:
            print(f"   Tier {tier:<19} no enabled agent produced scores")
            continue
        train_scores, _ = _compose(tier, on_train, config)
        metrics = _evaluate(y, scores, train_scores if train_scores is not None else scores,
                            keep=keep, train_keep=train_keep if train_scores is not None
                            else keep, **bootstrap)
        delta = {}
        if tier1 is not None and tier != 1 and evaluable:
            delta = _delta(y, scores, tier1, keep=keep, **bootstrap)
        disabled = [d for d in _TIER_ORDER[:tier] if not _enabled(config, d)]
        rows.append({"region": region, "method": f"Tier {tier}", "kind": "multi-agent",
                     "family": "tier", "agents": ",".join(used),
                     "agents_disabled": ",".join(disabled),
                     "fusion_weights": ",".join(
                         f"{d}={w:.4f}"
                         for d, w in _fusion_weights(tier, on_test, config).items()),
                     "circularity": _TIER_CIRCULARITY, "n_agents": len(used),
                     **metrics, **delta, **window})
        shown = (f" dAUC vs T1={delta['delta_auc']:+.3f} "
                 f"[{delta['delta_auc_ci_low']:+.3f}, {delta['delta_auc_ci_high']:+.3f}]"
                 if delta else "")
        print(f"   Tier {tier} ({len(used)} agents){'':<7} AUC={metrics['auc']!s:<7.7} "
              f"F1={metrics['f1']!s:<6.6} alert_rate={metrics['alert_rate']:.2f}"
              f"{shown} [{','.join(used)}]"
              + (f" disabled: {','.join(disabled)}" if disabled else ""))
    return rows


def _main_split(frame: pd.DataFrame, manifest: dict):
    """The fixed split and its scoring masks.

    Returns ``(train, test, keep, train_keep, y_test, window)``; ``keep`` and
    ``train_keep`` are False on masked days.
    """
    masked = _masked_days(frame, manifest)
    train, test = _split(frame)
    mask_of = pd.Series(~masked, index=frame.index)
    train_keep = mask_of.loc[train.index].to_numpy()
    keep = mask_of.loc[test.index].to_numpy()
    y_test = test["y_true"].astype(int).to_numpy()
    return train, test, keep, train_keep, y_test, _window(test, y_test, keep)


def run_region(region: str, frame: pd.DataFrame, manifest: dict, config: dict,
               **bootstrap) -> list[dict]:
    """Every baseline and every tier on the fixed split; masked days unscored."""
    features = [c for c in frame.columns if "__" in c]
    train, test, keep, train_keep, y_test, window = _main_split(frame, manifest)
    evaluable = len(np.unique(y_test[keep])) > 1

    print(f"\n[{region}] train={len(train)} test={len(test)} "
          f"[{window['test_start']} .. {window['test_end']}] "
          f"scored={window['n_test']} masked={window['n_masked']} "
          f"test_positives={window['test_positives']} "
          f"{'' if evaluable else '(FP harness only — no positives in test)'}")

    rows: list[dict] = []
    for name, (fn, kind) in BASELINES.items():
        family = _family(name, kind)
        # Row-wise models are fitted without the masked training days (their
        # news features would be zero-filled); the threshold then uses every
        # remaining training score.
        fit_train, fit_keep = ((train[train_keep], None) if name in _ROW_WISE
                               else (train, train_keep))
        try:
            scores = fn(fit_train, test, features)
        except _NotTrainable as exc:
            rows.append({"region": region, "method": name, "kind": kind,
                         "family": family,
                         "circularity": _CIRCULARITY.get(name, "unknown"),
                         "alert_rule": _alert_rule(name),
                         "auc": np.nan, "f1": np.nan,
                         "fpr": np.nan, "alert_rate": np.nan,
                         "not_applicable": str(exc), **window})
            print(f"   {name:<24} n/a — {exc}")
            continue
        except Exception as exc:
            print(f"   {name:<24} FAILED: {str(exc)[:60]}")
            continue
        scores = np.asarray(scores, dtype=float)
        # The controls' alerts are defined directly; everyone else's come from
        # the threshold rule. The full training window keeps the ORACLE's
        # first test day reading the last training day.
        alerts = _DIRECT_ALERTS[name][1](train, test) if name in _DIRECT_ALERTS else None
        metrics = _evaluate(y_test, scores[-len(test):], scores[: len(fit_train)],
                            keep=keep, train_keep=fit_keep, alerts=alerts, **bootstrap)
        rows.append({"region": region, "method": name, "kind": kind,
                     "family": family,
                     "circularity": _CIRCULARITY.get(name, "unknown"),
                     "alert_rule": _alert_rule(name),
                     **metrics, **window})
        print(f"   {name:<24} AUC={metrics['auc']!s:<7.7} "
              f"F1={metrics['f1']!s:<6.6} alert_rate={metrics['alert_rate']:.2f}")

    # Tiers always use the threshold rule. (In the rare case a tier has no
    # training composite, _tier_rows falls back to the test scores' p90; no
    # tier in the cached frames does.)
    rows += [{**row, "alert_rule": _THRESHOLD_RULE}
             for row in _tier_rows(region, config, train, test, keep, train_keep,
                                   tuple(range(1, len(_TIER_ORDER) + 1)), window,
                                   **bootstrap)]
    return rows


#: ``agent_flags`` of the appendix run that ignores the region's flags.
_ALL_AGENTS = "all agents enabled (sensitivity)"


def all_agents_sensitivity(region: str, frame: pd.DataFrame, manifest: dict,
                           config: dict, **bootstrap) -> list[dict]:
    """Every tier with every agent enabled, the region's flags ignored.

    For the appendix only: the main table honours ``agents.<name>.enabled``.
    Same split, masked days, threshold rule and paired delta vs Tier 1 as
    :func:`run_region`; only the flags differ.
    """
    train, test, keep, train_keep, _, window = _main_split(frame, manifest)
    print(f"\n[{region}] {_ALL_AGENTS}")
    rows = _tier_rows(region, _all_enabled(config), train, test, keep, train_keep,
                      tuple(range(1, len(_TIER_ORDER) + 1)), window, **bootstrap)

    def forced(method: str) -> str:
        tier = int(method.split()[-1])
        return ",".join(d for d in _TIER_ORDER[:tier] if not _enabled(config, d))

    return [{**row, "alert_rule": _THRESHOLD_RULE, "agent_flags": _ALL_AGENTS,
             "agents_forced_on": forced(row["method"])} for row in rows]


def _require_eq1(region: str, config: dict) -> None:
    """Refuse a region config with Eq. 1 already off.

    The ablations pair Eq. 1 off against on; with it off in the config they
    would compare off against off, and the main table would not be Eq. 1.
    """
    if not ((config.get("agents") or {}).get("shipping") or {}).get("level_shift", True):
        raise ValueError(f"{region}: agents.shipping.level_shift is false in the region "
                         "config; the main results and the ablations need it on.")


def _eq1_scores(config: dict, train, test) -> tuple[dict, dict, dict, dict, dict]:
    """Agent scores with Eq. 1 on, and with it off for the shipping agent only.

    Returns ``(on_test, on_train, off_config, off_test, off_train)``. The
    other agents do not read the shipping config, so their scores are reused.
    """
    on_test = _all_agent_scores(config, train, test)
    on_train = _all_agent_scores(config, train, train)
    off = _without_eq1(config)
    off_test = {**on_test, "shipping": _agent_scores("shipping", off, train, test)}
    off_train = {**on_train, "shipping": _agent_scores("shipping", off, train, train)}
    return on_test, on_train, off, off_test, off_train


#: ``eq1`` of the appendix rows with Eq. 1's duration term off.
_NO_EQ1 = "Eq. 1 duration term off (ablation)"


def no_eq1_ablation(region: str, frame: pd.DataFrame, manifest: dict, config: dict,
                    **bootstrap) -> list[dict]:
    """Every tier with Eq. 1's duration term off, paired against it on.

    Only the shipping agent is refitted without the term (:func:`_without_eq1`);
    the other agents are scored once with the main config and their scores
    reused (they do not read the shipping config). Each row carries the tier's metrics with the term off
    (its ``delta_auc`` is against Tier 1 of the same run, as in the main
    table), ``auc_with_eq1`` (the main table's AUC for that tier) and
    ``eq1_delta_auc``: the paired AUC(without) - AUC(with) on the same
    resampled blocks. Appendix only; the main table keeps the term on.

    Raises:
        ValueError: If ``config`` already has the term off: the pairing would
            compare off against off, and the main table would not be Eq. 1.
    """
    _require_eq1(region, config)
    train, test, keep, train_keep, y, window = _main_split(frame, manifest)
    tiers = tuple(range(1, len(_TIER_ORDER) + 1))
    on_test, on_train, off, off_test, off_train = _eq1_scores(config, train, test)
    print(f"\n[{region}] {_NO_EQ1}")
    rows = _tier_rows(region, off, train, test, keep, train_keep, tiers, window,
                      agent_scores=(off_test, off_train), **bootstrap)
    evaluable = len(np.unique(y[keep])) > 1
    out = []
    for row in rows:
        tier = int(row["method"].split()[-1])
        with_eq1, _ = _compose(tier, on_test, config)
        without, _ = _compose(tier, off_test, off)
        paired: dict = {}
        if evaluable and with_eq1 is not None and without is not None:
            d = _delta(y, without, with_eq1, keep=keep, **bootstrap)
            paired = {"auc_with_eq1": float(roc_auc_score(y[keep], np.nan_to_num(with_eq1[keep]))),
                      "eq1_delta_auc": d["delta_auc"],
                      "eq1_delta_auc_ci_low": d["delta_auc_ci_low"],
                      "eq1_delta_auc_ci_high": d["delta_auc_ci_high"],
                      "eq1_delta_valid": d["delta_valid"]}
            print(f"   Tier {tier} without - with Eq. 1: {d['delta_auc']:+.3f} "
                  f"[{d['delta_auc_ci_low']:+.3f}, {d['delta_auc_ci_high']:+.3f}]")
        out.append({**row, "circularity": _NO_EQ1_CIRCULARITY,
                    "alert_rule": _THRESHOLD_RULE, "eq1": _NO_EQ1, **paired})
    return out


#: ``family`` of the per-agent ablation rows.
_ABLATION = "agent_ablation"


def _ablation_circularity(agents: list[str], eq1: bool) -> str:
    """As the tiers: the shipping agent with Eq. 1 carries the label's own
    statistic ("high"), without it "medium"; a set without shipping reads
    other features only ("low")."""
    if "shipping" not in agents:
        return "low"
    return _TIER_CIRCULARITY if eq1 else _NO_EQ1_CIRCULARITY


def _ablation_rows(region: str, config: dict, on_test: dict, on_train: dict, y, keep,
                   train_keep, window: dict, eq1: bool, **bootstrap) -> list[dict]:
    """The full enabled set, each enabled agent alone, and each left out.

    Every row fuses through :func:`_compose` (enabled agents, renormalised
    weights), so the full set is the top tier and an agent alone is Tier 1's
    arithmetic. Leave-one-out rows carry ``loo_delta_auc``: the paired
    AUC(without the agent) - AUC(full set) on the same resampled blocks.
    Each row is thresholded on its own training composite.

    ``agent`` is the agent kept (``ablation == "single"``) or left out
    (``"leave-one-out"``); ``fusion_weights`` are the row's renormalised
    weights; ``agents_unavailable`` names enabled agents that produced no
    scores (they are in no row).
    """
    k = len(_TIER_ORDER)
    full, used = _compose(k, on_test, config)
    if full is None:
        return []
    evaluable = len(np.unique(y[keep])) > 1
    disabled = ",".join(d for d in _TIER_ORDER if not _enabled(config, d))
    unavailable = ",".join(d for d in _TIER_ORDER
                           if _enabled(config, d) and on_test.get(d) is None)
    label = "on" if eq1 else "off"
    print(f"\n[{region}] agent ablation, Eq. 1 {label}: full set [{','.join(used)}]")

    def row(method: str, ablation: str, agent: str, test_scores: dict,
            train_scores: dict) -> dict | None:
        scores, members = _compose(k, test_scores, config)
        if scores is None:
            return None
        reference, _ = _compose(k, train_scores, config)
        metrics = _evaluate(y, scores, reference if reference is not None else scores,
                            keep=keep, train_keep=train_keep if reference is not None
                            else keep, **bootstrap)
        paired: dict = {}
        if ablation == "leave-one-out" and evaluable:
            d = _delta(y, scores, full, keep=keep, **bootstrap)
            paired = {"loo_delta_auc": d["delta_auc"],
                      "loo_delta_auc_ci_low": d["delta_auc_ci_low"],
                      "loo_delta_auc_ci_high": d["delta_auc_ci_high"],
                      "loo_delta_valid": d["delta_valid"]}
        shown = (f" LOO - full={paired['loo_delta_auc']:+.3f} "
                 f"[{paired['loo_delta_auc_ci_low']:+.3f}, {paired['loo_delta_auc_ci_high']:+.3f}]"
                 if paired else "")
        print(f"   {method:<28} AUC={metrics['auc']!s:<7.7} "
              f"alert_rate={metrics['alert_rate']:.2f}{shown}")
        return {"region": region, "method": method, "kind": "agent-ablation",
                "family": _ABLATION, "ablation": ablation, "agent": agent,
                "agents": ",".join(members), "n_agents": len(members),
                "agents_disabled": disabled, "agents_unavailable": unavailable,
                "fusion_weights": ",".join(
                    f"{d}={w:.4f}" for d, w in _fusion_weights(k, test_scores, config).items()),
                "circularity": _ablation_circularity(members, eq1),
                "alert_rule": _THRESHOLD_RULE, "eq1": label,
                **metrics, **paired, **window}

    rows = [row("full enabled set", "full", "", on_test, on_train)]
    rows += [row(f"{a} alone", "single", a, {a: on_test[a]}, {a: on_train.get(a)})
             for a in used]
    rows += [row(f"without {a}", "leave-one-out", a,
                 {d: s for d, s in on_test.items() if d != a},
                 {d: s for d, s in on_train.items() if d != a}) for a in used]
    return [r for r in rows if r is not None]


def agent_ablation(region: str, frame: pd.DataFrame, manifest: dict, config: dict,
                   **bootstrap) -> tuple[list[dict], list[dict]]:
    """Each enabled agent alone and each left out, with Eq. 1 on and off.

    Returns ``(rows_with_eq1, rows_without)``. Same split, masked days,
    threshold rule and bootstrap blocks as :func:`run_region`; with Eq. 1 off
    only the shipping agent is refitted (:func:`_without_eq1`), the others'
    scores are reused. Appendix only.

    Raises:
        ValueError: If ``config`` already has Eq. 1 off (:func:`_require_eq1`).
    """
    _require_eq1(region, config)
    train, test, keep, train_keep, y, window = _main_split(frame, manifest)
    on_test, on_train, off, off_test, off_train = _eq1_scores(config, train, test)
    common = dict(y=y, keep=keep, train_keep=train_keep, window=window, **bootstrap)
    return (_ablation_rows(region, config, on_test, on_train, eq1=True, **common),
            _ablation_rows(region, off, off_test, off_train, eq1=False, **common))


#: Tiers re-scored in the split-date sensitivity table.
_SENSITIVITY_TIERS: tuple[int, ...] = (1, 4)


def split_sensitivity(region: str, frame: pd.DataFrame, manifest: dict, config: dict,
                      **bootstrap) -> list[dict]:
    """Tier 1 and Tier 4 re-scored at each settings.yaml sensitivity split date.

    Same harness as the main table (fixed end date, masked days unscored,
    training-window threshold, paired delta vs Tier 1); only the split moves.
    """
    masked = _masked_days(frame, manifest)
    mask_of = pd.Series(~masked, index=frame.index)
    rows: list[dict] = []
    for split_date in sensitivity_split_dates():
        train, test = split_by_date(frame, split_date=split_date)
        keep = mask_of.loc[test.index].to_numpy()
        train_keep = mask_of.loc[train.index].to_numpy()
        y = test["y_true"].astype(int).to_numpy()
        window = {**_window(test, y, keep),
                  "train_positives": int(train["y_true"].astype(int).sum())}
        print(f"\n[{region}] sensitivity split {split_date.date()}: "
              f"scored={window['n_test']} positives={window['test_positives']}")
        for row in _tier_rows(region, config, train, test, keep, train_keep,
                              _SENSITIVITY_TIERS, window, **bootstrap):
            rows.append({"split_date": str(split_date.date()), **row})
    return rows


# ------------------------------------------------------------------- run
def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    _OUT.mkdir(parents=True, exist_ok=True)

    regions = [p.stem for p in sorted(_CACHE.glob("*.csv"))]
    rows: list[dict] = []
    sensitivity: list[dict] = []
    all_agents: list[dict] = []
    no_eq1: list[dict] = []
    ablation_on: list[dict] = []
    ablation_off: list[dict] = []
    for region in regions:
        frame, manifest = _load(region)
        config = load_config_for_region(region)
        rows += run_region(region, frame, manifest, config)
        sensitivity += split_sensitivity(region, frame, manifest, config)
        all_agents += all_agents_sensitivity(region, frame, manifest, config)
        no_eq1 += no_eq1_ablation(region, frame, manifest, config)
        with_eq1, without_eq1 = agent_ablation(region, frame, manifest, config)
        ablation_on += with_eq1
        ablation_off += without_eq1

    results = pd.DataFrame(rows)
    results.to_csv(_OUT / "method_comparison_results.csv", index=False)
    print(f"\nWrote {_OUT / 'method_comparison_results.csv'} ({len(results)} rows)")
    table = pd.DataFrame(sensitivity)
    table.to_csv(_OUT / "split_sensitivity.csv", index=False)
    print(f"Wrote {_OUT / 'split_sensitivity.csv'} ({len(table)} rows)")
    every = pd.DataFrame(all_agents)
    every.to_csv(_OUT / "method_comparison_all_agents.csv", index=False)
    print(f"Wrote {_OUT / 'method_comparison_all_agents.csv'} ({len(every)} rows; "
          "appendix sensitivity, not the main result)")
    ablation = pd.DataFrame(no_eq1)
    ablation.to_csv(_OUT / "method_comparison_no_eq1.csv", index=False)
    print(f"Wrote {_OUT / 'method_comparison_no_eq1.csv'} ({len(ablation)} rows; "
          "Eq. 1 ablation, not the main result)")
    for name, part in (("agent_ablation.csv", ablation_on),
                       ("agent_ablation_no_eq1.csv", ablation_off)):
        pd.DataFrame(part).to_csv(_OUT / name, index=False)
        print(f"Wrote {_OUT / name} ({len(part)} rows; per-agent ablation, not the main result)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
