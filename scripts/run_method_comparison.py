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
contains no market signal despite being labelled as containing one.

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

Usage::

    python scripts/run_method_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
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

    ``train_keep`` drops masked training days from the reference.
    """
    reference = (np.asarray(scores, dtype=float) if train_scores is None
                 else np.asarray(train_scores, dtype=float))
    if train_scores is not None and train_keep is not None:
        reference = reference[np.asarray(train_keep, dtype=bool)]
    return float(np.quantile(np.nan_to_num(reference), 0.90))


def _evaluate(y_true: np.ndarray, scores: np.ndarray,
              train_scores: np.ndarray | None = None, *,
              keep: np.ndarray | None = None, train_keep: np.ndarray | None = None,
              **bootstrap) -> dict:
    """:func:`_score_metrics` plus moving-block bootstrap CIs on the test window.

    The threshold stays the training one in every resample. ``keep`` marks
    the test days that are scored and ``train_keep`` the training days the
    threshold is taken from; masked days (a news outage) count in neither.
    Bootstrap settings default to settings.yaml ``evaluation.bootstrap``, and
    keyword overrides (e.g. ``n_resamples``) are for tests.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    mask = np.ones(len(y), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    threshold = _threshold(s[mask] if train_scores is None else s, train_scores, train_keep)
    metrics = _score_metrics(y[mask], s[mask], threshold=threshold)
    cfg = {**bootstrap_config(), **bootstrap}
    ci = block_bootstrap_ci(y, s, threshold, keep=mask, **cfg)
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
    days = set(source.get("days_masked") or [])
    stamps = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d")
    return stamps.isin(days).to_numpy()


def _family(name: str, kind: str) -> str:
    """Row family: the oracle is a ceiling, not a baseline a tier competes with."""
    return "ceiling" if kind == "oracle" else "baseline"


def _score_metrics(
    y_true: np.ndarray, scores: np.ndarray, train_scores: np.ndarray | None = None,
    *, threshold: float | None = None,
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

    ``threshold`` passes in one already computed by :func:`_threshold`.
    """
    scores = np.nan_to_num(np.asarray(scores, dtype=float))
    if threshold is None:
        threshold = _threshold(scores, train_scores)
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "f1": np.nan, "fpr": np.nan,
                "alert_rate": float((scores > threshold).mean())}
    predicted = (scores > threshold).astype(int)
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
    """Trivial control: constant 1. Alerts every day."""
    return np.ones(len(train) + len(test))


def m2_never(train, test, features):
    """Trivial control: constant 0. Alerts never."""
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
    """
    joined = pd.concat([train, test])
    return joined["y_true"].astype(float).shift(1).fillna(0.0).to_numpy()


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


# ---------------------------------------------------------------- agents
def _agent_frame(frame: pd.DataFrame, domain: str) -> pd.DataFrame | None:
    """Strip the ``domain__`` prefix so an agent sees the columns it expects."""
    columns = [c for c in frame.columns if c.startswith(f"{domain}__")]
    if not columns:
        return None
    out = frame[["timestamp", *columns]].copy()
    return out.rename(columns={c: c.split("__", 1)[1] for c in columns})


def _agent_scores(domain: str, config: dict, train, test) -> np.ndarray | None:
    """Fit one agent on train and return its per-day anomaly score on test."""
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
    train_frame = _agent_frame(train, domain)
    test_frame = _agent_frame(test, domain)
    if train_frame is None or test_frame is None:
        return None

    try:
        agent = classes[domain](config=dict(config.get("agents", {}).get(domain, {})))
        agent.fit(train_frame)
        scored = agent.detect(agent.preprocess(test_frame))
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
    what the added agents change.
    """
    return {domain: _agent_scores(domain, config, train, test) for domain in domains}


def _compose(tier: int, scores: dict[str, np.ndarray | None],
             config: dict) -> tuple[np.ndarray | None, list[str]]:
    """Composite for a tier: the weighted mean of its available agents' scores."""
    weights = config.get("weights", {}) or {}
    total, mass, used = None, 0.0, []
    for domain in _TIER_ORDER[:tier]:
        agent = scores.get(domain)
        if agent is None:
            continue
        weight = float(weights.get(domain, 0.1))
        total = weight * agent if total is None else total + weight * agent
        mass += weight
        used.append(domain)
    if mass <= 0:
        return None, []
    return total / mass, used


def tier_scores(tier: int, config: dict, train, test) -> tuple[np.ndarray | None, list[str]]:
    """Composite score for a tier: the weighted mean of its agents' scores.

    Args:
        tier: How many agents from :data:`_TIER_ORDER` are enabled.

    Returns:
        ``(scores, agents_used)``. ``scores`` is ``None`` if no agent in the
        tier could be built.
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
               tiers: tuple[int, ...], window: dict, **bootstrap) -> list[dict]:
    """Scored rows for the given tiers, each with its paired delta vs Tier 1."""
    y = test["y_true"].astype(int).to_numpy()
    domains = _TIER_ORDER[:max(tiers)]
    on_test = _all_agent_scores(config, train, test, domains)
    on_train = _all_agent_scores(config, train, train, domains)
    tier1, _ = _compose(1, on_test, config)
    evaluable = len(np.unique(y[keep])) > 1
    rows = []
    for tier in tiers:
        scores, used = _compose(tier, on_test, config)
        if scores is None:
            print(f"   Tier {tier:<19} no agents available")
            continue
        train_scores, _ = _compose(tier, on_train, config)
        metrics = _evaluate(y, scores, train_scores if train_scores is not None else scores,
                            keep=keep, train_keep=train_keep if train_scores is not None
                            else keep, **bootstrap)
        delta = {}
        if tier1 is not None and tier != 1 and evaluable:
            delta = _delta(y, scores, tier1, keep=keep, **bootstrap)
        rows.append({"region": region, "method": f"Tier {tier}", "kind": "multi-agent",
                     "family": "tier", "agents": ",".join(used),
                     "circularity": _TIER_CIRCULARITY, "n_agents": len(used),
                     **metrics, **delta, **window})
        shown = (f" dAUC vs T1={delta['delta_auc']:+.3f} "
                 f"[{delta['delta_auc_ci_low']:+.3f}, {delta['delta_auc_ci_high']:+.3f}]"
                 if delta else "")
        print(f"   Tier {tier} ({len(used)} agents){'':<7} AUC={metrics['auc']!s:<7.7} "
              f"F1={metrics['f1']!s:<6.6} alert_rate={metrics['alert_rate']:.2f}"
              f"{shown} [{','.join(used)}]")
    return rows


def run_region(region: str, frame: pd.DataFrame, manifest: dict, config: dict,
               **bootstrap) -> list[dict]:
    """Every baseline and every tier on the fixed split; masked days unscored."""
    masked = _masked_days(frame, manifest)
    features = [c for c in frame.columns if "__" in c]
    train, test = _split(frame)
    mask_of = pd.Series(~masked, index=frame.index)
    train_keep = mask_of.loc[train.index].to_numpy()
    keep = mask_of.loc[test.index].to_numpy()
    y_test = test["y_true"].astype(int).to_numpy()
    evaluable = len(np.unique(y_test[keep])) > 1
    window = _window(test, y_test, keep)

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
                         "auc": np.nan, "f1": np.nan,
                         "fpr": np.nan, "alert_rate": np.nan,
                         "not_applicable": str(exc), **window})
            print(f"   {name:<24} n/a — {exc}")
            continue
        except Exception as exc:
            print(f"   {name:<24} FAILED: {str(exc)[:60]}")
            continue
        scores = np.asarray(scores, dtype=float)
        metrics = _evaluate(y_test, scores[-len(test):], scores[: len(fit_train)],
                            keep=keep, train_keep=fit_keep, **bootstrap)
        rows.append({"region": region, "method": name, "kind": kind,
                     "family": family,
                     "circularity": _CIRCULARITY.get(name, "unknown"),
                     **metrics, **window})
        print(f"   {name:<24} AUC={metrics['auc']!s:<7.7} "
              f"F1={metrics['f1']!s:<6.6} alert_rate={metrics['alert_rate']:.2f}")

    rows += _tier_rows(region, config, train, test, keep, train_keep,
                       tuple(range(1, len(_TIER_ORDER) + 1)), window, **bootstrap)
    return rows


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
    for region in regions:
        frame, manifest = _load(region)
        config = load_config_for_region(region)
        rows += run_region(region, frame, manifest, config)
        sensitivity += split_sensitivity(region, frame, manifest, config)

    results = pd.DataFrame(rows)
    results.to_csv(_OUT / "method_comparison_results.csv", index=False)
    print(f"\nWrote {_OUT / 'method_comparison_results.csv'} ({len(results)} rows)")
    table = pd.DataFrame(sensitivity)
    table.to_csv(_OUT / "split_sensitivity.csv", index=False)
    print(f"Wrote {_OUT / 'split_sensitivity.csv'} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
