"""Method comparison and measured agent ablation on the real evaluation set.

Reads the cached frames from ``scripts/build_eval_dataset.py`` — real connector
features, real level-shift labels — and scores every method on the same
held-out window.

Three design decisions carry most of the weight:

**One temporal split, applied to everyone.** The last 30% of each region's
series is the test window; nothing is fitted on it. A supervised model scored
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

logger = logging.getLogger(__name__)

_CACHE = _PROJECT_ROOT / "data" / "eval"
_OUT = _PROJECT_ROOT / "eval"

#: Fraction of each series held out. Temporal, not random.
_TEST_FRACTION = 0.30

#: Agents in the order tiers add them, with the feature prefix each reads.
_TIER_ORDER: tuple[str, ...] = (
    "shipping", "market", "geopolitical", "natural_disaster", "news_sentiment",
)


# --------------------------------------------------------------- utilities
def _load(region: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(_CACHE / f"{region}.csv", parse_dates=["timestamp"])
    manifest = json.loads((_CACHE / f"{region}.manifest.json").read_text(encoding="utf-8"))
    return frame, manifest


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(frame) * (1 - _TEST_FRACTION))
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def _score_metrics(
    y_true: np.ndarray, scores: np.ndarray, train_scores: np.ndarray | None = None
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
    """
    scores = np.nan_to_num(np.asarray(scores, dtype=float))
    reference = scores if train_scores is None else np.nan_to_num(train_scores)
    threshold = float(np.quantile(reference, 0.90))
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


BASELINES = {
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


def tier_scores(tier: int, config: dict, train, test) -> tuple[np.ndarray | None, list[str]]:
    """Composite score for a tier: the weighted mean of its agents' scores.

    Args:
        tier: How many agents from :data:`_TIER_ORDER` are enabled.

    Returns:
        ``(scores, agents_used)``. ``scores`` is ``None`` if no agent in the
        tier could be built.
    """
    weights = config.get("weights", {}) or {}
    total = np.zeros(len(test))
    mass = 0.0
    used: list[str] = []
    for domain in _TIER_ORDER[:tier]:
        scores = _agent_scores(domain, config, train, test)
        if scores is None:
            continue
        weight = float(weights.get(domain, 0.1))
        total += weight * scores
        mass += weight
        used.append(domain)
    if mass <= 0:
        return None, []
    return total / mass, used


# ------------------------------------------------------------------- run
def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    _OUT.mkdir(parents=True, exist_ok=True)

    regions = [p.stem for p in sorted(_CACHE.glob("*.csv"))]
    rows: list[dict] = []

    for region in regions:
        frame, manifest = _load(region)
        config = load_config_for_region(region)
        features = [c for c in frame.columns if "__" in c]
        train, test = _split(frame)
        y_test = test["y_true"].astype(int).to_numpy()
        evaluable = len(np.unique(y_test)) > 1

        print(f"\n[{region}] train={len(train)} test={len(test)} "
              f"test_positives={int(y_test.sum())} "
              f"{'' if evaluable else '(FP harness only — no positives in test)'}")

        for name, (fn, kind) in BASELINES.items():
            try:
                scores = fn(train, test, features)
            except _NotTrainable as exc:
                rows.append({"region": region, "method": name, "kind": kind,
                             "family": "baseline", "auc": np.nan, "f1": np.nan,
                             "fpr": np.nan, "alert_rate": np.nan,
                             "not_applicable": str(exc)})
                print(f"   {name:<24} n/a — {exc}")
                continue
            except Exception as exc:
                print(f"   {name:<24} FAILED: {str(exc)[:60]}")
                continue
            scores = np.asarray(scores, dtype=float)
            metrics = _score_metrics(
                y_test, scores[-len(test):], scores[: len(train)]
            )
            rows.append({"region": region, "method": name, "kind": kind,
                         "family": "baseline", **metrics})
            print(f"   {name:<24} AUC={metrics['auc']!s:<7.7} "
                  f"F1={metrics['f1']!s:<6.6} alert_rate={metrics['alert_rate']:.2f}")

        for tier in range(1, len(_TIER_ORDER) + 1):
            scores, used = tier_scores(tier, config, train, test)
            if scores is None:
                print(f"   Tier {tier:<19} no agents available")
                continue
            train_scores, _ = tier_scores(tier, config, train, train)
            metrics = _score_metrics(
                y_test, scores, train_scores if train_scores is not None else scores
            )
            rows.append({"region": region, "method": f"Tier {tier}", "kind": "multi-agent",
                         "family": "tier", "agents": ",".join(used),
                         "n_agents": len(used), **metrics})
            print(f"   Tier {tier} ({len(used)} agents){'':<7} AUC={metrics['auc']!s:<7.7} "
                  f"F1={metrics['f1']!s:<6.6} alert_rate={metrics['alert_rate']:.2f} "
                  f"[{','.join(used)}]")

    results = pd.DataFrame(rows)
    results.to_csv(_OUT / "method_comparison_results.csv", index=False)
    print(f"\nWrote {_OUT / 'method_comparison_results.csv'} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
