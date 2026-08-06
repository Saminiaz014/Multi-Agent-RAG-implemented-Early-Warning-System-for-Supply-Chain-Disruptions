"""Lightweight per-domain scoring for ablation experiments (R7).

These are benchmarking proxies — a rolling z-score, the same pattern as
``ZScoreBaseline`` in ``tier1_statistical.py`` — not production agents.
The ablation study measures whether the *aggregation strategy* (weighting
+ agreement bonus) adds value; it isn't a test of agent quality, so every
domain uses one simple, identical scoring rule. See
``docs/ABLATION_RATIONALE.md`` for why production agents (which need a
richer multi-column schema than R3 generates) aren't used here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmark.regions import KNOWN_DOMAINS


def _fill_missing(values: np.ndarray) -> np.ndarray:
    """Forward/back-fill sporadic missing days (R1's ~2% missing_data_rate).

    Same fix as ``tier1_statistical._fill_missing`` / ``tier2_classical._prepare_features``:
    a raw NaN reading would otherwise produce a NaN z-score for that one day.
    """
    return pd.Series(values).ffill().bfill().to_numpy()


class DomainScorer:
    """Rolling z-score on a single domain column.

    Args:
        column: Domain column name in the scenario DataFrame (e.g. ``"shipping"``).
        window: Rolling window length in days.
    """

    def __init__(self, column: str, window: int = 30):
        self.column = column
        self.window = window

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Score this domain, returning an array of shape ``(n_days,)`` in ``[0, 1]``.

        Args:
            df: Scenario DataFrame with this scorer's ``column``.

        Returns:
            Rolling z-score mapped from roughly [-3, 3] onto [0, 1].
        """
        values = _fill_missing(df[self.column].to_numpy())

        rolling_mean = pd.Series(values).rolling(self.window, min_periods=1).mean().to_numpy()
        rolling_std = pd.Series(values).rolling(self.window, min_periods=1).std().to_numpy()
        rolling_std = np.where(np.isnan(rolling_std) | (rolling_std == 0), 1e-6, rolling_std)

        z_scores = (values - rolling_mean) / rolling_std
        return np.clip((z_scores + 3) / 6, 0, 1)


# One scorer per domain, sharing the same rule — see module docstring.
DOMAIN_SCORERS: dict[str, DomainScorer] = {domain: DomainScorer(domain) for domain in KNOWN_DOMAINS}
