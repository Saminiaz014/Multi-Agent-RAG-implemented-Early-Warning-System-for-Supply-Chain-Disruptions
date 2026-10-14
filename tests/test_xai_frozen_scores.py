"""Detection is frozen: Path D must still reproduce the method-comparison CSV.

The explainability work reads agent scores; it must never change them. This
recomputes every Tier 1-5 composite with the script's own ``tier_scores`` and
``_score_metrics`` and checks AUC, F1, FPR and alert rate against
``eval/method_comparison_results.csv``. It also checks that the cached
per-agent scores the explanation layer uses are bit-identical to the script's.

``data/eval/`` is gitignored (rebuilt by ``scripts/build_eval_dataset.py``),
so the test skips on a checkout without it.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import path_d

pytestmark = pytest.mark.skipif(
    not path_d.eval_data_available(), reason="data/eval cache not built"
)

_TOL = 1e-9
_CSV = path_d._PROJECT_ROOT / "eval" / "method_comparison_results.csv"


@pytest.fixture(scope="module")
def csv() -> pd.DataFrame:
    return pd.read_csv(_CSV)


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= _TOL


@pytest.mark.parametrize("region", path_d.REGIONS)
def test_tier_metrics_match_csv(region: str, csv: pd.DataFrame) -> None:
    mod = path_d.comparison_module()
    config = path_d.csv_config(region)
    frame, _ = mod._load(region)
    train, test = mod._split(frame)
    y_test = test["y_true"].astype(int).to_numpy()
    cached = path_d.load_region(region)

    for tier in range(1, 6):
        scores, used = mod.tier_scores(tier, config, train, test)
        train_scores, _ = mod.tier_scores(tier, config, train, train)
        metrics = mod._score_metrics(y_test, scores, train_scores)
        row = csv[(csv.region == region) & (csv.method == f"Tier {tier}")].iloc[0]

        for key in ("auc", "f1", "fpr", "alert_rate"):
            assert _close(metrics[key], float(row[key])), (region, tier, key)
        assert ",".join(used) == row["agents"], (region, tier)

        # The explanation layer's cached scores are the script's, bit for bit.
        assert cached.tier_agents(tier) == used
        np.testing.assert_array_equal(cached.tier_scores(tier, "test"), scores)
        np.testing.assert_array_equal(cached.tier_scores(tier, "train"), train_scores)
