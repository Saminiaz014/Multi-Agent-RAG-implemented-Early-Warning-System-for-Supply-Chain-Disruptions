"""Runner for ablation experiments (R7), scored via lightweight domain scorers.

See ``docs/ABLATION_RATIONALE.md`` for why domain scorers stand in for the
production agents here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines.ablations import AblationConfig
from src.baselines.agreement_bonus import AgreementBonusCalculator
from src.baselines.baseline_base import BaselineRunner, best_f1_on_threshold_sweep
from src.baselines.domain_scorers import DOMAIN_SCORERS

logger = logging.getLogger(__name__)

# Configs whose weights are Optuna-tuned per scenario (rule 5) rather than
# used statically from ABLATIONS. A6/A7 differ from A5 only in bonus/RAG,
# so tuning each independently (same objective, same seed) naturally gives
# them A5's weights without extra plumbing to share state across configs.
_TUNED_CONFIG_IDS: frozenset[str] = frozenset({"A5", "A6", "A7"})

# Matches run_tier0/1/2_baselines.py.
_VAL_SLICE = slice(201, 281)


def tune_weights_optuna(
    domain_scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    agents: list[str],
    val_slice: slice = _VAL_SLICE,
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Optuna-search per-domain weights that maximize best-F1 on validation.

    Rule 5 calls for optimizing VUS-PR (D1) on the validation split. D1
    isn't implemented in :class:`~src.baselines.baseline_evaluator.BaselineEvaluator`
    (it needs ``tslearn``, which isn't a project dependency — see that
    module's documented NaN placeholders from R4). Best-achievable F1 on
    validation is used instead — the same substitute objective already
    used to tune EWMA's lambda / CUSUM's threshold in ``tier1_statistical.py``.

    Args:
        domain_scores: ``{domain: full-series scores}`` from :data:`DOMAIN_SCORERS`.
        y_true: Full-series ``y_disruption`` labels.
        agents: Domains to search weights over.
        val_slice: Validation window (default days 201-280).
        n_trials: Optuna trial budget (50, per rule 5).
        seed: Seeds both the sampler and the search for reproducibility.

    Returns:
        ``{domain: weight}`` summing to 1.0. Falls back to equal weights
        if no trial beat an all-zero F1 (e.g. this scenario has no
        positive days in the validation window at all — true for every
        non-``P_CRIT`` Hormuz scenario, since only P_CRIT's event window
        overlaps days 201-280).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_val = y_true[val_slice]

    def objective(trial: optuna.Trial) -> float:
        raw = {a: trial.suggest_float(f"w_{a}", 0.01, 1.0) for a in agents}
        total = sum(raw.values())
        weights = {a: v / total for a, v in raw.items()}

        composite_val = np.zeros(len(y_val))
        for a in agents:
            composite_val += weights[a] * domain_scores[a][val_slice]

        return best_f1_on_threshold_sweep(composite_val, y_val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_value <= 0.0:
        return {a: 1.0 / len(agents) for a in agents}

    raw = {a: study.best_params[f"w_{a}"] for a in agents}
    total = sum(raw.values())
    return {a: v / total for a, v in raw.items()}


class AblationRunner(BaselineRunner):
    """Run a single ablation configuration using lightweight domain scorers.

    Domain scorers are benchmarking proxies (rolling z-score per domain),
    not production agents. They answer: "does the aggregation strategy
    (weighting + agreement bonus) add value?"
    """

    def __init__(self, config: AblationConfig):
        super().__init__(config.name)
        self.config = config
        self.agreement_bonus = AgreementBonusCalculator() if config.use_agreement_bonus else None

    def run(self, df: pd.DataFrame, scenario_id: str, seed: int) -> tuple[np.ndarray, dict]:
        """Score selected domains, aggregate with configured (or tuned) weights.

        Args:
            df: Scenario DataFrame from R3 (one float column per domain).
            scenario_id: Scenario name, for logging.
            seed: Seeds the Optuna search for A5/A6/A7; unused otherwise.

        Returns:
            ``(anomaly_scores, metadata)``.
        """
        n_days = len(df)
        domain_scores: dict[str, np.ndarray] = {}
        for domain in self.config.agents:
            if domain not in DOMAIN_SCORERS:
                logger.warning("Domain '%s' not in scorer registry; skipping", domain)
                continue
            domain_scores[domain] = DOMAIN_SCORERS[domain].score(df)

        weights = dict(self.config.weights)
        tuned = False
        if self.config.config_id in _TUNED_CONFIG_IDS:
            y_true = df["y_disruption"].to_numpy()
            weights = tune_weights_optuna(domain_scores, y_true, self.config.agents, seed=seed)
            tuned = True

        anomaly_scores = np.zeros(n_days)
        for day_idx in range(n_days):
            day_scores: dict[str, float] = {}
            weight_sum = 0.0
            for domain in self.config.agents:
                if domain not in domain_scores:
                    continue
                score = float(domain_scores[domain][day_idx])
                weight = weights.get(domain, 0.0)
                day_scores[domain] = score
                anomaly_scores[day_idx] += score * weight
                weight_sum += weight

            if weight_sum > 0:
                anomaly_scores[day_idx] /= weight_sum

            if self.agreement_bonus is not None:
                anomaly_scores[day_idx], _ = self.agreement_bonus.apply(
                    anomaly_scores[day_idx], day_scores, self.config.agents
                )

        metadata = {
            "scenario_id": scenario_id,
            "ablation_config": self.config.config_id,
            "ablation_name": self.config.name,
            "seed": seed,
            "domains": self.config.agents,
            "weights": weights,
            "weights_tuned": tuned,
            "use_agreement_bonus": self.config.use_agreement_bonus,
            "use_rag": self.config.use_rag,
            "description": self.config.description,
            "note": (
                "Domain scores computed via lightweight proxies (rolling z-score), "
                "not production agents — see docs/ABLATION_RATIONALE.md"
            ),
        }
        logger.info(
            "Ablation %s (%s): %s seed=%d, mean score=%.4f, max score=%.4f%s",
            self.config.config_id, self.config.name, scenario_id, seed,
            anomaly_scores.mean(), anomaly_scores.max(),
            " [tuned]" if tuned else "",
        )
        return anomaly_scores, metadata
