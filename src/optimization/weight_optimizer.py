"""Optuna-driven weight optimizer for the supply-chain DSS.

Tunes *all* learnable weights across the three pipeline layers at once:

    Layer 1 — intra-agent feature weights (e.g. market oil/trade/freight)
    Layer 2 — inter-agent aggregation weights (RiskEngine)
    Layer 3 — risk + per-agent detection thresholds, agreement multipliers

Each trial proposes a full weight set, the pipeline is *fitted on the train
split and scored on the validation split*, and the resulting F1 / lead-time /
FPR blend is the value Optuna maximises. The **test split is held out** and
touched exactly once, in :meth:`optimize`, to produce the generalisation
number reported in the thesis. Best weights are written to
``config/optimized_weights.yaml`` (consumed by ``weight_mode: "optimized"``)
and a full results record to ``data/processed/optimization_results.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import yaml

from src.optimization.data_split import DataSplitManager
from src.optimization.pipeline_evaluator import EvalMetrics, PipelineEvaluator
from src.optimization.weight_config import (
    optimized_weights_path,
    resolve_active_weights,
)

logger = logging.getLogger(__name__)

_INVALID_OBJECTIVE: float = -1.0
_RESULTS_PATH = Path("data/processed/optimization_results.json")


def _safe_print(text: str) -> None:
    """Print box-drawing tables without crashing on cp1252 terminals."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - platform-specific
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


class WeightOptimizer:
    """Define the search space, run trials, and persist the best weights.

    Args:
        config: Full application config (reads the ``optimization`` block).
        data_manager: Optional pre-built :class:`DataSplitManager`. When
            ``None`` one is constructed from ``config`` and its splits are
            generated lazily.
    """

    def __init__(self, config: dict, data_manager: DataSplitManager | None = None) -> None:
        self.config = config
        opt_cfg = config.get("optimization", {}) or {}
        self.n_trials: int = int(opt_cfg.get("n_trials", 100))
        self.timeout: int = int(opt_cfg.get("timeout_seconds", 3600))
        self.objective_weights: dict = opt_cfg.get("objective_weights", {}) or {}
        space = opt_cfg.get("parameter_space", {}) or {}
        self._opt_inter: bool = bool(space.get("inter_agent_weights", True))
        self._opt_intra: bool = bool(space.get("intra_agent_weights", True))
        self._opt_thresholds: bool = bool(space.get("thresholds", True))

        self.data_manager: DataSplitManager = data_manager or DataSplitManager(config)
        self.evaluator = PipelineEvaluator(self.data_manager, self.objective_weights)

        sampler_name = str(opt_cfg.get("sampler", "tpe")).lower()
        sampler = (
            optuna.samplers.CmaEsSampler(seed=42)
            if sampler_name == "cmaes"
            else optuna.samplers.TPESampler(seed=42)
        )
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=10)
            if str(opt_cfg.get("pruner", "median")).lower() == "median"
            else optuna.pruners.NopPruner()
        )
        self.study = optuna.create_study(
            study_name="supply_chain_dss_weight_optimization",
            direction=str(opt_cfg.get("direction", "maximize")),
            sampler=sampler,
            pruner=pruner,
        )

    # ------------------------------------------------------------------
    # Parameter space
    # ------------------------------------------------------------------
    def define_parameter_space(self, trial: optuna.Trial) -> dict:
        """Propose a full weight set via Optuna's ``suggest`` API.

        Raw values are suggested and then renormalised so every weight group
        sums to 1.0 (a Dirichlet-style parameterisation that keeps the search
        space unconstrained while the injected weights stay valid). When a
        layer is disabled in ``optimization.parameter_space`` its hand-tuned
        values are used verbatim instead of being searched.

        Args:
            trial: Live Optuna trial (or a ``FixedTrial`` when replaying the
                best parameters).

        Returns:
            Structured params dict with ``inter_weights`` / ``intra`` /
            ``thresholds`` keys, ready for :meth:`PipelineEvaluator.evaluate`.
        """
        hand = resolve_active_weights({**self.config, "weight_mode": "hand_tuned"})

        # --- Layer 2: inter-agent weights (normalised to 1.0) ---
        if self._opt_inter:
            raw = {
                "shipping": trial.suggest_float("inter_shipping", 0.05, 0.50),
                "market": trial.suggest_float("inter_market", 0.05, 0.35),
                "geopolitical": trial.suggest_float("inter_geopolitical", 0.05, 0.50),
                "natural_disaster": trial.suggest_float("inter_disaster", 0.02, 0.25),
                "routing": trial.suggest_float("inter_routing", 0.05, 0.35),
                "news_sentiment": trial.suggest_float("inter_news", 0.02, 0.25),
            }
            total = sum(raw.values())
            inter_weights = {k: v / total for k, v in raw.items()}
        else:
            inter_weights = dict(hand["inter_agent_weights"])

        # --- Layer 1: intra-agent weights ---
        if self._opt_intra:
            shipping_if = trial.suggest_float("shipping_if_weight", 0.4, 0.9)
            routing_model = trial.suggest_float("routing_model_weight", 0.3, 0.8)

            m_oil = trial.suggest_float("market_oil", 0.2, 0.6)
            m_trade = trial.suggest_float("market_trade", 0.1, 0.5)
            m_freight = trial.suggest_float("market_freight", 0.1, 0.4)
            m_tot = m_oil + m_trade + m_freight

            g_sanc = trial.suggest_float("geo_sanctions", 0.15, 0.50)
            g_mil = trial.suggest_float("geo_military", 0.10, 0.40)
            g_dip = trial.suggest_float("geo_diplomatic", 0.10, 0.40)
            g_stab = trial.suggest_float("geo_stability", 0.05, 0.30)
            g_tot = g_sanc + g_mil + g_dip + g_stab

            d_eq = trial.suggest_float("disaster_earthquake", 0.15, 0.50)
            d_ts = trial.suggest_float("disaster_tsunami", 0.10, 0.45)
            d_cy = trial.suggest_float("disaster_cyclone", 0.05, 0.35)
            d_sw = trial.suggest_float("disaster_weather", 0.05, 0.30)
            d_tot = d_eq + d_ts + d_cy + d_sw

            n_sent = trial.suggest_float("news_sentiment_w", 0.20, 0.60)
            n_cons = trial.suggest_float("news_consensus_w", 0.10, 0.40)
            n_vel = trial.suggest_float("news_velocity_w", 0.05, 0.35)
            n_vol = trial.suggest_float("news_volume_w", 0.05, 0.30)
            n_tot = n_sent + n_cons + n_vel + n_vol

            intra = {
                "shipping": {
                    "isolation_forest": shipping_if,
                    "zscore": 1.0 - shipping_if,
                },
                "market": {
                    "oil": m_oil / m_tot,
                    "trade_volume": m_trade / m_tot,
                    "freight": m_freight / m_tot,
                },
                "geopolitical": {
                    "sanctions": g_sanc / g_tot,
                    "military": g_mil / g_tot,
                    "diplomatic": g_dip / g_tot,
                    "stability": g_stab / g_tot,
                },
                "natural_disaster": {
                    "earthquake": d_eq / d_tot,
                    "tsunami": d_ts / d_tot,
                    "cyclone": d_cy / d_tot,
                    "severe_weather": d_sw / d_tot,
                },
                "routing": {
                    "model_score": routing_model,
                    "transit_zscore": 1.0 - routing_model,
                },
                "news_sentiment": {
                    "sentiment": n_sent / n_tot,
                    "consensus": n_cons / n_tot,
                    "velocity": n_vel / n_tot,
                    "volume": n_vol / n_tot,
                },
            }
        else:
            intra = {k: dict(v) for k, v in hand["intra_agent_weights"].items()}

        # --- Layer 3: thresholds ---
        if self._opt_thresholds:
            thresholds = {
                "risk_high": trial.suggest_float("risk_high", 0.55, 0.85),
                "risk_medium": trial.suggest_float("risk_medium", 0.25, 0.55),
                "agreement_bonus_3": trial.suggest_float("agreement_bonus_3", 1.0, 1.3),
                "agreement_bonus_5": trial.suggest_float("agreement_bonus_5", 1.1, 1.5),
                "shipping_threshold": trial.suggest_float("shipping_threshold", 0.4, 0.8),
                "market_z_threshold": trial.suggest_float("market_z_threshold", 1.5, 3.5),
                "geopolitical_threshold": trial.suggest_float(
                    "geopolitical_threshold", 0.3, 0.7
                ),
                "disaster_threshold": trial.suggest_float("disaster_threshold", 0.15, 0.5),
                "disaster_single_event": trial.suggest_float(
                    "disaster_single_event", 0.25, 0.6
                ),
                "routing_threshold": trial.suggest_float("routing_threshold", 0.4, 0.8),
                "news_negative_threshold": trial.suggest_float(
                    "news_neg_threshold", -0.5, -0.1
                ),
                "news_consensus_threshold": trial.suggest_float(
                    "news_consensus_threshold", 0.2, 0.6
                ),
            }
        else:
            thresholds = dict(hand["thresholds"])

        return {
            "inter_weights": inter_weights,
            "intra": intra,
            "thresholds": thresholds,
        }

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective — fit on train, score on validation.

        Returns the F1/lead-time/FPR blend for the validation split, or
        :data:`_INVALID_OBJECTIVE` when a hard constraint is violated.
        """
        params = self.define_parameter_space(trial)
        thr = params["thresholds"]

        # Hard constraints — ordering of risk levels and agreement bonuses.
        if thr["risk_high"] <= thr["risk_medium"]:
            return _INVALID_OBJECTIVE
        if thr["agreement_bonus_5"] <= thr["agreement_bonus_3"]:
            return _INVALID_OBJECTIVE

        metrics = self.evaluator.evaluate(params, fit_split="train", eval_split="validation")

        trial.report(metrics.objective, step=0)
        if trial.should_prune():  # pragma: no cover - timing-dependent
            raise optuna.TrialPruned()

        trial.set_user_attr("f1", metrics.f1)
        trial.set_user_attr("lead_time_score", metrics.lead_time_score)
        trial.set_user_attr("fpr", metrics.fpr)
        logger.info(
            "[trial %d] objective=%.4f | f1=%.3f lead=%.3f fpr=%.3f",
            trial.number, metrics.objective, metrics.f1,
            metrics.lead_time_score, metrics.fpr,
        )
        return metrics.objective

    # ------------------------------------------------------------------
    # Optimize + finalise
    # ------------------------------------------------------------------
    def optimize(self, n_trials: int | None = None, timeout: int | None = None) -> dict:
        """Run the study, evaluate on test once, and persist results.

        Args:
            n_trials: Trial budget (defaults to the configured ``n_trials``).
            timeout: Wall-clock cap in seconds (defaults to configured value).

        Returns:
            The full results dict (also written to
            ``data/processed/optimization_results.json``).
        """
        n_trials = n_trials if n_trials is not None else self.n_trials
        timeout = timeout if timeout is not None else self.timeout

        # Ensure splits exist before timing/running so generation cost is not
        # charged against the trial timeout.
        self.data_manager.get_splits()
        logger.info(
            "[WeightOptimizer] starting study | n_trials=%d timeout=%ds", n_trials, timeout
        )
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        best = self.study.best_trial
        best_params = self.define_parameter_space(optuna.trial.FixedTrial(best.params))

        # Validation (re-evaluated for a clean record) + held-out TEST (once).
        val_metrics = self.evaluator.evaluate(best_params, "train", "validation")
        test_metrics = self.evaluator.evaluate(best_params, "train", "test")

        # Hand-tuned baseline on the same splits, for the comparison table.
        hand_params = self._hand_tuned_params()
        hand_val = self.evaluator.evaluate(hand_params, "train", "validation")
        hand_test = self.evaluator.evaluate(hand_params, "train", "test")

        results = self._assemble_results(
            best, best_params, val_metrics, test_metrics, hand_val, hand_test
        )
        self._save_optimized_weights(best_params, best, val_metrics)
        self._save_results(results)
        self._print_comparison(hand_test, test_metrics)
        return results

    def evaluate_on_test(self, params: dict) -> EvalMetrics:
        """Final, single-use evaluation on the held-out test split."""
        return self.evaluator.evaluate(params, fit_split="train", eval_split="test")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hand_tuned_params(self) -> dict:
        layout = resolve_active_weights({**self.config, "weight_mode": "hand_tuned"})
        return {
            "inter_weights": layout["inter_agent_weights"],
            "intra": layout["intra_agent_weights"],
            "thresholds": layout["thresholds"],
        }

    def _assemble_results(
        self,
        best: optuna.trial.FrozenTrial,
        best_params: dict,
        val: EvalMetrics,
        test: EvalMetrics,
        hand_val: EvalMetrics,
        hand_test: EvalMetrics,
    ) -> dict:
        history = [
            {
                "trial": t.number,
                "value": t.value,
                "state": str(t.state),
            }
            for t in self.study.trials
        ]
        n_completed = sum(
            1 for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        return {
            "best_trial": best.number,
            "n_trials_completed": n_completed,
            "best_objective_value": best.value,
            "validation_metrics": val.as_dict(),
            "test_metrics": test.as_dict(),
            "hand_tuned_metrics": {
                "validation": hand_val.as_dict(),
                "test": hand_test.as_dict(),
            },
            "improvement": {
                "f1_delta": test.f1 - hand_test.f1,
                "lead_time_delta": test.lead_time_days - hand_test.lead_time_days,
                "fpr_delta": test.fpr - hand_test.fpr,
            },
            "best_weights": {
                "inter_agent_weights": best_params["inter_weights"],
                "intra_agent_weights": best_params["intra"],
                "thresholds": best_params["thresholds"],
            },
            "optimization_history": history,
        }

    def _save_optimized_weights(
        self, best_params: dict, best: optuna.trial.FrozenTrial, val: EvalMetrics
    ) -> Path:
        path = optimized_weights_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# AUTO-GENERATED BY OPTUNA — DO NOT EDIT MANUALLY\n"
            f"# Optimization date: {datetime.now(timezone.utc).isoformat()}\n"
            f"# Best trial: {best.number}\n"
            f"# Validation score: {val.objective:.4f}\n"
            f"# Trials run: {len(self.study.trials)}\n\n"
        )
        body = yaml.safe_dump(
            {
                "inter_agent_weights": _round(best_params["inter_weights"]),
                "intra_agent_weights": {
                    k: _round(v) for k, v in best_params["intra"].items()
                },
                "thresholds": _round(best_params["thresholds"]),
            },
            sort_keys=False,
            default_flow_style=False,
        )
        path.write_text(header + body, encoding="utf-8")
        logger.info("[WeightOptimizer] wrote optimized weights to %s", path)
        return path

    @staticmethod
    def _save_results(results: dict) -> Path:
        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("[WeightOptimizer] wrote results to %s", _RESULTS_PATH)
        return _RESULTS_PATH

    @staticmethod
    def _print_comparison(hand: EvalMetrics, opt: EvalMetrics) -> None:
        def pct(old: float, new: float) -> str:
            if old == 0:
                return "  n/a   " if new == 0 else " +new   "
            return f"{(new - old) / abs(old) * 100:+6.1f}%"

        lines = [
            "",
            "┌─────────────────┬──────────────┬──────────────┬──────────────┐",
            "│ Metric          │ Hand-Tuned   │ Optimized    │ Improvement  │",
            "├─────────────────┼──────────────┼──────────────┼──────────────┤",
            f"│ F1 (test)       │ {hand.f1:>12.2f} │ {opt.f1:>12.2f} │ {pct(hand.f1, opt.f1):>12} │",
            f"│ Lead Time (days)│ {hand.lead_time_days:>12.2f} │ {opt.lead_time_days:>12.2f} │ "
            f"{opt.lead_time_days - hand.lead_time_days:>+10.2f}d │",
            f"│ FPR (test)      │ {hand.fpr:>12.2f} │ {opt.fpr:>12.2f} │ {pct(hand.fpr, opt.fpr):>12} │",
            "└─────────────────┴──────────────┴──────────────┴──────────────┘",
        ]
        _safe_print("\n".join(lines))


def _round(d: dict, ndigits: int = 6) -> dict:
    """Round float dict values for tidy YAML output."""
    return {k: round(float(v), ndigits) for k, v in d.items()}
