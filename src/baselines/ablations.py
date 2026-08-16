"""Ablation configurations for complexity justification (R7).

Each :class:`AblationConfig` answers a narrower question than the last:
does adding domains help, does weighting them help, does *tuning* those
weights help, and finally — the sharpest question — does an agreement
bonus on top of tuned weights help. See ``docs/ABLATION_RATIONALE.md``
for why these run against lightweight domain scorers rather than the
production agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.benchmark.regions import KNOWN_DOMAINS

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation run.

    Attributes:
        config_id: ``"A0"``..``"A7"``.
        name: Short slug, e.g. ``"shipping_only"``.
        agents: Domains included (subset of :data:`~src.benchmark.regions.KNOWN_DOMAINS`).
        weights: ``{domain: weight}``. Renormalized to sum to 1.0 if it doesn't already.
        use_agreement_bonus: Whether :class:`~src.baselines.agreement_bonus.AgreementBonusCalculator`
            is applied on top of the weighted composite.
        use_rag: Whether RAG retrieval runs. RAG is post-detection (see
            ``docs/ABLATION_RATIONALE.md``), so this never changes the
            anomaly score itself — it only gates explanation generation.
        description: One-line purpose, for logging/summary tables.
    """

    config_id: str
    name: str
    agents: list[str]
    weights: dict[str, float]
    use_agreement_bonus: bool = False
    use_rag: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            logger.warning(
                "Config %s: weights sum to %.4f, normalizing", self.config_id, weight_sum
            )
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}


_ALL_SIX: list[str] = list(KNOWN_DOMAINS)

# A5/A6/A7's weights below are placeholders/documentation only. Per rule 5,
# the actual weights used at evaluation time are Optuna-tuned per scenario
# on the validation split (days 201-280) — see
# ablation_runner.tune_weights_optuna(). A6 and A7 reuse whatever A5 tunes
# to for that scenario (they differ from A5 only in bonus/RAG), not these
# static numbers.
_TUNED_PLACEHOLDER: dict[str, float] = {
    "shipping": 0.22,
    "market": 0.18,
    "geopolitical": 0.19,
    "routing": 0.16,
    "news": 0.17,
    "disaster": 0.08,
}

ABLATIONS: dict[str, AblationConfig] = {
    "A0": AblationConfig(
        config_id="A0",
        name="shipping_only",
        agents=["shipping"],
        weights={"shipping": 1.0},
        description="Floor: single domain (shipping only)",
    ),
    "A1": AblationConfig(
        config_id="A1",
        name="2_agents",
        agents=["shipping", "market"],
        weights={"shipping": 0.6, "market": 0.4},
        description="Pair: shipping + market (hand-tuned weights)",
    ),
    "A2": AblationConfig(
        config_id="A2",
        name="4_agents",
        agents=["shipping", "market", "geopolitical", "routing"],
        weights={"shipping": 0.25, "market": 0.15, "geopolitical": 0.30, "routing": 0.30},
        description="Mid-rung: 4 agents (drop news, disaster)",
    ),
    "A3": AblationConfig(
        config_id="A3",
        name="6_equal",
        agents=list(_ALL_SIX),
        weights={a: 1 / 6 for a in _ALL_SIX},
        description="All 6 domains, equal weights (value of weighting)",
    ),
    "A4": AblationConfig(
        config_id="A4",
        name="6_handtuned",
        agents=list(_ALL_SIX),
        weights={
            "shipping": 0.20,
            "market": 0.15,
            "geopolitical": 0.20,
            "routing": 0.15,
            "news": 0.20,
            "disaster": 0.10,
        },
        description="All 6 domains, hand-tuned weights (existing config/settings.yaml values)",
    ),
    "A5": AblationConfig(
        config_id="A5",
        name="6_optuna",
        agents=list(_ALL_SIX),
        weights=dict(_TUNED_PLACEHOLDER),
        description="All 6 domains, Optuna-optimized weights (tuned per scenario on validation)",
    ),
    "A6": AblationConfig(
        config_id="A6",
        name="6_+bonus",
        agents=list(_ALL_SIX),
        weights=dict(_TUNED_PLACEHOLDER),
        use_agreement_bonus=True,
        description="A5 + agreement bonus (multi-domain consensus enforcement)",
    ),
    "A7": AblationConfig(
        config_id="A7",
        name="6_+bonus_-rag",
        agents=list(_ALL_SIX),
        weights=dict(_TUNED_PLACEHOLDER),
        use_agreement_bonus=True,
        use_rag=False,
        description="A6 but skip RAG (explanation only; detection score is identical to A6)",
    ),
}


def scope_to_domains(config: AblationConfig, active_domains: list[str]) -> AblationConfig:
    """Restrict ``config`` to domains actually active in a region.

    ``ABLATIONS`` is a fixed, region-agnostic dict — A2 hardcodes
    ``geopolitical`` and A3-A7 hardcode all six :data:`KNOWN_DOMAINS`, but
    only Hormuz has all six domains active (every other region is missing
    at least one, per ``Region.active_domains``). Scoring a domain a region
    never populates isn't a graceful no-op: :class:`~src.baselines.domain_scorers.DomainScorer`
    reads an all-``NaN`` column for it (``materialize_scenario`` still
    writes the column, just never fills it for inactive domains), and that
    ``NaN`` poisons the weighted composite for every day, collapsing
    ``D3_auc_pr`` to ``NaN`` and every threshold-based metric to a
    degenerate ``0.0`` — silently, with no error raised (found running A2-A7
    for bab_el_mandeb and panama, 2026-08-16; see
    docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6 gap 21).

    Args:
        config: A declared ``ABLATIONS`` entry.
        active_domains: The evaluating region's ``Region.active_domains``.

    Returns:
        A new :class:`AblationConfig` with ``agents``/``weights`` filtered
        to ``active_domains`` and weights renormalized over the survivors
        (via ``__post_init__``). For Hormuz (all six domains active) this
        is a no-op — every field is unchanged. Every region observed so
        far has ``shipping`` active, so ``agents`` never empties out; this
        function does not special-case a region where it would.
    """
    surviving_agents = [a for a in config.agents if a in active_domains]
    surviving_weights = {a: w for a, w in config.weights.items() if a in active_domains}
    return AblationConfig(
        config_id=config.config_id,
        name=config.name,
        agents=surviving_agents,
        weights=surviving_weights,
        use_agreement_bonus=config.use_agreement_bonus,
        use_rag=config.use_rag,
        description=config.description,
    )
