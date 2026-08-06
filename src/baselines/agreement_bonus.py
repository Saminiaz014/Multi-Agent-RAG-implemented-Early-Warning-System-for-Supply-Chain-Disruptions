"""Agreement bonus calculation for multi-domain consensus (R7)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_AGREEMENT_THRESHOLD: float = 0.65
_DEFAULT_BONUS_3: float = 0.15  # -> 1.15x multiplier
_DEFAULT_BONUS_5: float = 0.25  # -> 1.25x multiplier


class AgreementBonusCalculator:
    """Multi-domain agreement bonus: if N domains align, boost the composite score.

    Rationale: agreement across independent signal domains is confidence.
    If only one domain spikes, it might be noise; if >= 3 agree, the
    signal is more likely real.
    """

    def __init__(
        self,
        agreement_threshold: float = _DEFAULT_AGREEMENT_THRESHOLD,
        bonus_3_agents: float = _DEFAULT_BONUS_3,
        bonus_5_agents: float = _DEFAULT_BONUS_5,
    ):
        """
        Args:
            agreement_threshold: A domain score >= this counts as "agreeing".
            bonus_3_agents: Multiplier bump for >= 3 domains agreeing.
            bonus_5_agents: Multiplier bump for >= 5 domains agreeing.
        """
        self.agreement_threshold = agreement_threshold
        self.bonus_3_agents = bonus_3_agents
        self.bonus_5_agents = bonus_5_agents

    def apply(
        self,
        composite_score: float,
        domain_scores: dict[str, float],
        active_domains: list[str],
    ) -> tuple[float, dict]:
        """Apply the agreement bonus to a composite score.

        Args:
            composite_score: Weighted average before the bonus.
            domain_scores: ``{domain: anomaly_score}`` for this day.
            active_domains: Domains actually in play (subset of the 6).

        Returns:
            ``(boosted_score, bonus_metadata)`` — score clipped to ``[0, 1]``.
        """
        agreeing = sum(
            1
            for domain in active_domains
            if domain in domain_scores and domain_scores[domain] >= self.agreement_threshold
        )

        bonus_multiplier = 1.0
        bonus_reason = "no bonus"
        if agreeing >= 5:
            bonus_multiplier = 1.0 + self.bonus_5_agents
            bonus_reason = f"{agreeing}/{len(active_domains)} domains agree (high confidence)"
        elif agreeing >= 3:
            bonus_multiplier = 1.0 + self.bonus_3_agents
            bonus_reason = f"{agreeing}/{len(active_domains)} domains agree (moderate confidence)"

        boosted_score = min(composite_score * bonus_multiplier, 1.0)

        metadata = {
            "agreement_bonus_applied": True,
            "agreeing_domains": agreeing,
            "bonus_multiplier": bonus_multiplier,
            "bonus_reason": bonus_reason,
            "composite_before": composite_score,
            "composite_after": boosted_score,
        }

        if bonus_multiplier > 1.0:
            logger.debug(
                "Agreement bonus: %.4f -> %.4f (%s)", composite_score, boosted_score, bonus_reason
            )
        return boosted_score, metadata
