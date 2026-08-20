"""Plain-language explanations for risk spikes on the Decision view (Phase 12.5).

Mirrors the pattern :func:`src.dashboard.core.generate_risk_narrative` already
established: call Anthropic when ``ANTHROPIC_API_KEY`` is configured, otherwise
compose the paragraph from the live agent scores. Both paths return the same
``(text, source)`` shape, so the caller renders one thing and the UI works
identically with or without a key — which is what makes this testable here,
where no key is set.

**What an explanation may and may not say.** The timeline's dates are a rolling
display window, not the data's own timestamps (see
:func:`src.dashboard.core.timeline_dates`). A spike on "2026-06-02" is a
position in a synthetic evaluation series, *not* a real-world event on that
date. So the prompt forbids naming real incidents, dates, or actors, and asks
for an explanation of *the signal* — which agents moved and what that pattern
means operationally. The composed fallback is bound by the same rule by
construction: it only ever restates the agent scores it was given.

Explanations are cached per ``(region, day, level)`` for the process lifetime,
so re-clicking a spike never re-queries.
"""

from __future__ import annotations

import logging
import os

from src.core.regions import get_region

logger = logging.getLogger(__name__)

#: Anthropic model for the optional LLM path. Claude Opus 5 has thinking on by
#: default and ``max_tokens`` caps thinking *plus* response text together, so
#: the budget below is deliberately generous for a three-sentence answer —
#: sizing it to the visible output would truncate mid-sentence.
_MODEL = "claude-opus-5"
_MAX_TOKENS = 2000

#: `low` effort suits a short explanation. Note we keep thinking *on* rather
#: than disabling it: on this model disabling thinking can leak internal tags
#: into the visible response, and low effort already gets the cost saving.
_EFFORT = "low"

#: Process-lifetime cache: {(region, day, level): (text, source)}.
_CACHE: dict[tuple[str, int, str], tuple[str, str]] = {}

_SYSTEM = (
    "You explain supply-chain risk signals to a logistics manager. You are "
    "describing the output of a multi-agent anomaly-detection model running on "
    "SYNTHETIC evaluation data. Never name a real incident, organisation, "
    "vessel, or date, and never state that anything happened in the world — "
    "there is no real event behind this score. Explain which detection signals "
    "moved together and what that combination would mean operationally. Two to "
    "three sentences, plain language, no numbers, no jargon, no preamble."
)


def _format_drivers(drivers: list[tuple[str, float]]) -> str:
    """Render agent scores as readable names, strongest first."""
    return ", ".join(
        f"{name.replace('_', ' ')} ({score:.2f})" for name, score in drivers
    )


def _compose(
    region: str, level: str, drivers: list[tuple[str, float]], agreement: int
) -> str:
    """Build a deterministic explanation from the agent scores alone.

    This is the no-API-key path and the failure path. It is generated from one
    algorithm over the live drivers rather than per-combination canned strings,
    matching how :func:`core._compose_narrative` already works.

    Args:
        region: Region key.
        level: Threshold band the spike crossed.
        drivers: ``(agent, score)`` pairs, strongest first.
        agreement: How many agents scored above 0.5 that day.

    Returns:
        A two-to-three sentence explanation.
    """
    display = get_region(region).display_name
    if not drivers:
        return (
            f"Risk for the {display} moved into the {level.lower()} band, but no "
            "single agent stands out as the driver."
        )

    top = drivers[0][0].replace("_", " ")
    others = [name.replace("_", " ") for name, _ in drivers[1:3]]

    if others:
        joined = " and ".join(filter(None, [", ".join(others[:-1]), others[-1]]))
        lead = (
            f"Risk for the {display} rose into the {level.lower()} band, led by the "
            f"{top} signal with {joined} moving in the same direction."
        )
    else:
        lead = (
            f"Risk for the {display} rose into the {level.lower()} band, driven "
            f"almost entirely by the {top} signal."
        )

    if agreement >= 3:
        corroboration = (
            f" {agreement} detectors agree, so this is a broad shift rather than "
            "one noisy input — worth treating as a real change in conditions."
        )
    elif agreement == 2:
        corroboration = (
            " Two detectors agree, which is suggestive but thin; confirm against "
            "the underlying feed before acting."
        )
    else:
        corroboration = (
            " Only one detector is elevated, so this may be noise in a single "
            "input rather than a genuine change."
        )
    return lead + corroboration


def _call_anthropic(prompt: str) -> str | None:
    """Ask Claude for the explanation, or return ``None`` if unavailable.

    Returns ``None`` — rather than raising — for every failure mode, so the
    caller always has the composed fallback to fall back to: no key, package
    not installed, network error, or a safety refusal. Refusal is a realistic
    outcome here, not a theoretical one: these prompts describe sanctions,
    military and blockade signals.

    Args:
        prompt: The user-turn content.

    Returns:
        The model's text, or ``None``.
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            output_config={"effort": _EFFORT},
            messages=[{"role": "user", "content": prompt}],
        )

        # Check stop_reason before touching content: on a refusal the content
        # list is empty (or partial), so indexing it first would raise.
        if message.stop_reason == "refusal":
            logger.warning(
                "[llm_explanations] request declined by safety classifiers "
                "(%s) — using the composed explanation.",
                getattr(getattr(message, "stop_details", None), "category", None),
            )
            return None

        text = next(
            (b.text for b in message.content if getattr(b, "type", None) == "text"),
            "",
        ).strip()
        return text or None
    except Exception as exc:  # pragma: no cover - network/key/package dependent
        logger.warning(
            "[llm_explanations] Anthropic call failed (%s) — using the composed "
            "explanation.", exc,
        )
        return None


def explain_spike(
    region: str,
    spike: dict,
    drivers: list[tuple[str, float]],
    agreement: int,
) -> tuple[str, str]:
    """Explain one risk spike, cached per region/day/level.

    Args:
        region: Region key.
        spike: A spike dict from :func:`src.dashboard.core.detect_risk_spikes`.
        drivers: ``(agent, score)`` pairs for that day, strongest first.
        agreement: Number of agents scoring above 0.5 that day.

    Returns:
        ``(text, source)`` where source is ``"llm"`` or ``"composed"``.
    """
    key = (region, int(spike["day"]), str(spike["level"]))
    if key in _CACHE:
        return _CACHE[key]

    level = str(spike["level"])
    prompt = (
        f"Chokepoint: {get_region(region).display_name}. "
        f"Risk crossed into the {level} band. "
        f"Detection agents and their anomaly scores, strongest first: "
        f"{_format_drivers(drivers) or 'none elevated'}. "
        f"{agreement} of the region's agents are above their alert level. "
        "Explain what this pattern of signals means."
    )

    text = _call_anthropic(prompt)
    result = (
        (text, "llm") if text else (_compose(region, level, drivers, agreement), "composed")
    )
    _CACHE[key] = result
    return result


def clear_explanation_cache() -> None:
    """Drop cached explanations — for tests, or to force a re-query."""
    _CACHE.clear()


def cache_size() -> int:
    """Number of cached explanations."""
    return len(_CACHE)
