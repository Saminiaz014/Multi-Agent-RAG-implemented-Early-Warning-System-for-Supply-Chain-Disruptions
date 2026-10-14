"""Phase 6: deterministic narratives for every AER, and METRIC 9 E9.

Fills each AER's ``narrative`` field (template, decimal style) and replaces the
pending E9 rows in ``scorecard.csv`` with the template's narrative
faithfulness (supported claims / claims, per :func:`check_claims`).

Faithfulness of a template is 1.0 by construction, so E9 is only informative
together with the checker's sensitivity, which is measured here too: every
2- or 3-decimal number of every template (up to 8 per narrative) is shifted by
0.07 and the share of corrupted narratives the checker rejects is reported.
No API or LLM call.

Usage::

    python eval/explainability/run_narratives.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.explainability_metrics import episode_bootstrap  # noqa: E402
from src.explainability.narrative import check_claims, narrate  # noqa: E402

_OUT = _ROOT / "eval" / "explainability"
REGIONS = ("bab_el_mandeb", "hormuz", "malacca", "panama")
_DECIMAL = re.compile(r"(?<![\w.])[+-]?\d\.\d\d\d?\b")


def mutation_detection(text: str, aer: dict) -> tuple[int, int]:
    """(rejected, tried) for single-number corruptions of ``text``."""
    tried = rejected = 0
    for match in list(_DECIMAL.finditer(text))[:8]:
        value = match.group(0)
        digits = len(value.split(".")[1])
        corrupt = text[:match.start()] + f"{float(value) + 0.07:.{digits}f}" + text[match.end():]
        tried += 1
        rejected += check_claims(corrupt, aer)["faithfulness"] < 1.0
    return rejected, tried


def main() -> int:
    rows = []
    for region in REGIONS:
        path = _OUT / "aer" / f"{region}.jsonl"
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        by_episode: dict[str, list[float]] = defaultdict(list)
        rejected = tried = 0
        for rec in records:
            result = narrate(rec, "decimal")
            rec["narrative"] = {"text": result["text"], "claims": result["claims"],
                                "faithfulness": result["faithfulness"], "style": "decimal",
                                "source": result["source"]}
            by_episode[rec["episode_id"]].append(result["faithfulness"])
            r, t = mutation_detection(result["text"], rec)
            rejected, tried = rejected + r, tried + t
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        mean, lo, hi = episode_bootstrap([np.mean(v) for v in by_episode.values()])
        base = {"data_origin": "real_eval", "path": "D", "weight_mode": "hand_tuned", "tier": 5,
                "variant": "with_duration", "region": region, "n_episodes": len(by_episode),
                "n_alert_days": len(records), "metric_id": "E9", "method": "exact"}
        rows.append({**base, "metric": "narrative faithfulness (supported claims / claims)",
                     "level": "episode", "value": mean, "ci_lo": lo, "ci_hi": hi,
                     "n_days_used": len(records),
                     "note": "deterministic template; 1.0 by construction"})
        rows.append({**base, "metric": "claim-checker sensitivity (single-number corruptions rejected)",
                     "level": "day", "value": rejected / tried if tried else np.nan,
                     "ci_lo": np.nan, "ci_hi": np.nan, "n_days_used": tried,
                     "note": "each 2-3 decimal number shifted by +0.07; up to 8 per narrative"})
        print(f"{region}: faithfulness {mean:.3f}, checker sensitivity {rejected}/{tried}")

    scorecard = pd.read_csv(_OUT / "scorecard.csv")
    scorecard = scorecard[scorecard.metric_id != "E9"]
    pd.concat([scorecard, pd.DataFrame(rows)], ignore_index=True).to_csv(
        _OUT / "scorecard.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
