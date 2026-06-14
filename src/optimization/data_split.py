"""Train / validation / test split manager for weight optimization.

Synthetic supply-chain signals are *temporal* — each connector emits one
row per day with three fixed disruption scenarios injected at the same day
positions (days 60-74, 150-170, 280-290). You therefore cannot build an
honest train/val/test split by shuffling rows: that would leak the shape of
a disruption across splits and destroy the time ordering every agent's
rolling-window logic depends on.

Instead this manager generates three *independent realisations* of the same
world by re-seeding every connector. The disruption structure is identical
across splits (same scenarios, same days, so the task is the same), but the
day-to-day baseline noise is drawn from a different random stream
(``seed=42`` train, ``43`` validation, ``44`` test). A weight set that scores
well on the validation realisation but was tuned on the train realisation has
demonstrably generalised across noise rather than memorised one sample path —
which is exactly the property the optimizer is selecting for.

``is_disruption`` is carried through purely as an evaluation label and is
**never** exposed to an agent as an input feature.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion import (
    DisasterConnector,
    GeopoliticalConnector,
    MarketConnector,
    NewsConnector,
    RoutingConnector,
    ShippingConnector,
)

logger = logging.getLogger(__name__)

# Connector registry: split-key name → (class, constructor kwargs). The four
# event/domain connectors select synthetic mode via ``data_mode``; shipping
# and market via ``source_mode``.
_CONNECTORS: dict[str, tuple[type, dict]] = {
    "shipping": (ShippingConnector, {"source_mode": "synthetic"}),
    "market": (MarketConnector, {"source_mode": "synthetic"}),
    "geopolitical": (GeopoliticalConnector, {"config": {"data_mode": "synthetic"}}),
    "natural_disaster": (DisasterConnector, {"config": {"data_mode": "synthetic"}}),
    "routing": (RoutingConnector, {"config": {"data_mode": "synthetic"}}),
    "news_sentiment": (NewsConnector, {"config": {"data_mode": "synthetic"}}),
}

def _safe_print(text: str) -> None:
    """Print box-drawing summaries without exploding on cp1252 terminals."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


_SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
_DEFAULT_SEEDS: dict[str, int] = {"train": 42, "validation": 43, "test": 44}
_DEFAULT_DAYS: int = 365


class DataSplitManager:
    """Generate independent train/val/test realisations of the synthetic world.

    Args:
        config: Full application config. ``optimization.seeds`` overrides the
            default per-split seeds; ``optimization.days`` overrides the
            per-split row count.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        opt_cfg = self.config.get("optimization", {}) or {}
        seeds_cfg = opt_cfg.get("seeds", {}) or {}
        self.seeds: dict[str, int] = {
            "train": int(seeds_cfg.get("train", _DEFAULT_SEEDS["train"])),
            "validation": int(seeds_cfg.get("validation", _DEFAULT_SEEDS["validation"])),
            "test": int(seeds_cfg.get("test", _DEFAULT_SEEDS["test"])),
        }
        self.days: int = int(opt_cfg.get("days", _DEFAULT_DAYS))
        self._splits: dict[str, dict[str, pd.DataFrame]] | None = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_splits(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Generate all three splits across all six connectors.

        Returns:
            ``{split_name: {connector_name: DataFrame}}`` — each DataFrame is
            ``self.days`` rows with the same injected disruption windows and a
            split-specific noise realisation.
        """
        splits: dict[str, dict[str, pd.DataFrame]] = {}
        for split_name in _SPLIT_NAMES:
            seed = self.seeds[split_name]
            split: dict[str, pd.DataFrame] = {}
            for conn_name, (cls, kwargs) in _CONNECTORS.items():
                connector = cls(**kwargs)
                frame = connector.generate_dataset(days=self.days, seed=seed)
                split[conn_name] = frame.reset_index(drop=True)
            splits[split_name] = split
            logger.info(
                "[DataSplitManager] generated '%s' split (seed=%d) over %d "
                "connectors",
                split_name,
                seed,
                len(split),
            )
        self._splits = splits
        return splits

    def get_splits(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Return cached splits, generating them on first call."""
        if self._splits is None:
            self.generate_splits()
        assert self._splits is not None
        return self._splits

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    def get_ground_truth(self, split_name: str) -> pd.Series:
        """Return the shipping ``is_disruption`` label for a split.

        This is the single source of evaluation truth across the pipeline.
        It is deliberately taken from the shipping connector (the primary
        physical-flow signal) and must never be fed to an agent as input.

        Args:
            split_name: ``"train"`` / ``"validation"`` / ``"test"``.

        Returns:
            Boolean Series indexed by the shipping ``timestamp`` column.
        """
        splits = self.get_splits()
        if split_name not in splits:
            raise KeyError(f"Unknown split '{split_name}'. Expected {_SPLIT_NAMES}.")
        shipping = splits[split_name]["shipping"]
        gt = shipping["is_disruption"].astype(bool)
        gt.index = pd.to_datetime(shipping["timestamp"])
        gt.name = "is_disruption"
        return gt

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_splits(self) -> dict[str, Any]:
        """Sanity-check the generated splits and print a summary table.

        Confirms (a) every split/connector frame has ``self.days`` rows,
        (b) the disruption-day count matches across splits (same structure),
        and (c) the normal-day ``vessel_count`` series is genuinely
        decorrelated across splits (Pearson |r| < 0.5), i.e. the noise really
        does differ.

        Returns:
            Dict of collected check results (also printed as a table).
        """
        splits = self.get_splits()
        report: dict[str, Any] = {"rows_ok": True, "disruption_ok": True, "noise_ok": True}

        # (a) row counts
        for split_name, frames in splits.items():
            for conn_name, frame in frames.items():
                if len(frame) != self.days:
                    report["rows_ok"] = False
                    logger.warning(
                        "[DataSplitManager] %s/%s has %d rows (expected %d)",
                        split_name, conn_name, len(frame), self.days,
                    )

        # (b) disruption-day counts (shipping ground truth)
        disruption_counts = {
            s: int(self.get_ground_truth(s).sum()) for s in _SPLIT_NAMES
        }
        report["disruption_counts"] = disruption_counts
        if len(set(disruption_counts.values())) != 1:
            report["disruption_ok"] = False
            logger.warning(
                "[DataSplitManager] disruption-day counts differ across splits: %s",
                disruption_counts,
            )

        # (c) noise decorrelation on normal days (train vs validation)
        train_ship = splits["train"]["shipping"]
        val_ship = splits["validation"]["shipping"]
        normal_mask = ~train_ship["is_disruption"].astype(bool).to_numpy()
        train_vc = train_ship.loc[normal_mask, "vessel_count"].to_numpy(dtype=float)
        val_vc = val_ship.loc[normal_mask, "vessel_count"].to_numpy(dtype=float)
        if len(train_vc) > 2 and np.std(train_vc) > 0 and np.std(val_vc) > 0:
            corr = float(np.corrcoef(train_vc, val_vc)[0, 1])
        else:  # pragma: no cover - degenerate
            corr = 0.0
        report["normal_day_correlation"] = corr
        if abs(corr) >= 0.5:
            report["noise_ok"] = False
            logger.warning(
                "[DataSplitManager] normal-day vessel_count correlation %.3f "
                ">= 0.5 — splits may not be independent.",
                corr,
            )

        self._print_summary(splits, disruption_counts, corr, report)
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(
        splits: dict[str, dict[str, pd.DataFrame]],
        disruption_counts: dict[str, int],
        corr: float,
        report: dict[str, Any],
    ) -> None:
        lines = [
            "",
            "┌─────────────── DATA SPLIT SUMMARY ───────────────┐",
            f"│ {'Split':<12}{'Rows':>8}{'Disruption days':>20}     │",
            "├──────────────────────────────────────────────────┤",
        ]
        for split_name in _SPLIT_NAMES:
            rows = len(splits[split_name]["shipping"])
            ndis = disruption_counts[split_name]
            lines.append(f"│ {split_name:<12}{rows:>8}{ndis:>20}     │")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append(
            f"│ Normal-day corr (train vs val): {corr:>+7.3f}          │"
        )
        status = "PASS" if all(
            report[k] for k in ("rows_ok", "disruption_ok", "noise_ok")
        ) else "CHECK"
        lines.append(f"│ Independence / structure checks: {status:<12}     │")
        lines.append("└──────────────────────────────────────────────────┘")
        _safe_print("\n".join(lines))
