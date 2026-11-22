"""The fixed evaluation window, shared by every script that splits train/test.

``config/settings.yaml`` ``evaluation`` holds one split date and one end date
for all regions. The method comparison, the real-data explanation surrogate
and the live risk series all read them here, so they cannot drift apart.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.core.config_manager import load_base_config

logger = logging.getLogger(__name__)


def eval_window() -> dict[str, pd.Timestamp]:
    """The fixed split and end dates from settings.yaml ``evaluation``."""
    block = load_base_config()["evaluation"]
    return {"split_date": pd.Timestamp(block["split_date"]),
            "end_date": pd.Timestamp(block["end_date"])}


def bootstrap_config() -> dict:
    """Keyword arguments for :func:`~src.evaluation.block_bootstrap.block_bootstrap_ci`."""
    block = load_base_config()["evaluation"]["bootstrap"]
    return {"block": int(block["block_days"]), "n_resamples": int(block["n_resamples"]),
            "ci": float(block["ci"]), "seed": int(block["seed"])}


def sensitivity_split_dates() -> list[pd.Timestamp]:
    """Alternative split dates for the split-date sensitivity table."""
    block = load_base_config()["evaluation"]
    return [pd.Timestamp(d) for d in block["sensitivity_split_dates"]]


def split_by_date(
    frame: pd.DataFrame, split_date: str | pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train before ``split_date``; test from it through ``end_date``.

    The frame is sorted by ``timestamp`` first, so shuffled input splits the
    same way. A frame ending before ``end_date`` is accepted with a warning,
    since it simply has a shorter test window.

    Args:
        split_date: Overrides settings.yaml's ``split_date``, for the
            sensitivity table only. ``end_date`` always comes from settings.

    Raises:
        ValueError: If the frame has no days on one side of the split.
    """
    window = eval_window()
    if split_date is not None:
        window["split_date"] = pd.Timestamp(split_date)
    frame = frame.sort_values("timestamp", kind="stable")
    ts = pd.to_datetime(frame["timestamp"])
    train = frame[ts < window["split_date"]]
    test = frame[(ts >= window["split_date"]) & (ts <= window["end_date"])]
    if train.empty or test.empty:
        raise ValueError(
            f"frame [{ts.min().date()} .. {ts.max().date()}] does not span the "
            f"split at {window['split_date'].date()}"
        )
    if ts.max() < window["end_date"]:
        logger.warning(
            "frame ends %s, before end_date %s: test window is shorter",
            ts.max().date(), window["end_date"].date(),
        )
    return train.copy(), test.copy()
