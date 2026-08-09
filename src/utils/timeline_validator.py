"""Timeline validator — enforces the global minimum date floor.

Provides validation, filtering, and per-connector audit logging so every
DataFrame the pipeline ingests is guaranteed to fall within the
reproducibility window configured under ``config["global"]["min_date"]``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY_DATE_RANGE = (None, None)


class TimelineValidator:
    """Validates and enforces a global minimum date across all connectors."""

    def __init__(self, min_date: str = "2019-01-01") -> None:
        """
        Args:
            min_date: ISO format date string (e.g. "2019-01-01").
        """
        self.min_date = pd.Timestamp(min_date, tz="UTC")
        logger.info("Timeline validator initialized: min_date=%s", self.min_date.date())

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        connector_name: str,
        date_column: str = "timestamp",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate and filter a DataFrame against the minimum date floor.

        Args:
            df: Input DataFrame with ``date_column``. Never mutated — a
                filtered copy is returned.
            connector_name: Name of the connector (for logging/audit).
            date_column: Name of the timestamp column.

        Returns:
            ``(filtered_df, audit_dict)``. ``audit_dict`` always carries the
            same key set (``connector``, ``status``, ``rows_before``,
            ``rows_after``, ``rows_dropped``, ``pct_dropped``,
            ``date_range_before``, ``date_range_after``,
            ``min_date_enforced``) regardless of branch, so
            :meth:`audit_log_summary` can format every entry uniformly.
        """
        base_audit: dict[str, Any] = {
            "connector": connector_name,
            "rows_before": 0,
            "rows_after": 0,
            "rows_dropped": 0,
            "pct_dropped": 0.0,
            "date_range_before": _EMPTY_DATE_RANGE,
            "date_range_after": _EMPTY_DATE_RANGE,
            "min_date_enforced": str(self.min_date.date()),
        }

        if df.empty:
            return df, {**base_audit, "status": "empty", "note": "Input dataframe is empty"}

        if date_column not in df.columns:
            return df, {
                **base_audit,
                "status": "error",
                "rows_before": len(df),
                "error": f"Column '{date_column}' not found in {connector_name} output",
            }

        # Parsed as a side comparison series — the returned frame keeps the
        # caller's original dtype for date_column (tz-naive or tz-aware) so
        # downstream merges on that column never see a dtype it didn't
        # already have (Orchestrator.ingest merges shipping/market on
        # tz-naive 'timestamp').
        parsed = pd.to_datetime(df[date_column], utc=True)

        rows_before = len(df)
        min_before = parsed.min()
        max_before = parsed.max()

        df_filtered = df[parsed >= self.min_date].copy()
        parsed_after = parsed[parsed >= self.min_date]

        rows_after = len(df_filtered)
        rows_dropped = rows_before - rows_after
        pct_dropped = (rows_dropped / rows_before * 100) if rows_before > 0 else 0.0

        min_after = parsed_after.min() if not df_filtered.empty else None
        max_after = parsed_after.max() if not df_filtered.empty else None

        status = "warning" if pct_dropped > 50 else "ok"

        audit_dict: dict[str, Any] = {
            "connector": connector_name,
            "status": status,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_dropped": rows_dropped,
            "pct_dropped": round(pct_dropped, 2),
            "date_range_before": (str(min_before.date()), str(max_before.date())),
            "date_range_after": (
                str(min_after.date()) if min_after is not None else None,
                str(max_after.date()) if max_after is not None else None,
            ),
            "min_date_enforced": str(self.min_date.date()),
        }

        if status == "warning":
            logger.warning(
                "%s: %.1f%% of data dropped (>50%% threshold). Before: %s to %s, After: %s to %s",
                connector_name,
                pct_dropped,
                min_before.date(),
                max_before.date(),
                min_after.date() if min_after is not None else "empty",
                max_after.date() if max_after is not None else "empty",
            )
        else:
            logger.info(
                "%s: %d rows dropped (pre-%s). Retained: %d rows (%s to %s)",
                connector_name,
                rows_dropped,
                self.min_date.date(),
                rows_after,
                min_after.date() if min_after is not None else "empty",
                max_after.date() if max_after is not None else "empty",
            )

        return df_filtered, audit_dict

    def validate_date_string(self, date_str: str, connector_name: str) -> tuple[bool, str]:
        """Validate a single date string against the minimum date floor.

        Args:
            date_str: ISO format date (e.g. "2015-06-30").
            connector_name: Name of the connector (for logging).

        Returns:
            ``(is_valid, message)``.
        """
        try:
            date = pd.Timestamp(date_str, tz="UTC")
        except Exception as exc:
            return False, f"Invalid date format: {date_str} ({exc})"

        if date < self.min_date:
            msg = f"{connector_name}: date {date.date()} is before minimum {self.min_date.date()}"
            logger.warning(msg)
            return False, msg

        return True, f"Date {date.date()} is valid (>= {self.min_date.date()})"

    def audit_log_summary(self, audit_dicts: list[dict[str, Any]]) -> str:
        """Generate a summary audit report from multiple validations.

        Args:
            audit_dicts: List of ``audit_dict`` outputs from
                :meth:`validate_dataframe`.

        Returns:
            Formatted summary string for logging.
        """
        lines = [
            "",
            "=" * 70,
            "TIMELINE VALIDATION AUDIT REPORT",
            f"Min date enforced: {self.min_date.date()}",
            "=" * 70,
        ]

        for audit in audit_dicts:
            connector = audit.get("connector", "unknown")
            status = audit.get("status", "unknown")
            rows_after = audit.get("rows_after", 0)
            pct_dropped = audit.get("pct_dropped", 0.0)
            date_range = audit.get("date_range_after", _EMPTY_DATE_RANGE)

            status_icon = "OK" if status == "ok" else "!!" if status == "warning" else "XX"
            lines.append(
                f"[{status_icon}] {connector:20s} | {rows_after:6d} rows retained | "
                f"{pct_dropped:5.1f}% dropped | {date_range[0]} to {date_range[1]}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)
