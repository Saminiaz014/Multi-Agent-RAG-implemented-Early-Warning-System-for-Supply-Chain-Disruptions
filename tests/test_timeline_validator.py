"""Unit tests for TimelineValidator and its Orchestrator wiring."""

import pandas as pd
import pytest

from src.orchestrator import Orchestrator
from src.utils.timeline_validator import TimelineValidator


@pytest.fixture()
def validator() -> TimelineValidator:
    return TimelineValidator("2019-01-01")


def _mixed_dates_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2015-01-15",  # before min_date
                "2019-01-01",  # exactly min_date
                "2020-06-30",  # after min_date
                "2026-08-09",  # today
            ],
            "value": [1, 2, 3, 4],
        }
    )


def test_filters_rows_before_min_date(validator: TimelineValidator) -> None:
    df_filtered, audit = validator.validate_dataframe(_mixed_dates_df(), "TestConnector")

    assert len(df_filtered) == 3
    assert audit["rows_before"] == 4
    assert audit["rows_after"] == 3
    assert audit["rows_dropped"] == 1
    assert audit["pct_dropped"] == 25.0
    assert audit["status"] == "ok"
    assert audit["date_range_after"] == ("2019-01-01", "2026-08-09")


def test_boundary_date_is_inclusive(validator: TimelineValidator) -> None:
    df_filtered, _ = validator.validate_dataframe(_mixed_dates_df(), "TestConnector")
    assert "2019-01-01" in df_filtered["timestamp"].astype(str).tolist()


def test_does_not_mutate_input_dataframe(validator: TimelineValidator) -> None:
    original = _mixed_dates_df()
    original_dtype = original["timestamp"].dtype
    validator.validate_dataframe(original, "TestConnector")
    assert original["timestamp"].dtype == original_dtype
    assert len(original) == 4


def test_warning_status_when_majority_dropped(validator: TimelineValidator) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2010-01-01", "2011-01-01", "2012-01-01", "2020-01-01"],
            "value": [1, 2, 3, 4],
        }
    )
    _, audit = validator.validate_dataframe(df, "TestConnector")
    assert audit["status"] == "warning"
    assert audit["pct_dropped"] == 75.0


def test_empty_dataframe_short_circuits(validator: TimelineValidator) -> None:
    df_filtered, audit = validator.validate_dataframe(pd.DataFrame(), "TestConnector")
    assert df_filtered.empty
    assert audit["status"] == "empty"


def test_missing_date_column_reports_error(validator: TimelineValidator) -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    _, audit = validator.validate_dataframe(df, "TestConnector")
    assert audit["status"] == "error"
    assert "error" in audit


def test_validate_date_string_rejects_pre_floor_date(validator: TimelineValidator) -> None:
    is_valid, _ = validator.validate_date_string("2015-06-30", "TestConnector")
    assert is_valid is False


def test_validate_date_string_accepts_post_floor_date(validator: TimelineValidator) -> None:
    is_valid, _ = validator.validate_date_string("2020-06-30", "TestConnector")
    assert is_valid is True


def test_audit_log_summary_handles_every_status_branch(validator: TimelineValidator) -> None:
    _, ok_audit = validator.validate_dataframe(_mixed_dates_df(), "ok_connector")
    _, empty_audit = validator.validate_dataframe(pd.DataFrame(), "empty_connector")
    _, error_audit = validator.validate_dataframe(
        pd.DataFrame({"value": [1]}), "error_connector"
    )

    summary = validator.audit_log_summary([ok_audit, empty_audit, error_audit])

    assert "ok_connector" in summary
    assert "empty_connector" in summary
    assert "error_connector" in summary


def test_orchestrator_reads_min_date_from_config() -> None:
    config = {"ingestion": {}, "agents": {}, "global": {"min_date": "2020-01-01"}}
    orch = Orchestrator(config)
    assert orch.min_date == "2020-01-01"
    assert orch.timeline_validator.min_date == pd.Timestamp("2020-01-01", tz="UTC")


def test_orchestrator_defaults_min_date_when_global_section_absent() -> None:
    config = {"ingestion": {}, "agents": {}}
    orch = Orchestrator(config)
    assert orch.min_date == "2019-01-01"


def test_orchestrator_ingest_populates_timeline_audit() -> None:
    config = {"ingestion": {}, "agents": {}, "global": {"min_date": "2019-01-01"}}
    orch = Orchestrator(config)
    orch.ingest()

    connectors = {a["connector"] for a in orch._timeline_audit}
    assert connectors == {"shipping", "market"}
