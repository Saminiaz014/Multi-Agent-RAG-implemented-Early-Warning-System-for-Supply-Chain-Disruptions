"""The live series can reuse cached PortWatch columns instead of calling PortWatch again.

IMF PortWatch rate-limits (HTTP 429, a per-minute quota shared with other clients),
and every other domain is cheap to refresh. ``build_region(shipping=...)`` takes a
shipping frame in place of the fetch, and ``build_live_risk_series.py
--shipping-from-cache`` builds that frame from the previous ``data/live`` frame and
records, in the manifest, that it did.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bed = _load("build_eval_dataset_cache_test", "scripts/build_eval_dataset.py")
live = _load("build_live_risk_series_cache_test", "scripts/build_live_risk_series.py")

_DAYS = 400   # long enough for the label's 365-day baseline to exist


def _shipping(days: int = _DAYS) -> pd.DataFrame:
    count = np.full(days, 100.0)
    count[300:340] = 40.0                      # a sustained level shift: a labelled run
    return pd.DataFrame({
        "timestamp": pd.date_range("2019-01-01", periods=days, freq="D"),
        "vessel_count": count, "tanker_count": 10.0,
        "avg_delay_hours": np.linspace(1.0, 2.0, days), "congestion_index": 0.5,
        "is_disruption": np.arange(days) % 97 == 0})


def _patch(monkeypatch, module, fetched: list[str], frames: dict | None = None) -> None:
    frames = frames or {}

    def fetch(domain, config):
        fetched.append(domain)
        return frames.get(domain)

    monkeypatch.setattr(module, "load_config_for_region", lambda region: {})
    monkeypatch.setattr(module, "_fetch_domain", fetch)


def _fetched_frame(monkeypatch) -> tuple[pd.DataFrame, dict]:
    _patch(monkeypatch, bed, [], {"shipping": _shipping()})
    return bed.build_region("hormuz", drop_incomplete=False)


# ------------------------------------------------------------------ the pure functions
def test_shipping_from_frame_recovers_what_the_connector_returned(monkeypatch) -> None:
    merged, _ = _fetched_frame(monkeypatch)
    back = bed.shipping_from_frame(merged)
    pd.testing.assert_frame_equal(back, _shipping()[back.columns.tolist()], check_dtype=False)
    assert back["is_disruption"].dtype == bool
    assert back.columns[0] == "timestamp" and "y_true" not in back.columns


def test_shipping_from_frame_parses_text_timestamps_and_orders_the_days(monkeypatch) -> None:
    merged, _ = _fetched_frame(monkeypatch)
    scrambled = merged.sample(frac=1.0, random_state=0).copy()
    scrambled["timestamp"] = scrambled["timestamp"].astype(str)        # as read without parse_dates
    back = bed.shipping_from_frame(scrambled)
    assert pd.api.types.is_datetime64_any_dtype(back["timestamp"])
    pd.testing.assert_frame_equal(back, bed.shipping_from_frame(merged))
    assert back["timestamp"].is_monotonic_increasing


def test_shipping_from_frame_drops_days_with_no_vessel_count(monkeypatch) -> None:
    merged, _ = _fetched_frame(monkeypatch)
    merged.loc[5, "shipping__vessel_count"] = np.nan
    assert len(bed.shipping_from_frame(merged)) == len(merged) - 1


@pytest.mark.parametrize("drop", ["shipping__vessel_count", "y_true_shock"])
def test_shipping_from_frame_refuses_a_frame_it_did_not_write(monkeypatch, drop) -> None:
    merged, _ = _fetched_frame(monkeypatch)
    with pytest.raises(ValueError, match=drop):
        bed.shipping_from_frame(merged.drop(columns=[drop]))


# ------------------------------------------------------------------ build_region
def test_a_cached_shipping_frame_is_never_fetched(monkeypatch) -> None:
    fetched: list[str] = []
    _patch(monkeypatch, bed, fetched)
    source = {"source": "cache"}
    merged, manifest = bed.build_region("hormuz", drop_incomplete=False,
                                        shipping=_shipping(), shipping_source=source)
    assert "shipping" not in fetched
    assert manifest["shipping_source"] == source
    assert manifest["fetch_seconds"]["shipping"] == 0.0
    assert len(merged) == _DAYS and merged["y_true"].any()


def test_cached_and_fetched_shipping_build_the_same_frame(monkeypatch) -> None:
    fetched_frame, fetched_manifest = _fetched_frame(monkeypatch)
    cached = bed.shipping_from_frame(fetched_frame)
    _patch(monkeypatch, bed, [])
    cached_frame, cached_manifest = bed.build_region(
        "hormuz", drop_incomplete=False, shipping=cached, shipping_source={"source": "cache"})
    pd.testing.assert_frame_equal(cached_frame, fetched_frame)
    for key in ("rows", "positive_days", "positive_days_shock", "date_range", "feature_columns"):
        assert cached_manifest[key] == fetched_manifest[key], key


def test_a_fetched_build_records_no_shipping_source(monkeypatch) -> None:
    _, manifest = _fetched_frame(monkeypatch)
    assert "shipping_source" not in manifest


def test_a_shipping_source_without_a_shipping_frame_is_refused(monkeypatch) -> None:
    _patch(monkeypatch, bed, [], {"shipping": _shipping()})
    with pytest.raises(ValueError, match="shipping_source"):
        bed.build_region("hormuz", shipping_source={"source": "cache"})


@pytest.mark.parametrize("drop", ["vessel_count", "is_disruption", "timestamp"])
def test_a_shipping_frame_missing_a_connector_column_is_refused(monkeypatch, drop) -> None:
    _patch(monkeypatch, bed, [])
    with pytest.raises(ValueError, match=drop):
        bed.build_region("hormuz", shipping=_shipping().drop(columns=[drop]),
                         shipping_source={"source": "cache"})


# ------------------------------------------------------------------ the live builder
@pytest.fixture
def cached_live(tmp_path, monkeypatch):
    """A ``data/live`` directory holding a previous frame and its manifest."""
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    _patch(monkeypatch, bed, [], {"shipping": _shipping()})
    frame, manifest = bed.build_region("hormuz", drop_incomplete=False)
    frame.to_csv(tmp_path / "hormuz.csv", index=False)
    (tmp_path / "hormuz.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, frame


def test_live_fetch_from_cache_never_calls_portwatch_and_records_the_cache(
        cached_live, monkeypatch) -> None:
    tmp_path, previous = cached_live
    fetched: list[str] = []
    _patch(monkeypatch, live.build_eval_dataset, fetched)
    digest = hashlib.sha256((tmp_path / "hormuz.csv").read_bytes()).hexdigest()

    frame = live.fetch("hormuz", shipping_from_cache=True)

    assert "shipping" not in fetched and fetched          # the other domains were fetched
    assert frame["shipping__vessel_count"].tolist() == previous["shipping__vessel_count"].tolist()
    manifest = json.loads((tmp_path / "hormuz.manifest.json").read_text(encoding="utf-8"))
    source = manifest["shipping_source"]
    assert source["cache_sha256"] == digest
    assert source["cache"] == "data/live/hormuz.csv"
    assert source["cache_rows"] == len(previous)
    assert source["cache_last_day"] == str(previous["timestamp"].max())[:10]
    assert "not re-fetched" in source["source"]
    assert "shipping" in manifest["domains_present"]


def test_live_fetch_from_cache_keeps_the_provenance_of_a_cache_that_was_itself_cached(
        cached_live, monkeypatch) -> None:
    tmp_path, _ = cached_live
    _patch(monkeypatch, live.build_eval_dataset, [])
    live.fetch("hormuz", shipping_from_cache=True)
    live.fetch("hormuz", shipping_from_cache=True)         # now the cache is a cached frame
    source = json.loads((tmp_path / "hormuz.manifest.json").read_text(encoding="utf-8"))["shipping_source"]
    assert "inherited" in source and "cache_sha256" in source["inherited"]


def test_live_fetch_from_cache_without_a_cache_says_so_and_does_not_fetch(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    fetched: list[str] = []
    _patch(monkeypatch, live.build_eval_dataset, fetched)
    with pytest.raises(FileNotFoundError, match="--shipping-from-cache"):
        live.fetch("hormuz", shipping_from_cache=True)
    assert not fetched


def test_live_fetch_without_the_flag_still_fetches_shipping(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    fetched: list[str] = []
    _patch(monkeypatch, live.build_eval_dataset, fetched, {"shipping": _shipping()})
    live.fetch("hormuz")
    assert "shipping" in fetched
    manifest = json.loads((tmp_path / "hormuz.manifest.json").read_text(encoding="utf-8"))
    assert "shipping_source" not in manifest


def test_the_command_line_passes_the_flag_through(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(live, "fetch", lambda region, shipping_from_cache=False:
                        calls.append((region, shipping_from_cache)) or pd.DataFrame())
    monkeypatch.setattr(live, "score", lambda region, frame: None)
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    assert live.run(["--region", "panama", "--shipping-from-cache"]) == 0
    assert live.run(["--region", "panama"]) == 0
    assert calls == [("panama", True), ("panama", False)]


def test_the_flag_cannot_be_combined_with_score_only(capsys) -> None:
    with pytest.raises(SystemExit):
        live.run(["--score-only", "--shipping-from-cache"])
    assert "not allowed with argument" in capsys.readouterr().err


# ------------------------------------------------------------------ review fixes
def test_a_shipping_frame_without_a_source_is_refused(monkeypatch) -> None:
    _patch(monkeypatch, bed, [])
    with pytest.raises(ValueError, match="shipping_source"):
        bed.build_region("hormuz", shipping=_shipping())


def test_the_cached_shipping_is_bit_identical_after_the_csv_round_trip(cached_live) -> None:
    _, previous = cached_live
    cached, _ = live.cached_shipping("hormuz")
    pd.testing.assert_frame_equal(cached, bed.shipping_from_frame(previous), check_exact=True)


def test_a_failed_write_leaves_the_previous_live_frame_and_manifest_intact(
        cached_live, monkeypatch) -> None:
    tmp_path, _ = cached_live
    frame_before = (tmp_path / "hormuz.csv").read_bytes()
    manifest_before = (tmp_path / "hormuz.manifest.json").read_bytes()
    _patch(monkeypatch, live.build_eval_dataset, [])

    def broken_to_csv(self, path, *args, **kwargs):
        Path(path).write_bytes(b"timestamp,shipping__vessel_count\n2019-01-01,1")   # cut short
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", broken_to_csv)
    with pytest.raises(OSError, match="disk full"):
        live.fetch("hormuz", shipping_from_cache=True)
    assert (tmp_path / "hormuz.csv").read_bytes() == frame_before
    assert (tmp_path / "hormuz.manifest.json").read_bytes() == manifest_before
    assert not list(tmp_path.glob("*.tmp"))                # no half-written file left behind
