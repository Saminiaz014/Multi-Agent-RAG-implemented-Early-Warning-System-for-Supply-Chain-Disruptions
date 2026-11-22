"""NewsConnector ``gdelt_events`` mode and its wiring into the eval dataset.

The mode reads the per-region daily CSVs written by
``scripts/build_gdelt_news_csv.py`` (GDELT 1.0 events, frozen box filter).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.news_agent import NewsAgent
from src.ingestion.news_connector import NewsConnector

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_eval_dataset", _ROOT / "scripts" / "build_eval_dataset.py")
bed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bed)

_HEADER = ("date,news_volume,n_events,news_tone,goldstein_mean,n_material_conflict,"
           "total_events\n")


def _write_news(tmp: Path, region: str, rows: list[str], *,
                missing: list[str] | None = None, failed: list[str] | None = None,
                pending: list[str] | None = None, bad_hash: bool = False) -> Path:
    csv = tmp / f"{region}.csv"
    csv.write_text(_HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    (tmp / f"{region}.manifest.json").write_text(json.dumps({
        "filter_version": "v2", "filter_sha256": "f" * 64,
        "sha256": "0" * 64 if bad_hash else digest,
        "missing_days": missing or [], "failed_days": failed or [],
        "pending_days": pending or [], "region": region,
    }), encoding="utf-8")
    return tmp


def _connector(tmp: Path, region: str = "hormuz", **extra) -> NewsConnector:
    return NewsConnector(config={"data_mode": "gdelt_events", "region": region,
                                 "gdelt_events_dir": str(tmp), **extra})


_ROWS = [
    "2019-01-01,0,0,,,0,100000",          # peace time
    "2019-01-02,10,3,-4.0,-2.0,1,100000",
    "2019-01-03,4,2,-15.0,-8.0,2,50000",  # tone beyond the +-10 scale; half-size day
    "2019-01-05,2,1,3.0,1.0,0,200000",    # 2019-01-04 missing (GDELT 404)
]


# ------------------------------------------------------------- connector
def test_maps_export_csv_to_news_agent_schema(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS)).fetch()
    assert list(df.columns) == ["timestamp", "sentiment_score", "sentiment_magnitude",
                                "article_volume", "recency_weighted_score", "news_tone"]
    assert df["sentiment_score"].tolist() == [0.0, -0.4, -1.0, 0.3]
    assert df["sentiment_magnitude"].tolist() == [0.0, 0.4, 1.0, 0.3]


def test_volume_is_articles_per_100k_gdelt_events(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS)).fetch()
    # 10/100k, 4/50k, 2/200k events -> 10, 8, 1 per 100k
    assert df["article_volume"].tolist() == [0.0, 10.0, 8.0, 1.0]


def test_zero_volume_day_is_a_neutral_row_not_missing(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS)).fetch()
    first = df.iloc[0]
    assert first["timestamp"] == pd.Timestamp("2019-01-01")
    assert np.isnan(first["news_tone"])                     # tone undefined: NaN, not 0
    assert first["sentiment_score"] == 0.0                  # agent input: no signal
    assert first["article_volume"] == 0.0
    assert df.loc[1:, "news_tone"].tolist() == [-4.0, -15.0, 3.0]
    assert not df.drop(columns="news_tone").isna().any().any()


def test_missing_export_days_are_not_invented(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    assert pd.Timestamp("2019-01-04") not in set(df["timestamp"])
    assert df.attrs["news_source"]["missing_days"] == ["2019-01-04"]


def test_no_source_consensus_column(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS)).fetch()
    assert "source_consensus" not in df.columns   # not measurable; agent renormalises


def test_recency_weighting_uses_config_block(tmp_path) -> None:
    flat = {"fresh_days": 1, "fresh_weight": 1.0, "recent_days": 1,
            "recent_weight": 0.0, "old_weight": 0.0}
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS),
                    api={"recency_weights": flat}).fetch()
    np.testing.assert_allclose(df["recency_weighted_score"], df["sentiment_score"])


def test_provenance_travels_in_attrs(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS)).fetch()
    src = df.attrs["news_source"]
    assert src["filter_version"] == "v2" and src["filter_sha256"] == "f" * 64
    assert src["csv_sha256"] == hashlib.sha256(
        (tmp_path / "hormuz.csv").read_bytes()).hexdigest()


def test_csv_that_does_not_match_its_manifest_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="manifest"):
        _connector(_write_news(tmp_path, "hormuz", _ROWS, bad_hash=True)).fetch()


def test_missing_csv_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        _connector(tmp_path, region="panama").fetch()


def test_region_is_required(tmp_path) -> None:
    with pytest.raises(ValueError, match="region"):
        NewsConnector(config={"data_mode": "gdelt_events",
                              "gdelt_events_dir": str(tmp_path)}).fetch()


def test_articles_without_tone_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="tone"):
        _connector(_write_news(tmp_path, "hormuz", ["2019-01-01,5,1,,,0,1000"])).fetch()


def test_csv_without_total_events_is_refused(tmp_path) -> None:
    csv = tmp_path / "hormuz.csv"
    csv.write_text("date,news_volume,n_events,news_tone,goldstein_mean,"
                   "n_material_conflict\n2019-01-01,0,0,,,0\n", encoding="utf-8")
    (tmp_path / "hormuz.manifest.json").write_text(json.dumps(
        {"sha256": hashlib.sha256(csv.read_bytes()).hexdigest()}), encoding="utf-8")
    with pytest.raises(ValueError, match="total_events"):
        _connector(tmp_path).fetch()


def test_failed_download_days_are_refused(tmp_path) -> None:
    # A failed download is not "GDELT never published": it must not be filled.
    with pytest.raises(ValueError, match="failed"):
        _connector(_write_news(tmp_path, "hormuz", _ROWS, failed=["2019-01-04"])).fetch()


def test_pending_days_are_allowed_only_after_the_last_row(tmp_path) -> None:
    df = _connector(_write_news(tmp_path, "hormuz", _ROWS, pending=["2019-01-06"])).fetch()
    assert df["timestamp"].max() == pd.Timestamp("2019-01-05")
    with pytest.raises(ValueError, match="pending"):
        _connector(_write_news(tmp_path, "hormuz", _ROWS, pending=["2019-01-04"])).fetch()


def test_recency_counts_calendar_days_not_rows(tmp_path) -> None:
    rows = ["2019-01-01,5,1,-8.0,0,0,1000", "2019-01-02,5,1,-8.0,0,0,1000",
            "2019-01-10,5,1,2.0,0,0,1000"]                    # 01-03..01-09 unpublished
    fresh_only = {"fresh_days": 2, "fresh_weight": 1.0, "recent_days": 2,
                  "recent_weight": 0.0, "old_weight": 0.0}
    df = _connector(_write_news(tmp_path, "hormuz", rows,
                                missing=[f"2019-01-0{d}" for d in range(3, 10)]),
                    api={"recency_weights": fresh_only}).fetch()
    # Fresh window on 01-10 is {01-09, 01-10}; 01-09 is unpublished, so only 01-10
    # counts. Counting rows would average in 01-02 (-0.8) as if it were yesterday.
    assert df["recency_weighted_score"].iloc[-1] == pytest.approx(0.2)


def test_day_with_no_gdelt_events_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="total_events"):
        _connector(_write_news(tmp_path, "hormuz", ["2019-01-01,0,0,,,0,0"])).fetch()


# ------------------------------------------------------------------ agent
def test_news_agent_scores_and_runs_on_gdelt_events_frame(tmp_path) -> None:
    # 10 quiet days, a 10-day burst of hostile coverage, 8 quiet days.
    vols = [0] * 10 + [20] * 10 + [0] * 8
    rows = [f"2019-01-{d:02d},{v},5,-6.0,-5.0,2,100000" if v
            else f"2019-01-{d:02d},0,0,,,0,100000"
            for d, v in zip(range(1, 29), vols)]
    df = _connector(_write_news(tmp_path, "hormuz", rows)).fetch()
    agent = NewsAgent()
    agent.fit(df)
    scored = agent.detect(agent.preprocess(df))
    assert scored["anomaly_score"].between(0.0, 1.0).all()
    assert scored["anomaly_score"].iloc[10:20].mean() > scored["anomaly_score"].iloc[:10].mean()
    windows = agent.run(df)                  # output() must cope without consensus
    assert windows, "the burst should validate into at least one window"
    for w in windows:
        assert 0.0 <= w["confidence"] <= 1.0
        assert w["signals"]["source_consensus"] is None
        assert isinstance(w["signals"]["article_volume"], float)
        assert w["start_timestamp"] >= "2019-01-11"


# ------------------------------------------------------- eval dataset wiring
def test_fetch_domain_reads_gdelt_events_for_news(tmp_path) -> None:
    _write_news(tmp_path, "malacca", _ROWS)
    config = {"region": {"name": "malacca"},
              "agents": {"news_sentiment": {"data_mode": "synthetic",
                                            "gdelt_events_dir": str(tmp_path)}}}
    df = bed._fetch_domain("news_sentiment", config)
    assert df is not None and df["article_volume"].tolist() == [0.0, 10.0, 8.0, 1.0]


def _cached(days: int = 5) -> tuple[pd.DataFrame, dict]:
    ts = pd.date_range("2019-01-01", periods=days, freq="D")
    frame = pd.DataFrame({"timestamp": ts, "y_true": [False] * days,
                          "shipping__vessel_count": np.arange(days, dtype=float),
                          "news_sentiment__sentiment_score": [9.0] * days})
    manifest = {"region": "hormuz", "rows": days,
                "domains_present": {"shipping": ["vessel_count"],
                                    "news_sentiment": ["sentiment_score"]},
                "domains_missing": [],
                "feature_columns": ["shipping__vessel_count",
                                    "news_sentiment__sentiment_score"]}
    return frame, manifest


def test_replace_news_changes_only_news_columns(tmp_path) -> None:
    frame, manifest = _cached()
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    out, m = bed.replace_news(frame, manifest, news)
    pd.testing.assert_frame_equal(out[["timestamp", "y_true", "shipping__vessel_count"]],
                                  frame[["timestamp", "y_true", "shipping__vessel_count"]])
    # 01-04 unpublished: masked (NaN), neither forward-filled nor interpolated.
    vol = out["news_sentiment__article_volume"]
    assert vol.isna().tolist() == [False, False, False, True, False]
    assert vol.dropna().tolist() == [0, 10, 8, 1]
    assert out.loc[3, [f"news_sentiment__{c}" for c in
                       bed._DOMAIN_FEATURES["news_sentiment"]]].isna().all()
    assert "news_sentiment__news_tone" not in out.columns     # NaN tone never reaches models
    assert m["domains_present"]["news_sentiment"] == list(bed._DOMAIN_FEATURES["news_sentiment"])
    assert m["feature_columns"] == ["shipping__vessel_count",
                                    *[f"news_sentiment__{c}" for c in
                                      bed._DOMAIN_FEATURES["news_sentiment"]]]
    assert m["news_source"]["filter_version"] == "v2"
    assert m["news_source"]["days_masked"] == ["2019-01-04"]
    assert "days_forward_filled" not in m["news_source"]
    assert m["news_source"]["gap_policy"].startswith("masked")
    assert m["rows"] == len(frame)


def test_replace_news_refuses_gaps_gdelt_did_not_report_missing(tmp_path) -> None:
    frame, manifest = _cached()
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS)).fetch()   # 01-04 not listed
    with pytest.raises(ValueError, match="not listed"):
        bed.replace_news(frame, manifest, news)


def test_replace_news_adds_news_to_a_region_that_had_none(tmp_path) -> None:
    frame, manifest = _cached()
    frame = frame.drop(columns="news_sentiment__sentiment_score")
    manifest["domains_present"].pop("news_sentiment")
    manifest["domains_missing"] = ["news_sentiment"]
    manifest["feature_columns"] = ["shipping__vessel_count"]
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    _, m = bed.replace_news(frame, manifest, news)
    assert m["domains_missing"] == [] and "news_sentiment" in m["domains_present"]


def test_replace_news_refuses_days_outside_news_coverage(tmp_path) -> None:
    frame, manifest = _cached(days=7)            # runs past the news CSV's last day
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    with pytest.raises(ValueError, match="coverage"):
        bed.replace_news(frame, manifest, news)


def _shipping(days: int) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2019-01-01", periods=days, freq="D"),
        "vessel_count": np.arange(days, dtype=float) + 50, "tanker_count": 10.0,
        "avg_delay_hours": 1.0, "congestion_index": 0.5, "is_disruption": False})


def test_build_region_attaches_news_like_replace_news(tmp_path, monkeypatch) -> None:
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    frames = {"shipping": _shipping(7), "news_sentiment": news}   # 7 days: 2 past news
    monkeypatch.setattr(bed, "load_config_for_region", lambda region: {})
    monkeypatch.setattr(bed, "_fetch_domain", lambda domain, config: frames.get(domain))
    merged, m = bed.build_region("hormuz")                  # drop_incomplete=True
    assert merged["timestamp"].max() == pd.Timestamp("2019-01-05")   # outside news: dropped
    # The masked day stays in the frame (other domains intact) with NaN news.
    assert len(merged) == 5
    vol = merged["news_sentiment__article_volume"]
    assert vol.isna().tolist() == [False, False, False, True, False]
    assert m["news_source"]["days_masked"] == ["2019-01-04"]
    assert m["rows_news_masked"] == 1
    live, _ = bed.build_region("hormuz", drop_incomplete=False)      # live series: NaN, no raise
    assert live["news_sentiment__article_volume"].iloc[5:].isna().all()


def test_refresh_news_on_disk_leaves_other_columns_byte_identical(tmp_path, monkeypatch) -> None:
    cache, news_dir = tmp_path / "eval", tmp_path / "news"
    cache.mkdir()
    news_dir.mkdir()
    _write_news(news_dir, "hormuz", _ROWS, missing=["2019-01-04"])
    awkward = [0.1 + 0.2, 1e-17, 123456.78901234567, -0.0, 2.5e300]
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2019-01-01", periods=5, freq="D"),
        "y_true": [True, False, False, True, False],
        "shipping__vessel_count": awkward,
        "news_sentiment__sentiment_score": [9.0] * 5})
    frame.to_csv(cache / "hormuz.csv", index=False)
    (cache / "hormuz.manifest.json").write_text(json.dumps(
        {"region": "hormuz", "rows": 5, "domains_present": {}, "domains_missing": [],
         "feature_columns": []}), encoding="utf-8")
    before = pd.read_csv(cache / "hormuz.csv", dtype=str)
    monkeypatch.setattr(bed, "_CACHE_DIR", cache)
    bed.refresh_news("hormuz", news_dir=news_dir)
    after = pd.read_csv(cache / "hormuz.csv", dtype=str)
    keep = ["timestamp", "y_true", "shipping__vessel_count"]
    pd.testing.assert_frame_equal(before[keep], after[keep])
    assert after["news_sentiment__article_volume"].fillna("").tolist() == [
        "0.0", "10.0", "8.0", "", "1.0"]                      # 01-04 masked


def test_refresh_news_surfaces_the_real_error(tmp_path, monkeypatch) -> None:
    cache, news_dir = tmp_path / "eval", tmp_path / "news"
    cache.mkdir()
    news_dir.mkdir()
    _write_news(news_dir, "hormuz", _ROWS, bad_hash=True)
    frame, manifest = _cached()
    frame.to_csv(cache / "hormuz.csv", index=False)
    (cache / "hormuz.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(bed, "_CACHE_DIR", cache)
    with pytest.raises(ValueError, match="manifest"):
        bed.refresh_news("hormuz", news_dir=news_dir)


def test_build_region_forward_fills_other_domains_never_interpolates(tmp_path, monkeypatch) -> None:
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    shipping = _shipping(5)
    shipping.loc[2, "congestion_index"] = np.nan            # a hole inside the series
    shipping.loc[3, "congestion_index"] = 0.9
    frames = {"shipping": shipping, "news_sentiment": news}
    monkeypatch.setattr(bed, "load_config_for_region", lambda region: {})
    monkeypatch.setattr(bed, "_fetch_domain", lambda domain, config: frames.get(domain))
    merged, m = bed.build_region("hormuz")
    # Linear interpolation would give 0.7, using the next day; forward-fill
    # carries the last known value.
    assert merged["shipping__congestion_index"].tolist()[:4] == [0.5, 0.5, 0.5, 0.9]
    assert m["gap_fill"]["cells_filled"]["shipping__congestion_index"] == 1
    assert m["gap_fill"]["policy"].startswith("forward-fill")


def test_masked_news_day_with_another_domain_missing_is_still_dropped(tmp_path, monkeypatch) -> None:
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    shipping = _shipping(5).drop(index=3)                  # no shipping row on 01-04
    frames = {"shipping": shipping, "news_sentiment": news}
    monkeypatch.setattr(bed, "load_config_for_region", lambda region: {})
    monkeypatch.setattr(bed, "_fetch_domain", lambda domain, config: frames.get(domain))
    merged, m = bed.build_region("hormuz")
    assert pd.Timestamp("2019-01-04") not in set(merged["timestamp"])
    assert m["rows_news_masked"] == 0


def test_manifest_records_how_market_values_are_dated(tmp_path, monkeypatch) -> None:
    news = _connector(_write_news(tmp_path, "hormuz", _ROWS, missing=["2019-01-04"])).fetch()
    market = pd.DataFrame({"timestamp": pd.date_range("2019-01-01", periods=5, freq="D"),
                           "brent_crude_usd": 60.0, "trade_volume_index": 0.5,
                           "freight_rate_index": 100.0})
    market.attrs["market_source"] = {"dating": "first-release date (ALFRED)"}
    frames = {"shipping": _shipping(5), "news_sentiment": news, "market": market}
    monkeypatch.setattr(bed, "load_config_for_region", lambda region: {})
    monkeypatch.setattr(bed, "_fetch_domain", lambda domain, config: frames.get(domain))
    _, m = bed.build_region("hormuz")
    assert m["market_source"]["dating"].startswith("first-release")


# --------------------------------------------- NaN-aware tone (no articles)
def test_recency_tone_mean_skips_days_without_articles(tmp_path) -> None:
    rows = ["2019-01-01,0,0,,,0,1000",                  # quiet, before any coverage
            "2019-01-02,5,1,-8.0,0,0,1000",
            "2019-01-03,0,0,,,0,1000", "2019-01-04,0,0,,,0,1000"]
    df = _connector(_write_news(tmp_path, "hormuz", rows)).fetch()
    rec = df.set_index("timestamp")["recency_weighted_score"]
    assert rec.loc["2019-01-01"] == 0.0                 # no tone anywhere yet: no signal
    # Quiet days carry no tone, so they do not pull the mean toward 0.
    assert rec.loc["2019-01-03"] == pytest.approx(-0.8)
    assert rec.loc["2019-01-04"] == pytest.approx(-0.8)
    assert df.set_index("timestamp").loc["2019-01-03", "sentiment_score"] == 0.0


def _agent_frame(tone, volume) -> pd.DataFrame:
    n = len(tone)
    return pd.DataFrame({"timestamp": pd.date_range("2019-01-01", periods=n, freq="D"),
                         "sentiment_score": tone, "sentiment_magnitude": np.abs(tone),
                         "article_volume": volume,
                         "recency_weighted_score": tone})


def _preprocessed(frame: pd.DataFrame) -> pd.DataFrame:
    agent = NewsAgent()
    agent.fit(frame)
    return agent.preprocess(frame)


def test_agent_tone_mean_skips_days_without_articles() -> None:
    tone = np.array([-0.5, 0.0, 0.0, 0.0, -0.3])
    out = _preprocessed(_agent_frame(tone, [5.0, 0.0, 0.0, 0.0, 5.0]))
    assert out["sentiment_rolling_7d"].iloc[3] == pytest.approx(-0.5)   # not -0.125
    assert out["sentiment_rolling_7d"].iloc[4] == pytest.approx(-0.4)


def test_agent_velocity_keeps_its_original_daily_definition() -> None:
    # Only the tone *mean* is NaN-aware. Velocity stays the 3-day change in
    # the daily sentiment_score (0 on quiet days), as before; a day without
    # any news data gives no velocity signal instead of NaN.
    out = _preprocessed(_agent_frame(np.array([0.0, 0.0, 0.0, -0.6, np.nan, -0.6, -0.6]),
                                     [0.0, 0.0, 0.0, 5.0, np.nan, 5.0, 5.0]))
    assert out["sentiment_velocity"].iloc[3] == pytest.approx(-0.6)
    assert out["sentiment_velocity"].notna().all()
    assert out["sentiment_velocity"].iloc[4] == 0.0


def test_agent_scores_nan_on_a_day_with_no_news_data_at_all() -> None:
    tone = np.array([-0.2, -0.4, np.nan, np.nan, -0.6, -0.6, -0.6])
    volume = np.array([5.0, 5.0, np.nan, np.nan, 5.0, 5.0, 5.0])
    frame = _agent_frame(tone, volume)
    agent = NewsAgent()
    agent.fit(frame)
    scored = agent.detect(agent.preprocess(frame))
    assert scored["anomaly_score"].iloc[2:4].isna().all()     # masked: no observation
    rest = scored["anomaly_score"].drop(index=[2, 3])
    assert rest.notna().all() and rest.between(0.0, 1.0).all()
    assert not scored["is_anomaly"].iloc[2:4].any()
