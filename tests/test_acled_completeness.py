"""ACLED completeness is recorded in each frame's manifest and enforced on use.

* ``build_region`` copies the connector's ``geopolitical_source`` block (row
  counts per country-year, a content hash) into the manifest.
* The comparison harness refuses a frame with no such block, or one with a
  country-year that is a round multiple of the page size (the old one-page cap).
* ``test_cached_manifests_have_no_round_cap_country_year`` fails on any local
  frame whose ACLED events look cut at a page.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bed = _load("build_eval_dataset", "scripts/build_eval_dataset.py")
rmc = _load("run_method_comparison", "scripts/run_method_comparison.py")

_POLICY = {"gap_fill": {"policy": "forward-fill from the last earlier value"},
           "market_source": {"dating": "first-release date (ALFRED)"}}


def _source(*counts: int, page_size: int = 5000, round_cap: list | None = None) -> dict:
    rows = [{"country": "Yemen", "year": 2023 + i, "events": n, "pages": max(1, -(-n // page_size))}
            for i, n in enumerate(counts)]
    return {"page_size": page_size, "country_years": rows, "events_total": sum(counts),
            "events_sha256": "0" * 64, "round_cap_country_years": round_cap or []}


# ------------------------------------------------------------------ manifest
def test_build_region_copies_the_geopolitical_source_block(monkeypatch) -> None:
    days = pd.date_range("2019-01-01", periods=5, freq="D")
    shipping = pd.DataFrame({"timestamp": days, "vessel_count": 50.0, "tanker_count": 10.0,
                             "avg_delay_hours": 1.0, "congestion_index": 0.5,
                             "is_disruption": False})
    geo = pd.DataFrame({"timestamp": days, "military_activity_index": 0.2,
                        "diplomatic_incident_score": 0.1, "regime_stability_index": 0.9})
    geo.attrs["geopolitical_source"] = _source(7627, 40)
    frames = {"shipping": shipping, "geopolitical": geo}
    monkeypatch.setattr(bed, "load_config_for_region", lambda region: {})
    monkeypatch.setattr(bed, "_fetch_domain", lambda domain, config: frames.get(domain))
    _, manifest = bed.build_region("hormuz")
    assert manifest["geopolitical_source"] == _source(7627, 40)


# ------------------------------------------------------------------- harness
def _manifest(**extra) -> dict:
    return {"region": "x", "news_source": {"days_masked": []}, **_POLICY,
            "domains_present": {"geopolitical": ["military_activity_index"]}, **extra}


def test_a_frame_without_the_source_block_is_refused() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3)})
    with pytest.raises(ValueError, match="geopolitical_source"):
        rmc._masked_days(frame, _manifest())


def test_a_round_cap_country_year_is_refused() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3)})
    capped = _source(5000, 40, round_cap=[{"country": "Yemen", "year": 2023, "events": 5000}])
    with pytest.raises(ValueError, match="cap"):
        rmc._masked_days(frame, _manifest(geopolitical_source=capped))


def test_a_complete_frame_and_a_region_without_geopolitics_pass() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3)})
    assert not rmc._masked_days(frame, _manifest(geopolitical_source=_source(7627, 40))).any()
    no_geo = {**_manifest(), "domains_present": {"shipping": ["vessel_count"]}}
    assert not rmc._masked_days(frame, no_geo).any()


# ---------------------------------------------------- the cached local frames
def _cached_manifests() -> list[Path]:
    return sorted((_ROOT / "data" / "eval").glob("*.manifest.json"))


def test_cached_manifests_have_no_round_cap_country_year() -> None:
    """Fails on any local frame whose ACLED events look cut at a page.

    A count that is a positive multiple of the page size is what the old
    one-page fetch produced for every country-year over the cap. Frames built
    before the cursor fetch have no source block and fail here too: rebuild
    them with ``scripts/build_eval_dataset.py --force``.
    """
    paths = _cached_manifests()
    if not paths:
        pytest.skip("no cached evaluation frames in data/eval")
    problems: list[str] = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "geopolitical" not in (manifest.get("domains_present") or {}):
            continue
        source = manifest.get("geopolitical_source")
        if not source:
            problems.append(f"{path.stem}: no geopolitical_source (built before the cursor fetch)")
            continue
        size = source["page_size"]
        capped = [f"{c['country']} {c['year']}: {c['events']}" for c in source["country_years"]
                  if c["events"] > 0 and c["events"] % size == 0]
        if capped or source["round_cap_country_years"]:
            problems.append(f"{path.stem}: round counts {capped}")
        if source["events_total"] != sum(c["events"] for c in source["country_years"]):
            problems.append(f"{path.stem}: events_total does not match its country-years")
    assert not problems, "; ".join(problems)


# ------------------------------------------------------- review follow-ups
def test_the_guard_recomputes_round_counts_instead_of_trusting_the_recorded_list() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3)})
    lying = _source(5000, 40, round_cap=[])                       # manifest says "no caps"
    with pytest.raises(ValueError, match="page_size"):            # error text names the way out
        rmc._masked_days(frame, _manifest(geopolitical_source=lying))


def test_require_complete_acled_is_one_helper_for_every_frame_consumer() -> None:
    rmc.require_complete_acled({"domains_present": {"shipping": ["x"]}}, "x")      # no ACLED
    rmc.require_complete_acled(_manifest(geopolitical_source=_source(7627, 40)), "x")
    with pytest.raises(ValueError, match="geopolitical_source"):
        rmc.require_complete_acled(_manifest(), "x")


def test_the_surrogate_script_refuses_a_stale_frame(tmp_path, monkeypatch) -> None:
    rrs = _load("run_real_surrogate", "scripts/run_real_surrogate.py")
    pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3), "y_true": 0}).to_csv(
        tmp_path / "x.csv", index=False)
    monkeypatch.setattr(rrs, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rrs, "_EVAL_CACHE", tmp_path)
    monkeypatch.setattr(rrs, "_FALLBACK_CACHE", tmp_path / "none")
    with pytest.raises(ValueError, match="no manifest"):           # unverifiable: refused
        rrs.load_region_frame("x")
    (tmp_path / "x.manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    with pytest.raises(ValueError, match="geopolitical_source"):
        rrs.load_region_frame("x")
    (tmp_path / "x.manifest.json").write_text(
        json.dumps(_manifest(geopolitical_source=_source(7627, 40))), encoding="utf-8")
    frame, _ = rrs.load_region_frame("x")
    assert len(frame) == 3
