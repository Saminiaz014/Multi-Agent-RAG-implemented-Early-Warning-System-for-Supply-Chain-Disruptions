"""Tests for the EVAL01 region registry and scenario spec loader."""

from __future__ import annotations

import copy

import yaml
import pytest

from src.benchmark.regions import (
    CANONICAL_REGIONS,
    REGION_ALIASES,
    REGION_STATUS,
    load_region,
    resolve_region_key,
)
from src.benchmark.scenario_generator import load_scenario, materialize_scenario


def test_load_region() -> None:
    region = load_region("hormuz")
    assert region.name == "hormuz"
    assert "shipping" in region.active_domains
    assert region.reroutable == False


@pytest.mark.parametrize(
    "region_key",
    [key for key, status in REGION_STATUS.items() if status == "populated"],
)
def test_load_region_succeeds_for_every_populated_region(region_key: str) -> None:
    """Every region REGION_STATUS marks 'populated' must have a
    config/benchmark/{key}.yaml that actually loads — a status flip not
    backed by a valid, passing YAML should fail this test immediately,
    not surface later as a silent gap. Parametrized off REGION_STATUS
    itself, so this extends automatically as more regions go populated —
    today that's just hormuz."""
    region = load_region(region_key)
    assert region.name == region_key


def test_load_scenario() -> None:
    spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
    assert spec.region == "hormuz"
    assert spec.event["onset_day"] == 240


def test_materialize_hormuz_4_scenarios() -> None:
    """Hormuz 4 scenarios x 1 seed each."""
    for scenario_file in ["hormuz_P_CRIT", "hormuz_P_HIGH", "hormuz_N_QUIET", "hormuz_N_DECOY"]:
        spec = load_scenario(f"config/benchmark/scenarios/{scenario_file}.yaml")
        region = load_region(spec.region)
        df = materialize_scenario(spec, region)
        assert len(df) == 365
        assert set(region.active_domains).issubset(df.columns)
        if scenario_file == "hormuz_P_CRIT":
            assert df.loc[240:300, "y_disruption"].sum() > 0  # event window has positives


def test_silent_agent_handling() -> None:
    """Disaster agent effect is null; verify it stays at baseline."""
    spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
    region = load_region(spec.region)
    df = materialize_scenario(spec, region)
    # disaster signal should be flat ~0.02 +/- noise, no ramp
    disaster_mean = df["disaster"].mean()
    assert 0.01 < disaster_mean < 0.04


def test_scenario_split() -> None:
    """Train 0-200, val 201-280, test 281-364."""
    spec = load_scenario("config/benchmark/scenarios/hormuz_P_CRIT.yaml")
    region = load_region(spec.region)
    df = materialize_scenario(spec, region)
    train, val, test = df[0:201], df[201:281], df[281:365]
    assert len(train) == 201 and len(val) == 80 and len(test) == 84


def test_resolve_region_key_canonical_hit() -> None:
    assert resolve_region_key("hormuz") == "hormuz"
    assert resolve_region_key("malacca") == "malacca"


def test_resolve_region_key_alias_hit() -> None:
    """red_sea (the pre-existing settings.yaml/extraction.chokepoints key)
    resolves to the canonical bab_el_mandeb key."""
    assert resolve_region_key("red_sea") == "bab_el_mandeb"


def test_resolve_region_key_display_name_hit() -> None:
    assert resolve_region_key("Strait of Hormuz") == "hormuz"
    assert resolve_region_key("Bab el-Mandeb") == "bab_el_mandeb"


def test_resolve_region_key_case_insensitive() -> None:
    assert resolve_region_key("HORMUZ") == "hormuz"
    assert resolve_region_key("Red_Sea") == "bab_el_mandeb"


def test_resolve_region_key_malacca_is_canonical() -> None:
    """malacca is a full canonical key, not an alias — it already agrees
    with extraction.chokepoints, so no aliasing is needed."""
    assert "malacca" in CANONICAL_REGIONS
    assert "malacca" not in REGION_ALIASES
    assert resolve_region_key("malacca") == "malacca"


def test_resolve_region_key_unknown_raises_with_options() -> None:
    with pytest.raises(ValueError, match="Unknown region"):
        resolve_region_key("nonexistent_region")


def test_resolve_region_key_taiwan_strait_not_supported() -> None:
    """taiwan_strait is explicitly out of scope for this project (per an
    earlier external plan this codebase never implemented) and must raise."""
    with pytest.raises(ValueError):
        resolve_region_key("taiwan_strait")


def test_canonical_regions_status() -> None:
    """Only hormuz is populated; the other four canonical keys are declared
    but not pretending to have data."""
    assert REGION_STATUS["hormuz"] == "populated"
    for key in ("bab_el_mandeb", "panama", "suez", "malacca"):
        assert REGION_STATUS[key] == "planned"
    assert set(REGION_STATUS) == set(CANONICAL_REGIONS)


def test_status_filtering_returns_only_hormuz() -> None:
    """The exact filter src/dashboard/core.py's AVAILABLE_REGIONS applies
    (CANONICAL_REGIONS filtered to status == "populated") yields exactly
    hormuz today — mirrored here as a direct unit test of the filtering
    mechanism itself, independent of the dashboard module."""
    populated = {
        display_name: key
        for key, display_name in CANONICAL_REGIONS.items()
        if REGION_STATUS.get(key) == "populated"
    }
    assert populated == {"Strait of Hormuz": "hormuz"}


# ===========================================================================
# Fix 3/4/6 (docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md §6): load-time
# validation that converts previously-silent corruption into a loud
# ValueError. Fix 5 (scalar baseline_transits_per_day) tests follow.
# ===========================================================================

_HORMUZ_P_CRIT_YAML = "config/benchmark/scenarios/hormuz_P_CRIT.yaml"


def _write_scenario(tmp_path, overrides: dict) -> str:
    """hormuz_P_CRIT.yaml with a shallow top-level override, written to a
    temp file — e.g. _write_scenario(tmp_path, {"class": "P-CRT"})."""
    base = yaml.safe_load(open(_HORMUZ_P_CRIT_YAML, encoding="utf-8"))
    spec = copy.deepcopy(base)
    spec.update(overrides)
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(path)


def test_load_scenario_bad_peak_band_raises(tmp_path) -> None:
    base = yaml.safe_load(open(_HORMUZ_P_CRIT_YAML, encoding="utf-8"))
    bad_event = dict(base["event"])
    bad_event["peak_band"] = "catastrophic"
    path = _write_scenario(tmp_path, {"event": bad_event})
    with pytest.raises(ValueError, match="peak_band"):
        load_scenario(path)


def test_load_scenario_unrecognized_class_raises(tmp_path) -> None:
    path = _write_scenario(tmp_path, {"class": "P-CRT"})  # typo of P-CRIT
    with pytest.raises(ValueError, match="class"):
        load_scenario(path)


def test_load_scenario_out_of_range_domain_raises(tmp_path) -> None:
    """A raw percentage (45) landing in a bounded 0-1 domain (geopolitical)
    must raise, not silently distort the signal ~100x — the exact incident
    BENCHMARK_SCHEMA_REFERENCE.md §2/§6 was commissioned to prevent."""
    base = yaml.safe_load(open(_HORMUZ_P_CRIT_YAML, encoding="utf-8"))
    bad_signals = dict(base["signals"])
    bad_signals["geopolitical"] = {"baseline": {"mean": 45, "std": 0.02}, "effect": None}
    path = _write_scenario(tmp_path, {"signals": bad_signals})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_scenario(path)


def test_load_scenario_unbounded_domain_accepts_large_values(tmp_path) -> None:
    """shipping/market are NOT range-checked to [0, 1] — only bounded
    domains are. A large raw vessel count must still load fine."""
    base = yaml.safe_load(open(_HORMUZ_P_CRIT_YAML, encoding="utf-8"))
    signals = dict(base["signals"])
    signals["shipping"] = {"baseline": {"mean": 500, "std": 20}, "effect": None}
    path = _write_scenario(tmp_path, {"signals": signals})
    spec = load_scenario(path)  # must not raise
    assert spec.signals["shipping"]["baseline"]["mean"] == 500


def test_load_scenario_event_null_raises_helpful_message(tmp_path) -> None:
    path = _write_scenario(tmp_path, {"event": None})
    with pytest.raises(ValueError, match="hormuz_N_QUIET"):
        load_scenario(path)


def test_load_region_scalar_baseline_transits(tmp_path, monkeypatch) -> None:
    """A bare scalar baseline_transits_per_day is accepted and normalized
    to a one-element list, matching the Region docstring (previously this
    crashed with an uncaught TypeError)."""
    monkeypatch.setattr("src.benchmark.regions._CONFIG_DIR", tmp_path)
    (tmp_path / "scalartest.yaml").write_text(
        yaml.safe_dump({
            "scalartest": {
                "center": {"lat": 1.0, "lng": 2.0},
                "baseline_transits_per_day": 70,
                "active_domains": [],
                "reroutable": False,
                "loss_scaling": "linear",
                "disaster_relevance": "low",
            }
        }),
        encoding="utf-8",
    )
    region = load_region("scalartest")
    assert region.baseline_transits_per_day == [70]


def test_load_region_bad_baseline_transits_type_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.benchmark.regions._CONFIG_DIR", tmp_path)
    (tmp_path / "badtest.yaml").write_text(
        yaml.safe_dump({
            "badtest": {
                "center": {"lat": 1.0, "lng": 2.0},
                "baseline_transits_per_day": "sixty to eighty",
                "active_domains": [],
                "reroutable": False,
                "loss_scaling": "linear",
                "disaster_relevance": "low",
            }
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline_transits_per_day"):
        load_region("badtest")
