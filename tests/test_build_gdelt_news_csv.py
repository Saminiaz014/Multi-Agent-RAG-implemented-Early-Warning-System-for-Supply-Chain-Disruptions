"""Tests for scripts/build_gdelt_news_csv.py (GDELT 1.0 corridor news, filter v2)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_gdelt_news_csv.py"
_spec = importlib.util.spec_from_file_location("build_gdelt_news_csv", _SCRIPT)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def _line(lat: float, lon: float, geo_type: str = "4", name: str = "Somewhere",
          n_art: int = 10, tone: float = -2.0, gold: float = -5.0,
          quad: str = "4") -> str:
    p = [""] * g.N_COLS
    p[g.C_QUAD], p[g.C_GOLD] = quad, str(gold)
    p[g.C_NART], p[g.C_TONE] = str(n_art), str(tone)
    p[g.C_AGEO_TYPE], p[g.C_AGEO_NAME] = geo_type, name
    p[g.C_AGEO_LAT], p[g.C_AGEO_LON] = str(lat), str(lon)
    return "\t".join(p)


def _zip(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20190101.export.CSV", "\n".join(lines) + "\n")
    return buf.getvalue()


def _centre(region: str) -> tuple[float, float]:
    a, b, c, d = g.BBOXES[region][0]
    return (a + b) / 2, (c + d) / 2


# ---------------------------------------------------------------- filter
@pytest.mark.parametrize("geo_type,kept", [("1", False), ("2", False), ("3", True),
                                           ("4", True), ("5", False), ("", False)])
def test_only_point_geocodes_types_3_and_4_are_kept(geo_type, kept) -> None:
    lat, lon = _centre("hormuz")
    agg = g.process(_zip([_line(lat, lon, geo_type=geo_type)]))["agg"]
    assert ("hormuz" in agg) is kept


def test_panama_city_is_excluded_by_name_even_inside_box() -> None:
    lat, lon = 8.96667, -79.5333                     # GDELT's Panama City point
    assert g.region_of(lat, lon) == "panama"
    lines = [_line(lat, lon, name="Panama City, Panamá, Panama"),
             _line(lat, lon, name="panama city"),
             _line(9.33333, -79.9167, name="Panama Canal, Panama (general), Panama", n_art=3)]
    agg = g.process(_zip(lines))["agg"]
    assert agg["panama"]["n_events"] == 1
    assert agg["panama"]["n_articles"] == 3


def test_panama_city_name_exclusion_applies_only_to_panama() -> None:
    lat, lon = _centre("hormuz")
    agg = g.process(_zip([_line(lat, lon, name="Panama City")]))["agg"]
    assert agg["hormuz"]["n_events"] == 1


def test_gdelt_gazetteer_points_fall_in_their_region() -> None:
    """v2 reason: GDELT's own coordinates for the corridor features."""
    assert g.region_of(9.33333, -79.9167) == "panama"     # "Panama Canal"
    assert g.region_of(2.196, 102.241) == "malacca"       # "Malacca"/"Melaka"


def test_named_strait_points_fall_inside_their_box() -> None:
    points = {"hormuz": [(26.6, 56.5), (26.567, 56.25)],
              "bab_el_mandeb": [(12.583, 43.333)],
              "panama": [(9.133, -79.683)],
              "malacca": [(4.0, 100.0)]}
    for region, pts in points.items():
        for lat, lon in pts:
            assert g.region_of(lat, lon) == region


def test_known_cities_fall_outside_every_box() -> None:
    cities = {
        "Dubai": (25.20, 55.27), "Bandar Abbas": (27.19, 56.27),
        "Ras al-Khaimah": (25.79, 55.94), "Singapore": (1.35, 103.82),
        "Kuala Lumpur": (3.14, 101.69), "Port Klang": (3.00, 101.39),
        "Seremban": (2.72, 101.94), "Penang": (5.41, 100.33),
        "Djibouti": (11.59, 43.15), "Aden": (12.80, 45.03), "Mocha": (13.32, 43.25),
        "San Miguelito": (9.03, -79.50), "Tocumen": (9.08, -79.38),
    }
    for name, (lat, lon) in cities.items():
        assert g.region_of(lat, lon) is None, name


def test_every_region_box_is_well_formed() -> None:
    for region, boxes in g.BBOXES.items():
        assert boxes, region
        for a, b, c, d in boxes:
            assert a < b and c < d, region


def test_tone_is_article_weighted_and_nan_without_articles() -> None:
    lat, lon = _centre("hormuz")
    agg = g.process(_zip([_line(lat, lon, n_art=1, tone=-4.0),
                          _line(lat, lon, n_art=3, tone=0.0)]))["agg"]
    r = g.row(dt.date(2019, 1, 1), agg["hormuz"])
    assert r["news_volume"] == 4 and r["news_tone"] == -1.0
    zero = g.row(dt.date(2019, 1, 1), g.empty_agg())
    assert zero["news_volume"] == 0 and zero["news_tone"] == ""   # peace time, not missing


def test_malformed_numbers_count_as_bad_rows() -> None:
    lat, lon = _centre("hormuz")
    bad = _line(lat, lon).split("\t")
    bad[g.C_NART] = "abc"
    res = g.process(_zip(["\t".join(bad), _line(lat, lon)]))
    assert res["bad_rows"] == 1 and res["agg"]["hormuz"]["n_events"] == 1


# ------------------------------------------------------------- fetch
class _Resp:
    def __init__(self, body: bytes) -> None:
        self._b = body
    def read(self) -> bytes:
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a) -> None:
        return None


def _http404(url):
    return HTTPError(url, 404, "nf", {}, None)


def test_pending_cutoff_is_three_days(monkeypatch) -> None:
    monkeypatch.setattr(g, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http404("u")))
    today = dt.date(2026, 11, 21)
    assert g.fetch(today - dt.timedelta(days=3), today=today) is g.PENDING
    assert g.fetch(today - dt.timedelta(days=4), today=today) is None


def test_truncated_zip_is_retried(monkeypatch) -> None:
    good = _zip([_line(*_centre("hormuz"))])
    bodies = iter([good[:50], good])
    monkeypatch.setattr(g, "urlopen", lambda *a, **k: _Resp(next(bodies)))
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    assert g.fetch(dt.date(2019, 1, 1), today=dt.date(2026, 11, 21)) == good


def test_non_404_http_error_is_retried(monkeypatch) -> None:
    good = _zip([_line(*_centre("hormuz"))])
    calls = iter([HTTPError("u", 503, "x", {}, None), _Resp(good)])

    def fake(*a, **k):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(g, "urlopen", fake)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    assert g.fetch(dt.date(2019, 1, 1), today=dt.date(2026, 11, 21)) == good


def test_persistent_failure_raises(monkeypatch) -> None:
    monkeypatch.setattr(g, "urlopen", lambda *a, **k: _Resp(b"not a zip"))
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        g.fetch(dt.date(2019, 1, 1), today=dt.date(2026, 11, 21))


def test_url_is_https() -> None:
    assert g.URL.startswith("https://")


# ------------------------------------------------------ progress/ingest
def test_manifest_counts_only_missing_days_in_range() -> None:
    missing = ["2018-12-31", "2019-01-02", "2019-02-01"]
    assert g.missing_in_range(missing, dt.date(2019, 1, 1), dt.date(2019, 1, 10)) == ["2019-01-02"]


@pytest.mark.parametrize("workers", [1, 3])
def test_ingest_persists_missing_not_pending_or_failed(monkeypatch, tmp_path, workers) -> None:
    blob = _zip([_line(*_centre("hormuz"))])
    outcomes = {dt.date(2019, 1, 1): blob, dt.date(2019, 1, 2): None,
                dt.date(2019, 1, 3): g.PENDING, dt.date(2019, 1, 4): RuntimeError("boom")}
    seen = []

    def fake_fetch(day):
        seen.append(day)
        out = outcomes[day]
        if isinstance(out, Exception):
            raise out
        return out
    monkeypatch.setattr(g, "fetch", fake_fetch)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    prog_f = tmp_path / "_progress.json"
    prog = g._load_progress(prog_f)
    pending, failed = g._ingest(prog, prog_f, dt.date(2019, 1, 1), dt.date(2019, 1, 4), workers)
    assert pending == ["2019-01-03"] and failed == ["2019-01-04"]
    saved = g._load_progress(prog_f)
    assert list(saved["done"]) == ["2019-01-01"] and saved["missing"] == ["2019-01-02"]
    seen.clear()
    g._ingest(saved, prog_f, dt.date(2019, 1, 1), dt.date(2019, 1, 4), workers)
    assert sorted(seen) == [dt.date(2019, 1, 3), dt.date(2019, 1, 4)]   # only these refetched


def test_progress_from_other_filter_is_refused(tmp_path) -> None:
    prog_f = tmp_path / "_progress.json"
    prog_f.write_text('{"filter_version": "v1", "filter_sha256": "other", "done": {}, "missing": []}')
    with pytest.raises(SystemExit):
        g._load_progress(prog_f)


def test_zero_share_counts_zero_volume_days() -> None:
    prog = {"done": {"2019-01-01": {"agg": {"hormuz": {**g.empty_agg(), "n_articles": 5}}},
                     "2019-01-02": {"agg": {}}}}
    share = g._zero_share(prog, dt.date(2019, 1, 1), dt.date(2019, 1, 2))
    assert share["hormuz"] == (1, 2) and share["malacca"] == (2, 2)


def test_filter_version_is_v2() -> None:
    assert g.FILTER_VERSION == "v2"
