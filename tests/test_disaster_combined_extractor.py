"""Tests for the combined GDACS + USGS monthly cap (Phase 7).

The cap is the only thing standing between the knowledge base and tens of
thousands of hazard alerts, so its edge cases are worth pinning: it must bound
*both* sources, bucket by calendar month rather than by position in the list,
and prefer severe events over routine ones when it has to drop some.

No network: documents are built by hand in the shape ``_normalize_document``
emits.
"""

from __future__ import annotations

from src.extractors.disaster_combined_extractor import DisasterCombinedExtractor as DCE


def _doc(doc_id: str, date: str, severity: str = "low", source: str = "gdacs") -> dict:
    return {
        "id": f"{source}_{doc_id}",
        "text": f"event {doc_id}",
        "metadata": {"source_api": source, "event_date": date, "severity": severity},
    }


class TestMonthlyCap:
    def test_caps_each_month_independently(self) -> None:
        """A busy month must not eat a quiet month's allowance."""
        docs = [_doc(f"a{i}", "2024-03-15") for i in range(40)]
        docs += [_doc("b1", "2024-04-02"), _doc("b2", "2024-04-20")]

        kept = DCE._apply_monthly_cap(docs)

        assert len(kept) == DCE.DOCS_PER_MONTH + 2
        by_month = [DCE._month_key(d) for d in kept]
        assert by_month.count((2024, 3)) == DCE.DOCS_PER_MONTH
        assert by_month.count((2024, 4)) == 2

    def test_cap_applies_to_gdacs_not_only_usgs(self) -> None:
        """Capping only the second source would leave the total unbounded.

        GDACS is normally the noisy one, so a cap that trusts it to be under
        the limit does nothing at all.
        """
        gdacs_only = [_doc(f"g{i}", "2024-05-10", source="gdacs") for i in range(50)]

        assert len(DCE._apply_monthly_cap(gdacs_only)) == DCE.DOCS_PER_MONTH

    def test_keeps_severe_events_over_routine_ones(self) -> None:
        """Severity decides what survives — not arrival order from the API."""
        docs = [_doc(f"low{i}", "2024-06-01", "low") for i in range(20)]
        docs += [_doc("red", "2024-06-14", "high"), _doc("orange", "2024-06-20", "medium")]

        kept = DCE._apply_monthly_cap(docs)
        kept_ids = {d["id"] for d in kept}

        assert len(kept) == DCE.DOCS_PER_MONTH
        assert "gdacs_red" in kept_ids, "a Red alert must never be dropped for Green ones"
        assert "gdacs_orange" in kept_ids

    def test_both_sources_can_survive_the_same_month(self) -> None:
        """The point of two sources is that both reach the knowledge base."""
        docs = [_doc(f"g{i}", "2024-07-05", "medium", "gdacs") for i in range(10)]
        docs += [_doc(f"u{i}", "2024-07-06", "high", "usgs") for i in range(10)]

        kept = DCE._apply_monthly_cap(docs)
        sources = {d["metadata"]["source_api"] for d in kept}

        assert len(kept) == DCE.DOCS_PER_MONTH
        assert sources == {"gdacs", "usgs"}

    def test_undated_documents_cannot_evict_dated_ones(self) -> None:
        """A missing date must not silently displace a real month's events."""
        docs = [_doc(f"u{i}", "", "high") for i in range(20)]
        docs += [_doc("dated", "2024-08-03", "low")]

        kept = DCE._apply_monthly_cap(docs)
        kept_ids = {d["id"] for d in kept}

        assert "gdacs_dated" in kept_ids
        assert sum(1 for d in kept if DCE._month_key(d) == (0, 0)) == DCE.DOCS_PER_MONTH

    def test_selection_is_deterministic(self) -> None:
        """Two runs over the same data must keep the same documents."""
        docs = [_doc(f"x{i}", "2024-09-11", "medium") for i in range(30)]

        first = [d["id"] for d in DCE._apply_monthly_cap(docs)]
        second = [d["id"] for d in DCE._apply_monthly_cap(list(reversed(docs)))]

        assert first == second

    def test_empty_input_is_not_an_error(self) -> None:
        assert DCE._apply_monthly_cap([]) == []


class TestMonthKey:
    def test_parses_iso_dates(self) -> None:
        assert DCE._month_key(_doc("a", "2019-12-31")) == (2019, 12)

    def test_malformed_dates_fall_into_the_undated_bucket(self) -> None:
        for bad in ("", "not-a-date", "2019", None):
            doc = {"metadata": {"event_date": bad}}
            assert DCE._month_key(doc) == (0, 0), bad

    def test_missing_metadata_is_tolerated(self) -> None:
        assert DCE._month_key({}) == (0, 0)


class TestGDACSRegionFiltering:
    """GDACS scoping is client-side, and must actually scope.

    Regression test for a real failure: the endpoint accepts ``countrycode``,
    ``iso3`` and ``countrylist`` and silently ignores all three, returning an
    identical global list every time. Relying on it put 635 global events —
    Madagascar cyclones, Philippine volcanoes — into the knowledge base
    labelled ``region=hormuz``. Filtering now happens on each event's
    ``affectedcountries``, and these tests pin that it discriminates.
    """

    @staticmethod
    def _feature(event_id: str, iso2: list[str], date: str = "2024-03-04") -> dict:
        return {
            "properties": {
                "eventid": event_id,
                "eventtype": "FL",
                "alertlevel": "Orange",
                "fromdate": date,
                "country": ", ".join(iso2),
                "affectedcountries": [{"iso2": c, "iso3": c + "X"} for c in iso2],
            },
            "geometry": {"coordinates": [56.0, 26.0]},
        }

    def _extractor(self, monkeypatch, features: list[dict]):
        from src.core.config_manager import load_base_config
        from src.extractors.gdacs_extractor import GDACSExtractor

        extractor = GDACSExtractor(load_base_config())
        monkeypatch.setattr(
            extractor, "_global_events", lambda start, end: features
        )
        return extractor

    def test_only_events_touching_the_region_are_kept(self, monkeypatch) -> None:
        features = [
            self._feature("1", ["IR"]),        # hormuz
            self._feature("2", ["MG"]),        # Madagascar — the old bug
            self._feature("3", ["PH"]),        # Philippines — the old bug
            self._feature("4", ["AE", "OM"]),  # hormuz
        ]
        extractor = self._extractor(monkeypatch, features)

        docs = extractor.extract_historical("hormuz", start_year=2024, end_year=2024)
        matched = {d["metadata"]["gdacs_event_id"] for d in docs}

        assert matched == {"1", "4"}, "global events must not reach a region"

    def test_multi_country_events_match_on_any_affected_country(
        self, monkeypatch
    ) -> None:
        """The primary country is not the only one that counts.

        A cyclone listing Madagascar first but also affecting a region's
        country is genuinely relevant to that region.
        """
        features = [self._feature("9", ["MG", "MZ", "OM"])]
        extractor = self._extractor(monkeypatch, features)

        docs = extractor.extract_historical("hormuz", start_year=2024, end_year=2024)
        assert len(docs) == 1

    def test_regions_receive_different_events(self, monkeypatch) -> None:
        """Two regions returning identical sets is the signature of the bug."""
        features = [
            self._feature("a", ["IR"]),
            self._feature("b", ["ID"]),
            self._feature("c", ["PA"]),
        ]
        extractor = self._extractor(monkeypatch, features)

        per_region = {
            region: {
                d["metadata"]["gdacs_event_id"]
                for d in extractor.extract_historical(
                    region, start_year=2024, end_year=2024
                )
            }
            for region in ("hormuz", "malacca", "panama")
        }

        assert per_region["hormuz"] == {"a"}
        assert per_region["malacca"] == {"b"}
        assert per_region["panama"] == {"c"}

    def test_document_ids_are_region_qualified(self, monkeypatch) -> None:
        """A shared event must survive as one document per affected region.

        The builder deduplicates by id, so an unqualified id would drop the
        second region's copy — and its region metadata with it.
        """
        shared = [self._feature("shared", ["SA"])]  # in hormuz and bab_el_mandeb
        extractor = self._extractor(monkeypatch, shared)

        ids = set()
        for region in ("hormuz", "bab_el_mandeb"):
            docs = extractor.extract_historical(region, start_year=2024, end_year=2024)
            if docs:  # only if SA is genuinely in that region's iso2 list
                ids.add(docs[0]["id"])

        assert len(ids) == len([
            r for r in ("hormuz", "bab_el_mandeb")
            if "SA" in extractor._iso2_codes(r)
        ])

    def test_events_without_affected_countries_are_not_assigned_anywhere(
        self, monkeypatch
    ) -> None:
        """An unattributable event must be dropped, not handed to every region."""
        features = [{"properties": {"eventid": "x", "eventtype": "EQ",
                                    "fromdate": "2024-01-01"}, "geometry": {}}]
        extractor = self._extractor(monkeypatch, features)

        for region in ("hormuz", "malacca", "panama", "bab_el_mandeb"):
            assert extractor.extract_historical(
                region, start_year=2024, end_year=2024
            ) == []


class TestFailedWindowRecovery:
    """A window that cannot be fetched must not become a silent hole.

    Regression test for a real run: the window ``2024-10-01..2026-12-31``
    timed out, was logged and dropped, and every GDACS event after October
    2024 went missing from the knowledge base without the run failing.
    """

    def _extractor(self, monkeypatch, fetch):
        from src.core.config_manager import load_base_config
        from src.extractors import gdacs_extractor as mod

        extractor = mod.GDACSExtractor(load_base_config())
        monkeypatch.setattr(extractor, "_rate_limit_wait", lambda: None)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(extractor, "_fetch", fetch)
        return extractor

    def test_unfetchable_window_is_split_and_retried(self, monkeypatch) -> None:
        """The halves must still be attempted after the whole fails."""
        attempted: list[tuple[str, str]] = []

        def fetch(from_date, to_date, attempt=1):
            attempted.append((from_date, to_date))
            # The full-span request fails; anything shorter succeeds.
            if from_date == "2024-01-01" and to_date == "2024-12-31":
                return None
            return [{"properties": {"eventid": from_date, "eventtype": "FL",
                                    "fromdate": from_date,
                                    "affectedcountries": [{"iso2": "IR"}]},
                     "geometry": {}}]

        extractor = self._extractor(monkeypatch, fetch)
        out: list[dict] = []
        extractor._collect(2024 * 12, 2024 * 12 + 11, out)

        assert len(attempted) > 1, "a failed window must be retried in halves"
        assert out, "the halves' events must still reach the caller"

    def test_retries_are_attempted_before_giving_up(self, monkeypatch) -> None:
        """A transient timeout should not cost the window."""
        from src.core.config_manager import load_base_config
        from src.extractors import gdacs_extractor as mod

        calls = {"n": 0}

        class _Response:
            headers = {"content-type": "application/json"}

            @staticmethod
            def raise_for_status() -> None: ...

            @staticmethod
            def json() -> dict:
                return {"features": [{"properties": {"eventid": "ok"}}]}

        def flaky_get(*_a, **_kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("read timed out")
            return _Response()

        extractor = mod.GDACSExtractor(load_base_config())
        monkeypatch.setattr(extractor, "_rate_limit_wait", lambda: None)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(mod.requests, "get", flaky_get)

        assert extractor._fetch("2024-01-01", "2024-12-31") is not None
        assert calls["n"] == 3

    def test_persistent_failure_returns_none_not_empty(self, monkeypatch) -> None:
        """``None`` means "unknown"; ``[]`` would claim a genuinely quiet span."""
        from src.core.config_manager import load_base_config
        from src.extractors import gdacs_extractor as mod

        extractor = mod.GDACSExtractor(load_base_config())
        monkeypatch.setattr(extractor, "_rate_limit_wait", lambda: None)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            mod.requests, "get",
            lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("down")),
        )

        assert extractor._fetch("2024-01-01", "2024-01-31") is None


class TestWindowBounds:
    """Month-ordinal windowing in the GDACS extractor."""

    def test_bounds_cover_whole_months(self) -> None:
        from src.extractors.gdacs_extractor import GDACSExtractor

        # 2024-01 .. 2024-12
        assert GDACSExtractor._window_bounds(2024 * 12, 2024 * 12 + 11) == (
            "2024-01-01", "2024-12-31",
        )

    def test_february_end_respects_leap_years(self) -> None:
        """A truncated February would drop the 29th's events entirely."""
        from src.extractors.gdacs_extractor import GDACSExtractor

        feb_2024 = 2024 * 12 + 1
        feb_2023 = 2023 * 12 + 1
        assert GDACSExtractor._window_bounds(feb_2024, feb_2024)[1] == "2024-02-29"
        assert GDACSExtractor._window_bounds(feb_2023, feb_2023)[1] == "2023-02-28"

    def test_windows_can_span_a_year_boundary(self) -> None:
        """Halving lands on arbitrary months, not just January starts."""
        from src.extractors.gdacs_extractor import GDACSExtractor

        assert GDACSExtractor._window_bounds(2021 * 12 + 10, 2022 * 12 + 2) == (
            "2021-11-01", "2022-03-31",
        )
