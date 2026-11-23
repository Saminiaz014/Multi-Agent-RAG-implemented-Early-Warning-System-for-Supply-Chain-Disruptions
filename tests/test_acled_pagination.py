"""ACLED fetch: every country-year in full, never cut at one page.

The connector used to send ``limit=5000`` and read one page, so a country-year
with more events lost the rest silently (Yemen 2023: 5,000 of 7,627). The API
paginates with a cursor: ``next_cursor`` in the response, passed back as
``cursor``. (``page`` is rejected with HTTP 400.)
"""

from __future__ import annotations

import pytest

from src.extractors.acled_extractor import ACLEDExtractor, ACLEDFetchError

USER, PASSWORD = "analyst@example.org", "s3cret-pw"


def _extractor(monkeypatch, pages: list, calls: list[dict],
               user: str = USER, password: str = PASSWORD) -> ACLEDExtractor:
    """An extractor whose transport serves ``pages`` in order and records params.

    A queue entry that is an exception is raised when its turn comes.
    """
    extractor = ACLEDExtractor({"api_keys": {"acled_username": user, "acled_password": password}})
    monkeypatch.setattr(extractor, "_rate_limit_wait", lambda: None)
    queue = [pages] if isinstance(pages, Exception) else list(pages)

    class _Data:
        endpoint = "/acled/read"

        @staticmethod
        def _get(endpoint, params=None):
            calls.append(dict(params))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        @staticmethod
        def _parse_event(row):
            return dict(row)

    class _Client:
        _acled_data_client = _Data()

    monkeypatch.setattr(extractor, "_get_client", lambda: _Client())
    return extractor


def _page(start: int, n: int, cursor) -> dict:
    rows = [{"event_id_cnty": f"YEM{start + i}", "event_date": "2023-01-01",
             "event_type": "Protests", "fatalities": 0} for i in range(n)]
    return {"success": True, "data": rows, "next_cursor": cursor, "count": None,
            "total_count": None}


def test_follows_the_cursor_to_the_last_page(monkeypatch) -> None:
    calls: list[dict] = []
    pages = [_page(0, 5000, 111), _page(5000, 2627, None)]       # Yemen 2023
    events, n_pages = _extractor(monkeypatch, pages, calls).fetch_all_events(
        "Yemen", 2023, page_size=5000)
    assert len(events) == 7627 and n_pages == 2
    assert [c.get("cursor") for c in calls] == [None, 111]
    assert all(c["limit"] == 5000 and c["country"] == "Yemen" and c["year"] == 2023
               for c in calls)
    assert all("page" not in c for c in calls)                    # the API answers 400


def test_a_single_short_page_is_one_request(monkeypatch) -> None:
    calls: list[dict] = []
    events, n_pages = _extractor(monkeypatch, [_page(0, 12, None)], calls).fetch_all_events(
        "Djibouti", 2024)
    assert len(events) == 12 and n_pages == 1 and len(calls) == 1


def test_an_exact_full_page_is_complete_only_when_the_api_says_so(monkeypatch) -> None:
    """5,000 rows with no next_cursor is a true 5,000; with one it is not."""
    done = _extractor(monkeypatch, [_page(0, 5000, None)], []).fetch_all_events("X", 2023)
    assert len(done[0]) == 5000 and done[1] == 1
    more = _extractor(monkeypatch, [_page(0, 5000, 7), _page(5000, 1, None)], [])
    assert len(more.fetch_all_events("X", 2023)[0]) == 5001


def test_a_failing_page_raises_and_never_returns_a_partial_result(monkeypatch) -> None:
    """The second request fails: the first page must not come back as "the events"."""
    pages = [_page(0, 5000, 111),
             RuntimeError("HTTP 503 for url https://acleddata.com/api/acled/read?cursor=111")]
    with pytest.raises(ACLEDFetchError, match="Yemen/2023"):
        _extractor(monkeypatch, pages, []).fetch_all_events("Yemen", 2023)


def test_error_text_never_contains_the_credentials(monkeypatch) -> None:
    boom = RuntimeError(f"auth failed for {USER} with {PASSWORD} at https://x/?u={USER}")
    with pytest.raises(ACLEDFetchError) as caught:
        _extractor(monkeypatch, boom, []).fetch_all_events("Yemen", 2023)
    message = str(caught.value)
    assert USER not in message and PASSWORD not in message and "***" in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_an_api_error_response_raises(monkeypatch) -> None:
    bad = {"success": False, "error": [{"message": "no access"}], "data": []}
    with pytest.raises(ACLEDFetchError, match="no access"):
        _extractor(monkeypatch, [bad], []).fetch_all_events("Yemen", 2023)


def test_missing_credentials_raise_instead_of_returning_nothing(monkeypatch) -> None:
    with pytest.raises(ACLEDFetchError, match="credentials"):
        _extractor(monkeypatch, [], [], user="", password="").fetch_all_events("Yemen", 2023)


def test_a_repeated_cursor_cannot_loop_forever(monkeypatch) -> None:
    pages = [_page(0, 10, 5), _page(10, 10, 5), _page(20, 10, 5)]
    with pytest.raises(ACLEDFetchError, match="cursor"):
        _extractor(monkeypatch, pages, []).fetch_all_events("Yemen", 2023)


def test_overlapping_pages_are_refused(monkeypatch) -> None:
    pages = [_page(0, 10, 5), _page(5, 10, None)]                  # ids 5..9 twice
    with pytest.raises(ACLEDFetchError, match="duplicate"):
        _extractor(monkeypatch, pages, []).fetch_all_events("Yemen", 2023)


def test_a_runaway_page_count_is_refused(monkeypatch) -> None:
    pages = [_page(i * 10, 10, i + 1) for i in range(20)]
    with pytest.raises(ACLEDFetchError, match="pages"):
        _extractor(monkeypatch, pages, []).fetch_all_events("Yemen", 2023, max_pages=5)


def test_a_client_without_the_expected_transport_is_named(monkeypatch) -> None:
    extractor = ACLEDExtractor({"api_keys": {"acled_username": USER, "acled_password": PASSWORD}})
    monkeypatch.setattr(extractor, "_get_client", lambda: object())
    with pytest.raises(ACLEDFetchError, match="acled"):
        extractor.fetch_all_events("Yemen", 2023)


# ---------------------------------------------------- the connector's source block
def _connector(monkeypatch, counts: dict[tuple[str, int], int], page_size: int = 5000):
    """A GeopoliticalConnector whose extractor serves ``counts[(country, year)]`` events."""
    import src.ingestion.geopolitical_connector as mod

    monkeypatch.setattr(mod.GeopoliticalConnector, "_acled_config", lambda self: {})

    class _Stub:
        def __init__(self, _cfg) -> None: ...

        def fetch_all_events(self, country, year, page_size=5000, max_pages=200):
            n = counts[(country, year)]
            events = [{"event_id_cnty": f"{country}{year}-{i}", "event_date": f"{year}-01-{1 + i % 28:02d}",
                       "event_type": "Battles" if i % 2 else "Protests", "fatalities": i % 3}
                      for i in range(n)]
            return events, max(1, -(-n // page_size))

    monkeypatch.setattr("src.extractors.acled_extractor.ACLEDExtractor", _Stub)
    years = sorted({y for _, y in counts})
    return mod.GeopoliticalConnector(config={
        "data_mode": "api", "acled_countries": sorted({c for c, _ in counts}),
        "api": {"start_year": years[0], "end_year": years[-1], "page_size": page_size}})


def test_the_frame_carries_row_counts_and_a_content_hash(monkeypatch) -> None:
    counts = {("Yemen", 2023): 7627, ("Yemen", 2024): 40, ("Djibouti", 2023): 8, ("Djibouti", 2024): 0}
    source = _connector(monkeypatch, counts).fetch_api().attrs["geopolitical_source"]
    by = {(c["country"], c["year"]): c for c in source["country_years"]}
    assert by[("Yemen", 2023)] == {"country": "Yemen", "year": 2023, "events": 7627, "pages": 2}
    assert by[("Djibouti", 2024)]["events"] == 0
    assert source["events_total"] == sum(counts.values()) and source["page_size"] == 5000
    assert len(source["events_sha256"]) == 64
    assert source["round_cap_country_years"] == []


def test_the_content_hash_follows_the_events(monkeypatch) -> None:
    base = {("Yemen", 2023): 30}
    a = _connector(monkeypatch, base).fetch_api().attrs["geopolitical_source"]["events_sha256"]
    again = _connector(monkeypatch, base).fetch_api().attrs["geopolitical_source"]["events_sha256"]
    other = _connector(monkeypatch, {("Yemen", 2023): 31}).fetch_api() \
        .attrs["geopolitical_source"]["events_sha256"]
    assert a == again and a != other


def test_a_round_cap_count_is_flagged_not_hidden(monkeypatch) -> None:
    counts = {("Yemen", 2023): 5000, ("Sudan", 2023): 10000, ("Somalia", 2023): 4999}
    source = _connector(monkeypatch, counts).fetch_api().attrs["geopolitical_source"]
    assert source["round_cap_country_years"] == [
        {"country": "Sudan", "year": 2023, "events": 10000},
        {"country": "Yemen", "year": 2023, "events": 5000}]


def test_round_cap_detection_uses_the_page_size() -> None:
    from src.ingestion.geopolitical_connector import round_cap_country_years

    rows = [{"country": "A", "year": 2020, "events": 1000}, {"country": "B", "year": 2020, "events": 999},
            {"country": "C", "year": 2020, "events": 0}]
    assert [r["country"] for r in round_cap_country_years(rows, 1000)] == ["A"]
    assert round_cap_country_years(rows, 5000) == []


def test_a_failed_country_year_stops_the_fetch(monkeypatch) -> None:
    import src.ingestion.geopolitical_connector as mod

    monkeypatch.setattr(mod.GeopoliticalConnector, "_acled_config", lambda self: {})

    class _Stub:
        def __init__(self, _cfg) -> None: ...

        def fetch_all_events(self, country, year, page_size=5000, max_pages=200):
            raise ACLEDFetchError(f"ACLED {country}/{year}: boom")

    monkeypatch.setattr("src.extractors.acled_extractor.ACLEDExtractor", _Stub)
    connector = mod.GeopoliticalConnector(config={
        "data_mode": "api", "acled_countries": ["Yemen"], "api": {"start_year": 2023, "end_year": 2023}})
    with pytest.raises(ACLEDFetchError, match="Yemen/2023"):
        connector.fetch_api()


# --------------------------------------------------------------- review follow-ups
def test_a_failure_on_page_two_is_labelled_page_two(monkeypatch) -> None:
    class _Broken(dict):
        def get(self, key, default=None):
            if key == "data":
                raise ValueError("bad payload")
            return super().get(key, default)

    pages = [_page(0, 10, 5), _Broken(success=True, next_cursor=6)]
    with pytest.raises(ACLEDFetchError, match="page 2 "):
        _extractor(monkeypatch, pages, []).fetch_all_events("Yemen", 2023)


def test_overlap_is_caught_on_the_page_it_happens(monkeypatch) -> None:
    pages = [_page(0, 10, 5), _page(5, 10, 6)] + [_page(100 + i * 10, 10, 7 + i) for i in range(20)]
    calls: list[dict] = []
    with pytest.raises(ACLEDFetchError, match="duplicate"):
        _extractor(monkeypatch, pages, calls).fetch_all_events("Yemen", 2023)
    assert len(calls) == 2                       # stopped on page 2, not after 200 pages


def test_a_client_setup_failure_is_masked_too(monkeypatch) -> None:
    extractor = ACLEDExtractor({"api_keys": {"acled_username": USER, "acled_password": PASSWORD}})

    def boom():
        raise RuntimeError(f"token exchange failed for {USER} / {PASSWORD}")

    monkeypatch.setattr(extractor, "_get_client", boom)
    with pytest.raises(ACLEDFetchError) as caught:
        extractor.fetch_all_events("Yemen", 2023)
    assert USER not in str(caught.value) and PASSWORD not in str(caught.value)


def test_the_installed_acled_client_still_has_the_transport_we_use() -> None:
    """No network: the private pieces fetch_all_events relies on (acled 1.0.x)."""
    import inspect

    from acled.clients.acled_data_client import AcledDataClient
    from acled.clients.client import AcledClient

    assert callable(getattr(AcledDataClient, "_get", None))
    assert callable(getattr(AcledDataClient, "_parse_event", None))
    assert "self.endpoint" in inspect.getsource(AcledDataClient.__init__)
    assert "_acled_data_client" in inspect.getsource(AcledClient.__init__)


def test_the_page_size_setting_falls_back_to_its_old_name() -> None:
    import src.ingestion.geopolitical_connector as mod

    old = mod.GeopoliticalConnector(config={"api": {"events_per_year": 1234}})
    both = mod.GeopoliticalConnector(config={"api": {"page_size": 99, "events_per_year": 1234}})
    neither = mod.GeopoliticalConnector(config={})
    assert (old.api_page_size, both.api_page_size, neither.api_page_size) == (1234, 99, 5000)


def test_the_digest_covers_event_ids_only_and_a_field_revision_is_not_detected() -> None:
    """Documents the limit: ACLED can revise fields in place without changing ids."""
    from src.ingestion.geopolitical_connector import _events_digest

    a = [{"event_id_cnty": "A1", "event_date": "2024-01-01", "event_type": "Battles",
          "fatalities": 1}]
    assert _events_digest(a) == _events_digest([{**a[0], "fatalities": 9}])      # same id
    assert _events_digest(a) != _events_digest([{**a[0], "event_id_cnty": "A2"}])
    no_id = [{"event_date": "2024-01-01", "event_type": "Battles", "fatalities": 1}]
    assert _events_digest(no_id) != _events_digest([{**no_id[0], "fatalities": 2}])  # fields hashed
