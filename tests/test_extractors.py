"""Tests for live API extractors and the knowledge base builder.

These tests mock all HTTP calls — no real API requests are made, so they
run safely in CI without API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import yaml

from src.extractors.acled_extractor import ACLEDExtractor
from src.extractors.ambee_extractor import AmbeeExtractor, EVENT_TYPE_TO_AGENT_FEATURE
from src.extractors.base_extractor import BaseExtractor, resolve_env_value
from src.extractors.fred_extractor import FREDExtractor
from src.extractors.knowledge_base_builder import KnowledgeBaseBuilder
from src.extractors.newsapi_extractor import NewsAPIExtractor
from src.extractors.reliefweb_extractor import ReliefWebExtractor
from src.extractors.serpapi_extractor import HISTORICAL_QUERIES, SerpAPIExtractor

CONFIG = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


class _MockExtractor(BaseExtractor):
    @property
    def source_name(self) -> str:
        return "mock"

    def extract_historical(self, region, **kwargs):
        return []


class TestBaseExtractor:
    def test_normalize_document_schema(self):
        extractor = _MockExtractor(CONFIG)
        doc = extractor._normalize_document(
            doc_id="test_123",
            text="Test disruption event in Strait of Hormuz",
            event_date="2024-01-15",
            region="hormuz",
            countries=["Iran", "Oman"],
            primary_agents=["shipping", "geopolitical"],
            event_type="conflict",
            severity="high",
        )
        assert doc["id"] == "mock_test_123"
        assert "Strait of Hormuz" in doc["text"]
        assert doc["metadata"]["source_api"] == "mock"
        assert doc["metadata"]["region"] == "hormuz"
        assert doc["metadata"]["severity"] == "high"
        assert "shipping" in doc["metadata"]["primary_agents"]

    def test_rate_limiting(self):
        import time

        config = {**CONFIG, "extraction": {"rate_limits": {"mock": 600}}}
        extractor = _MockExtractor(config)
        start = time.time()
        extractor._rate_limit_wait()
        extractor._rate_limit_wait()
        assert time.time() - start >= 0.05

    def test_resolve_env_value_placeholder(self, monkeypatch):
        monkeypatch.setenv("SOME_TEST_KEY", "secret123")
        assert resolve_env_value("${SOME_TEST_KEY}") == "secret123"

    def test_resolve_env_value_missing(self, monkeypatch):
        monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
        assert resolve_env_value("${MISSING_TEST_KEY}") == ""

    def test_resolve_env_value_plain_string(self):
        assert resolve_env_value("plain-value") == "plain-value"


class TestNewsAPIExtractor:
    @patch("src.extractors.newsapi_extractor.requests.get")
    def test_search_articles(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Tanker attacked in Strait of Hormuz",
                    "description": "Two oil tankers were attacked near Iran",
                    "content": "The attacks caused oil prices to spike...",
                    "source": {"name": "Reuters"},
                    "publishedAt": "2024-06-15T12:00:00Z",
                    "url": "https://example.com/article1",
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {**CONFIG, "api_keys": {"newsapi": "test_key"}}
        extractor = NewsAPIExtractor(config)
        docs = extractor.extract_historical("hormuz")

        assert len(docs) > 0
        assert docs[0]["metadata"]["source_api"] == "newsapi"
        assert docs[0]["metadata"]["region"] == "hormuz"
        assert "news_sentiment" in docs[0]["metadata"]["primary_agents"]

    def test_no_api_key_returns_empty(self):
        config = {**CONFIG, "api_keys": {"newsapi": ""}}
        extractor = NewsAPIExtractor(config)
        assert extractor.extract_historical("hormuz") == []


class TestReliefWebExtractor:
    """Test ReliefWeb extractor (kept as fallback; primary is now AmbeeExtractor)."""

    @patch("src.extractors.reliefweb_extractor.requests.post")
    def test_search_disasters(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": 12345,
                    "fields": {
                        "name": "Cyclone Gonu - Jun 2007",
                        "date": {"created": "2007-06-03T00:00:00+00:00"},
                        "type": [{"name": "Tropical Cyclone"}],
                        "country": [{"name": "Oman"}],
                        "status": "past",
                        "glide": "TC-2007-000073-OMN",
                        "description-html": "<p>Cyclone Gonu struck Oman...</p>",
                    },
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        extractor = ReliefWebExtractor(CONFIG)
        disasters = extractor._search_disasters(query="cyclone", country="Oman")

        assert len(disasters) == 1
        assert "Cyclone Gonu" in disasters[0]["name"]
        assert "Oman" in disasters[0]["countries"]


class TestFREDExtractor:
    @patch("src.extractors.fred_extractor.requests.get")
    def test_get_observations(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2019-06-01", "value": "62.50"},
                {"date": "2019-06-15", "value": "65.20"},
                {"date": "2019-06-20", "value": "72.80"},
                {"date": "2019-06-25", "value": "68.10"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {**CONFIG, "api_keys": {"fred": "test_key"}}
        extractor = FREDExtractor(config)
        obs = extractor._get_series_observations("DCOILBRENTEU", "2019-06-01", "2019-06-30")

        assert len(obs) == 4
        assert obs[0]["value"] == 62.50
        assert obs[2]["value"] == 72.80

    def test_disruption_metrics(self):
        config = {**CONFIG, "api_keys": {"fred": "test_key"}}
        extractor = FREDExtractor(config)
        observations = [{"date": f"2019-06-{i:02d}", "value": 65.0 + (i * 0.5)} for i in range(1, 31)]
        observations.append({"date": "2019-07-01", "value": 95.0})

        metrics = extractor._compute_disruption_metrics(observations)
        assert metrics["spike_pct"] > 0
        assert metrics["max_value"] == 95.0


class TestACLEDExtractor:
    def test_risk_profile_high(self):
        config = {**CONFIG, "api_keys": {"acled_username": "test", "acled_password": "test"}}
        extractor = ACLEDExtractor(config)
        events = [
            {"event_type": "Battles", "fatalities": 5},
            {"event_type": "Battles", "fatalities": 12},
            {"event_type": "Explosions/Remote violence", "fatalities": 3},
            {"event_type": "Violence against civilians", "fatalities": 2},
            {"event_type": "Strategic developments", "fatalities": 0},
        ] * 25  # 125 events total

        profile = extractor._compute_risk_profile(events)
        assert profile["risk_level"] == "high"
        assert profile["total_events"] == 125
        assert profile["military_events"] == 75

    def test_risk_profile_low(self):
        config = {**CONFIG, "api_keys": {"acled_username": "test", "acled_password": "test"}}
        extractor = ACLEDExtractor(config)
        events = [
            {"event_type": "Strategic developments", "fatalities": 0},
            {"event_type": "Strategic developments", "fatalities": 0},
        ]
        profile = extractor._compute_risk_profile(events)
        assert profile["risk_level"] == "low"

    def test_no_credentials_returns_empty(self):
        config = {**CONFIG, "api_keys": {"acled_username": "", "acled_password": ""}}
        extractor = ACLEDExtractor(config)
        assert extractor.extract_historical("hormuz") == []

    @patch("src.extractors.acled_extractor.ACLEDExtractor._get_client")
    def test_fetch_events_uses_acled_client(self, mock_get_client):
        """_fetch_events delegates to AcledClient.get_data (OAuth transport)."""
        mock_client = MagicMock()
        mock_client.get_data.return_value = [
            {"event_type": "Battles", "fatalities": 5},
            {"event_type": "Strategic developments", "fatalities": 0},
        ]
        mock_get_client.return_value = mock_client

        config = {**CONFIG, "api_keys": {"acled_username": "user@example.com", "acled_password": "pw"}}
        extractor = ACLEDExtractor(config)
        events = extractor._fetch_events("Iran", 2019, limit=50)

        assert len(events) == 2
        mock_client.get_data.assert_called_once_with(country="Iran", year=2019, limit=50)

    @patch("src.extractors.acled_extractor.ACLEDExtractor._get_client")
    def test_fetch_events_client_error_returns_empty(self, mock_get_client):
        mock_get_client.side_effect = RuntimeError("token exchange failed")
        config = {**CONFIG, "api_keys": {"acled_username": "user@example.com", "acled_password": "pw"}}
        extractor = ACLEDExtractor(config)
        assert extractor._fetch_events("Iran", 2019) == []


class TestAISStreamMonitor:
    def test_extract_historical_returns_empty(self):
        from src.extractors.aisstream_monitor import AISStreamMonitor

        config = {**CONFIG, "api_keys": {"aisstream": "test_key"}}
        monitor = AISStreamMonitor(config)
        assert monitor.extract_historical("hormuz") == []

    def test_compute_metrics_empty_state(self):
        from src.extractors.aisstream_monitor import AISStreamMonitor

        config = {**CONFIG, "api_keys": {"aisstream": "test_key"}}
        monitor = AISStreamMonitor(config)
        metrics = monitor.compute_current_metrics()
        assert metrics["vessel_count"] == 0
        assert metrics["congestion_index"] == 0.0


class TestAmbeeExtractor:
    """Test Ambee Disasters API extractor with mocked responses.

    Note: Ambee's ``/latest`` endpoint keys events under ``"result"``;
    ``/history`` keys them under ``"data"``. ``extract_historical`` tries
    history first, so a mock response with only a ``"result"`` key
    correctly exercises the fallback to ``/latest``.
    """

    @staticmethod
    def _config_with_points(points: dict) -> dict:
        return {
            **CONFIG,
            "agents": {
                **CONFIG["agents"],
                "natural_disaster": {
                    **CONFIG["agents"]["natural_disaster"],
                    "api": {**CONFIG["agents"]["natural_disaster"]["api"], "ambee_api_key": "test_key"},
                    "monitoring_points": points,
                },
            },
        }

    @patch("src.extractors.ambee_extractor.requests.get")
    def test_fetch_disasters_falls_back_to_latest(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "event_name": "Cyclone Shaheen",
                    "event_type": "CY",
                    "date": "2021-10-03",
                    "lat": 23.61,
                    "lng": 57.58,
                    "proximity_severity_level": "High Risk",
                    "default_alert_levels": "Red",
                },
                {
                    "event_name": "Minor Tremor",
                    "event_type": "EQ",
                    "date": "2023-05-15",
                    "lat": 26.0,
                    "lng": 56.0,
                    "proximity_severity_level": "Low",
                    "default_alert_levels": "Green",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = self._config_with_points({"hormuz": [{"name": "Test Point", "lat": 26.5, "lng": 56.2}]})
        extractor = AmbeeExtractor(config)
        docs = extractor.extract_historical("hormuz")

        assert len(docs) == 2
        assert mock_get.call_count == 2  # history attempt, then latest fallback
        assert docs[0]["metadata"]["source_api"] == "ambee"
        assert docs[0]["metadata"]["region"] == "hormuz"
        assert "natural_disaster" in docs[0]["metadata"]["primary_agents"]

    def test_severity_computation(self):
        config = self._config_with_points({})
        extractor = AmbeeExtractor(config)

        event_high = {"proximity_severity_level": "High Risk", "default_alert_levels": "Red"}
        assert abs(extractor._compute_severity(event_high) - 0.87) < 0.01

        event_low = {"proximity_severity_level": "Low", "default_alert_levels": "Green"}
        assert abs(extractor._compute_severity(event_low) - 0.16) < 0.01

        event_mid = {"proximity_severity_level": "Moderate", "default_alert_levels": "Yellow"}
        assert abs(extractor._compute_severity(event_mid) - 0.44) < 0.01

    def test_severity_classification(self):
        config = self._config_with_points({})
        extractor = AmbeeExtractor(config)
        assert extractor._classify_severity(0.87) == "high"
        assert extractor._classify_severity(0.44) == "medium"
        assert extractor._classify_severity(0.16) == "low"

    def test_event_type_mapping(self):
        assert EVENT_TYPE_TO_AGENT_FEATURE["EQ"] == "earthquake_severity"
        assert EVENT_TYPE_TO_AGENT_FEATURE["CY"] == "cyclone_severity"
        assert EVENT_TYPE_TO_AGENT_FEATURE["FL"] == "severe_weather_index"
        assert EVENT_TYPE_TO_AGENT_FEATURE["SW"] == "severe_weather_index"

    @patch("src.extractors.ambee_extractor.requests.get")
    def test_deduplication_across_points(self, mock_get):
        same_event = {
            "event_name": "Regional Earthquake",
            "event_type": "EQ",
            "date": "2024-01-15",
            "lat": 26.5,
            "lng": 56.2,
            "proximity_severity_level": "Moderate",
            "default_alert_levels": "Yellow",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": [same_event]}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = self._config_with_points({
            "hormuz": [
                {"name": "Point A", "lat": 26.5, "lng": 56.2},
                {"name": "Point B", "lat": 26.6, "lng": 56.3},
            ]
        })
        extractor = AmbeeExtractor(config)
        docs = extractor.extract_historical("hormuz")

        # Same event from two points -> should appear only once.
        assert len(docs) == 1

    def test_no_api_key_returns_empty(self):
        config = self._config_with_points({"hormuz": [{"name": "P", "lat": 26.5, "lng": 56.2}]})
        config["agents"]["natural_disaster"]["api"]["ambee_api_key"] = ""
        extractor = AmbeeExtractor(config)
        assert extractor.extract_historical("hormuz") == []


class TestSerpAPIExtractor:
    """Test SerpAPI historical Google News extractor with mocked responses.

    Real response shape (verified against the live API): items carry
    ``title``/``link``/``source``/``date``/``iso_date`` with no ``snippet``
    field in practice; some items group coverage under a nested ``stories``
    list instead of a flat ``link``.
    """

    @patch("src.extractors.serpapi_extractor.requests.get")
    def test_search_google_news(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "news_results": [
                {
                    "title": "Iran tanker attack raises tensions near Hormuz",
                    "link": "https://example.com/article1",
                    "source": {"name": "Reuters"},
                    "date": "06/20/2019, 07:00 AM, +0000 UTC",
                    "iso_date": "2019-06-20T07:00:00Z",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {
            **CONFIG,
            "api_keys": {**CONFIG["api_keys"], "serpapi": "test_key"},
            "extraction": {**CONFIG["extraction"], "rate_limits": {**CONFIG["extraction"]["rate_limits"], "serpapi": 6000}},
        }
        extractor = SerpAPIExtractor(config)
        docs = extractor.extract_historical("hormuz")

        assert docs
        assert docs[0]["metadata"]["source_api"] == "serpapi"
        assert docs[0]["metadata"]["region"] == "hormuz"
        assert docs[0]["metadata"]["severity"] == "high"
        assert docs[0]["metadata"]["event_date"] == "2019-06-20"

    @patch("src.extractors.serpapi_extractor.requests.get")
    def test_grouped_stories_handling(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "news_results": [
                {
                    "title": "Top stories",
                    "stories": [
                        {
                            "title": "Story A about Hormuz tankers",
                            "link": "https://example.com/a",
                            "source": {"name": "AP"},
                            "iso_date": "2019-06-21T00:00:00Z",
                        },
                        {
                            "title": "Story B about Hormuz tankers",
                            "link": "https://example.com/b",
                            "source": {"name": "BBC"},
                            "iso_date": "2019-06-22T00:00:00Z",
                        },
                    ],
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {
            **CONFIG,
            "api_keys": {**CONFIG["api_keys"], "serpapi": "test_key"},
            "extraction": {**CONFIG["extraction"], "rate_limits": {**CONFIG["extraction"]["rate_limits"], "serpapi": 6000}},
        }
        extractor = SerpAPIExtractor(config)
        docs = extractor.extract_historical("hormuz")

        # 6 hormuz-region cases x 2 queries each = 12 calls hitting the same
        # mock, each returning 2 flattened stories -> 24 docs total.
        hormuz_queries = sum(
            len(c["queries"]) for c in HISTORICAL_QUERIES if c["region"] == "hormuz"
        )
        assert len(docs) == hormuz_queries * 2
        titles = {d["metadata"]["title"] for d in docs}
        assert "Story A about Hormuz tankers" in titles
        assert "Story B about Hormuz tankers" in titles

    @patch("src.extractors.serpapi_extractor.requests.get")
    def test_extract_all_cases(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "news_results": [
                {"title": "Some historical article", "link": "https://example.com/x",
                 "source": {"name": "Reuters"}, "iso_date": "2020-01-01T00:00:00Z"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {
            **CONFIG,
            "api_keys": {**CONFIG["api_keys"], "serpapi": "test_key"},
            "extraction": {**CONFIG["extraction"], "rate_limits": {**CONFIG["extraction"]["rate_limits"], "serpapi": 6000}},
        }
        extractor = SerpAPIExtractor(config)
        docs = extractor.extract_all_cases()

        total_queries = sum(len(case["queries"]) for case in HISTORICAL_QUERIES)
        assert total_queries == 20
        assert mock_get.call_count == total_queries
        assert len(docs) == total_queries  # 1 result per query in this mock

    def test_no_key_returns_empty(self):
        config = {**CONFIG, "api_keys": {**CONFIG["api_keys"], "serpapi": ""}}
        extractor = SerpAPIExtractor(config)
        assert extractor.extract_historical("hormuz") == []
        assert extractor.extract_all_cases() == []

    def test_historical_queries_cover_all_cases(self):
        assert len(HISTORICAL_QUERIES) == 10
        for case in HISTORICAL_QUERIES:
            assert len(case["queries"]) >= 2

    def test_all_regions_covered(self):
        regions = {case["region"] for case in HISTORICAL_QUERIES}
        assert regions == {"hormuz", "red_sea", "suez", "malacca"}

    def test_all_agent_domains_covered(self):
        all_agents: set[str] = set()
        for case in HISTORICAL_QUERIES:
            all_agents.update(case["primary_agents"])
        assert all_agents == {
            "shipping", "market", "geopolitical", "natural_disaster", "routing", "news_sentiment",
        }


class TestKnowledgeBaseBuilder:
    def test_deduplication(self):
        builder = KnowledgeBaseBuilder({**CONFIG, "extraction": {**CONFIG["extraction"], "enabled_extractors": []}})
        docs = [
            {"id": "doc_1", "text": "First", "metadata": {}},
            {"id": "doc_2", "text": "Second", "metadata": {}},
            {"id": "doc_1", "text": "Duplicate", "metadata": {}},
            {"id": "doc_3", "text": "Third", "metadata": {}},
            {"id": "doc_2", "text": "Another dup", "metadata": {}},
        ]
        unique = builder._deduplicate(docs)
        assert len(unique) == 3
        assert [d["id"] for d in unique] == ["doc_1", "doc_2", "doc_3"]


class TestRAGCompositeThreshold:
    def test_below_threshold_returns_none(self):
        from src.rag.context_retriever import ContextRetriever

        rag_cfg = {**CONFIG["rag"], "collection_name": "test_threshold_below"}
        retriever = ContextRetriever(rag_cfg, persist_directory="data/knowledge_base/.chromadb_test_below")
        result = retriever.query_gated({"shipping": 0.9}, composite_risk_score=0.10)
        assert result is None

    def test_above_threshold_triggers(self):
        from src.rag.context_retriever import ContextRetriever

        rag_cfg = {**CONFIG["rag"], "collection_name": "test_threshold_above"}
        retriever = ContextRetriever(rag_cfg, persist_directory="data/knowledge_base/.chromadb_test_above")
        retriever.build_index("data/knowledge_base/disruption_cases.json")
        result = retriever.query_gated({"shipping": 0.9, "geopolitical": 0.85}, composite_risk_score=0.80)
        assert result is not None
        assert result["triggered"] is True
        assert result["composite_score"] == 0.80
