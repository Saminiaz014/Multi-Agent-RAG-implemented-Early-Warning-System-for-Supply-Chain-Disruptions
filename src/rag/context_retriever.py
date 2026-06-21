"""ChromaDB-based historical disruption context retriever.

Embeds query signals using ChromaDB's built-in DefaultEmbeddingFunction
(ONNX-backed all-MiniLM-L6-v2) and retrieves the most semantically similar
historical disruption cases from a local ChromaDB collection.

Two ingestion paths:

* :meth:`load_knowledge_base` — legacy path, loads from individual
  ``*.json`` files (``{"text": ..., "metadata": {...}}`` per file).

* :meth:`build_index` — primary path for Phase 2F.3+; reads the single
  ``data/knowledge_base/disruption_cases.json`` array and rebuilds the
  collection whenever the case count changes.

Two query paths:

* :meth:`retrieve` — low-level; accepts a raw query string.

* :meth:`query` — high-level; accepts a 6-domain signal profile dict
  ``{agent_name: score}`` and builds the query string automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

# Anomaly score threshold above which a domain is considered "active".
_SIGNAL_BASELINE: float = 0.40

# Natural-language description templates for each agent domain.
_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "shipping": "high shipping disruption with vessel count reduction and transit delays",
    "market": "elevated market stress with oil price spike and freight rate increase",
    "geopolitical": "elevated geopolitical tension with sanctions, military activity, or diplomatic incidents",
    "natural_disaster": "natural disaster impacting port infrastructure and regional maritime access",
    "routing": "significant vessel rerouting, route deviation, and alternative corridor congestion",
    "news_sentiment": "negative news sentiment and high media volume about the Strait of Hormuz region",
}

# Fields required on every disruption case.
_REQUIRED_CASE_FIELDS: frozenset[str] = frozenset(
    [
        "id",
        "event",
        "date",
        "region",
        "description",
        "features",
        "impact",
        "duration_days",
        "recovery_days",
        "primary_agents",
        "lessons",
    ]
)


class ContextRetriever:
    """Retrieve historically similar disruption cases for a given query.

    Uses a local persistent ChromaDB collection populated from JSON files
    in the knowledge base directory.  Embeddings are generated with
    ChromaDB's built-in ONNX-backed ``DefaultEmbeddingFunction``
    (all-MiniLM-L6-v2) — no external API or HuggingFace download required
    after the first-use ONNX cache is populated.

    Args:
        config: RAG configuration block from settings.yaml (``collection_name``
            and ``top_k`` keys).
        persist_directory: Directory for ChromaDB persistence.
    """

    def __init__(
        self,
        config: dict,
        persist_directory: str = "data/knowledge_base/.chromadb",
    ) -> None:
        self.config = config
        self.collection_name: str = config["collection_name"]
        self.top_k: int = config.get("top_k", 3)

        # Composite-threshold-gated lookup settings (Phase: live extraction).
        self.composite_threshold: float = float(config.get("composite_threshold", 0.65))
        self.min_similarity: float = float(config.get("min_similarity", 0.55))
        self._live_collection_name: str = (
            config.get("collections", {}).get("live_context", "live_extracted_context")
        )

        self._ef = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._live_collection = self._client.get_or_create_collection(
            name=self._live_collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ContextRetriever ready — collection='%s', top_k=%d",
            self.collection_name,
            self.top_k,
        )

    # ------------------------------------------------------------------
    # Legacy ingestion path (individual *.json files)
    # ------------------------------------------------------------------

    def load_knowledge_base(self, kb_directory: str) -> int:
        """Index all JSON disruption case files into the ChromaDB collection.

        Each JSON file must contain a ``"text"`` key (narrative description)
        and an optional ``"metadata"`` dict.  Already-indexed IDs are skipped
        to support incremental updates.

        Args:
            kb_directory: Path to the directory containing ``*.json`` case files.

        Returns:
            Number of new documents added.
        """
        kb_path = Path(kb_directory)
        files = [f for f in kb_path.glob("*.json") if f.name != "disruption_cases.json"]
        if not files:
            logger.warning("No individual JSON files found in knowledge base: %s", kb_path)
            return 0

        existing_ids: set[str] = set(self._collection.get()["ids"])
        docs, ids, metadatas = [], [], []

        for fp in files:
            doc_id = fp.stem
            if doc_id in existing_ids:
                continue
            case: dict[str, Any] = json.loads(fp.read_text(encoding="utf-8"))
            docs.append(case["text"])
            ids.append(doc_id)
            metadatas.append(case.get("metadata", {}))

        if docs:
            self._collection.add(documents=docs, ids=ids, metadatas=metadatas)
            logger.info("Indexed %d new cases into '%s'.", len(docs), self.collection_name)

        return len(docs)

    # ------------------------------------------------------------------
    # Primary ingestion path (single disruption_cases.json array)
    # ------------------------------------------------------------------

    def build_index(self, kb_json_path: str = "data/knowledge_base/disruption_cases.json") -> int:
        """Index cases from a single JSON file, rebuilding when count changes.

        Reads the disruption-case array from ``kb_json_path``.  If the
        collection already holds exactly the same number of documents, the
        index is considered fresh and nothing is re-embedded (fast path).
        When the counts differ (new cases added, or collection empty) the
        entire collection is cleared and rebuilt from scratch.

        Args:
            kb_json_path: Path to the JSON array of disruption cases.

        Returns:
            Number of documents added (0 when already up-to-date).
        """
        path = Path(kb_json_path)
        if not path.exists():
            logger.warning("[ContextRetriever.build_index] File not found: %s", path)
            return 0

        cases: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        json_count = len(cases)
        collection_count = self._collection.count()

        if collection_count == json_count and json_count > 0:
            logger.info(
                "[ContextRetriever.build_index] Index up-to-date (%d cases).",
                json_count,
            )
            return 0

        # Clear stale entries before rebuilding.
        if collection_count > 0:
            existing_ids = self._collection.get()["ids"]
            if existing_ids:
                self._collection.delete(ids=existing_ids)
            logger.info(
                "[ContextRetriever.build_index] Cleared %d stale entries.",
                collection_count,
            )

        docs, ids, metadatas = [], [], []
        for case in cases:
            docs.append(self._case_to_text(case))
            ids.append(case["id"])
            metadatas.append(self._case_to_metadata(case))

        if docs:
            # ChromaDB auto-embeds via DefaultEmbeddingFunction.
            self._collection.add(documents=docs, ids=ids, metadatas=metadatas)
            logger.info(
                "[ContextRetriever.build_index] Indexed %d case(s) into '%s'.",
                len(docs),
                self.collection_name,
            )

        return len(docs)

    # ------------------------------------------------------------------
    # Low-level query (raw text)
    # ------------------------------------------------------------------

    def retrieve(self, query_text: str, top_k: int | None = None) -> list[dict]:
        """Retrieve the most similar historical cases for a query string.

        Args:
            query_text: Natural-language description of the current signal
                state, e.g. "Vessel traffic down 40%% near Hormuz, oil +12%%".
            top_k: Override the default number of results to return.

        Returns:
            List of result dicts, each with keys:
                - ``id`` (str): Case identifier.
                - ``document`` (str): Original case text.
                - ``distance`` (float): Cosine distance (lower = more similar).
                - ``similarity`` (float): ``1 - distance`` (higher = more similar).
                - ``metadata`` (dict): Case metadata from the JSON file.
        """
        k = top_k if top_k is not None else self.top_k
        if self._collection.count() == 0:
            logger.warning(
                "[ContextRetriever.retrieve] Collection is empty — call "
                "build_index() or load_knowledge_base() first."
            )
            return []
        k = min(k, self._collection.count())
        # ChromaDB auto-embeds the query_texts via DefaultEmbeddingFunction.
        results = self._collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        output = []
        for doc, dist, meta, doc_id in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["ids"][0],
        ):
            output.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "distance": float(dist),
                    "similarity": round(1.0 - float(dist), 4),
                    "metadata": meta,
                }
            )
        return output

    # ------------------------------------------------------------------
    # High-level query (6-domain signal profile)
    # ------------------------------------------------------------------

    def query(
        self,
        current_signals: dict[str, float],
        top_k: int | None = None,
    ) -> list[dict]:
        """Retrieve cases matching a 6-agent signal profile.

        Builds a natural-language query string from the active signal
        domains (those whose anomaly score exceeds the baseline threshold)
        and delegates to :meth:`retrieve`.

        Args:
            current_signals: Mapping of agent name → anomaly score in
                ``[0, 1]``.  Keys may be any subset of the six agent domains;
                absent domains are treated as 0.0 (below baseline).
            top_k: Override default number of results.

        Returns:
            Same structure as :meth:`retrieve`.
        """
        query_text = self._build_query_string(current_signals)
        logger.debug("[ContextRetriever.query] query='%s'", query_text)
        return self.retrieve(query_text, top_k=top_k)

    # ------------------------------------------------------------------
    # Dual-collection, composite-threshold-gated query (live extraction)
    # ------------------------------------------------------------------

    def build_both_indexes(
        self, kb_json_path: str = "data/knowledge_base/disruption_cases.json"
    ) -> dict[str, int]:
        """Ensure both the static and live ChromaDB collections are populated.

        Args:
            kb_json_path: Path to the static disruption-cases JSON array.

        Returns:
            ``{"static_cases": <count>, "live_context": <count>}`` document
            counts currently held in each collection.
        """
        self.build_index(kb_json_path)
        return {
            "static_cases": self._collection.count(),
            "live_context": self._live_collection.count(),
        }

    def query_gated(
        self,
        current_signals: dict[str, float],
        composite_risk_score: float,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> dict | None:
        """Retrieve historical precedents only when risk clears the threshold.

        Queries both the static ``disruption_cases`` collection and the live
        ``live_extracted_context`` collection (populated by
        :class:`~src.extractors.knowledge_base_builder.KnowledgeBaseBuilder`),
        merges results by similarity, and filters out anything below
        ``min_similarity``.

        Args:
            current_signals: Mapping of agent name -> anomaly score in
                ``[0, 1]``.
            composite_risk_score: The aggregated composite risk score for the
                current pipeline run.
            top_k: Override the default number of merged results to return.
            min_similarity: Override the default minimum cosine similarity.

        Returns:
            ``None`` when ``composite_risk_score`` is below
            ``self.composite_threshold``. Otherwise a dict::

                {
                    "triggered": True,
                    "composite_score": float,
                    "threshold": float,
                    "matches": [{"source": "static"|"live", "text": str,
                                  "similarity": float, "metadata": dict}, ...],
                    "formatted_summary": str,
                }
        """
        if composite_risk_score < self.composite_threshold:
            logger.debug(
                "[ContextRetriever.query_gated] not triggered: composite=%.3f < threshold=%.3f",
                composite_risk_score,
                self.composite_threshold,
            )
            return None

        k = top_k if top_k is not None else self.top_k
        sim_floor = min_similarity if min_similarity is not None else self.min_similarity
        query_text = self._build_query_string(current_signals)

        matches: list[dict] = []
        for source, collection in (("static", self._collection), ("live", self._live_collection)):
            if collection.count() == 0:
                continue
            n = min(k, collection.count())
            results = collection.query(
                query_texts=[query_text],
                n_results=n,
                include=["documents", "distances", "metadatas"],
            )
            for doc, dist, meta in zip(
                results["documents"][0], results["distances"][0], results["metadatas"][0]
            ):
                similarity = round(1.0 - float(dist), 4)
                if similarity < sim_floor:
                    continue
                matches.append(
                    {"source": source, "text": doc, "similarity": similarity, "metadata": meta}
                )

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        matches = matches[:k]

        return {
            "triggered": True,
            "composite_score": composite_risk_score,
            "threshold": self.composite_threshold,
            "matches": matches,
            "formatted_summary": self._format_gated_matches(matches),
        }

    @staticmethod
    def _format_gated_matches(matches: list[dict]) -> str:
        """Readable summary of :meth:`query_gated` matches for dashboard display."""
        if not matches:
            return "No historical precedents found above the similarity threshold."
        lines = ["Historical Precedents:"]
        for i, m in enumerate(matches, 1):
            meta = m.get("metadata", {})
            event = meta.get("event") or meta.get("disaster_name") or meta.get("title") or "Untitled event"
            date = meta.get("date") or meta.get("event_date") or ""
            preview = (m.get("text") or "")[:200].replace("\n", " ")
            lines.append(
                f"{i}. [{m['source']}] [{date}] {event} (similarity: {m['similarity']:.2f})\n   {preview}..."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_context(self, results: list[dict]) -> str:
        """Format retrieval results as a readable multi-line string.

        Args:
            results: Output of :meth:`query` or :meth:`retrieve`.

        Returns:
            Human-readable precedent summary with event names and similarity
            scores, suitable for inclusion in API responses or logging.
        """
        if not results:
            return "No historical precedents found."

        lines = ["Historical Precedents:"]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            event = meta.get("event", r["id"])
            date = meta.get("date", "")
            sim = r.get("similarity", 0.0)

            agents_raw = meta.get("primary_agents", "[]")
            try:
                agents: list[str] = (
                    json.loads(agents_raw)
                    if isinstance(agents_raw, str)
                    else agents_raw
                )
            except (json.JSONDecodeError, TypeError):
                agents = []
            agents_str = ", ".join(agents) if agents else "unknown"

            doc_preview = (r.get("document") or "")[:220].replace("\n", " ")
            lines.append(
                f"{i}. [{date}] {event} (similarity: {sim:.2f})"
                f" [Domains: {agents_str}]\n   {doc_preview}..."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _case_to_text(case: dict) -> str:
        """Build the embedding text from a disruption-case dict."""
        agents = ", ".join(case.get("primary_agents", []))
        feats = case.get("features", {})
        geo_risk = feats.get("geopolitical_risk_level", "unknown")
        disaster = "with natural disaster involvement" if feats.get("natural_disaster_involved") else ""
        return (
            f"{case.get('event', '')} ({case.get('date', '')}). "
            f"Region: {case.get('region', '')}. "
            f"{case.get('description', '')} "
            f"Impact: {case.get('impact', '')} "
            f"Geopolitical risk: {geo_risk}. {disaster} "
            f"Duration: {case.get('duration_days', 0)} days. "
            f"Primary signal domains: {agents}. "
            f"Key lesson: {case.get('lessons', '')}"
        ).strip()

    @staticmethod
    def _case_to_metadata(case: dict) -> dict:
        """Extract ChromaDB-safe metadata from a disruption-case dict."""
        feats = case.get("features", {})
        return {
            "event": str(case.get("event", "")),
            "date": str(case.get("date", "")),
            "region": str(case.get("region", "")),
            # ChromaDB metadata values must be str/int/float/bool — serialise list as JSON.
            "primary_agents": json.dumps(case.get("primary_agents", [])),
            "duration_days": int(case.get("duration_days", 0)),
            "recovery_days": int(case.get("recovery_days", 0)),
            "geopolitical_risk_level": str(feats.get("geopolitical_risk_level", "low")),
            "natural_disaster_involved": bool(feats.get("natural_disaster_involved", False)),
            "vessel_count_drop_pct": float(feats.get("vessel_count_drop_pct", 0.0)),
            "oil_price_spike_pct": float(feats.get("oil_price_spike_pct", 0.0)),
        }

    @staticmethod
    def _build_query_string(signals: dict[str, float]) -> str:
        """Build a natural-language query from a 6-domain signal dict.

        Only domains whose score exceeds :data:`_SIGNAL_BASELINE` are
        included in the query.  When no domain is active, a generic
        normal-conditions string is returned.
        """
        parts: list[str] = []
        for domain, description in _DOMAIN_DESCRIPTIONS.items():
            if float(signals.get(domain, 0.0)) > _SIGNAL_BASELINE:
                parts.append(description)

        if not parts:
            return (
                "Normal shipping and market conditions near the Strait of Hormuz "
                "with no elevated disruption signals."
            )
        return "Supply chain disruption signals: " + "; ".join(parts) + "."
