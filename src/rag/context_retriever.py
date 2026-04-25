"""ChromaDB-based historical disruption context retriever.

Embeds query signals using sentence-transformers and retrieves the most
semantically similar historical disruption cases from a local ChromaDB
collection, providing analysts with relevant precedents.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


class ContextRetriever:
    """Retrieve historically similar disruption cases for a given query.

    Uses a local persistent ChromaDB collection populated from JSON files
    in the knowledge base directory.  Embeddings are generated with a
    lightweight sentence-transformer model — no external API required.

    Args:
        config: RAG configuration block from settings.yaml (``collection_name``
            and ``top_k`` keys).
        persist_directory: Directory for ChromaDB persistence.
        embed_model_name: HuggingFace sentence-transformer model identifier.
    """

    def __init__(
        self,
        config: dict,
        persist_directory: str = "data/knowledge_base/.chromadb",
        embed_model_name: str = _DEFAULT_EMBED_MODEL,
    ) -> None:
        self.collection_name: str = config["collection_name"]
        self.top_k: int = config.get("top_k", 3)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embed_model = SentenceTransformer(embed_model_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ContextRetriever ready — collection='%s', top_k=%d",
            self.collection_name,
            self.top_k,
        )

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
        files = list(kb_path.glob("*.json"))
        if not files:
            logger.warning("No JSON files found in knowledge base: %s", kb_path)
            return 0

        existing_ids: set[str] = set(self._collection.get()["ids"])
        docs, embeddings, ids, metadatas = [], [], [], []

        for fp in files:
            doc_id = fp.stem
            if doc_id in existing_ids:
                continue
            case: dict[str, Any] = json.loads(fp.read_text(encoding="utf-8"))
            text: str = case["text"]
            docs.append(text)
            embeddings.append(self._embed_model.encode(text).tolist())
            ids.append(doc_id)
            metadatas.append(case.get("metadata", {}))

        if docs:
            self._collection.add(
                documents=docs,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            logger.info("Indexed %d new cases into '%s'.", len(docs), self.collection_name)

        return len(docs)

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
                - ``metadata`` (dict): Case metadata from the JSON file.
        """
        k = top_k if top_k is not None else self.top_k
        query_embedding = self._embed_model.encode(query_text).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
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
                {"id": doc_id, "document": doc, "distance": dist, "metadata": meta}
            )
        return output
