"""Combined GDACS + USGS extractor with a shared monthly quota.

The two disaster sources are complementary (see
:mod:`src.extractors.usgs_extractor`) but their volumes are wildly different:
a single busy country-year can return hundreds of GDACS alerts, while a quiet
bounding box returns no earthquakes at all. Left unbounded, hazard documents
would swamp a knowledge base whose other sources contribute a few hundred
documents in total, and retrieval would return flood alerts for every query.

This wrapper bounds that: at most :attr:`DOCS_PER_MONTH` documents per region
per calendar month, across *both* sources. The cap is applied to the fetched
results rather than by issuing one request per month — the extractors already
window adaptively, and monthly requests would cost thousands of calls against
two free public services to return the same events.

Selection within a month is by severity, not by arrival order, so the cap
drops routine alerts and keeps the events a disruption model needs.
"""

from __future__ import annotations

import logging

from src.extractors.gdacs_extractor import GDACSExtractor
from src.extractors.usgs_extractor import USGSExtractor

logger = logging.getLogger(__name__)

#: Severity -> sort rank. Higher survives the monthly cap.
_SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}

#: Bucket for documents carrying no parseable event date. Kept separate rather
#: than folded into a real month, so undated documents cannot evict dated ones.
_UNDATED = (0, 0)


class DisasterCombinedExtractor:
    """Fetch GDACS + USGS for a region, capped per calendar month.

    Not a :class:`~src.extractors.base_extractor.BaseExtractor`: it owns no
    endpoint of its own and emits documents already normalized by the two
    extractors it wraps. Both keep their own ``source`` metadata, so the mix
    of sources stays visible in the knowledge base.
    """

    #: Combined ceiling per region per month.
    DOCS_PER_MONTH = 15

    def __init__(self, config: dict) -> None:
        """Args:
            config: Full application config, forwarded to both extractors.
                They read ``extraction.chokepoints``, ``extraction.gdacs``
                and ``extraction.usgs`` from it.
        """
        self.config = config
        self.gdacs = GDACSExtractor(config)
        self.usgs = USGSExtractor(config)

    def extract_historical(
        self, region: str, start_year: int | None = None,
        end_year: int | None = None, **kwargs,
    ) -> list[dict]:
        """Extract both sources for ``region`` and apply the monthly cap.

        Args:
            region: Chokepoint key.
            start_year: Passed through; defaults to the configured range.
            end_year: Passed through; defaults to the configured range.

        Returns:
            Documents from both sources, at most :attr:`DOCS_PER_MONTH` per
            calendar month.
        """
        gdacs_docs = self._safe_extract(self.gdacs, region, start_year, end_year)
        usgs_docs = self._safe_extract(self.usgs, region, start_year, end_year)

        combined = self._apply_monthly_cap(gdacs_docs + usgs_docs)
        logger.info(
            "[%s] gdacs=%d usgs=%d -> %d after the %d/month cap",
            region, len(gdacs_docs), len(usgs_docs), len(combined),
            self.DOCS_PER_MONTH,
        )
        return combined

    @staticmethod
    def _safe_extract(
        extractor, region: str, start_year: int | None, end_year: int | None
    ) -> list[dict]:
        """Run one extractor, logging rather than raising on failure.

        One source failing must not cost the other its results — that is the
        whole reason there are two.
        """
        try:
            return extractor.extract_historical(
                region, start_year=start_year, end_year=end_year
            )
        except Exception as exc:
            logger.error(
                "%s failed for region=%s: %s", type(extractor).__name__, region, exc
            )
            return []

    @classmethod
    def _month_key(cls, doc: dict) -> tuple[int, int]:
        """``(year, month)`` from a document's ``event_date``.

        Dates are the ISO ``YYYY-MM-DD`` written by ``_normalize_document``.
        Anything unparseable lands in :data:`_UNDATED`.
        """
        raw = str((doc.get("metadata") or {}).get("event_date") or "")
        try:
            return int(raw[:4]), int(raw[5:7])
        except ValueError:
            return _UNDATED

    @classmethod
    def _rank(cls, doc: dict) -> tuple[int, str]:
        """Sort key: severity first, then id so the selection is deterministic."""
        severity = str((doc.get("metadata") or {}).get("severity") or "low")
        return _SEVERITY_RANK.get(severity, 0), str(doc.get("id", ""))

    @classmethod
    def _apply_monthly_cap(cls, documents: list[dict]) -> list[dict]:
        """Keep the top :attr:`DOCS_PER_MONTH` documents of each month.

        Applied across both sources rather than to USGS alone: capping only
        the second source would leave the total unbounded whenever the first
        one is the noisy one, which is the usual case here.

        Args:
            documents: Documents from both sources, in any order.

        Returns:
            The capped set, ordered by month then severity.
        """
        by_month: dict[tuple[int, int], list[dict]] = {}
        for doc in documents:
            by_month.setdefault(cls._month_key(doc), []).append(doc)

        undated = len(by_month.get(_UNDATED, []))
        if undated:
            logger.warning(
                "%d document(s) had no parseable event_date and were capped "
                "as a single bucket.", undated,
            )

        kept: list[dict] = []
        for month in sorted(by_month):
            ranked = sorted(by_month[month], key=cls._rank, reverse=True)
            dropped = len(ranked) - cls.DOCS_PER_MONTH
            if dropped > 0:
                logger.debug(
                    "%04d-%02d: dropped %d of %d over the cap",
                    month[0], month[1], dropped, len(ranked),
                )
            kept.extend(ranked[: cls.DOCS_PER_MONTH])
        return kept
