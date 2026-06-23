"""GDELT adapter (keyless) — recent local news / "what's happening".

Queries the GDELT 2.0 DOC API for recent articles mentioning the city plus the
query topics. Returns `article` evidence (not geo-ranked). Keyless; needs a city
name to anchor the query, so it no-ops (with a note) when geo has no city.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from ..config import get_settings
from ..logging import get_logger
from ..ratelimit import RateBudget
from ..schemas import GeoContext, QuerySpec, SourceResult
from .base import AccessTier, SourceAdapter, register

log = get_logger(__name__)

_LIMIT = 8


def _query_terms(spec: QuerySpec) -> str:
    parts: list[str] = []
    if spec.entity:
        parts.append(spec.entity)
    parts.extend(spec.topics or [])
    return " ".join(parts).strip()


class GdeltAdapter(SourceAdapter):
    name = "gdelt_news"
    access_tier = AccessTier.OFFICIAL_API
    supported_categories = {"news"}

    async def search(
        self, spec: QuerySpec, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        if not geo.city:
            budget.note("gdelt: no city resolved; skipped local news")
            return []
        if not await budget.allow(self.name):
            return []

        settings = get_settings()
        terms = _query_terms(spec)
        # GDELT query: quoted city + optional topic terms. Keep it short.
        query = f'"{geo.city}" {terms}'.strip()
        headers = {"User-Agent": settings.nominatim_user_agent}
        now = datetime.now(UTC)

        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            resp = await client.get(
                settings.gdelt_base_url,
                params={
                    "query": query,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": str(_LIMIT),
                    "sort": "DateDesc",
                    "timespan": "1w",
                },
            )
            resp.raise_for_status()
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001 — GDELT returns plain-text on bad queries
                budget.note("gdelt: non-JSON response; skipped")
                return []

        results: list[SourceResult] = []
        for art in payload.get("articles", []):
            title = (art.get("title") or "").strip()
            url = art.get("url")
            if not title or not url:
                continue
            domain = art.get("domain", "")
            seendate = art.get("seendate", "")
            results.append(
                SourceResult(
                    source="gdelt_news",
                    url=url,
                    title=title,
                    snippet=f"{domain} · {seendate}".strip(" ·"),
                    kind="article",
                    geo=None,
                    fetched_at=now,
                    raw={},
                )
            )
        log.info("gdelt.search", query=query, results=len(results))
        return results


register(GdeltAdapter())
