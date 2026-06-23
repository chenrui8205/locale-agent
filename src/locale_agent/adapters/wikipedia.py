"""Wikipedia adapter (keyless MediaWiki API) — neighborhood/landmark context.

Geosearch for articles near the address, then pull intro extracts. Returns
`context` evidence (geolocated, but not "go here" options) the synthesizer can
use to describe an area. Fully keyless; requires only a descriptive User-Agent.
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

_LIMIT = 6
_GEOSEARCH_MAX_RADIUS_M = 10000  # MediaWiki gsradius hard cap


class WikipediaAdapter(SourceAdapter):
    name = "wikipedia"
    access_tier = AccessTier.OFFICIAL_API
    supported_categories = {"context"}

    async def search(
        self, spec: QuerySpec, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        if not await budget.allow(self.name):
            return []

        settings = get_settings()
        base = settings.wikipedia_base_url
        headers = {"User-Agent": settings.nominatim_user_agent}
        radius = min(geo.default_radius_m, _GEOSEARCH_MAX_RADIUS_M)
        now = datetime.now(UTC)

        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            geo_resp = await client.get(
                base,
                params={
                    "action": "query",
                    "list": "geosearch",
                    "gscoord": f"{geo.lat}|{geo.lng}",
                    "gsradius": str(radius),
                    "gslimit": str(_LIMIT),
                    "format": "json",
                    "formatversion": "2",
                },
            )
            geo_resp.raise_for_status()
            pages = geo_resp.json().get("query", {}).get("geosearch", [])
            if not pages:
                return []

            ids = "|".join(str(p["pageid"]) for p in pages)
            ex_resp = await client.get(
                base,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": "1",
                    "explaintext": "1",
                    "pageids": ids,
                    "format": "json",
                    "formatversion": "2",
                    "exlimit": "max",
                },
            )
            ex_resp.raise_for_status()
            extracts = {
                p["pageid"]: p.get("extract", "")
                for p in ex_resp.json().get("query", {}).get("pages", [])
            }

        results: list[SourceResult] = []
        for p in pages:
            pid = p["pageid"]
            snippet = (extracts.get(pid) or "").strip().replace("\n", " ")[:300]
            results.append(
                SourceResult(
                    source="wikipedia",
                    url=f"https://en.wikipedia.org/?curid={pid}",
                    title=p["title"],
                    snippet=snippet or "Wikipedia article",
                    kind="context",
                    geo=(p.get("lat"), p.get("lon")) if p.get("lat") is not None else None,
                    fetched_at=now,
                    raw={},
                )
            )
        log.info("wikipedia.search", radius_m=radius, results=len(results))
        return results


register(WikipediaAdapter())
