"""Wikipedia adapter (keyless MediaWiki API) — neighborhood/landmark context.

First hop (`search`): geosearch for articles near the address, then pull intro
extracts. Second hop (`search_text`): full-text `list=search` for a free-text
query (e.g. a place name), then the same intro extracts. Both return `context`
evidence (not "go here" options) the synthesizer can use to describe an area.
Fully keyless; requires only a descriptive User-Agent.

Caveat for the second hop: MediaWiki full-text search matches words, not
entities. Only landmarks and chains have articles; for an ordinary business
name it returns unrelated word-match hits rather than nothing, so re-planning
should prefer Reddit unless the place is plausibly notable.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime

import httpx

from ..config import get_settings
from ..logging import get_logger
from ..ratelimit import RateBudget
from ..schemas import GeoContext, QuerySpec, SourceResult
from .base import AccessTier, SourceAdapter, register

log = get_logger(__name__)

_LIMIT = 6
_FOLLOWUP_LIMIT = 5              # second-hop articles per follow-up query
_GEOSEARCH_MAX_RADIUS_M = 10000  # MediaWiki gsradius hard cap
_HTTP_TIMEOUT_S = 15.0
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_search_snippet(raw: str) -> str:
    """Strip MediaWiki's `<span class="searchmatch">` markup and HTML entities."""
    return html.unescape(_TAG_RE.sub("", raw)).replace("\n", " ").strip()


async def _fetch_extracts(client: httpx.AsyncClient, base: str, pageids: list[int]) -> dict[int, str]:
    if not pageids:
        return {}
    resp = await client.get(
        base,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "pageids": "|".join(str(pid) for pid in pageids),
            "format": "json",
            "formatversion": "2",
            "exlimit": "max",
        },
    )
    resp.raise_for_status()
    return {p["pageid"]: p.get("extract", "") for p in resp.json().get("query", {}).get("pages", [])}


def _to_result(page: dict, snippet: str, now: datetime) -> SourceResult:
    return SourceResult(
        source="wikipedia",
        url=f"https://en.wikipedia.org/?curid={page['pageid']}",
        title=page["title"],
        snippet=snippet[:300] or "Wikipedia article",
        kind="context",
        geo=(page.get("lat"), page.get("lon")) if page.get("lat") is not None else None,
        fetched_at=now,
        raw={},
    )


class WikipediaAdapter(SourceAdapter):
    name = "wikipedia"
    access_tier = AccessTier.OFFICIAL_API
    supported_categories = {"context"}

    def _client(self) -> httpx.AsyncClient:
        settings = get_settings()
        return httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT_S, headers={"User-Agent": settings.nominatim_user_agent}
        )

    async def search(
        self, spec: QuerySpec, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        if not await budget.allow(self.name):
            return []

        base = get_settings().wikipedia_base_url
        radius = min(geo.default_radius_m, _GEOSEARCH_MAX_RADIUS_M)
        now = datetime.now(UTC)

        async with self._client() as client:
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
            extracts = await _fetch_extracts(client, base, [p["pageid"] for p in pages])

        results = [
            _to_result(p, (extracts.get(p["pageid"]) or "").strip().replace("\n", " "), now)
            for p in pages
        ]
        log.info("wikipedia.search", radius_m=radius, results=len(results))
        return results

    async def search_text(
        self, query: str, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        """Second hop: MediaWiki full-text search for `query`, intro extracts as snippets.

        Degrades to [] + note on any HTTP error; never raises.
        """
        query = query.strip()
        if not query:
            budget.note("wikipedia: empty follow-up query; skipped")
            return []
        if not await budget.allow(self.name):
            return []

        base = get_settings().wikipedia_base_url
        now = datetime.now(UTC)
        try:
            async with self._client() as client:
                resp = await client.get(
                    base,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": str(_FOLLOWUP_LIMIT),
                        "format": "json",
                        "formatversion": "2",
                    },
                )
                resp.raise_for_status()
                hits = resp.json().get("query", {}).get("search", [])
                if not hits:
                    return []
                extracts = await _fetch_extracts(client, base, [h["pageid"] for h in hits])
        except httpx.HTTPError as e:
            budget.note(f"wikipedia: follow-up '{query}' failed: {e}")
            log.warning("wikipedia.search_text_error", query=query, error=str(e))
            return []
        except Exception as e:  # noqa: BLE001 — adapters never raise into the graph
            budget.note(f"wikipedia: follow-up '{query}' failed: {e}")
            log.warning("wikipedia.search_text_unexpected_error", query=query, error=str(e))
            return []

        results: list[SourceResult] = []
        for h in hits:
            snippet = (extracts.get(h["pageid"]) or "").strip().replace("\n", " ")
            if not snippet:
                snippet = _clean_search_snippet(h.get("snippet", ""))
            results.append(_to_result(h, snippet, now))
        log.info("wikipedia.search_text", query=query, results=len(results))
        return results


register(WikipediaAdapter())
