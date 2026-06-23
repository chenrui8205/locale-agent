"""Reddit adapter — sources local community discussion via Apify.

Reddit's official Data API is gated behind partner approval, so we pull posts
through Apify's `reddit-scraper-lite` Actor: Apify runs the scrape on its own
infrastructure and returns structured JSON; our code only calls its REST API
with a token. Results are `discussion` evidence (not rankable places). No-ops
gracefully (empty + note) when no APIFY_TOKEN is configured.

Project decision: this routes around Reddit's official API via a commercial
scraping vendor. Keep volume low and persist nothing — `SourceResult.raw` stays
empty per the no-hoarding rule.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..apify import ApifyError, run_actor_sync
from ..config import get_settings
from ..logging import get_logger
from ..ratelimit import RateBudget
from ..schemas import GeoContext, QuerySpec, SourceResult
from .base import AccessTier, SourceAdapter, register

log = get_logger(__name__)

_LIMIT = 8


def _city_slug(city: str) -> str:
    return "".join(ch for ch in city.lower() if ch.isalnum())


def _search_terms(spec: QuerySpec) -> str:
    parts: list[str] = []
    if spec.entity:
        parts.append(spec.entity)
    parts.extend(spec.topics or [])
    if not parts:
        parts.append(spec.raw_query)
    return " ".join(parts).strip()[:200]


def _to_result(item: dict, now: datetime) -> SourceResult | None:
    title = (item.get("title") or "").strip()
    url = item.get("url") or item.get("link")
    if not title or not url:
        return None
    body = (item.get("body") or "").strip().replace("\n", " ")
    community = item.get("communityName") or item.get("parsedCommunityName") or ""
    if body:
        snippet = body[:300]
    else:
        bits = [b for b in (
            community,
            f"{item['upVotes']} upvotes" if item.get("upVotes") is not None else "",
            f"{item['numberOfComments']} comments" if item.get("numberOfComments") is not None else "",
        ) if b]
        snippet = " · ".join(bits) or "Reddit post"
    return SourceResult(
        source="reddit",
        url=str(url),
        title=title,
        snippet=snippet,
        kind="discussion",
        geo=None,
        fetched_at=now,
        raw={},  # ephemeral; never persist post content
    )


class RedditAdapter(SourceAdapter):
    name = "reddit"
    access_tier = AccessTier.OFFICIAL_API  # sourced via the Apify vendor
    supported_categories = {"community", "discussion", "place", "service", "listing"}

    async def search(
        self, spec: QuerySpec, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        settings = get_settings()
        if not settings.has_apify:
            budget.note("reddit: no APIFY_TOKEN configured; skipped")
            return []
        if not await budget.allow(self.name):
            return []

        terms = _search_terms(spec)
        query = f"{geo.city} {terms}".strip() if geo.city else terms
        run_input = {
            "searches": [query],
            "searchPosts": True,
            "searchComments": False,
            "searchCommunities": False,
            "searchUsers": False,
            "sort": "relevance",
            "time": "year",
            "maxItems": _LIMIT,
        }
        now = datetime.now(UTC)
        try:
            items = await run_actor_sync(
                settings.apify_reddit_actor,
                run_input,
                token=settings.apify_token,
                base_url=settings.apify_base_url,
                timeout_s=settings.apify_timeout_s,
            )
        except ApifyError as e:
            budget.note(f"reddit (apify): {e}")
            log.warning("reddit.apify_error", error=str(e))
            return []

        results: list[SourceResult] = []
        seen: set[str] = set()
        for item in items:
            r = _to_result(item, now)
            if r is None or r.url in seen:
                continue
            seen.add(str(r.url))
            results.append(r)
        log.info("reddit.search", query=query, results=len(results))
        return results


register(RedditAdapter())
