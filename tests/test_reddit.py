"""Reddit adapter (Apify-backed) — no-op without token + parsing of mocked Apify items."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import respx

from locale_agent.adapters.reddit import RedditAdapter, _city_slug, _to_result
from locale_agent.config import Settings
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import GeoContext, QueryArchetype, QuerySpec

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=12000)
_SPEC = QuerySpec(
    archetype=QueryArchetype.FIND_COMMUNITY, raw_query="tennis partner", address="x", entity="tennis partner"
)

_ITEMS = [
    {
        "dataType": "post", "title": "Looking for a tennis partner in SJ",
        "url": "https://www.reddit.com/r/SanJose/comments/abc",
        "body": "3.5 level, weekday evenings", "communityName": "SanJose",
        "upVotes": 12, "numberOfComments": 5,
    },
    {
        "dataType": "post", "title": "Best courts near downtown?",
        "url": "https://www.reddit.com/r/SanJose/comments/def",
        "body": "", "communityName": "SanJose", "upVotes": 3, "numberOfComments": 8,
    },
    {"dataType": "post", "title": "no url here", "body": "x"},  # dropped (no url)
]

_ENDPOINT = "https://api.apify.com/v2/acts/practicaltools~apify-reddit-api/run-sync-get-dataset-items"


def test_city_slug() -> None:
    assert _city_slug("San Jose") == "sanjose"


def test_to_result_prefers_body_then_metadata() -> None:
    now = datetime.now(UTC)
    r1 = _to_result(_ITEMS[0], now)
    assert r1 is not None and r1.kind == "discussion" and "3.5 level" in r1.snippet
    r2 = _to_result(_ITEMS[1], now)
    assert r2 is not None and "comments" in r2.snippet  # empty body → metadata snippet
    assert _to_result(_ITEMS[2], now) is None  # no url


async def test_reddit_noop_without_token(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token=""))
    budget = RateBudget(None, cost_cap=12)
    results = await RedditAdapter().search(_SPEC, _GEO, budget)
    assert results == []
    assert any("APIFY_TOKEN" in n for n in budget.notes)
    assert budget.calls_made == 0  # never consumed a call slot


@respx.mock
async def test_reddit_parses_apify_items(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(
        rmod, "get_settings",
        lambda: Settings(apify_token="tok", apify_reddit_actor="practicaltools/apify-reddit-api"),
    )
    route = respx.post(url__startswith=_ENDPOINT).mock(return_value=httpx.Response(200, json=_ITEMS))
    budget = RateBudget(None, cost_cap=12)
    results = await RedditAdapter().search(_SPEC, _GEO, budget)
    assert route.called
    assert len(results) == 2  # the no-url item is dropped
    assert all(r.source == "reddit" and r.kind == "discussion" for r in results)
    assert all(r.raw == {} for r in results)  # no-hoarding: nothing stashed


@respx.mock
async def test_reddit_graceful_on_apify_error(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token="tok"))
    respx.post(url__startswith=_ENDPOINT).mock(return_value=httpx.Response(402, text="payment required"))
    budget = RateBudget(None, cost_cap=12)
    results = await RedditAdapter().search(_SPEC, _GEO, budget)
    assert results == []
    assert any("apify" in n.lower() for n in budget.notes)
