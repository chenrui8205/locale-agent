"""Second-hop `search_text` on adapters — offline (respx-mocked), no keys.

Covers the re-planning contract: Reddit scopes to the city's subreddit over all
time (the only query shape that measured relevant + local hits) and caps at 5 items,
Wikipedia does a full-text MediaWiki search, both degrade to [] + a budget note
instead of raising, and the base class refuses politely.
"""

from __future__ import annotations

import json

import httpx
import respx

from locale_agent.adapters.base import AccessTier, SourceAdapter
from locale_agent.adapters.reddit import _FOLLOWUP_LIMIT, RedditAdapter, _followup_query, _strip_city
from locale_agent.adapters.wikipedia import WikipediaAdapter
from locale_agent.config import Settings, get_settings
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import GeoContext, QuerySpec, SourceResult

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=8000)
_GEO_NO_CITY = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", default_radius_m=8000)
_QUERY = "Adobe Animal Hospital"

_APIFY_ENDPOINT = "https://api.apify.com/v2/acts/practicaltools~apify-reddit-api/run-sync-get-dataset-items"
_REDDIT_ITEMS = [
    {
        "dataType": "post", "title": "Adobe Animal Hospital — worth it?",
        "url": "https://www.reddit.com/r/SanJose/comments/aaa",
        "body": "Took my dog there twice, vets were great but pricey.", "communityName": "SanJose",
        "upVotes": 20, "numberOfComments": 9,
    },
    {
        "dataType": "post", "title": "Vet recommendations near Los Altos",
        "url": "https://www.reddit.com/r/bayarea/comments/bbb",
        "body": "", "communityName": "bayarea", "upVotes": 4, "numberOfComments": 11,
    },
    {"dataType": "post", "title": "no url", "body": "dropped"},
]

_WIKI_SEARCH = {
    "query": {
        "searchinfo": {"totalhits": 2},
        "search": [
            {"pageid": 62599348, "title": "St. James Park (San Jose, California)",
             "snippet": '<span class="searchmatch">St</span>. James Park is a park in &quot;downtown&quot;'},
            {"pageid": 1574470, "title": "Downtown San Jose",
             "snippet": "Downtown <span class=\"searchmatch\">San</span> Jose"},
        ],
    }
}
_WIKI_EXTRACTS = {
    "query": {
        "pages": [
            {"pageid": 62599348, "title": "St. James Park (San Jose, California)",
             "extract": "St. James Park is a historic urban park in downtown San Jose."},
            {"pageid": 1574470, "title": "Downtown San Jose", "extract": ""},  # falls back to search snippet
        ]
    }
}


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class _PlacesOnly(SourceAdapter):
    name = "placesonly"
    access_tier = AccessTier.PUBLIC_API
    supported_categories = {"place"}

    async def search(self, spec: QuerySpec, geo: GeoContext, budget: RateBudget) -> list[SourceResult]:
        return []


async def test_base_search_text_default_is_unsupported() -> None:
    budget = RateBudget(None, cost_cap=12)
    results = await _PlacesOnly().search_text(_QUERY, _GEO, budget)
    assert results == []
    assert "placesonly: follow-up search not supported" in budget.notes
    assert budget.calls_made == 0


# --------------------------------------------------------------------------- #
# Reddit
# --------------------------------------------------------------------------- #
@respx.mock
async def test_reddit_search_text_scopes_to_city_subreddit_and_followup_limit(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(
        rmod, "get_settings",
        lambda: Settings(apify_token="tok", apify_reddit_actor="practicaltools/apify-reddit-api"),
    )
    route = respx.post(url__startswith=_APIFY_ENDPOINT).mock(return_value=httpx.Response(200, json=_REDDIT_ITEMS))
    budget = RateBudget(None, cost_cap=12)
    results = await RedditAdapter().search_text(_QUERY, _GEO, budget)

    assert route.called and route.call_count == 1
    sent = json.loads(route.calls.last.request.content)
    assert sent["searches"] == ["subreddit:sanjose Adobe Animal Hospital"]
    assert sent["maxItems"] == _FOLLOWUP_LIMIT == 5
    assert sent["searchPosts"] is True and sent["searchComments"] is False
    assert sent["sort"] == "relevance" and sent["time"] == "all"
    assert budget.calls_made == 1  # charged exactly once

    assert len(results) == 2  # no-url item dropped
    assert all(r.source == "reddit" and r.kind == "discussion" for r in results)
    assert all(r.raw == {} for r in results)
    assert all(getattr(r, "about", None) is None for r in results)  # caller stamps `about`
    assert "pricey" in results[0].snippet


@respx.mock
async def test_reddit_search_text_without_city_sends_bare_query(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token="tok"))
    route = respx.post(url__startswith=_APIFY_ENDPOINT).mock(return_value=httpx.Response(200, json=[]))
    await RedditAdapter().search_text(f"  {_QUERY} ", _GEO_NO_CITY, RateBudget(None, cost_cap=12))
    assert json.loads(route.calls.last.request.content)["searches"] == [_QUERY]


@respx.mock
async def test_reddit_search_text_strips_city_from_query(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token="tok"))
    route = respx.post(url__startswith=_APIFY_ENDPOINT).mock(return_value=httpx.Response(200, json=[]))
    # Both planner paths may name the city ("<name> <city>" fallback, LLM free text);
    # the city is expressed exactly once, as the subreddit scope.
    await RedditAdapter().search_text(f"{_QUERY} san jose reviews", _GEO, RateBudget(None, cost_cap=12))
    assert json.loads(route.calls.last.request.content)["searches"] == [f"subreddit:sanjose {_QUERY} reviews"]


def test_followup_query_shapes() -> None:
    assert _strip_city("Banfield Pet Hospital San Jose", "San Jose") == "Banfield Pet Hospital"
    assert _strip_city("San Jose Banfield San Jose emergency", "San Jose") == "Banfield emergency"
    assert _strip_city("San Josefina Bakery", "San Jose") == "San Josefina Bakery"  # whole words only
    assert _strip_city("  Philz Coffee ", None) == "Philz Coffee"
    assert _followup_query("Philz Coffee", "San Jose") == "subreddit:sanjose Philz Coffee"
    assert _followup_query("Philz Coffee", "Los Altos") == "subreddit:losaltos Philz Coffee"
    assert _followup_query("Philz Coffee San Jose", None) == "Philz Coffee San Jose"


async def test_reddit_search_text_noop_without_token(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token=""))
    budget = RateBudget(None, cost_cap=12)
    results = await RedditAdapter().search_text(_QUERY, _GEO, budget)
    assert results == []
    assert any("APIFY_TOKEN" in n for n in budget.notes)
    assert budget.calls_made == 0  # never consumed a call slot


@respx.mock
async def test_reddit_search_text_graceful_on_apify_error(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token="tok"))
    respx.post(url__startswith=_APIFY_ENDPOINT).mock(return_value=httpx.Response(402, text="payment required"))
    budget = RateBudget(None, cost_cap=12)
    results = await RedditAdapter().search_text(_QUERY, _GEO, budget)
    assert results == []
    assert any("apify" in n.lower() for n in budget.notes)


async def test_reddit_search_text_respects_budget(monkeypatch) -> None:
    import locale_agent.adapters.reddit as rmod

    monkeypatch.setattr(rmod, "get_settings", lambda: Settings(apify_token="tok"))
    budget = RateBudget(None, cost_cap=0)
    assert await RedditAdapter().search_text(_QUERY, _GEO, budget) == []
    assert any("cost cap" in n for n in budget.notes)


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #
@respx.mock
async def test_wikipedia_search_text_returns_context_results() -> None:
    base = get_settings().wikipedia_base_url
    route = respx.get(base).mock(
        side_effect=[httpx.Response(200, json=_WIKI_SEARCH), httpx.Response(200, json=_WIKI_EXTRACTS)]
    )
    budget = RateBudget(None, cost_cap=12)
    results = await WikipediaAdapter().search_text("St. James Park San Jose", _GEO, budget)

    assert route.call_count == 2
    first = route.calls[0].request.url.params
    assert first["list"] == "search" and first["srsearch"] == "St. James Park San Jose"
    assert first["srlimit"] == "5"
    assert budget.calls_made == 1

    assert len(results) == 2
    assert all(r.source == "wikipedia" and r.kind == "context" for r in results)
    assert all(r.url and "curid=" in r.url for r in results)
    assert "historic urban park" in results[0].snippet
    # empty extract → cleaned search snippet (no MediaWiki markup, entities unescaped)
    assert results[1].snippet == "Downtown San Jose"
    assert "<span" not in results[1].snippet
    assert all(getattr(r, "about", None) is None for r in results)


@respx.mock
async def test_wikipedia_search_text_degrades_on_http_error() -> None:
    base = get_settings().wikipedia_base_url
    respx.get(base).mock(return_value=httpx.Response(503, text="backend unavailable"))
    budget = RateBudget(None, cost_cap=12)
    results = await WikipediaAdapter().search_text(_QUERY, _GEO, budget)
    assert results == []
    assert any("wikipedia" in n and _QUERY in n for n in budget.notes)


@respx.mock
async def test_wikipedia_search_text_no_hits_returns_nothing() -> None:
    base = get_settings().wikipedia_base_url
    route = respx.get(base).mock(return_value=httpx.Response(200, json={"query": {"search": []}}))
    results = await WikipediaAdapter().search_text(_QUERY, _GEO, RateBudget(None, cost_cap=12))
    assert results == []
    assert route.call_count == 1  # no extracts call when there is nothing to extract


@respx.mock
async def test_wikipedia_search_text_respects_budget() -> None:
    base = get_settings().wikipedia_base_url
    route = respx.get(base).mock(return_value=httpx.Response(200, json=_WIKI_SEARCH))
    assert await WikipediaAdapter().search_text(_QUERY, _GEO, RateBudget(None, cost_cap=0)) == []
    assert not route.called
