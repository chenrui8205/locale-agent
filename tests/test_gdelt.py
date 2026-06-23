"""GDELT news adapter — mocked article list (offline)."""

from __future__ import annotations

import httpx
import respx

from locale_agent.adapters.gdelt import GdeltAdapter
from locale_agent.config import get_settings
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import GeoContext, QueryArchetype, QuerySpec

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=5000)
_NO_CITY = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="", state="", default_radius_m=5000)
_SPEC = QuerySpec(archetype=QueryArchetype.LOCAL_FEED, raw_query="what's happening", address="x", topics=["events"])

_ARTICLES = {
    "articles": [
        {"url": "https://news.example/a", "title": "San Jose festival draws thousands", "domain": "news.example", "seendate": "20260620T120000Z"},
        {"url": "https://news.example/b", "title": "Road work begins on N 1st St", "domain": "news.example", "seendate": "20260619T080000Z"},
        {"title": "no url — skipped", "domain": "x"},
    ]
}


@respx.mock
async def test_gdelt_parses_articles() -> None:
    settings = get_settings()
    respx.get(settings.gdelt_base_url).mock(return_value=httpx.Response(200, json=_ARTICLES))
    results = await GdeltAdapter().search(_SPEC, _GEO, RateBudget(None, cost_cap=12))
    assert len(results) == 2  # third (no url) dropped
    assert all(r.kind == "article" for r in results)
    assert all(r.source == "gdelt_news" for r in results)
    assert any("festival" in r.title for r in results)


async def test_gdelt_noop_without_city() -> None:
    budget = RateBudget(None, cost_cap=12)
    results = await GdeltAdapter().search(_SPEC, _NO_CITY, budget)
    assert results == []
    assert any("city" in n for n in budget.notes)


@respx.mock
async def test_gdelt_handles_non_json() -> None:
    settings = get_settings()
    respx.get(settings.gdelt_base_url).mock(return_value=httpx.Response(200, text="rate limited"))
    budget = RateBudget(None, cost_cap=12)
    results = await GdeltAdapter().search(_SPEC, _GEO, budget)
    assert results == []
    assert any("non-JSON" in n for n in budget.notes)
