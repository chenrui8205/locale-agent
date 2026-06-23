"""Wikipedia adapter — mocked geosearch + extracts (offline)."""

from __future__ import annotations

import httpx
import respx

from locale_agent.adapters.wikipedia import WikipediaAdapter
from locale_agent.config import get_settings
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import GeoContext, QueryArchetype, QuerySpec

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=8000)
_SPEC = QuerySpec(archetype=QueryArchetype.FIND_COMMUNITY, raw_query="what's this area like", address="x")

_GEOSEARCH = {
    "query": {
        "geosearch": [
            {"pageid": 111, "title": "St. James Park", "lat": 37.337, "lon": -121.888},
            {"pageid": 222, "title": "San Jose Museum of Art", "lat": 37.333, "lon": -121.890},
        ]
    }
}
_EXTRACTS = {
    "query": {
        "pages": [
            {"pageid": 111, "title": "St. James Park", "extract": "St. James Park is a historic urban park."},
            {"pageid": 222, "title": "San Jose Museum of Art", "extract": "A modern and contemporary art museum."},
        ]
    }
}


@respx.mock
async def test_wikipedia_parses_geosearch_and_extracts() -> None:
    settings = get_settings()
    respx.get(settings.wikipedia_base_url).mock(
        side_effect=[httpx.Response(200, json=_GEOSEARCH), httpx.Response(200, json=_EXTRACTS)]
    )
    results = await WikipediaAdapter().search(_SPEC, _GEO, RateBudget(None, cost_cap=12))
    assert len(results) == 2
    assert all(r.kind == "context" for r in results)
    assert all(r.source == "wikipedia" for r in results)
    assert any("historic urban park" in r.snippet for r in results)
    assert all(r.url and "curid=" in r.url for r in results)


@respx.mock
async def test_wikipedia_empty_geosearch_returns_nothing() -> None:
    settings = get_settings()
    respx.get(settings.wikipedia_base_url).mock(
        return_value=httpx.Response(200, json={"query": {"geosearch": []}})
    )
    results = await WikipediaAdapter().search(_SPEC, _GEO, RateBudget(None, cost_cap=12))
    assert results == []


@respx.mock
async def test_wikipedia_respects_budget() -> None:
    settings = get_settings()
    route = respx.get(settings.wikipedia_base_url).mock(return_value=httpx.Response(200, json=_GEOSEARCH))
    results = await WikipediaAdapter().search(_SPEC, _GEO, RateBudget(None, cost_cap=0))
    assert results == []
    assert not route.called
