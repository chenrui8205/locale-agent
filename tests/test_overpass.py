"""Overpass adapter — mocked responses + query construction."""

from __future__ import annotations

import httpx
import respx

from locale_agent.adapters.overpass import OverpassAdapter, build_query
from locale_agent.config import get_settings
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import GeoContext, QueryArchetype, QuerySpec

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=5000)
_SPEC = QuerySpec(
    archetype=QueryArchetype.FIND_PLACE, raw_query="emergency vet", address="x", entity="vet"
)

_OVERPASS_FIXTURE = {
    "elements": [
        {
            "type": "node",
            "id": 1,
            "lat": 37.34,
            "lon": -121.89,
            "tags": {"name": "City Vet", "amenity": "veterinary", "phone": "+1-408-555-0100"},
        },
        {
            "type": "way",
            "id": 2,
            "center": {"lat": 37.35, "lon": -121.90},
            "tags": {"name": "Animal ER", "amenity": "veterinary"},
        },
        {  # unnamed — must be skipped
            "type": "node",
            "id": 3,
            "lat": 37.36,
            "lon": -121.91,
            "tags": {"amenity": "veterinary"},
        },
    ]
}


def test_build_query_uses_tag_filter_without_name_regex() -> None:
    q = build_query(_SPEC, _GEO, radius_m=5000, limit=30)
    assert '["amenity"="veterinary"]' in q
    assert '["name"~' not in q  # mapped entity → no slow name-regex fallback
    assert "around:5000,37.33,-121.88" in q


def test_build_query_name_regex_for_unmapped_entity() -> None:
    spec = QuerySpec(
        archetype=QueryArchetype.FIND_PLACE, raw_query="x", address="x", entity="cheese monger"
    )
    q = build_query(spec, _GEO, radius_m=5000, limit=30)
    assert '["name"~"cheese monger",i]' in q


@respx.mock
async def test_search_parses_and_skips_unnamed() -> None:
    settings = get_settings()
    respx.post(settings.overpass_base_url).mock(
        return_value=httpx.Response(200, json=_OVERPASS_FIXTURE)
    )
    results = await OverpassAdapter().search(_SPEC, _GEO, RateBudget(None, cost_cap=12))
    assert len(results) == 2  # unnamed element dropped
    names = {r.title for r in results}
    assert names == {"City Vet", "Animal ER"}
    assert all(r.source == "openstreetmap" for r in results)
    assert all(r.url and r.url.startswith("https://www.openstreetmap.org/") for r in results)


@respx.mock
async def test_search_respects_budget() -> None:
    settings = get_settings()
    route = respx.post(settings.overpass_base_url).mock(
        return_value=httpx.Response(200, json=_OVERPASS_FIXTURE)
    )
    budget = RateBudget(None, cost_cap=0)  # no calls allowed
    results = await OverpassAdapter().search(_SPEC, _GEO, budget)
    assert results == []
    assert not route.called  # never hit the network when budget is exhausted
