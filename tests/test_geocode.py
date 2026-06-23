"""Geocoding via Nominatim — mocked (no live network)."""

from __future__ import annotations

import httpx
import respx

from locale_agent.geocode import geocode

_FIXTURE = [
    {
        "lat": "37.3686860",
        "lon": "-121.9149470",
        "display_name": "1729, North 1st Street, San Jose, CA, USA",
        "address": {"city": "San Jose", "state": "California", "neighbourhood": "Northside"},
    }
]


@respx.mock
async def test_geocode_parses_components() -> None:
    respx.get(url__regex=r"https://nominatim\.openstreetmap\.org/search.*").mock(
        return_value=httpx.Response(200, json=_FIXTURE)
    )
    g = await geocode("1729 N 1st St, San Jose, CA")
    assert g is not None
    assert round(g.lat, 3) == 37.369
    assert g.city == "San Jose"
    assert g.state == "California"
    assert g.neighborhood == "Northside"


@respx.mock
async def test_geocode_empty_returns_none() -> None:
    respx.get(url__regex=r"https://nominatim\.openstreetmap\.org/search.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert await geocode("nowhere at all") is None
