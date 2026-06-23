"""OpenStreetMap / Overpass adapter (keyless, OFFICIAL_API open).

Searches OSM for places near the user's geo. Maps common entity phrases to OSM
tag filters, and always adds a name-regex fallback so arbitrary entities still
return something. Volatile fields like open_now are NOT available from OSM; the
synthesis layer is told not to claim them.
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

# entity phrase -> list of Overpass tag filter clauses (each is one nwr query)
_OSM_FILTERS: dict[str, list[str]] = {
    "vet": ['["amenity"="veterinary"]'],
    "veterinary": ['["amenity"="veterinary"]'],
    "pet hospital": ['["amenity"="veterinary"]'],
    "animal hospital": ['["amenity"="veterinary"]'],
    "tennis": ['["leisure"="pitch"]["sport"="tennis"]', '["sport"="tennis"]'],
    "tennis court": ['["leisure"="pitch"]["sport"="tennis"]'],
    "pharmacy": ['["amenity"="pharmacy"]'],
    "hospital": ['["amenity"="hospital"]'],
    "urgent care": ['["healthcare"="centre"]', '["amenity"="clinic"]'],
    "clinic": ['["amenity"="clinic"]'],
    "restaurant": ['["amenity"="restaurant"]'],
    "cafe": ['["amenity"="cafe"]'],
    "coffee": ['["amenity"="cafe"]'],
    "bar": ['["amenity"="bar"]'],
    "gym": ['["leisure"="fitness_centre"]'],
    "park": ['["leisure"="park"]'],
    "grocery": ['["shop"="supermarket"]'],
    "supermarket": ['["shop"="supermarket"]'],
    "library": ['["amenity"="library"]'],
    "dentist": ['["amenity"="dentist"]'],
    "doctor": ['["amenity"="doctors"]'],
    "pharmacy 24h": ['["amenity"="pharmacy"]["opening_hours"~"24/7"]'],
    "hardware": ['["shop"="hardware"]'],
    "plumber": ['["craft"="plumber"]', '["shop"="hardware"]'],
    "school": ['["amenity"="school"]'],
}


def _filters_for(entity: str | None) -> list[str]:
    if not entity:
        return []
    e = entity.lower().strip()
    clauses: list[str] = []
    for key, filt in _OSM_FILTERS.items():
        if key in e or e in key:
            clauses.extend(filt)
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in clauses:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _escape_regex(text: str) -> str:
    # Overpass regex is RE2-ish; escape quotes/backslashes for the QL string.
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_query(spec: QuerySpec, geo: GeoContext, radius_m: int, limit: int) -> str:
    around = f"around:{radius_m},{geo.lat},{geo.lng}"
    tag_clauses = [f"nwr{filt}({around});" for filt in _filters_for(spec.entity)]
    if tag_clauses:
        # Mapped entity → tag filters are precise and fast. Skip the name-regex
        # fallback, which scans every named object in the radius and is slow.
        clauses = tag_clauses
    elif spec.entity:
        kw = _escape_regex(spec.entity.strip())
        clauses = [f'nwr["name"~"{kw}",i]({around});']
    else:
        clauses = [f'nwr["amenity"]({around});']
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:50];\n(\n  {body}\n);\nout center tags {limit};"


def _snippet(tags: dict[str, str]) -> str:
    bits: list[str] = []
    cat = (
        tags.get("amenity")
        or tags.get("leisure")
        or tags.get("shop")
        or tags.get("craft")
        or tags.get("healthcare")
    )
    if cat:
        bits.append(cat.replace("_", " "))
    street = " ".join(x for x in (tags.get("addr:housenumber"), tags.get("addr:street")) if x)
    if street:
        bits.append(street)
    if tags.get("opening_hours"):
        bits.append(f"hours: {tags['opening_hours']}")
    if tags.get("phone") or tags.get("contact:phone"):
        bits.append(f"phone: {tags.get('phone') or tags.get('contact:phone')}")
    return " · ".join(bits) or "OpenStreetMap place"


class OverpassAdapter(SourceAdapter):
    name = "openstreetmap"
    access_tier = AccessTier.OFFICIAL_API
    supported_categories = {"place"}

    async def search(
        self, spec: QuerySpec, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        if not await budget.allow(self.name):
            return []

        settings = get_settings()
        radius_m = geo.default_radius_m
        query = build_query(spec, geo, radius_m=radius_m, limit=30)
        headers = {"User-Agent": settings.nominatim_user_agent}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                settings.overpass_base_url, data={"data": query}, headers=headers
            )
            resp.raise_for_status()
            payload = resp.json()

        now = datetime.now(UTC)
        results: list[SourceResult] = []
        for el in payload.get("elements", []):
            tags = el.get("tags") or {}
            name = tags.get("name")
            if not name:
                continue  # skip unnamed geometry — not user-actionable
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            osm_type = el.get("type", "node")
            results.append(
                SourceResult(
                    source=self.name,
                    url=f"https://www.openstreetmap.org/{osm_type}/{el.get('id')}",
                    title=name,
                    snippet=_snippet(tags),
                    geo=(lat, lon) if lat is not None and lon is not None else None,
                    fetched_at=now,
                    raw=el,  # request-scoped only; never persisted
                )
            )
        log.info("overpass.search", entity=spec.entity, radius_m=radius_m, results=len(results))
        return results


# Register a singleton instance (not the class) so the registry yields bound adapters.
register(OverpassAdapter())
