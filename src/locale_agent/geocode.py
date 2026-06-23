"""Address geocoding via Nominatim (keyless OSM). Fallback to Google later."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import get_settings
from .logging import get_logger

log = get_logger(__name__)


@dataclass
class GeoResult:
    lat: float
    lng: float
    city: str = ""
    state: str = ""
    neighborhood: str | None = None
    display_name: str = ""


async def geocode(address: str) -> GeoResult | None:
    """Resolve a free-text address to coordinates + components, or None."""
    settings = get_settings()
    params = {
        "q": address,
        "format": "jsonv2",
        "addressdetails": "1",
        "limit": "1",
    }
    headers = {"User-Agent": settings.nominatim_user_agent}
    url = f"{settings.nominatim_base_url}/search"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if not data:
        log.warning("geocode.empty", address=address)
        return None

    top = data[0]
    addr = top.get("address", {})
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("municipality") or ""
    neighborhood = addr.get("neighbourhood") or addr.get("suburb") or addr.get("quarter")
    result = GeoResult(
        lat=float(top["lat"]),
        lng=float(top["lon"]),
        city=city,
        state=addr.get("state", ""),
        neighborhood=neighborhood,
        display_name=top.get("display_name", ""),
    )
    log.info("geocode.ok", address=address, lat=result.lat, lng=result.lng, city=result.city)
    return result
