"""Source adapter contract + registry.

Every source implements the same interface and declares an access tier. The
planner selects only OFFICIAL_API / PUBLIC_API adapters; BROWSER_AGENT /
UNAVAILABLE are skipped (recorded as a note). MVP V0 ships one real adapter
(Overpass); browser-agent stubs (FB Marketplace, Thumbtack) land in a later
milestone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from ..ratelimit import RateBudget
from ..schemas import GeoContext, QuerySpec, SourceResult


class AccessTier(str, Enum):
    OFFICIAL_API = "official_api"   # implement for real
    PUBLIC_API = "public_api"       # implement for real
    BROWSER_AGENT = "browser_agent"  # stub in MVP
    UNAVAILABLE = "unavailable"     # never attempt


IMPLEMENTED_TIERS = {AccessTier.OFFICIAL_API, AccessTier.PUBLIC_API}


class SourceAdapter(ABC):
    name: str
    access_tier: AccessTier
    supported_categories: set[str]

    @abstractmethod
    async def search(
        self, spec: QuerySpec, geo: GeoContext, budget: RateBudget
    ) -> list[SourceResult]:
        ...


_REGISTRY: list[SourceAdapter] = []


def register(adapter: SourceAdapter) -> SourceAdapter:
    _REGISTRY.append(adapter)
    return adapter


def all_adapters() -> list[SourceAdapter]:
    return list(_REGISTRY)


def adapters_for(category: str) -> list[SourceAdapter]:
    """Implemented adapters whose supported_categories include `category`."""
    return [
        a
        for a in _REGISTRY
        if category in a.supported_categories and a.access_tier in IMPLEMENTED_TIERS
    ]
