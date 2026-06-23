"""Adapter registry. Importing this package registers all implemented adapters."""

from __future__ import annotations

from .base import (
    AccessTier,
    SourceAdapter,
    adapters_for,
    all_adapters,
    register,
)
from .gdelt import GdeltAdapter
from .overpass import OverpassAdapter
from .reddit import RedditAdapter
from .wikipedia import WikipediaAdapter

__all__ = [
    "AccessTier",
    "SourceAdapter",
    "OverpassAdapter",
    "RedditAdapter",
    "WikipediaAdapter",
    "GdeltAdapter",
    "adapters_for",
    "all_adapters",
    "register",
]
