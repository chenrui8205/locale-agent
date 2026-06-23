"""Plan node archetype→source routing + mixed-evidence split (offline)."""

from __future__ import annotations

from datetime import UTC, datetime

from locale_agent.agent.graph import plan, resolve_entities
from locale_agent.schemas import GeoContext, QueryArchetype, QuerySpec, SourceResult

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=8000)


def _spec(arch: QueryArchetype) -> QuerySpec:
    return QuerySpec(archetype=arch, raw_query="q", address="x", entity="thing")


async def _routed(arch: QueryArchetype) -> set[str]:
    out = await plan({"spec": _spec(arch), "geo": _GEO, "notes": []})
    return {s.adapter for s in out["plan"].sources}


async def test_find_place_routes_to_places_and_reddit() -> None:
    chosen = await _routed(QueryArchetype.FIND_PLACE)
    assert "openstreetmap" in chosen
    assert "reddit" in chosen
    assert "gdelt_news" not in chosen  # news is not relevant to find-a-place


async def test_local_feed_routes_to_news_reddit_wiki() -> None:
    chosen = await _routed(QueryArchetype.LOCAL_FEED)
    assert "gdelt_news" in chosen
    assert "reddit" in chosen
    assert "wikipedia" in chosen
    assert "openstreetmap" not in chosen


async def test_community_routes_to_reddit_places_wiki() -> None:
    chosen = await _routed(QueryArchetype.FIND_COMMUNITY)
    assert {"reddit", "openstreetmap", "wikipedia"} <= chosen


def _sr(source: str, kind: str, title: str, geo=None) -> SourceResult:
    return SourceResult(
        source=source, url=f"https://x/{title}", title=title, snippet="…",
        kind=kind, geo=geo, fetched_at=datetime.now(UTC), raw={},
    )


async def test_resolve_entities_splits_places_from_evidence() -> None:
    results = [
        _sr("openstreetmap", "place", "City Vet", geo=(37.34, -121.89)),
        _sr("reddit", "discussion", "Best vet in SJ?"),
        _sr("gdelt_news", "article", "New animal hospital opens"),
        _sr("wikipedia", "context", "San Jose"),
        _sr("reddit", "discussion", "Best vet in SJ?"),  # dup URL → deduped
    ]
    out = await resolve_entities({"results": results, "geo": _GEO, "notes": []})
    assert [e.name for e in out["entities"]] == ["City Vet"]  # only the place
    ctx_titles = [c.title for c in out["context"]]
    assert "City Vet" not in ctx_titles
    assert len(out["context"]) == 3  # 4 non-place results, one duplicate collapsed
    assert {c.kind for c in out["context"]} == {"discussion", "article", "context"}
