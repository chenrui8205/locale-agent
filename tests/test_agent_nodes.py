"""Entity dedup/ranking + synthesis grounding completeness (offline, no LLM)."""

from __future__ import annotations

from datetime import UTC, datetime

from locale_agent.agent.graph import resolve_entities, synthesize
from locale_agent.schemas import GeoContext, QueryArchetype, QuerySpec, SourceResult

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=8000)
_SPEC = QuerySpec(archetype=QueryArchetype.FIND_PLACE, raw_query="vet", address="x", entity="veterinary")


def _sr(name: str, lat: float, lng: float) -> SourceResult:
    return SourceResult(
        source="openstreetmap",
        url=f"https://www.openstreetmap.org/node/{abs(hash(name)) % 1000}",
        title=name,
        snippet="veterinary",
        geo=(lat, lng),
        fetched_at=datetime.now(UTC),
        raw={"tags": {"name": name, "amenity": "veterinary", "phone": "+1-408-555-0000"}},
    )


async def test_resolve_entities_dedups_and_ranks() -> None:
    results = [
        _sr("Banfield", 37.50, -121.50),       # far
        _sr("Banfield", 37.34, -121.89),       # near duplicate name
        _sr("Animal ER", 37.40, -121.80),      # mid
    ]
    out = await resolve_entities({"results": results, "geo": _GEO, "notes": []})
    ents = out["entities"]
    names = [e.name for e in ents]
    assert len(names) == len(set(n.lower() for n in names)), "duplicates not collapsed"
    dists = [e.distance_m for e in ents]
    assert dists == sorted(dists), "not ranked by distance"
    # nearest Banfield kept
    banfield = next(e for e in ents if e.name == "Banfield")
    assert banfield.distance_m < 5000


async def test_synthesis_grounding_completeness() -> None:
    """With options present, the answer must carry at least one citation."""
    results = [_sr("City Vet", 37.34, -121.89), _sr("Animal ER", 37.40, -121.80)]
    ent_out = await resolve_entities({"results": results, "geo": _GEO, "notes": []})
    state = {
        "raw_query": "emergency vet",
        "address": "1 Main St, San Jose, CA",
        "spec": _SPEC,
        "geo": _GEO,
        "entities": ent_out["entities"],
        "notes": [],
    }
    out = await synthesize(state)
    ans = out["answer"]
    assert ans.body.strip(), "empty answer body"
    assert len(ans.citations) >= 1, "factual answer with no citation (grounding violation)"
    assert len(ans.options) >= 2


async def test_synthesis_no_results_makes_no_uncited_claims() -> None:
    state = {
        "raw_query": "emergency vet",
        "address": "1 Main St",
        "spec": _SPEC,
        "geo": _GEO,
        "entities": [],
        "notes": [],
    }
    out = await synthesize(state)
    ans = out["answer"]
    assert ans.citations == []  # no places claimed → no citations needed
    assert ans.confidence < 0.5
