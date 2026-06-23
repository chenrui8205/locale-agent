"""Intent extraction: fallback classification + QuerySpec schema conformance."""

from __future__ import annotations

from locale_agent.agent.graph import _fallback_spec
from locale_agent.schemas import QueryArchetype, QuerySpec


def test_fallback_classifies_place_and_entity() -> None:
    spec = _fallback_spec("My dog is sick, I need an emergency vet ASAP", "1 Main St")
    assert isinstance(spec, QuerySpec)
    assert spec.archetype is QueryArchetype.FIND_PLACE
    assert spec.entity == "veterinary"


def test_fallback_classifies_service_pro() -> None:
    spec = _fallback_spec("need a plumber tomorrow for a leak", "1 Main St")
    assert spec.archetype is QueryArchetype.FIND_SERVICE_PRO


def test_fallback_classifies_community() -> None:
    spec = _fallback_spec("looking for a tennis coach or hitting partner", "1 Main St")
    assert spec.archetype is QueryArchetype.FIND_COMMUNITY
    assert spec.entity == "tennis court"


def test_fallback_unknown_entity_uses_raw_query() -> None:
    spec = _fallback_spec("artisanal cheese monger", "1 Main St")
    assert spec.entity == "artisanal cheese monger"  # name-regex fallback path
