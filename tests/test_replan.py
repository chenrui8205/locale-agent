"""Dependent re-planning loop (second hop): replan + execute_follow_ups + synthesize
integration + graph wiring. Offline: no LLM key, fake adapter, stubbed geocode."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from locale_agent.adapters import base as adapters_base
from locale_agent.adapters.base import AccessTier, SourceAdapter
from locale_agent.agent import graph as graph_mod
from locale_agent.agent.graph import (
    build_graph,
    execute_follow_ups,
    replan,
    resolve_entities,
    run_agent,
    synthesize,
)
from locale_agent.config import get_settings
from locale_agent.geocode import GeoResult
from locale_agent.llm import LLMError
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import (
    Citation,
    FollowUp,
    GeoContext,
    QueryArchetype,
    QuerySpec,
    ReplanDecision,
    ResolvedEntity,
    SourceResult,
)

_GEO = GeoContext(lat=37.33, lng=-121.88, h3_cell="x", city="San Jose", state="CA", default_radius_m=8000)
_SPEC = QuerySpec(archetype=QueryArchetype.FIND_PLACE, raw_query="emergency vet", address="1 Main St", entity="veterinary")


def _place(name: str, lat: float, lng: float) -> SourceResult:
    return SourceResult(
        source="fake_places",
        url=f"https://fake/{name.replace(' ', '_')}",
        title=name,
        snippet="veterinary",
        kind="place",
        geo=(lat, lng),
        fetched_at=datetime.now(UTC),
        raw={"tags": {"name": name, "amenity": "veterinary"}},
    )


def _entity(name: str, dist: float | None) -> ResolvedEntity:
    return ResolvedEntity(name=name, distance_m=dist, attributes={"snippet": "veterinary"},
                          sources=[Citation(source="fake_places", url=f"https://fake/{name}", snippet="veterinary")])


class FakeAdapter(SourceAdapter):
    """Returns 3 places on `search` and 2 opinions mentioning the query on `search_text`."""

    name = "fake_places"
    access_tier = AccessTier.PUBLIC_API
    supported_categories = {"place"}

    def __init__(self) -> None:
        self.text_queries: list[str] = []

    async def search(self, spec: QuerySpec, geo: GeoContext, budget: RateBudget) -> list[SourceResult]:
        await budget.allow(self.name)
        return [
            _place("Adobe Animal Hospital", 37.34, -121.89),  # nearest
            _place("Banfield", 37.40, -121.80),
            _place("Far Vet", 37.60, -121.50),
        ]

    async def search_text(self, query: str, geo: GeoContext, budget: RateBudget) -> list[SourceResult]:
        await budget.allow(self.name)
        self.text_queries.append(query)
        place = query.split(" San Jose")[0]
        now = datetime.now(UTC)
        return [
            SourceResult(source="reddit", url=f"https://reddit/{place}/1", title=f"Is {place} any good?",
                         snippet="great vet, pricey", kind="discussion", fetched_at=now),
            SourceResult(source="reddit", url=f"https://reddit/{place}/2", title=f"{place} saved my dog",
                         snippet="fast ER", kind="discussion", fetched_at=now),
        ]


@pytest.fixture
def fake_adapter() -> Iterator[FakeAdapter]:
    a = FakeAdapter()
    adapters_base.register(a)
    try:
        yield a
    finally:
        adapters_base._REGISTRY.remove(a)


class _NoLLM:
    """Stand-in for LLMClient with no key: `enabled` False, `extract` raises."""

    enabled = False

    async def extract(self, **_: object) -> dict[str, object]:
        raise LLMError("LLM disabled (test)")


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Hermetic knobs: replan on, default cap, and the LLM forced OFF even when a
    real key sits in .env (the graph module's `get_llm` is what the nodes call)."""
    s = get_settings()
    monkeypatch.setattr(s, "replan_enabled", True)
    monkeypatch.setattr(s, "replan_max_follow_ups", 3)
    monkeypatch.setattr(graph_mod, "get_llm", lambda: _NoLLM())
    return s


# --------------------------------------------------------------------------- #
# replan
# --------------------------------------------------------------------------- #
async def test_fallback_plans_nearest_n_on_reddit(settings) -> None:  # type: ignore[no-untyped-def]
    assert not graph_mod.get_llm().enabled, "tests must run without a key"
    # deliberately out of distance order: nearest are B (100), A (200), D (300)
    ents = [_entity("A", 200), _entity("B", 100), _entity("C", None), _entity("D", 300)]
    out = await replan({"entities": ents, "geo": _GEO, "notes": [], "budget": RateBudget(None, cost_cap=12)})
    d = out["replan"]
    assert isinstance(d, ReplanDecision) and d.stop_reason == ""
    assert [f.entity_index for f in d.follow_ups] == [1, 0, 3]
    assert all(f.adapter == "reddit" for f in d.follow_ups)
    assert [f.query for f in d.follow_ups] == ["B", "A", "D"]  # bare names; adapter adds the city
    assert "replan: 3 follow-up(s) planned via fallback" in out["notes"]


async def test_fallback_respects_max_follow_ups_and_missing_city(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "replan_max_follow_ups", 1)
    geo = _GEO.model_copy(update={"city": ""})
    out = await replan({"entities": [_entity("A", 1), _entity("B", 2)], "geo": geo, "notes": []})
    assert [f.query for f in out["replan"].follow_ups] == ["A"]  # no city, no trailing space


async def test_replan_skips_when_disabled(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "replan_enabled", False)
    out = await replan({"entities": [_entity("A", 1)], "geo": _GEO, "notes": []})
    assert out["replan"].follow_ups == []
    assert out["replan"].stop_reason == "disabled by config"
    assert "replan: skipped — disabled by config" in out["notes"]


async def test_replan_skips_when_no_entities(settings) -> None:  # type: ignore[no-untyped-def]
    out = await replan({"entities": [], "geo": _GEO, "notes": []})
    assert out["replan"].stop_reason == "no places to follow up on"
    assert any(n.startswith("replan: skipped — no places") for n in out["notes"])


async def test_replan_skips_when_budget_exhausted(settings) -> None:  # type: ignore[no-untyped-def]
    budget = RateBudget(None, cost_cap=1)
    assert await budget.allow("x")
    out = await replan({"entities": [_entity("A", 1)], "geo": _GEO, "notes": [], "budget": budget})
    assert out["replan"].stop_reason == "budget exhausted"
    assert "replan: skipped — budget exhausted" in out["notes"]


async def test_replan_drops_unknown_adapter_with_note(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(graph_mod, "_FALLBACK_FOLLOW_UP_ADAPTER", "nope")
    out = await replan({"entities": [_entity("A", 1)], "geo": _GEO, "notes": []})
    assert out["replan"].follow_ups == []
    assert out["replan"].stop_reason
    assert any("unknown/unsupported adapter 'nope'" in n for n in out["notes"])


# --------------------------------------------------------------------------- #
# execute_follow_ups
# --------------------------------------------------------------------------- #
async def test_execute_follow_ups_attaches_opinions_and_dedups_context(fake_adapter: FakeAdapter) -> None:
    ents = [_entity("Adobe Animal Hospital", 100), _entity("Banfield", 200)]
    now = datetime.now(UTC)
    # context already holds one of the URLs the follow-up will return → must not duplicate
    pre = SourceResult(source="reddit", url="https://reddit/Banfield/1", title="old", snippet="x",
                       kind="discussion", fetched_at=now)
    decision = ReplanDecision(follow_ups=[
        FollowUp(entity_index=1, adapter="fake_places", query="Banfield San Jose"),
        FollowUp(entity_index=0, adapter="fake_places", query="Adobe Animal Hospital San Jose"),
        FollowUp(entity_index=0, adapter="missing_adapter", query="whatever"),
    ])
    out = await execute_follow_ups({
        "entities": ents, "context": [pre], "geo": _GEO, "replan": decision, "notes": [],
        "budget": RateBudget(None, cost_cap=12),
    })
    e0, e1 = out["entities"]
    assert [o.snippet for o in e0.opinions] == [
        "Is Adobe Animal Hospital any good? — great vet, pricey",
        "Adobe Animal Hospital saved my dog — fast ER",
    ]
    assert len(e1.opinions) == 2 and all(o.source == "reddit" for o in e1.opinions)
    assert sorted(fake_adapter.text_queries) == ["Adobe Animal Hospital San Jose", "Banfield San Jose"]
    # about stamped on the raw results and they landed in context, deduped by url
    ctx = out["context"]
    assert ctx[0] is pre
    assert {r.url for r in ctx} == {
        "https://reddit/Banfield/1", "https://reddit/Banfield/2",
        "https://reddit/Adobe Animal Hospital/1", "https://reddit/Adobe Animal Hospital/2",
    }
    assert {r.about for r in ctx[1:]} == {"Adobe Animal Hospital", "Banfield"}
    notes = out["notes"]
    assert "follow-up 'Banfield San Jose' on fake_places: 2 result(s), 2 relevant" in notes
    assert any("on missing_adapter failed" in n for n in notes)


class NoisyAdapter(FakeAdapter):
    """Reddit-shaped: one hit about the place, two city-level noise posts."""

    name = "noisy"

    async def search_text(self, query: str, geo: GeoContext, budget: RateBudget) -> list[SourceResult]:
        await budget.allow(self.name)
        self.text_queries.append(query)
        now = datetime.now(UTC)
        mk = lambda i, t, s: SourceResult(source="reddit", url=f"https://reddit/noisy/{i}", title=t,  # noqa: E731
                                          snippet=s, kind="discussion", fetched_at=now)
        return [
            mk(1, "World Cup 2026 San Jose viewing party", "who's going to Levi's?"),
            mk(2, "Took my dog to Banfield's on N 1st last night", "ER was quick, pricey"),
            mk(3, "Regional Medical Center San Jose wait times?", "spent 4h in the ER"),
        ]


async def test_execute_follow_ups_filters_irrelevant_results(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    noisy = NoisyAdapter()
    adapters_base.register(noisy)
    try:
        ents = [_entity("Banfield Pet Hospital", 100)]
        decision = ReplanDecision(follow_ups=[FollowUp(entity_index=0, adapter="noisy", query="Banfield Pet Hospital")])
        out = await execute_follow_ups({
            "entities": ents, "context": [], "geo": _GEO, "replan": decision, "notes": [],
            "budget": RateBudget(None, cost_cap=12),
        })
    finally:
        adapters_base._REGISTRY.remove(noisy)
    (e,) = out["entities"]
    assert [o.url for o in e.opinions] == ["https://reddit/noisy/2"]
    assert [r.url for r in out["context"]] == ["https://reddit/noisy/2"]  # dropped hits never enter context
    assert "follow-up 'Banfield Pet Hospital' on noisy: 3 result(s), 1 relevant" in out["notes"]


# --------------------------------------------------------------------------- #
# _is_about (relevance rule)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        # distinctive token present
        ("Took my dog to Banfield's on N 1st last night — ER was quick", "Banfield Pet Hospital", True),
        ("Is Adobe Animal Hospital any good? great vet", "Adobe Animal Hospital", True),
        # city-level noise: no distinctive token
        ("World Cup 2026 San Jose viewing party — who's going?", "Banfield Pet Hospital", False),
        ("Regional Medical Center San Jose wait times? spent 4h in the ER", "Banfield Pet Hospital", False),
        # generic words alone never count ("pet", "hospital" appear, "banfield" does not)
        ("best pet hospital in town?", "Banfield Pet Hospital", False),
        # name made only of generic words: needs ALL of its own tokens
        ("the pet emergency center on Saratoga was great", "Pet Emergency Center", True),
        ("any good emergency vet?", "Pet Emergency Center", False),  # only 1 of its tokens
        ("my pet is sick", "Pet Emergency Center", False),
        # live false positive 2026-09-08: the city shelter is not the clinic
        ("URGENT - San Jose Animal Care Center new policy announced today", "Animal Health Center", False),
        ("Animal Health Center on Story Rd stayed open late for us", "Animal Health Center", True),
        # whole-word match, not substring
        ("Adobe Acrobat crashed again", "Adobe Animal Hospital", True),   # same token, accepted by design
        ("Banfields are great", "Banfield", False),
        ("", "Banfield", False),
        ("anything", "", False),
    ],
)
def test_is_about(text: str, name: str, expected: bool) -> None:
    assert graph_mod._is_about(text, name) is expected


def test_is_about_treats_city_tokens_as_generic() -> None:
    # "San Jose Animal Hospital" must not match every San Jose post once the city is generic
    assert graph_mod._is_about("San Jose traffic is awful", "San Jose Animal Hospital") is True
    assert graph_mod._is_about("San Jose traffic is awful", "San Jose Animal Hospital",
                               extra_generic=frozenset({"san", "jose"})) is False
    assert graph_mod._is_about("the animal hospital in san jose saved my cat", "San Jose Animal Hospital",
                               extra_generic=frozenset({"san", "jose"})) is True


async def test_execute_follow_ups_noop_without_decision() -> None:
    ents = [_entity("A", 1)]
    out = await execute_follow_ups({"entities": ents, "context": [], "geo": _GEO, "notes": []})
    assert out["entities"][0].opinions == [] and out["context"] == []


# --------------------------------------------------------------------------- #
# synthesize
# --------------------------------------------------------------------------- #
async def test_synthesize_fallback_mentions_opinions(settings) -> None:  # type: ignore[no-untyped-def]
    e = _entity("Adobe Animal Hospital", 100)
    e.opinions = [
        Citation(source="reddit", url="https://reddit/1", snippet="Adobe saved my dog — fast ER"),
        Citation(source="reddit", url="https://reddit/2", snippet="Is Adobe any good? — pricey"),
        Citation(source="reddit", url="https://reddit/3", snippet="third one — hidden"),
    ]
    out = await synthesize({"raw_query": "emergency vet", "address": "1 Main St", "spec": _SPEC,
                            "geo": _GEO, "entities": [e], "context": [], "notes": []})
    body = out["answer"].body
    assert "What locals say: Adobe saved my dog" in body
    assert "What locals say: Is Adobe any good?" in body
    assert "third one" not in body  # capped at 2 per place
    assert out["answer"].options[0].opinions == e.opinions  # serialisable for the API
    # prompt rendering used by the LLM path
    assert "   · (Reddit) Adobe saved my dog — fast ER" in graph_mod._candidate_text([e])


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_graph_wires_new_nodes_in_order() -> None:
    g = build_graph().get_graph()
    nodes = list(g.nodes)
    for n in ("resolve_entities", "replan", "execute_follow_ups", "synthesize"):
        assert n in nodes
    edges = {(e.source, e.target) for e in g.edges}
    assert ("resolve_entities", "replan") in edges
    assert ("replan", "execute_follow_ups") in edges
    assert ("execute_follow_ups", "synthesize") in edges
    assert ("resolve_entities", "synthesize") not in edges


async def test_run_agent_end_to_end_offline(fake_adapter: FakeAdapter, settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def _geocode(_: str) -> GeoResult:
        return GeoResult(lat=37.33, lng=-121.88, city="San Jose", state="CA")

    monkeypatch.setattr(graph_mod, "geocode", _geocode)
    monkeypatch.setattr(graph_mod, "_FALLBACK_FOLLOW_UP_ADAPTER", fake_adapter.name)
    # Only the fake adapter exists for this run — never touch Overpass/Apify.
    monkeypatch.setattr(graph_mod, "adapters_for", lambda cat: [fake_adapter] if cat in fake_adapter.supported_categories else [])
    monkeypatch.setattr(graph_mod, "all_adapters", lambda: [fake_adapter])
    monkeypatch.setattr(settings, "replan_max_follow_ups", 2)

    final = await run_agent(query_id="q1", raw_query="emergency vet", address="1 Main St, San Jose, CA", spec=_SPEC)
    assert final["replan"].stop_reason == ""
    assert [f.query for f in final["replan"].follow_ups] == ["Adobe Animal Hospital", "Banfield"]
    ents = final["entities"]
    assert len(ents[0].opinions) == 2 and len(ents[1].opinions) == 2 and ents[2].opinions == []
    assert "replan: 2 follow-up(s) planned via fallback" in final["notes"]
    assert "What locals say" in final["answer"].body
    assert final["answer"].options[0].opinions
