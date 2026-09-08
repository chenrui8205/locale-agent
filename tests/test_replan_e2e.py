"""Second hop, end to end through the FastAPI app (offline).

POST /ask → run_agent → replan → execute_follow_ups → synthesize → AskResponse.
A fake adapter answers both hops, geocode is stubbed, the LLM is off (conftest),
Redis is absent (RateBudget degrades open) and persistence is a no-op — so the
test exercises the real graph wiring and the API serialisation and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

from locale_agent.adapters import base as adapters_base
from locale_agent.adapters.base import AccessTier, SourceAdapter
from locale_agent.agent import graph as graph_mod
from locale_agent.api import main as api_mod
from locale_agent.config import get_settings
from locale_agent.geocode import GeoResult
from locale_agent.ratelimit import RateBudget
from locale_agent.schemas import GeoContext, QuerySpec, SourceResult


def _place(name: str, lat: float, lng: float) -> SourceResult:
    return SourceResult(
        source="fake_places",
        url=f"https://fake/{name.replace(' ', '_')}",
        title=name,
        snippet="veterinary",
        kind="place",
        geo=(lat, lng),
        fetched_at=datetime.now(UTC),
        raw={"tags": {"name": name, "amenity": "veterinary", "phone": "+1-408-555-0100"}},
    )


class FakeAdapter(SourceAdapter):
    """3 places on `search`; 2 opinions about the queried place on `search_text`."""

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
def offline_stack(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAdapter]:
    """Same recipe as tests/test_replan.py::test_run_agent_end_to_end_offline, plus
    the app-level seams (Redis handle, persistence)."""
    fake = FakeAdapter()
    adapters_base.register(fake)

    async def _geocode(_: str) -> GeoResult:
        return GeoResult(lat=37.33, lng=-121.88, city="San Jose", state="CA")

    async def _no_persist(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(graph_mod, "geocode", _geocode)
    monkeypatch.setattr(graph_mod, "_FALLBACK_FOLLOW_UP_ADAPTER", fake.name)
    monkeypatch.setattr(graph_mod, "adapters_for", lambda cat: [fake] if cat in fake.supported_categories else [])
    monkeypatch.setattr(graph_mod, "all_adapters", lambda: [fake])
    settings = get_settings()
    monkeypatch.setattr(settings, "replan_enabled", True)
    monkeypatch.setattr(settings, "replan_max_follow_ups", 2)
    # ASGITransport does not run the lifespan; RateBudget(None) degrades open.
    monkeypatch.setattr(api_mod.app.state, "redis", None, raising=False)
    monkeypatch.setattr(api_mod, "persist_ask", _no_persist)
    try:
        yield fake
    finally:
        adapters_base._REGISTRY.remove(fake)


async def _post_ask(payload: dict[str, object]) -> httpx.Response:
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/ask", json=payload)


async def test_ask_surfaces_second_hop(offline_stack: FakeAdapter) -> None:
    assert not graph_mod.get_llm().enabled, "conftest must keep the LLM off"

    res = await _post_ask({
        "address": "1 Main St, San Jose, CA",
        "question": "emergency vet near me ASAP",
        "archetype": "find_place",
        "entity": "veterinary",
        "urgency": "high",
    })
    assert res.status_code == 200, res.text
    data = res.json()

    # replan decision serialised on the response
    replan = data["replan"]
    assert replan["stop_reason"] == ""
    assert [f["query"] for f in replan["follow_ups"]] == ["Adobe Animal Hospital San Jose", "Banfield San Jose"]
    assert all(f["adapter"] == "fake_places" for f in replan["follow_ups"])
    assert sorted(offline_stack.text_queries) == ["Adobe Animal Hospital San Jose", "Banfield San Jose"]

    # per-place opinions ride on the options
    opts = data["answer"]["options"]
    assert opts[0]["name"] == "Adobe Animal Hospital"
    assert len(opts[0]["opinions"]) == 2 and len(opts[1]["opinions"]) == 2 and opts[2]["opinions"] == []
    op = opts[0]["opinions"][0]
    assert op["source"] == "reddit" and op["url"] == "https://reddit/Adobe Animal Hospital/1"
    assert op["snippet"] == "Is Adobe Animal Hospital any good? — great vet, pricey"

    # notes discipline
    notes = data["notes"]
    assert "replan: 2 follow-up(s) planned via fallback" in notes
    assert any(n.startswith("follow-up 'Adobe Animal Hospital San Jose' on fake_places: 2 result(s)") for n in notes)

    # the answer uses the opinions and cites at least one of them
    body = data["answer"]["body"]
    assert "What locals say: Is Adobe Animal Hospital any good?" in body
    cited_urls = {c["url"] for c in data["answer"]["citations"]}
    assert cited_urls & {"https://reddit/Adobe Animal Hospital/1", "https://reddit/Adobe Animal Hospital/2",
                         "https://reddit/Banfield/1", "https://reddit/Banfield/2"}, "no citation from the follow-up"

    # follow-up results also landed in context (kind=discussion, about=place)
    followup_ctx = [c for c in data["context"] if c["about"]]
    assert {c["about"] for c in followup_ctx} == {"Adobe Animal Hospital", "Banfield"}


async def test_ask_reports_skip_when_disabled(offline_stack: FakeAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "replan_enabled", False)
    res = await _post_ask({"address": "1 Main St, San Jose, CA", "question": "emergency vet near me",
                           "archetype": "find_place", "entity": "veterinary"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["replan"] == {"follow_ups": [], "stop_reason": "disabled by config"}
    assert "replan: skipped — disabled by config" in data["notes"]
    assert offline_stack.text_queries == []
    assert all(o["opinions"] == [] for o in data["answer"]["options"])
