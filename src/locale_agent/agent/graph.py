"""The LangGraph StateGraph: extract_intent → resolve_geo → plan → execute_tools
→ resolve_entities → replan → execute_follow_ups → synthesize.

`replan` + `execute_follow_ups` are the dependent second hop: once places are
known, ask a free-text follow-up about each ("what do locals say about X") —
bounded to one hop and `settings.replan_max_follow_ups` queries.

Every node degrades gracefully: partial failures append to `notes` and never
raise out of the graph. Grounding is enforced in `synthesize` (every factual
claim carries a citation; uncited output is repaired + noted).
"""

from __future__ import annotations

import asyncio
import re
from math import asin, cos, radians, sin, sqrt
from typing import Any

import h3
from langgraph.graph import END, START, StateGraph

from ..adapters import adapters_for, all_adapters
from ..adapters.base import IMPLEMENTED_TIERS
from ..config import get_settings
from ..geocode import geocode
from ..llm import get_llm
from ..logging import get_logger
from ..ratelimit import RateBudget
from ..schemas import (
    Answer,
    Citation,
    ExecutionPlan,
    FollowUp,
    GeoContext,
    PlannedSource,
    QueryArchetype,
    QuerySpec,
    ReplanDecision,
    ResolvedEntity,
    SourceResult,
)
from .state import AgentState

log = get_logger(__name__)

H3_RES = 9
PER_CALL_TIMEOUT_S = 65.0

# Archetype → search radius (m). Urgent place lookups want a wide net.
_RADIUS_BY_ARCHETYPE: dict[QueryArchetype, int] = {
    QueryArchetype.FIND_PLACE: 8000,
    QueryArchetype.FIND_SERVICE_PRO: 15000,
    QueryArchetype.FIND_COMMUNITY: 12000,
    QueryArchetype.FIND_LISTING: 20000,
    QueryArchetype.LOCAL_FEED: 5000,
}

# Archetype → source categories to fan out across. The planner unions every
# implemented adapter whose supported_categories intersect this list.
_CATEGORIES_BY_ARCHETYPE: dict[QueryArchetype, list[str]] = {
    QueryArchetype.FIND_PLACE: ["place"],                         # Overpass + Reddit
    QueryArchetype.FIND_SERVICE_PRO: ["place", "service"],        # Overpass + Reddit
    QueryArchetype.FIND_COMMUNITY: ["community", "place", "context"],  # Reddit + Overpass + Wikipedia
    QueryArchetype.FIND_LISTING: ["listing"],                     # Reddit (FB stub later)
    QueryArchetype.LOCAL_FEED: ["news", "community", "context"],  # GDELT + Reddit + Wikipedia
}

_MAX_EVIDENCE = 10

# Second hop: adapter the no-LLM fallback follows up on (the only adapter that can
# answer "what do locals say about <place>"), and how many opinion titles the
# template answer lists per place.
_FALLBACK_FOLLOW_UP_ADAPTER = "reddit"
_MAX_OPINIONS_IN_TEMPLATE = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6_371_000.0
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _budget(state: AgentState) -> RateBudget:
    b = state.get("budget")
    if isinstance(b, RateBudget):
        return b
    return RateBudget(None, cost_cap=get_settings().cost_cap_external_calls)


def _notes(state: AgentState) -> list[str]:
    return list(state.get("notes", []))


def _budget_remaining(budget: RateBudget) -> int:
    return budget.cost_cap - budget.calls_made


# --------------------------------------------------------------------------- #
# 1. extract_intent
# --------------------------------------------------------------------------- #
_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "archetype": {"type": "string", "enum": [a.value for a in QueryArchetype]},
        "entity": {
            "type": ["string", "null"],
            "description": "The concrete thing sought, phrased for a maps/OSM search "
            "(e.g. 'veterinary', 'tennis court', 'plumber', 'pharmacy').",
        },
        "topics": {"type": "array", "items": {"type": "string"}},
        "budget_usd": {"type": ["number", "null"]},
        "deadline_days": {"type": ["integer", "null"]},
        "urgency": {"type": ["string", "null"], "description": "'high' or 'low'"},
        "specifics": {"type": ["string", "null"]},
    },
    "required": [
        "archetype",
        "entity",
        "topics",
        "budget_usd",
        "deadline_days",
        "urgency",
        "specifics",
    ],
}

_INTENT_SYSTEM = (
    "You extract structured intent from a hyperlocal research question anchored to an "
    "address. Classify the archetype, then identify the concrete entity the user wants, "
    "phrased so it can drive a maps/OpenStreetMap search (a short noun like 'veterinary', "
    "'tennis court', 'plumber', 'pharmacy' — not a full sentence). Capture any budget, "
    "deadline, and urgency. Be precise and literal; do not invent constraints."
)

# Keyword → entity for the no-LLM fallback.
_FALLBACK_ENTITY = {
    "vet": "veterinary", "pet": "veterinary", "animal hospital": "veterinary",
    "plumber": "plumber", "pharmacy": "pharmacy", "tennis": "tennis court",
    "coffee": "cafe", "cafe": "cafe", "restaurant": "restaurant", "gym": "gym",
    "park": "park", "library": "library", "dentist": "dentist", "doctor": "doctor",
    "hospital": "hospital", "grocery": "supermarket", "supermarket": "supermarket",
    "urgent care": "urgent care", "school": "school",
}
_FALLBACK_ARCHETYPE = [
    (("plumber", "electrician", "handyman", "contractor"), QueryArchetype.FIND_SERVICE_PRO),
    (("coach", "partner", "club", "group", "meetup"), QueryArchetype.FIND_COMMUNITY),
    (("used", "buy", "sell", "marketplace", "listing", "secondhand"), QueryArchetype.FIND_LISTING),
    (("happening", "events near", "this week", "this weekend"), QueryArchetype.LOCAL_FEED),
]


def _fallback_spec(raw_query: str, address: str) -> QuerySpec:
    q = raw_query.lower()
    archetype = QueryArchetype.FIND_PLACE
    for kws, arch in _FALLBACK_ARCHETYPE:
        if any(k in q for k in kws):
            archetype = arch
            break
    entity: str | None = None
    for kw, ent in _FALLBACK_ENTITY.items():
        if kw in q:
            entity = ent
            break
    if entity is None:
        entity = raw_query.strip()  # last resort: name-regex search on the raw text
    return QuerySpec(archetype=archetype, raw_query=raw_query, address=address, entity=entity)


async def extract_intent(state: AgentState) -> AgentState:
    settings = get_settings()
    llm = get_llm()
    raw_query, address = state["raw_query"], state["address"]
    notes = _notes(state)

    # Structured form input already gave us a full spec — skip the LLM call (thrift).
    prefilled = state.get("spec")
    if prefilled is not None:
        notes.append("intent: used structured form input (skipped the LLM extraction call)")
        return {"spec": prefilled, "notes": notes}

    if not llm.enabled:
        spec = _fallback_spec(raw_query, address)
        notes.append("intent: no LLM key — used rule-based fallback extraction")
        return {"spec": spec, "notes": notes}

    try:
        data = await llm.extract(
            model=settings.intent_model,
            system=_INTENT_SYSTEM,
            user=f"Address: {address}\nQuestion: {raw_query}",
            tool_name="emit_query_spec",
            tool_description="Emit the structured intent for this hyperlocal query.",
            input_schema=_INTENT_SCHEMA,
            max_tokens=512,
        )
        constraints: dict[str, Any] = {}
        if data.get("budget_usd") is not None:
            constraints["budget_usd"] = data["budget_usd"]
        if data.get("deadline_days") is not None:
            constraints["deadline_days"] = data["deadline_days"]
        if data.get("urgency"):
            constraints["urgency"] = data["urgency"]
        if data.get("specifics"):
            constraints["specifics"] = data["specifics"]
        spec = QuerySpec(
            archetype=QueryArchetype(data["archetype"]),
            raw_query=raw_query,
            address=address,
            entity=data.get("entity"),
            constraints=constraints,
            topics=data.get("topics") or [],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("extract_intent.fallback", error=str(e))
        spec = _fallback_spec(raw_query, address)
        notes.append(f"intent: LLM extraction failed ({e}); used fallback")

    return {"spec": spec, "notes": notes}


# --------------------------------------------------------------------------- #
# 2. resolve_geo
# --------------------------------------------------------------------------- #
async def resolve_geo(state: AgentState) -> AgentState:
    spec = state.get("spec")
    address = state["address"]
    notes = _notes(state)
    budget = _budget(state)

    if not await budget.allow("nominatim"):
        notes.append("geo: budget exhausted before geocoding")
        return {"geo": None, "notes": notes}

    try:
        g = await geocode(address)
    except Exception as e:  # noqa: BLE001
        log.warning("resolve_geo.error", error=str(e))
        g = None
        notes.append(f"geo: geocoding failed ({e})")

    if g is None:
        notes.append(f"geo: could not resolve address '{address}'")
        return {"geo": None, "notes": notes}

    radius = _RADIUS_BY_ARCHETYPE.get(
        spec.archetype if spec else QueryArchetype.FIND_PLACE, 8000
    )
    geo = GeoContext(
        lat=g.lat,
        lng=g.lng,
        h3_cell=h3.latlng_to_cell(g.lat, g.lng, H3_RES),
        neighborhood=g.neighborhood,
        city=g.city,
        state=g.state,
        default_radius_m=radius,
    )
    return {"geo": geo, "notes": notes}


# --------------------------------------------------------------------------- #
# 3. plan
# --------------------------------------------------------------------------- #
async def plan(state: AgentState) -> AgentState:
    spec = state.get("spec")
    geo = state.get("geo")
    notes = _notes(state)

    if geo is None or spec is None:
        return {"plan": ExecutionPlan(sources=[], notes=["no geo — skipping source search"]), "notes": notes}

    # Archetype-aware routing: union every implemented adapter across the
    # archetype's categories, deduped, order preserved.
    categories = _CATEGORIES_BY_ARCHETYPE.get(spec.archetype, ["place"])
    chosen: dict[str, Any] = {}
    for cat in categories:
        for a in adapters_for(cat):
            chosen.setdefault(a.name, a)

    plan_notes: list[str] = []
    sources = [
        PlannedSource(adapter=a.name, radius_m=geo.default_radius_m, limit=30)
        for a in chosen.values()
    ]
    if sources:
        plan_notes.append(f"routing {spec.archetype.value} → {', '.join(chosen.keys())}")
    else:
        plan_notes.append("no implemented adapters matched the query categories")

    return {"plan": ExecutionPlan(sources=sources, notes=plan_notes), "notes": notes}


# --------------------------------------------------------------------------- #
# 4. execute_tools (parallel fan-out)
# --------------------------------------------------------------------------- #
def _adapter_by_name(name: str):  # type: ignore[no-untyped-def]
    for a in all_adapters():
        if a.name == name:
            return a
    return None


async def execute_tools(state: AgentState) -> AgentState:
    spec = state.get("spec")
    geo = state.get("geo")
    plan_ = state.get("plan")
    notes = _notes(state)
    budget = _budget(state)

    if geo is None or spec is None or plan_ is None or not plan_.sources:
        notes.extend(plan_.notes if plan_ else [])
        return {"results": [], "notes": notes}

    async def _run(ps: PlannedSource) -> tuple[str, list[SourceResult] | Exception]:
        adapter = _adapter_by_name(ps.adapter)
        if adapter is None:
            return ps.adapter, RuntimeError("adapter not found")
        try:
            res = await asyncio.wait_for(
                adapter.search(spec, geo, budget), timeout=PER_CALL_TIMEOUT_S
            )
            return ps.adapter, res
        except Exception as e:  # noqa: BLE001 — circuit-breaker: isolate failures
            return ps.adapter, e

    outcomes = await asyncio.gather(*(_run(ps) for ps in plan_.sources))

    results: list[SourceResult] = []
    for name, outcome in outcomes:
        if isinstance(outcome, Exception):
            notes.append(f"source '{name}' failed: {outcome}")
        else:
            results.extend(outcome)

    notes.extend(plan_.notes)
    notes.extend(budget.notes)
    return {"results": results, "notes": notes}


# --------------------------------------------------------------------------- #
# 5. resolve_entities (light dedup + distance + rank)
# --------------------------------------------------------------------------- #
def _attrs_from_raw(raw: dict) -> dict[str, Any]:
    tags = raw.get("tags") or {}
    out: dict[str, Any] = {}
    for key, dst in (
        ("phone", "phone"),
        ("contact:phone", "phone"),
        ("website", "website"),
        ("contact:website", "website"),
        ("opening_hours", "opening_hours"),
    ):
        if tags.get(key) and dst not in out:
            out[dst] = tags[key]
    cat = tags.get("amenity") or tags.get("leisure") or tags.get("shop") or tags.get("craft")
    if cat:
        out["category"] = cat
    return out


async def resolve_entities(state: AgentState) -> AgentState:
    results = state.get("results", [])
    geo = state.get("geo")
    notes = _notes(state)

    place_results = [r for r in results if r.kind == "place"]
    other_results = [r for r in results if r.kind != "place"]

    by_name: dict[str, ResolvedEntity] = {}
    for r in place_results:
        lat = r.geo[0] if r.geo else None
        lng = r.geo[1] if r.geo else None
        dist = (
            _haversine_m(geo.lat, geo.lng, lat, lng)
            if geo is not None and lat is not None and lng is not None
            else None
        )
        attrs = _attrs_from_raw(r.raw)
        attrs["snippet"] = r.snippet
        entity = ResolvedEntity(
            name=r.title,
            categories=[attrs["category"]] if "category" in attrs else [],
            lat=lat,
            lng=lng,
            address=None,
            distance_m=dist,
            attributes=attrs,
            sources=[Citation(source=r.source, url=r.url, snippet=r.snippet)],
        )
        key = r.title.strip().lower()
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = entity
        else:
            # keep nearest; merge the source citation
            existing.sources.extend(entity.sources)
            if (entity.distance_m or 1e18) < (existing.distance_m or 1e18):
                entity.sources = existing.sources
                by_name[key] = entity

    entities = sorted(
        by_name.values(),
        key=lambda e: e.distance_m if e.distance_m is not None else 1e18,
    )

    # Non-place evidence (Reddit / news / Wikipedia): dedup by URL, cap.
    seen: set[str] = set()
    context: list[SourceResult] = []
    for r in other_results:
        key = r.url or r.title
        if key in seen:
            continue
        seen.add(key)
        context.append(r)
    context = context[:_MAX_EVIDENCE]

    if place_results and len(place_results) != len(entities):
        notes.append(
            f"entities: deduped {len(place_results)} place results into {len(entities)} places"
        )
    if context:
        n_sources = len({r.source for r in context})
        notes.append(
            f"evidence: {len(context)} discussion/news/context items from {n_sources} source(s)"
        )
    return {"entities": entities, "context": context, "notes": notes}


# --------------------------------------------------------------------------- #
# 5b. replan (dependent second hop: which places, what to ask, whether to stop)
# --------------------------------------------------------------------------- #
_REPLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "follow_ups": {
            "type": "array",
            "description": "Follow-up queries to run, one per place, most useful first.",
            "items": {
                "type": "object",
                "properties": {
                    "entity_index": {
                        "type": "integer",
                        "description": "Index [P#] of the place this follow-up is about.",
                    },
                    "adapter": {
                        "type": "string",
                        "description": "Which source to ask (one of the ADAPTERS listed).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Short free-text search query, e.g. '<place name> <city> reviews'.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["entity_index", "adapter", "query", "reason"],
            },
        },
        "stop": {
            "type": "boolean",
            "description": "True if no follow-up is worth its cost (then follow_ups is empty).",
        },
        "stop_reason": {
            "type": "string",
            "description": "Why you stopped (empty if you planned follow-ups).",
        },
    },
    "required": ["follow_ups", "stop", "stop_reason"],
}

_REPLAN_SYSTEM = (
    "You are the planner for Locale, a hyperlocal research assistant. A first search "
    "already found the PLACES below for the user's question. Decide the second hop: "
    "which places deserve a follow-up search for what locals say about them, and what "
    "to ask. Pick at most MAX places — prefer the ones most likely to be recommended "
    "(nearest, best matching the question) and skip generic or irrelevant ones. Each "
    "query is a short free-text search (the place name, plus the city, plus what the "
    "user cares about, e.g. 'Adobe Animal Hospital San Jose emergency reviews'). Only use "
    "the ADAPTERS listed. If no follow-up is worth its cost (e.g. the places are all "
    "chains the user won't need opinions on, or there is nothing to rank), set stop=true "
    "with a one-line stop_reason."
)


def _follow_up_adapters() -> dict[str, Any]:
    """Adapters that may be chosen for a follow-up: registered AND implemented-tier."""
    return {a.name: a for a in all_adapters() if a.access_tier in IMPLEMENTED_TIERS}


def _nearest(entities: list[ResolvedEntity], n: int) -> list[int]:
    def _dist(i: int) -> float:
        d = entities[i].distance_m
        return d if d is not None else 1e18

    return sorted(range(len(entities)), key=_dist)[:n]


def _fallback_follow_ups(
    entities: list[ResolvedEntity], geo: GeoContext | None, max_n: int
) -> list[FollowUp]:
    city = geo.city if geo else ""
    return [
        FollowUp(
            entity_index=i,
            adapter=_FALLBACK_FOLLOW_UP_ADAPTER,
            query=f"{entities[i].name} {city}".strip(),
            reason="nearest place (no LLM — deterministic fallback)",
        )
        for i in _nearest(entities, max_n)
    ]


def _validate_follow_ups(
    raw: list[FollowUp],
    entities: list[ResolvedEntity],
    allowed: dict[str, Any],
    max_n: int,
    notes: list[str],
) -> list[FollowUp]:
    """Drop invalid indices/adapters/empty queries (each with a note), dedup by
    (entity, adapter), and cap at max_n."""
    out: list[FollowUp] = []
    seen: set[tuple[int, str]] = set()
    for fu in raw:
        if not 0 <= fu.entity_index < len(entities):
            notes.append(f"replan: dropped follow-up with invalid place index {fu.entity_index}")
            continue
        if fu.adapter not in allowed:
            notes.append(f"replan: dropped follow-up on unknown/unsupported adapter '{fu.adapter}'")
            continue
        if not fu.query.strip():
            notes.append(f"replan: dropped follow-up with empty query for '{entities[fu.entity_index].name}'")
            continue
        key = (fu.entity_index, fu.adapter)
        if key in seen:
            continue
        seen.add(key)
        out.append(fu)
        if len(out) >= max_n:
            break
    return out


async def _llm_follow_ups(
    state: AgentState,
    entities: list[ResolvedEntity],
    allowed: dict[str, Any],
    max_n: int,
) -> tuple[list[FollowUp], str]:
    """Ask the model for follow-ups. Returns (follow_ups, stop_reason); raises on LLM error."""
    settings = get_settings()
    geo = state.get("geo")
    where = geo.city if geo and geo.city else state.get("address", "")
    adapters_txt = ", ".join(sorted(allowed))
    data = await get_llm().extract(
        model=settings.replan_model,
        system=_REPLAN_SYSTEM,
        user=(
            f"Question: {state.get('raw_query', '')}\n"
            f"User address: {state.get('address', '')} ({where})\n"
            f"MAX: {max_n}\n"
            f"ADAPTERS: {adapters_txt}\n\n"
            "PLACES (cite as [P#]):\n" + _candidate_text(entities)
        ),
        tool_name="emit_replan",
        tool_description="Emit the second-hop follow-up plan (or stop).",
        input_schema=_REPLAN_SCHEMA,
        max_tokens=600,
    )
    follow_ups: list[FollowUp] = []
    for item in data.get("follow_ups") or []:
        if not isinstance(item, dict):
            continue
        try:
            follow_ups.append(
                FollowUp(
                    entity_index=int(item.get("entity_index", -1)),
                    adapter=str(item.get("adapter", "")).strip().lower(),
                    query=str(item.get("query", "")).strip(),
                    reason=str(item.get("reason", "")).strip(),
                )
            )
        except (TypeError, ValueError):
            continue
    stop_reason = str(data.get("stop_reason") or "").strip()
    if data.get("stop") or not follow_ups:
        return [], stop_reason or "model chose to stop"
    return follow_ups, ""


async def replan(state: AgentState) -> AgentState:
    settings = get_settings()
    llm = get_llm()
    entities = state.get("entities", [])
    notes = _notes(state)
    budget = _budget(state)
    max_n = settings.replan_max_follow_ups

    def _skip(reason: str) -> AgentState:
        notes.append(f"replan: skipped — {reason}")
        return {"replan": ReplanDecision(stop_reason=reason), "notes": notes}

    if not settings.replan_enabled:
        return _skip("disabled by config")
    if not entities:
        return _skip("no places to follow up on")
    if _budget_remaining(budget) < 1:
        return _skip("budget exhausted")
    if max_n < 1:
        return _skip("replan_max_follow_ups is 0")

    allowed = _follow_up_adapters()
    candidates = entities[:_MAX_OPTIONS]  # same window the synthesizer will see
    via = "fallback"
    stop_reason = ""
    raw: list[FollowUp] = []
    if llm.enabled:
        try:
            raw, stop_reason = await _llm_follow_ups(state, candidates, allowed, max_n)
            via = "llm"
        except Exception as e:  # noqa: BLE001
            log.warning("replan.fallback", error=str(e))
            notes.append(f"replan: LLM planning failed ({e}); used fallback")
            raw, stop_reason, via = [], "", "fallback"
    if via == "fallback":
        raw = _fallback_follow_ups(candidates, state.get("geo"), max_n)

    if via == "llm" and stop_reason and not raw:
        return _skip(stop_reason)

    follow_ups = _validate_follow_ups(raw, candidates, allowed, max_n, notes)
    if not follow_ups:
        return _skip("no valid follow-ups (no follow-up capable adapter?)")

    notes.append(f"replan: {len(follow_ups)} follow-up(s) planned via {via}")
    return {"replan": ReplanDecision(follow_ups=follow_ups), "notes": notes}


# --------------------------------------------------------------------------- #
# 5c. execute_follow_ups (parallel second hop; attaches opinions per place)
# --------------------------------------------------------------------------- #
async def execute_follow_ups(state: AgentState) -> AgentState:
    decision = state.get("replan")
    entities = list(state.get("entities", []))
    context = list(state.get("context", []))
    geo = state.get("geo")
    notes = _notes(state)
    budget = _budget(state)

    if decision is None or not decision.follow_ups or geo is None:
        if decision is not None and decision.follow_ups and geo is None:
            notes.append("follow-ups: skipped — no geo context")
        return {"entities": entities, "context": context, "notes": notes}

    async def _run(fu: FollowUp) -> tuple[FollowUp, list[SourceResult] | Exception]:
        adapter = _adapter_by_name(fu.adapter)
        if adapter is None:
            return fu, RuntimeError("adapter not found")
        try:
            res = await asyncio.wait_for(
                adapter.search_text(fu.query, geo, budget), timeout=PER_CALL_TIMEOUT_S
            )
            return fu, list(res)
        except Exception as e:  # noqa: BLE001 — isolate failures per follow-up
            return fu, e

    before = len(budget.notes)
    outcomes = await asyncio.gather(*(_run(fu) for fu in decision.follow_ups))

    seen_urls: set[str] = {r.url or r.title for r in context}
    for fu, outcome in outcomes:
        if isinstance(outcome, Exception):
            notes.append(f"follow-up '{fu.query}' on {fu.adapter} failed: {outcome}")
            continue
        if not 0 <= fu.entity_index < len(entities):
            notes.append(f"follow-up '{fu.query}': place index {fu.entity_index} out of range")
            continue
        entity = entities[fu.entity_index]
        for r in outcome:
            r.about = entity.name
            entity.opinions.append(
                Citation(source=r.source, url=r.url, snippet=f"{r.title} — {r.snippet}"[:300])
            )
            key = r.url or r.title
            if key not in seen_urls:
                seen_urls.add(key)
                context.append(r)
        notes.append(f"follow-up '{fu.query}' on {fu.adapter}: {len(outcome)} result(s)")

    notes.extend(budget.notes[before:])  # only the notes this hop produced
    return {"entities": entities, "context": context, "notes": notes}


# --------------------------------------------------------------------------- #
# 6. synthesize (grounded, cited)
# --------------------------------------------------------------------------- #
_SYNTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body": {
            "type": "string",
            "description": "The answer: a short ranked shortlist of places and/or a "
            "synthesis of what locals and local news say.",
        },
        "cited_places": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices [P#] of the place options you recommend.",
        },
        "cited_evidence": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices [E#] of the discussion/news/context items you drew on.",
        },
        "confidence": {"type": "number", "description": "0..1 confidence in the answer."},
    },
    "required": ["body", "cited_places", "cited_evidence", "confidence"],
}

_SYNTH_SYSTEM = (
    "You are Locale, a hyperlocal research assistant. Answer using ONLY the provided "
    "PLACES and EVIDENCE — never invent places, phone numbers, hours, or facts. PLACES "
    "are nearby venues: rank the most relevant first and say how to reach each (phone or "
    "website) when given. EVIDENCE is what locals say (Reddit), recent local news, and "
    "area context (Wikipedia): weave in the relevant points and attribute them. "
    "Reference places and sources by NAME in your prose — do NOT print bracket indices "
    "like [P0] or [E1] in the answer text; instead report the indices you used in the "
    "cited_places and cited_evidence fields. Do NOT claim a place is 'open now' — hours "
    "may be stale; tell the user to call ahead. If sources disagree, say so. "
    "When a PLACE has indented '·' lines beneath it, those are what locals say about "
    "THAT specific place — use them to rank and to justify each recommendation. Be "
    "concise and practical."
)

_MAX_OPTIONS = 8

_EVIDENCE_TAG = {"discussion": "Reddit", "article": "Local news", "context": "Wikipedia"}
_SOURCE_TAG = {"reddit": "Reddit", "gdelt_news": "Local news", "wikipedia": "Wikipedia"}

# Strip internal citation markers ([P0], [E1, E2], …) if the model leaks them into prose.
_INDEX_MARKER = re.compile(r"\s?\[[PE]\d+(?:\s*,\s*[PE]?\d+)*\]")


def _candidate_text(cands: list[ResolvedEntity]) -> str:
    lines: list[str] = []
    for i, c in enumerate(cands):
        dist = f"{c.distance_m / 1000:.1f} km" if c.distance_m is not None else "distance unknown"
        extras = []
        if c.attributes.get("phone"):
            extras.append(f"phone {c.attributes['phone']}")
        if c.attributes.get("website"):
            extras.append(f"web {c.attributes['website']}")
        if c.attributes.get("opening_hours"):
            extras.append(f"listed hours {c.attributes['opening_hours']}")
        extra = f" | {'; '.join(extras)}" if extras else ""
        lines.append(f"[P{i}] {c.name} — {c.attributes.get('snippet', '')} ({dist}){extra}")
        for op in c.opinions:
            lines.append(f"   · ({_SOURCE_TAG.get(op.source, op.source)}) {op.snippet}")
    return "\n".join(lines)


def _evidence_text(evidence: list[SourceResult]) -> str:
    lines: list[str] = []
    for i, e in enumerate(evidence):
        tag = _EVIDENCE_TAG.get(e.kind, e.source)
        lines.append(f"[E{i}] ({tag}) {e.title} — {e.snippet}")
    return "\n".join(lines)


def _citation_for(c: ResolvedEntity) -> Citation:
    src = c.sources[0] if c.sources else None
    return Citation(
        source=src.source if src else "openstreetmap",
        url=src.url if src else None,
        snippet=f"{c.name} — {c.attributes.get('snippet', '')}",
    )


def _evidence_citation(e: SourceResult) -> Citation:
    return Citation(source=e.source, url=e.url, snippet=f"{e.title} — {e.snippet}"[:300])


async def synthesize(state: AgentState) -> AgentState:
    settings = get_settings()
    llm = get_llm()
    spec = state.get("spec")
    geo = state.get("geo")
    entities = state.get("entities", [])
    evidence_all = state.get("context", [])
    notes = _notes(state)

    entity_name = spec.entity if spec and spec.entity else "options"
    where = geo.city if geo and geo.city else state["address"]
    cands = entities[:_MAX_OPTIONS]
    evidence = evidence_all[:_MAX_EVIDENCE]

    if not cands and not evidence:
        body = (
            f"I couldn't find any {entity_name} or local discussion within range of "
            f"{state['address']}. Try widening the area or rephrasing what you're after."
        )
        return {
            "answer": Answer(body=body, citations=[], confidence=0.2, options=[]),
            "notes": notes,
        }

    body, cited_p, cited_e, confidence = "", [], [], 0.5
    if llm.enabled:
        try:
            sections: list[str] = []
            if cands:
                sections.append("PLACES (cite as [P#]):\n" + _candidate_text(cands))
            if evidence:
                sections.append("EVIDENCE (cite as [E#]):\n" + _evidence_text(evidence))
            data = await llm.extract(
                model=settings.synthesis_model,
                system=_SYNTH_SYSTEM,
                user=(
                    f"User address: {state['address']} ({where})\n"
                    f"Question: {state['raw_query']}\n\n" + "\n\n".join(sections)
                ),
                tool_name="emit_answer",
                tool_description="Emit the grounded, cited answer.",
                input_schema=_SYNTH_SCHEMA,
                max_tokens=1200,
            )
            body = str(data.get("body", "")).strip()
            cited_p = [
                i for i in (data.get("cited_places") or [])
                if isinstance(i, int) and 0 <= i < len(cands)
            ]
            cited_e = [
                i for i in (data.get("cited_evidence") or [])
                if isinstance(i, int) and 0 <= i < len(evidence)
            ]
            confidence = float(data.get("confidence", 0.6))
        except Exception as e:  # noqa: BLE001
            log.warning("synthesize.fallback", error=str(e))
            notes.append(f"synthesis: LLM failed ({e}); used template answer")
            body, cited_p, cited_e = "", [], []

    # Deterministic body when there is no LLM (or it failed).
    if not body:
        lines: list[str] = []
        if cands:
            lines.append(f"Here are {min(len(cands), 5)} {entity_name} near {state['address']}:")
            for i, c in enumerate(cands[:5]):
                dist = f"{c.distance_m / 1000:.1f} km away" if c.distance_m is not None else ""
                contact = c.attributes.get("phone") or c.attributes.get("website") or ""
                contact = f" — {contact}" if contact else ""
                lines.append(f"{i + 1}. {c.name} ({dist}){contact}")
                for op in c.opinions[:_MAX_OPINIONS_IN_TEMPLATE]:
                    title = op.snippet.split(" — ", 1)[0]
                    lines.append(f"   · What locals say: {title}")
            lines.append("Call ahead to confirm hours and availability.")
            cited_p = list(range(min(5, len(cands))))
        if evidence:
            lines.append("\nWhat locals & local news say:")
            for ev in evidence[:4]:
                lines.append(f"- {ev.title}")
            cited_e = list(range(min(4, len(evidence))))
        body = "\n".join(lines) if lines else "No answer."

    body = _INDEX_MARKER.sub("", body).strip()
    citations = [_citation_for(cands[i]) for i in cited_p]
    citations += [_evidence_citation(evidence[i]) for i in cited_e]

    # Grounding post-check: a non-empty answer must carry at least one citation.
    if body and (cands or evidence) and not citations:
        citations = [_citation_for(c) for c in cands[:3]] or [
            _evidence_citation(e) for e in evidence[:3]
        ]
        notes.append("grounding: answer had no citations; auto-cited the top sources")
        confidence = min(confidence, 0.4)

    confidence = max(0.0, min(1.0, confidence))
    answer = Answer(body=body, citations=citations, confidence=confidence, options=cands)
    return {"answer": answer, "notes": notes}


# --------------------------------------------------------------------------- #
# graph assembly
# --------------------------------------------------------------------------- #
def build_graph():  # type: ignore[no-untyped-def]
    g = StateGraph(AgentState)
    g.add_node("extract_intent", extract_intent)
    g.add_node("resolve_geo", resolve_geo)
    g.add_node("plan", plan)
    g.add_node("execute_tools", execute_tools)
    g.add_node("resolve_entities", resolve_entities)
    g.add_node("replan", replan)
    g.add_node("execute_follow_ups", execute_follow_ups)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "extract_intent")
    g.add_edge("extract_intent", "resolve_geo")
    g.add_edge("resolve_geo", "plan")
    g.add_edge("plan", "execute_tools")
    g.add_edge("execute_tools", "resolve_entities")
    g.add_edge("resolve_entities", "replan")
    g.add_edge("replan", "execute_follow_ups")
    g.add_edge("execute_follow_ups", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_GRAPH = None


def get_graph():  # type: ignore[no-untyped-def]
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def run_agent(
    *,
    query_id: str,
    raw_query: str,
    address: str,
    spec: QuerySpec | None = None,
    budget: RateBudget | None = None,
) -> AgentState:
    """Run the graph. If `spec` is given (structured form input), extract_intent
    short-circuits and no intent LLM call is made."""
    settings = get_settings()
    initial: AgentState = {
        "query_id": query_id,
        "raw_query": raw_query,
        "address": address,
        "notes": [],
        "budget": budget or RateBudget(None, cost_cap=settings.cost_cap_external_calls),
    }
    if spec is not None:
        initial["spec"] = spec
    final: AgentState = await get_graph().ainvoke(initial)
    return final
