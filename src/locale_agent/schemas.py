"""Pydantic v2 domain schemas threaded through the agent.

These are the deterministic backbone of the system: LLM nodes emit these shapes
(never freeform), adapters return these, and the API serializes these.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
class QueryArchetype(str, Enum):
    FIND_SERVICE_PRO = "find_service_pro"
    FIND_COMMUNITY = "find_community"
    FIND_LISTING = "find_listing"
    FIND_PLACE = "find_place"
    LOCAL_FEED = "local_feed"


class QuerySpec(BaseModel):
    """Structured intent extracted from the raw query. The agent's spine."""

    archetype: QueryArchetype
    raw_query: str
    address: str
    entity: str | None = None  # "tennis coach", "emergency vet", "used road bike"
    constraints: dict = Field(default_factory=dict)  # budget_usd, deadline_days, urgency, ...
    topics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Geo
# --------------------------------------------------------------------------- #
class GeoContext(BaseModel):
    lat: float
    lng: float
    h3_cell: str  # res 8-9
    neighborhood: str | None = None
    city: str = ""
    state: str = ""
    default_radius_m: int = 5000  # archetype-tunable


# --------------------------------------------------------------------------- #
# Source results
# --------------------------------------------------------------------------- #
ResultKind = Literal["place", "discussion", "article", "context"]


class SourceResult(BaseModel):
    """One raw hit from a source adapter. `raw` is request-scoped and never persisted.

    `kind` tells the agent how to use it: a `place` is a rankable "go here" option
    (Overpass); `discussion` (Reddit), `article` (news), and `context` (Wikipedia)
    are supporting evidence the synthesizer cites but does not rank by distance.
    """

    source: str
    url: str | None = None
    title: str
    snippet: str
    kind: ResultKind = "place"
    geo: tuple[float, float] | None = None  # (lat, lng)
    fetched_at: datetime
    raw: dict = Field(default_factory=dict)  # EPHEMERAL — never written to the DB


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
class PlannedSource(BaseModel):
    adapter: str
    radius_m: int
    limit: int = 20


class ExecutionPlan(BaseModel):
    sources: list[PlannedSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Entities + answer
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    source: str
    url: str | None = None
    snippet: str


class ResolvedEntity(BaseModel):
    """A canonical option surfaced to the user. In V0 this is ~1:1 with a SourceResult;
    entity dedup across sources lands in a later milestone."""

    name: str
    categories: list[str] = Field(default_factory=list)
    lat: float | None = None
    lng: float | None = None
    address: str | None = None
    distance_m: float | None = None
    attributes: dict = Field(default_factory=dict)  # phone, website, opening_hours, ...
    sources: list[Citation] = Field(default_factory=list)


class Answer(BaseModel):
    body: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0  # 0..1
    options: list[ResolvedEntity] = Field(default_factory=list)


class ProposedAction(BaseModel):
    """HITL: the agent drafts, a human confirms. Never executes a side effect."""

    kind: str  # "email" | "sms" | "booking_link" | ...
    target: str | None = None
    draft: str
    requires_confirmation: bool = True


# --------------------------------------------------------------------------- #
# Live feed (schema defined now; ingestion + /feed land in a later milestone)
# --------------------------------------------------------------------------- #
class Signal(BaseModel):
    h3_cell: str
    category: str
    summary: str
    source: str
    source_url: str | None = None
    event_time: datetime | None = None
    fetched_at: datetime
    ttl_s: int
    derived_from: Literal["first_party_query", "third_party_content"]
