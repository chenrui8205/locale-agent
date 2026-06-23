"""AgentState threaded through the graph."""

from __future__ import annotations

from typing import Any, TypedDict

from ..schemas import (
    Answer,
    ExecutionPlan,
    GeoContext,
    ProposedAction,
    QuerySpec,
    ResolvedEntity,
    SourceResult,
)


class AgentState(TypedDict, total=False):
    query_id: str
    raw_query: str
    address: str
    spec: QuerySpec | None
    geo: GeoContext | None
    plan: ExecutionPlan | None
    results: list[SourceResult]
    entities: list[ResolvedEntity]
    context: list[SourceResult]  # non-place evidence (Reddit/news/wiki) for synthesis
    answer: Answer | None
    actions: list[ProposedAction]
    notes: list[str]
    budget: Any  # RateBudget — runtime only, not serialized in V0
