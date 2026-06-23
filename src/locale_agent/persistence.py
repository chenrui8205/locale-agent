"""Write first-party signal (query_log) + the answer on each /ask.

Provenance/no-hoarding: we persist only the query skeleton + the synthesized
answer. SourceResult.raw (full third-party payloads, OSM tags, etc.) is never
written here — it stays request-scoped in agent state.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from .models import Answer as AnswerRow
from .models import QueryLog
from .schemas import Answer, GeoContext, QuerySpec


async def persist_ask(
    session: AsyncSession,
    *,
    query_id: str,
    raw_query: str,
    spec: QuerySpec | None,
    geo: GeoContext | None,
    answer: Answer | None,
) -> None:
    ql = QueryLog(
        id=uuid.UUID(query_id),
        raw_query=raw_query,
        query_spec=spec.model_dump(mode="json") if spec else {},
        h3_cell=geo.h3_cell if geo else None,
        derived_from="first_party_query",  # always first-party; enforced by DB CHECK
    )
    session.add(ql)
    await session.flush()

    if answer is not None:
        session.add(
            AnswerRow(
                query_id=ql.id,
                body=answer.body,
                citations=[c.model_dump(mode="json") for c in answer.citations],
                confidence=answer.confidence,
            )
        )
    await session.commit()
