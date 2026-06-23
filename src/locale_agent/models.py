"""SQLAlchemy 2.0 ORM models.

Schema is created by the Alembic baseline (hand-written raw DDL); these ORM
classes mirror it for reads/writes. Keep the two in sync.

Provenance/no-hoarding notes (enforced by DB constraints + tests):
- `query_log.derived_from` is CHECK-constrained to 'first_party_query'.
- `signal.derived_from` is CHECK-constrained to the two allowed values.
- No third-party review text or full Places/source payloads are persisted here;
  `SourceResult.raw` stays request-scoped and never reaches these tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class Place(Base):
    """Canonical place entity. Skeleton (cacheable) + volatile cache (short TTL)."""

    __tablename__ = "place"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    external_ids: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    name: Mapped[str] = mapped_column(Text)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    categories: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    geom: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326), nullable=True
    )
    # Volatile cache — short TTL, re-hydrated at query time.
    hours: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_now: Mapped[bool | None] = mapped_column(nullable=True)
    volatile_fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    volatile_ttl_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[object | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class QueryLog(Base):
    """First-party signal source — written on every /ask."""

    __tablename__ = "query_log"
    __table_args__ = (
        CheckConstraint(
            "derived_from = 'first_party_query'", name="ck_query_log_first_party"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_query: Mapped[str] = mapped_column(Text)
    query_spec: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    h3_cell: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    derived_from: Mapped[str] = mapped_column(
        String, default="first_party_query", server_default="first_party_query"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Signal(Base):
    """Warm hyperlocal signal store for the live feed."""

    __tablename__ = "signal"
    __table_args__ = (
        CheckConstraint(
            "derived_from IN ('first_party_query', 'third_party_content')",
            name="ck_signal_derived_from",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    h3_cell: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ttl_s: Mapped[int] = mapped_column(Integer)
    derived_from: Mapped[str] = mapped_column(String)
    geom: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326), nullable=True
    )
    embedding: Mapped[object | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)


class Answer(Base):
    __tablename__ = "answer"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("query_log.id", ondelete="CASCADE"), index=True
    )
    body: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    confidence: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
