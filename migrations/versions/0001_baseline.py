"""baseline schema: extensions, place, query_log, signal, answer

Revision ID: 0001
Revises:
Create Date: 2026-06-20
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- place: canonical entity; skeleton + short-TTL volatile cache only ---
    op.execute(
        """
        CREATE TABLE place (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            external_ids jsonb NOT NULL DEFAULT '{}'::jsonb,
            name text NOT NULL,
            address text,
            categories jsonb NOT NULL DEFAULT '[]'::jsonb,
            geom geometry(Point, 4326),
            hours jsonb,
            rating double precision,
            open_now boolean,
            volatile_fetched_at timestamptz,
            volatile_ttl_s integer,
            embedding vector(1024),
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_place_geom ON place USING gist (geom)")

    # --- query_log: first-party signal source (every /ask) ---
    op.execute(
        """
        CREATE TABLE query_log (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id text,
            raw_query text NOT NULL,
            query_spec jsonb NOT NULL DEFAULT '{}'::jsonb,
            h3_cell text,
            derived_from text NOT NULL DEFAULT 'first_party_query',
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_query_log_first_party CHECK (derived_from = 'first_party_query')
        )
        """
    )
    op.execute("CREATE INDEX ix_query_log_h3_cell ON query_log (h3_cell)")

    # --- signal: warm hyperlocal store; provenance firewall via CHECK ---
    op.execute(
        """
        CREATE TABLE signal (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            h3_cell text NOT NULL,
            category text NOT NULL,
            summary text NOT NULL,
            source text NOT NULL,
            source_url text,
            event_time timestamptz,
            fetched_at timestamptz NOT NULL DEFAULT now(),
            ttl_s integer NOT NULL,
            derived_from text NOT NULL,
            geom geometry(Point, 4326),
            embedding vector(1024),
            CONSTRAINT ck_signal_derived_from
                CHECK (derived_from IN ('first_party_query', 'third_party_content'))
        )
        """
    )
    op.execute("CREATE INDEX ix_signal_h3_cell ON signal (h3_cell)")
    op.execute("CREATE INDEX ix_signal_category_fetched_at ON signal (category, fetched_at)")
    op.execute("CREATE INDEX ix_signal_geom ON signal USING gist (geom)")

    # --- answer ---
    op.execute(
        """
        CREATE TABLE answer (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            query_id uuid NOT NULL REFERENCES query_log(id) ON DELETE CASCADE,
            body text NOT NULL,
            citations jsonb NOT NULL DEFAULT '[]'::jsonb,
            confidence double precision NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_answer_query_id ON answer (query_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS answer")
    op.execute("DROP TABLE IF EXISTS signal")
    op.execute("DROP TABLE IF EXISTS query_log")
    op.execute("DROP TABLE IF EXISTS place")
    # Extensions left in place intentionally (other DBs in the cluster may use them).
