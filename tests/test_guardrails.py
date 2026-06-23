"""Guardrails: rate/cost budget degradation + provenance / no-hoarding schema."""

from __future__ import annotations

from locale_agent.models import Answer, Place, QueryLog, Signal
from locale_agent.ratelimit import RateBudget


async def test_cost_cap_exhaustion_returns_false_with_note() -> None:
    budget = RateBudget(None, cost_cap=2)
    assert await budget.allow("src") is True
    assert await budget.allow("src") is True
    assert await budget.allow("src") is False  # exceeds cap → graceful, not an exception
    assert any("cost cap" in n for n in budget.notes)


def test_no_hoarding_persisted_tables_have_no_raw_payload() -> None:
    # The persisted tables must not carry raw third-party payloads / review text.
    for model in (QueryLog, Answer):
        cols = set(model.__table__.columns.keys())
        assert "raw" not in cols
        assert "review_text" not in cols


def test_signal_provenance_check_constraint_exists() -> None:
    checks = [
        c for c in Signal.__table__.constraints if type(c).__name__ == "CheckConstraint"
    ]
    assert any("derived_from" in str(c.sqltext) for c in checks), "missing provenance firewall"


def test_query_log_is_first_party_only() -> None:
    checks = [
        c for c in QueryLog.__table__.constraints if type(c).__name__ == "CheckConstraint"
    ]
    assert any("first_party_query" in str(c.sqltext) for c in checks)


def test_place_keeps_only_skeleton_plus_volatile_cache() -> None:
    # Sanity: place has the volatile-cache fields, with a short TTL column, and an
    # external_ids skeleton — but no column for raw Places/Yelp payloads.
    cols = set(Place.__table__.columns.keys())
    assert {"external_ids", "volatile_ttl_s", "volatile_fetched_at"} <= cols
    assert "raw" not in cols and "reviews" not in cols
