"""Internal citation markers must never leak into the user-facing answer body."""

from __future__ import annotations

from locale_agent.agent.graph import _INDEX_MARKER


def test_index_markers_stripped() -> None:
    s = "Try Starbucks [P0] and Peet's [P1]; transit [E0, E4, E5] nearby."
    out = _INDEX_MARKER.sub("", s)
    assert "[" not in out and "]" not in out
    assert "Starbucks" in out and "Peet's" in out


def test_plain_brackets_preserved() -> None:
    # Only P#/E# index groups are stripped — ordinary brackets stay.
    s = "Open Mon–Fri [by appointment]."
    assert _INDEX_MARKER.sub("", s) == s
