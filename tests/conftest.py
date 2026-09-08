"""Hermetic defaults for the whole suite.

This checkout's `.env` may hold real Anthropic/Apify keys. Without this file,
`get_llm().enabled` is True inside pytest and the synthesize/intent/replan nodes
silently call Haiku live (they *pass*, because every LLM failure degrades to the
template path — so nothing else would ever notice). The autouse fixture below:

- swaps `get_llm` (both the definition in `locale_agent.llm` and the name the
  graph nodes bound at import) for a disabled stub whose `extract` raises
  `LLMError`, and resets the module-level cached client;
- blanks `anthropic_api_key` / `apify_token` on the lru_cached `Settings`
  instance, so `settings.has_llm` / `has_apify` are False everywhere else.

Tests that need a key patch it back explicitly (e.g. `Settings(apify_token="tok")`
handed to the reddit module) — those still work because they build their own
Settings and never go through the cached instance. Nothing here touches the
adapter registry: graph-level tests replace `adapters_for` / `all_adapters`
themselves (see tests/test_replan.py), and adapter tests are respx-mocked.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from locale_agent import llm as llm_mod
from locale_agent.agent import graph as graph_mod
from locale_agent.config import get_settings
from locale_agent.llm import LLMError


class DisabledLLM:
    """Stand-in for `LLMClient` with no key: `enabled` is False, `extract` raises."""

    enabled = False

    async def extract(self, **_: object) -> dict[str, object]:
        raise LLMError("LLM disabled (tests are hermetic; see tests/conftest.py)")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    stub = DisabledLLM()
    monkeypatch.setattr(llm_mod, "_client", None)
    monkeypatch.setattr(llm_mod, "get_llm", lambda: stub)
    monkeypatch.setattr(graph_mod, "get_llm", lambda: stub)

    settings = get_settings()  # lru_cached singleton — patch attributes in place
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "apify_token", "")
    yield
