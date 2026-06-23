"""Single Anthropic client wrapper for the whole app.

- Haiku for intent/slot extraction (fast, structured).
- Sonnet for synthesis (grounded RAG).

Structured output is obtained via *forced tool use* (`tool_choice` pinned to a
single tool) — the most broadly-compatible structured-output path across SDK
versions — then validated by the caller with Pydantic. Retries/backoff are
delegated to the SDK via `max_retries`.

When no API key is configured (`settings.has_llm` is False) the client is
`enabled == False`; callers fall back to deterministic behavior so the service
still boots and answers without a key.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from .config import Settings, get_settings
from .logging import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: AsyncAnthropic | None = None
        if self._settings.has_llm:
            self._client = AsyncAnthropic(
                api_key=self._settings.anthropic_api_key,
                max_retries=3,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def extract(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Force the model to emit one tool call and return its validated input dict."""
        if self._client is None:
            raise LLMError("LLM disabled (no ANTHROPIC_API_KEY configured)")

        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == tool_name:
                return dict(block.input)  # type: ignore[arg-type]
        raise LLMError(f"model did not return a '{tool_name}' tool call")


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
