"""Per-request rate/cost budget.

Combines two guardrails from the spec:
  - Cost cap per /ask: a hard ceiling on billable external calls (config).
  - Per-source Redis rate budget: a fixed-window counter per source.

Degrades gracefully: if Redis is unavailable, it proceeds (open) but records a
note. Exhaustion returns False so callers can produce partial results + a note
rather than a 5xx.
"""

from __future__ import annotations

import time
from typing import Any


class RateBudget:
    def __init__(
        self,
        redis: Any | None,
        *,
        cost_cap: int,
        per_source_per_min: int = 30,
        window_s: int = 60,
    ) -> None:
        self._redis = redis
        self.cost_cap = cost_cap
        self.per_source_per_min = per_source_per_min
        self.window_s = window_s
        self.calls_made = 0
        self._notes: list[str] = []

    def note(self, msg: str) -> None:
        self._notes.append(msg)

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    async def allow(self, source: str) -> bool:
        """Return True if a billable call to `source` is permitted right now."""
        if self.calls_made >= self.cost_cap:
            self.note(f"cost cap ({self.cost_cap} calls) reached; skipped {source}")
            return False

        if self._redis is not None:
            try:
                bucket = int(time.time() // self.window_s)
                key = f"budget:{source}:{bucket}"
                count = await self._redis.incr(key)
                if count == 1:
                    await self._redis.expire(key, self.window_s)
                if count > self.per_source_per_min:
                    self.note(f"rate budget for {source} exhausted ({self.per_source_per_min}/min)")
                    return False
            except Exception:  # noqa: BLE001 — degrade open, never crash the request
                self.note(f"rate limiter unavailable for {source}; proceeding without it")

        self.calls_made += 1
        return True
