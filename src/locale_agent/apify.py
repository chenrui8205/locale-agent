"""Thin async client for the Apify run-sync API.

Used to source data from sites without a usable official API (Reddit now;
Nextdoor later) via Apify's hosted Actors. The scraping runs on Apify's
infrastructure — we just POST an input and receive structured dataset items.
"""

from __future__ import annotations

from typing import Any

import httpx

from .logging import get_logger

log = get_logger(__name__)


class ApifyError(RuntimeError):
    pass


async def run_actor_sync(
    actor_id: str,
    run_input: dict[str, Any],
    *,
    token: str,
    base_url: str = "https://api.apify.com/v2",
    timeout_s: int = 60,
) -> list[dict[str, Any]]:
    """Run an Apify Actor and return its dataset items in a single call.

    Uses `run-sync-get-dataset-items`: one POST that starts the Actor, waits for
    it, and streams back the resulting items. Raises `ApifyError` on any failure
    or timeout so callers can degrade gracefully (partial results + a note).
    """
    actor_path = actor_id.replace("/", "~")  # Apify uses '~' in API paths
    url = f"{base_url}/acts/{actor_path}/run-sync-get-dataset-items"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, params={"token": token}, json=run_input)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise ApifyError(f"actor '{actor_id}' HTTP {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise ApifyError(f"actor '{actor_id}' request failed: {e}") from e

    if not isinstance(data, list):
        raise ApifyError(f"actor '{actor_id}' returned a non-list payload")
    return data
