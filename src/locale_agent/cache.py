"""Redis client factory (async)."""

from __future__ import annotations

import redis.asyncio as aioredis

from .config import get_settings


def make_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)
