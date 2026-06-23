"""Celery app foundation (Redis broker).

The live-feed ingestion (SourceMonitor tasks + Celery Beat schedule) lands in a
later milestone; this just wires the app so the worker can start.
"""

from __future__ import annotations

from celery import Celery

from .config import get_settings

_settings = get_settings()

celery_app = Celery(
    "locale",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
)
celery_app.conf.update(timezone="UTC", enable_utc=True)


@celery_app.task(name="locale.ping")
def ping() -> str:
    """Trivial liveness task to confirm the worker is wired up."""
    return "pong"
