"""Structured JSON logging via structlog, correlated by query_id."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit JSON lines to stdout."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    # httpx logs every request line at INFO including the query string — for Apify
    # that is `?token=…`. Never let a vendor credential reach the logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(*args: object, **kwargs: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(*args, **kwargs)  # type: ignore[no-any-return]


def bind_query_id(query_id: str) -> None:
    """Bind query_id into the contextvars so every subsequent log line carries it."""
    structlog.contextvars.bind_contextvars(query_id=query_id)
