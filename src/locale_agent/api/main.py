"""FastAPI app: /healthz, POST /ask, and the web UI at /."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from ..cache import make_redis
from ..config import get_settings
from ..db import SessionLocal, engine
from ..logging import bind_query_id, configure_logging, get_logger
from ..persistence import persist_ask
from ..ratelimit import RateBudget
from ..schemas import Answer, GeoContext, QueryArchetype, QuerySpec, SourceResult
from ..agent import run_agent

log = get_logger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# --------------------------------------------------------------------------- #
# request/response models
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    address: str = Field(..., min_length=3, description="The user's exact address.")
    question: str = Field(..., min_length=3, description="Natural-language hyperlocal question.")
    # Optional structured input from the composer form. When `archetype` is set we
    # build the QuerySpec directly and skip the intent LLM call (thrift).
    archetype: QueryArchetype | None = None
    entity: str | None = None
    topics: list[str] = Field(default_factory=list)
    budget_usd: float | None = None
    deadline_days: int | None = None
    urgency: str | None = None


class AskResponse(BaseModel):
    query_id: str
    answer: Answer | None
    spec: QuerySpec | None
    geo: GeoContext | None
    context: list[SourceResult] = Field(default_factory=list)
    notes: list[str]


def _spec_from_request(req: AskRequest) -> QuerySpec | None:
    """Build a QuerySpec from structured form fields, or None for free-text."""
    if req.archetype is None:
        return None
    constraints: dict = {}
    if req.budget_usd is not None:
        constraints["budget_usd"] = req.budget_usd
    if req.deadline_days is not None:
        constraints["deadline_days"] = req.deadline_days
    if req.urgency:
        constraints["urgency"] = req.urgency
    return QuerySpec(
        archetype=req.archetype,
        raw_query=req.question,
        address=req.address,
        entity=req.entity,
        constraints=constraints,
        topics=req.topics,
    )


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.redis = make_redis()
    log.info("startup", llm_enabled=settings.has_llm)
    yield
    try:
        await app.state.redis.aclose()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="Locale", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    settings = get_settings()
    redis_ok = True
    db_ok = True
    try:
        await app.state.redis.ping()
    except Exception:  # noqa: BLE001
        redis_ok = False
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False
    status = "ok" if (redis_ok and db_ok) else "degraded"
    return {"status": status, "llm": settings.has_llm, "redis": redis_ok, "db": db_ok}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    settings = get_settings()
    query_id = str(uuid.uuid4())
    bind_query_id(query_id)
    log.info("ask.start", address=req.address, question=req.question)

    spec = _spec_from_request(req)
    budget = RateBudget(app.state.redis, cost_cap=settings.cost_cap_external_calls)
    state = await run_agent(
        query_id=query_id,
        raw_query=req.question,
        address=req.address,
        spec=spec,
        budget=budget,
    )
    notes = list(state.get("notes", []))

    try:
        async with SessionLocal() as session:
            await persist_ask(
                session,
                query_id=query_id,
                raw_query=req.question,
                spec=state.get("spec"),
                geo=state.get("geo"),
                answer=state.get("answer"),
            )
    except Exception as e:  # noqa: BLE001 — DB optional for the live answer; never 5xx
        log.warning("ask.persist_failed", error=str(e))
        notes.append(f"persistence: could not write query_log/answer ({e})")

    log.info("ask.done", results=len(state.get("results", [])))
    return AskResponse(
        query_id=query_id,
        answer=state.get("answer"),
        spec=state.get("spec"),
        geo=state.get("geo"),
        context=state.get("context", []),
        notes=notes,
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
