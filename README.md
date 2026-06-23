# Locale — hyperlocal research agent (V0)

Locale answers *hyperlocal* questions anchored to your exact address — "I need an
emergency vet near me ASAP", "where's the closest pharmacy" — with a **grounded,
cited** answer instead of a generic web search.

This is the **V0 MVP**: the deterministic `FIND_PLACE` spine, end-to-end.

```
POST /ask  →  extract_intent → resolve_geo → plan → execute_tools
              → resolve_entities → synthesize (grounded, cited)
```

## What's in V0

- **FastAPI** service: `POST /ask`, `GET /healthz`, and a small web UI at `/`.
- **LangGraph** agent spine with six nodes (above).
- **One real, keyless source**: OpenStreetMap — Nominatim (geocoding) + Overpass
  (place search). No API keys, no billing.
- **Claude** for intent extraction (Haiku) and answer synthesis (Sonnet), with a
  **deterministic fallback** so it runs with no key.
- **Postgres 16 + PostGIS + pgvector** and **Redis** via Docker Compose; **Celery**
  app wired up. SQLAlchemy 2.0 async + Alembic.
- **Guardrails**: grounding post-check (no uncited claims), per-source Redis rate
  budget + per-query cost cap, provenance firewall (`derived_from` CHECK), and
  no third-party payload hoarding (only the query skeleton + answer are persisted).

Everything degrades gracefully: if Redis, Postgres, or the LLM key is missing, the
request still returns a grounded answer and records a `note` — never a 5xx.

> **Deferred (later milestones):** the async live `/feed` + Celery ingestion, the
> other archetypes (service-pro, community, listings), more sources (Reddit, Yelp,
> Nextdoor, FB/Thumbtack stubs), entity-embedding tie-break, multi-turn
> checkpointing, and the eval harness.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (installed) — manages Python 3.12 + deps.
- **Docker Desktop** — for Postgres + Redis. (Optional to *try* it; required for
  persistence + rate limiting.)
- An **Anthropic API key** — optional; without it the agent uses its fallback.

## Setup

```bash
uv sync                      # install deps into .venv (Python 3.12)
cp .env.example .env         # then edit .env
```

Edit `.env`:
- `ANTHROPIC_API_KEY` — paste your key from https://console.anthropic.com
- `NOMINATIM_USER_AGENT` — **must** include a real contact email (OSM policy);
  placeholder values get a 403.

## Run

**Quick start (no Docker)** — works immediately; persistence + rate limiting
degrade to notes:

```bash
uv run uvicorn locale_agent.api.main:app --reload
# open http://localhost:8000
```

**Full stack:**

```bash
docker compose up -d --build      # Postgres (PostGIS+pgvector) + Redis
uv run alembic upgrade head       # create the schema
uv run uvicorn locale_agent.api.main:app --reload
# optional background worker (no scheduled tasks yet):
# uv run celery -A locale_agent.worker.celery_app worker -l info
```

Then visit **http://localhost:8000**, or:

```bash
curl -s localhost:8000/healthz | jq
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"address":"1729 N 1st St, San Jose, CA","question":"emergency vet near me ASAP"}' | jq
```

Interactive API docs: **http://localhost:8000/docs**.

## Develop

```bash
uv run pytest        # offline tests (respx-mocked; no live network)
uv run mypy src/locale_agent
```

## Layout

```
src/locale_agent/
  api/main.py        FastAPI app (/ask, /healthz, /)
  agent/graph.py     LangGraph nodes + StateGraph
  adapters/          SourceAdapter ABC, registry, Overpass adapter
  geocode.py         Nominatim geocoding
  schemas.py         Pydantic domain models (QuerySpec, Answer, ...)
  models.py          SQLAlchemy ORM (place, query_log, signal, answer)
  ratelimit.py       per-request cost cap + Redis rate budget
  llm.py             single Anthropic client (Haiku + Sonnet) + fallback
  worker.py          Celery app foundation
  web/index.html     the UI
migrations/          Alembic (baseline = extensions + all tables + CHECKs)
```
