# Locale — hyperlocal research agent (V0)

Locale answers *hyperlocal* questions anchored to your exact address — "I need an
emergency vet near me ASAP", "where's the closest pharmacy" — with a **grounded,
cited** answer instead of a generic web search.

This is the **V0 MVP**: the deterministic `FIND_PLACE` spine, end-to-end.

```
POST /ask  →  extract_intent → resolve_geo → plan → execute_tools
              → resolve_entities → replan → execute_follow_ups
              → synthesize (grounded, cited)
```

## What's in V0

- **FastAPI** service: `POST /ask`, `GET /healthz`, and a small web UI at `/`.
- **LangGraph** agent spine with eight nodes (above).
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

### Second hop (re-planning)

The first hop is a fixed workflow: archetype → source categories → one parallel
fan-out. That cannot ask a question whose *arguments depend on first-hop results*
("what do locals say about **Adobe Animal Hospital**?"). So after places are
resolved the graph runs one bounded, dependent hop:

- **`replan`** — Haiku sees the resolved places (`[P#]` lines) and makes exactly
  three decisions via one forced tool call: *which* places deserve a follow-up,
  *what* free-text query to ask about each, and *whether to stop*. Everything
  else stays on rails: at most `replan_max_follow_ups` queries, only registered
  adapters, invalid picks dropped with a note. With no key (or on any model error)
  a deterministic fallback follows up on the nearest N places with
  `"<place name> <city>"` on Reddit.
- **`execute_follow_ups`** — runs the follow-ups in parallel through the new
  `SourceAdapter.search_text(query, geo, budget)` method (Reddit via Apify as
  `subreddit:<cityslug> <place>` over all time — the only query shape that
  measured on-topic *local* hits; Wikipedia full-text; other adapters answer
  "not supported" + `[]`). Results are
  stamped `about=<place>`, attached to each place as `ResolvedEntity.opinions`
  (citations), and also appended to `context` so nothing is lost. They are
  charged against the same per-request `cost_cap_external_calls` budget.
- **`synthesize`** sees the opinions inline under each place and is told to use
  them when ranking; the template answer lists "What locals say: …" per place.

It surfaces in the API as `AskResponse.replan` (`follow_ups[{entity_index,
adapter, query, reason}]`, `stop_reason`) and `options[i].opinions`; the web UI
renders a "What locals say" list under each place card and the second-hop plan
in the agent notes. Every decision is also a note: `replan: 3 follow-up(s)
planned via llm|fallback`, `replan: skipped — <reason>`, `follow-up '<query>' on
<adapter>: <k> result(s)`.

Config knobs (`.env` / `config.py`):

| Setting | Default | Meaning |
|---|---|---|
| `REPLAN_ENABLED` | `true` | Turn the second hop off entirely (`replan: skipped — disabled by config`). |
| `REPLAN_MAX_FOLLOW_UPS` | `3` | Hard cap on follow-up queries per request (also the fallback's "nearest N"). |
| `REPLAN_MODEL` | `claude-haiku-4-5` | Model for the `replan` decision. |

> **Deferred (later milestones):** the async live `/feed` + Celery ingestion, the
> other archetypes (service-pro, community, listings) — including a follow-up hop
> for `FIND_SERVICE_PRO` when Overpass returns nothing — more sources (Yelp,
> Nextdoor, FB/Thumbtack stubs), a title-overlap filter for Wikipedia follow-ups,
> multi-hop re-planning (today it is exactly one hop), entity-embedding tie-break,
> multi-turn checkpointing, and the eval harness.

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

The suite is hermetic even when `.env` holds real keys: `tests/conftest.py`
swaps the LLM client for a disabled stub and blanks `ANTHROPIC_API_KEY` /
`APIFY_TOKEN` on the cached settings for every test. `tests/test_replan_e2e.py`
drives the second hop end-to-end through the FastAPI app with a fake adapter.

## Layout

```
src/locale_agent/
  api/main.py        FastAPI app (/ask, /healthz, /)
  agent/graph.py     LangGraph nodes + StateGraph (incl. replan / execute_follow_ups)
  adapters/          SourceAdapter ABC (search + search_text), registry, Overpass/Wikipedia/GDELT/Reddit
  geocode.py         Nominatim geocoding
  schemas.py         Pydantic domain models (QuerySpec, Answer, ...)
  models.py          SQLAlchemy ORM (place, query_log, signal, answer)
  ratelimit.py       per-request cost cap + Redis rate budget
  llm.py             single Anthropic client (Haiku + Sonnet) + fallback
  worker.py          Celery app foundation
  web/index.html     the UI
migrations/          Alembic (baseline = extensions + all tables + CHECKs)
```
