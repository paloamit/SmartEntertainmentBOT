# Smart Entertainment BOT 🎬

A ChatGPT-style assistant that recommends **movies, web series and animation** from a
live catalog, asks you structured clarifying questions before recommending, recognises a
title from a poster you upload, and remembers your chats between visits.

- **Frontend:** React (Vite) — talks to the backend over HTTP only.
- **Backend:** FastAPI — all AI, catalog, caching and session logic.
- The two are **fully decoupled**: point `VITE_API_BASE_URL` at any deployment.

---

## How it works

```
React app ──HTTP/SSE──▶ FastAPI backend ──▶ Groq LLM (two-tier)
                             │
                             ├─▶ TMDB  (primary catalog: movies, series, anime)
                             ├─▶ OMDb  (fallback + real IMDb ratings)
                             ├─▶ TVMaze / Jikan  (no-key fallbacks)
                             ├─▶ Redis (24h cache) or in-process fallback
                             └─▶ SQLite (or Postgres) — visitors, chats, usage
```

### Grounding, not hallucination
Every title the bot mentions comes back from `find_titles`. The keys the model picks are
**validated against the catalog before display** (`cards_for_keys`) — anything invented
resolves to nothing and is silently dropped. The model is also forbidden from showing
internal keys like `tmdb:movie:27205`.

### Human-in-the-loop clarifying questions
When key details are missing the model calls `ask_user` with a **structured payload**
(`question` + `fields`), and the frontend renders real controls — dropdown, radio buttons
or checkboxes — instead of prose. Answers come back as structured values, never parsed
from free text. The `ask_user` tool is *removed* from the tool schema after the user
answers, which makes a second round physically impossible.

### Scope guardrail
Only entertainment questions are answered. A keyword fast path handles most messages with
zero LLM calls; anything ambiguous goes to the small model, and the verdict is cached for
24h. Everything else is declined warmly.

---

## Cost control

| Technique | Effect |
|---|---|
| **Answer cache (24h)** | A similar question — from the same *or a different* user — is served with **zero LLM calls**. Keys are normalised (case, punctuation, filler words); digits are preserved so "5 movies" never collides with "3 movies". |
| **API response cache (24h)** | TMDB/OMDb/TVMaze/Jikan responses are cached, protecting provider quotas. |
| **Cached greeting** | Generated *once*, then personalised per visitor with a `{name}` placeholder — a returning user is greeted by name for free. |
| **Two-tier routing** | `llm_model_small` for greeting, titles, scope checks, name extraction, plot summaries; `llm_model_large` only for recommendations, comparisons and criticism. Chosen by keyword heuristics, so routing itself costs nothing. |
| **Prose in the tool turn** | The model writes its reply *and* calls `recommend_titles` in the same turn, removing an entire extra LLM round-trip per recommendation. |
| **Lazy detail loading** | Expanded card data is fetched by the frontend (`/api/titles/{key}`), never pushed through the model — big token saving. |
| **Trimmed tool payloads** | Overviews clipped before they reach the prompt. |
| **Bounded history** | Only the last `MAX_HISTORY_MESSAGES` turns are replayed. |

Cache is Redis when `REDIS_URL` is set, otherwise an in-process TTL map with the same
24h TTL. If Redis fails mid-flight the app logs once and transparently falls back.

---

## Identity — returning users

Each browser stores a random `visitor_id` in `localStorage` and sends it as
`X-Visitor-Id`. Sessions are scoped to it, so a returning user is greeted by name and
their most recent chat is restored automatically.

Optionally, a **sync code** (`ENABLE_IDENTITY_UNLOCK=true`) lets the same person reopen
their chats on another device — the backend stores only a SHA-256 hash.

---

## Limits (all configurable in `.env`)

| Var | Default | Meaning |
|---|---|---|
| `MAX_SESSIONS` | `5` | Hard cap per user. Creating a 6th chat fails with a friendly `429` until one is deleted. |
| `MAX_QUERIES_PER_SESSION` | `10` | Hard cap of user questions per chat. The 11th returns a friendly `429`. |
| `MAX_RECOMMENDATIONS` | `5` | Never show more than this many titles at once. |

Both limits return a structured error (`{"code": "...", "message": "..."}`) that the UI
renders as a banner.

---

## Poster recognition (images only)

Tap **+** and upload a poster or still. A vision model identifies the title, which is
then **grounded against the catalog** — the model's guess is never trusted on its own.
Results are cached against a hash of the image, so re-uploading the same poster is free.

> Groq's free tier currently serves **no** image-capable model. Set
> `LLM_VISION_API_KEY` + `LLM_VISION_BASE_URL` to a vision provider (Gemini's
> OpenAI-compatible endpoint works) to enable uploads. Everything else works regardless.

---

## Setup

### Backend
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your keys
uvicorn app.main:app --reload --port 8000
```
Docs at `http://localhost:8000/docs`.

### Frontend
```bash
cd frontend
npm install
cp .env.example .env      # VITE_API_BASE_URL
npm run dev
```
Open `http://localhost:5173`.

---

## Keys you need

| Key | Required? | Where |
|---|---|---|
| `LLM_API_KEY` | **yes** | https://console.groq.com/keys |
| `TMDB_API_KEY` | strongly recommended | https://www.themoviedb.org/settings/api — sign in → avatar → **Settings** → *API Key (v3 auth)* |
| `OMDB_API_KEY` | optional | http://www.omdbapi.com/apikey.aspx — adds real **IMDb ratings** and a no-TMDB fallback |
| `LANGFUSE_*` | optional | https://cloud.langfuse.com → Settings → API Keys |
| `REDIS_URL` | optional | Upstash / Render Redis |
| `DATABASE_URL` | optional locally, **required on free-tier Render** | Render Postgres or Neon |

Without TMDB the bot still works for title lookups via OMDb, but genre/year/country
*discovery* ("best Korean thrillers of 2024") needs TMDB.

---

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/health` | models, sources, cache, limits |
| `GET` | `/api/bootstrap` | identity + sessions + greeting in one call |
| `POST` | `/api/chat` | one turn, full JSON |
| `POST` | `/api/chat/stream` | one turn as SSE |
| `POST` | `/api/sessions` | create a chat (respects the cap) |
| `GET` | `/api/sessions` | this visitor's chats |
| `GET` | `/api/sessions/{id}` | full transcript |
| `DELETE` | `/api/sessions/{id}` | delete a chat |
| `POST` | `/api/identity/unlock` | set / restore a sync code |
| `POST` | `/api/vision/identify` | identify a poster (images only) |
| `GET` | `/api/titles/{key}` | full record incl. where to watch |
| `GET` | `/api/browse/trending` | trending titles |
| `POST` | `/api/feedback` | satisfied / dissatisfied + review text |
| `GET` | `/api/observability/usage` | tokens, USD cost, cache hits, questions asked |
| `GET` | `/metrics` | Prometheus |

**SSE events** from `/api/chat/stream`:
`meta`, `status`, `token`, `titles`, `clarify`, `done`.

---

## Observability

### Langfuse
Every turn is a trace; the root span's input/output is the user's message and the reply.
Inside it you get **one `generation` per model call** (never aggregated), typed tool
observations — `retriever` for lookups like `find_titles`, `tool` for actions — a
`guardrail` for the scope check, and an `identify-entertainment-poster` generation for
vision. Sessions group a conversation, users are the visitor id, and satisfied/
dissatisfied feedback is recorded as a `user_satisfaction` score.

Uploaded images are traced as media, not as raw base64.

### Prometheus (`/metrics`)
| Metric | Meaning |
|---|---|
| `sebot_llm_calls_total{model,task,status}` | calls per model and task tier |
| `sebot_llm_tokens_total{model,direction}` | prompt vs completion tokens |
| `sebot_llm_cost_usd_total{model}` | estimated spend |
| `sebot_llm_latency_seconds` | per-model latency histogram |
| `sebot_cache_total{result}` | answer-cache hits vs misses |
| `sebot_tool_calls_total{tool}` | tool usage |
| `sebot_scope_rejections_total` | off-topic questions declined |
| `sebot_feedback_total{rating}` | satisfied vs dissatisfied |
| `sebot_chat_outcomes_total{kind}` | answer vs clarify |

`GET /api/observability/usage?hours=24` returns the same roll-up as JSON, plus the
recent questions users asked. Optionally: `cd monitoring && docker compose up -d`.

---

## Deploy to Render

`render.yaml` defines two free services: `entertainmentbot-api` (FastAPI) and
`entertainmentbot-web` (static React).

1. Push this directory to GitHub and create a Render **Blueprint**.
2. Set the secret env vars (keys are never committed).
3. Set `DATABASE_URL` — Render's free tier wipes local disk on restart, so without it
   your chats disappear.
4. Deploy the API, copy its URL into the web service's `VITE_API_BASE_URL`.

---

## Configuration

See `backend/.env.example` — every knob (provider, models, limits, cache TTL, costs)
lives there, so swapping providers needs no code changes.
