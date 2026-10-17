"""LLM orchestration: Groq client, two-tier routing and the tool-calling loop.

Two models
----------
`llm_model_small` handles cheap deterministic work (greeting, session titles,
scope checks, cast lists, plot summaries). `llm_model_large` handles the
reasoning-heavy work (recommendations, comparisons, criticism). Routing is done
with keyword heuristics rather than a classifier, so choosing a model costs no
extra LLM call.

Cost control
------------
* Every upstream response is cached for 24h, so a similar question — asked by
  the same or a different user — is served without touching the LLM at all.
* The model writes its recommendation prose *in the same turn* as the
  `recommend_titles` tool call, which removes an entire extra LLM round-trip.
* Tool payloads are trimmed (`entertainment.to_compact`) to keep prompts small.
* Token counts and USD cost are recorded per call for the usage dashboard.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

from openai import OpenAI

from . import cache, db, entertainment, metrics, observability as obs
from .config import get_settings

logger = logging.getLogger("sebot.llm")
_settings = get_settings()

_client = None
_vision = None
_vision_key: tuple = ()
_FEEDBACK_SCORE = {"dissatisfied": 0.0, "satisfied": 1.0}


def _client_instance():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=_settings.llm_api_key or "missing-key",
            base_url=_settings.llm_base_url,
            timeout=60.0,
            max_retries=2,
        )
    return _client


# ------------------------------------------------------- provider failover
# A chain entry is either a bare model id (primary provider) or
# `provider:model` (secondary provider). `_resolve_target` is the single place
# that turns either form into a live client plus a model name, so the failover
# loops keep treating the chain as a plain list of strings.
_fallback_client = None
_fallback_key: tuple | None = None


def _fallback_client_instance():
    global _fallback_client, _fallback_key
    key = (
        _settings.fallback_api_key(),
        _settings.fallback_base_url(),
    )
    if _fallback_client is None or _fallback_key != key:
        _fallback_client = OpenAI(
            api_key=key[0] or "missing-key",
            base_url=key[1],
            timeout=_settings.llm_fallback_timeout_ms / 1000.0,
            max_retries=2,
        )
        _fallback_key = key
    return _fallback_client


def _resolve_target(target: str):
    """(client, model) for a chain entry.

    A bare id keeps the legacy meaning — the primary provider, using
    LLM_API_KEY/LLM_BASE_URL — so an existing LLM_FAILOVER_MODELS list needs no
    changes. `provider:model` selects the configured secondary provider.
    """
    provider, sep, model = target.partition(":")
    if not sep:
        return _client_instance(), target
    return _fallback_client_instance(), model


def chain_report() -> list[dict]:
    """The failover chain with each entry's real provider, for /api/health."""
    out: list[dict] = []
    for target in _failover_chain("large"):
        provider, sep, model = target.partition(":")
        if not sep:
            provider, model = "primary", target
        try:
            client, model_id = _resolve_target(target)
            out.append({"provider": provider, "model": model_id,
                        "base_url": str(client.base_url)})
        except Exception:
            out.append({"provider": provider, "model": model, "error": True})
    return out


def _extra() -> dict:
    """Extra request body shared by every model call.

    `reasoning_effort` looks like it should help here — the original note said
    gpt-oss buries its answer in a `reasoning` field unless told to keep
    thinking short. Measured against Groq it does the opposite and much worse:
    passing ANY value (low/medium/high) makes the model emit **nothing at
    all** — no content, no reasoning, and crucially no tool calls. The whole
    app is built on tool calling, so a turn would silently return zero titles.

        no reasoning_effort       -> tools=['f']
        reasoning_effort=low      -> content=0 reasoning=0 tools=[]

    So it is off by default. Leave LLM_REASONING_EFFORT blank unless you have
    verified the specific provider still calls tools with it set.
    """
    effort = (_settings.llm_reasoning_effort or "").strip()
    if not effort:
        return {}
    logger.warning(
        "LLM_REASONING_EFFORT=%s is being sent; this has been observed to "
        "disable tool calling on Groq's gpt-oss models", effort,
    )
    return {"extra_body": {"reasoning_effort": effort}}


def has_key() -> bool:
    return bool(_settings.llm_api_key.strip())


def _key_missing_msg() -> str:
    """Deployment-aware 'no key' message.

    Saying "edit backend/.env" is actively wrong on Render: that file is
    git-ignored and does not exist on the server, so the advice can never be
    acted on. Detect a hosted deploy and point at the real control panel.
    """
    if os.environ.get("RENDER"):
        return (
            "⚠️ My LLM key isn't configured on this server. Add `LLM_API_KEY` to the "
            "**Environment** tab of this service in Render, then redeploy "
            "(free key: https://console.groq.com/keys)."
        )
    return (
        "⚠️ My LLM key isn't configured yet. Add `LLM_API_KEY` to `backend/.env` "
        "(a free Groq key: https://console.groq.com/keys) and restart the server."
    )


_NO_SOURCE_MSG = (
    "⚠️ I don't have an entertainment data source connected yet. "
    + (
        "Add `WATCHMODE_API_KEY` (or `TMDB_API_KEY`) to this service's "
        "**Environment** tab in Render, then redeploy."
        if os.environ.get("RENDER")
        else (
            "Add `TMDB_API_KEY` to `backend/.env` "
            "(free key: https://www.themoviedb.org/settings/api) so I can "
            "recommend real movies, series and anime."
        )
    )
)

# ============================================================== system prompt
SYSTEM_PROMPT = """You are **Smart Entertainment BOT** 🎬 — a warm, human movie, TV-series and
animation recommender. You are deeply knowledgeable about cinema and television,
and you recommend titles that genuinely exist.

## Your scope
Movies, web series, anime/animation, actors, directors, genres, plots, ratings,
critics, streaming availability, "what should I watch", comparisons and rewatches.
If the user asks about ANYTHING else (coding, maths, general knowledge, news,
health, sports…), do not answer it and do not invent anything. Simply say, warmly
and briefly, that you're all about movies, series and animation, and offer to help
with that instead. Never lecture.

## Grounding — never break this
- Every title you mention must come back from the `find_titles` tool in this
  conversation. NEVER invent a title, rating, cast member, year or streaming
  service, even if you are sure you know it.
- NEVER show or mention internal title keys (e.g. "tmdb:movie:27205"). Refer to
  titles by their name only.
- If a tool returns nothing useful, say what you couldn't find and ask what
  they'd like instead. Never pad a recommendation with made-up titles.

## When you find nothing — be honest, don't improvise
If `find_titles` comes back with `not_found: true`, the catalogs genuinely have
nothing for that request. You MUST:
- Say plainly that you couldn't find it. "I couldn't find a movie called X" is
  a perfectly good answer.
- Ask whether they'd like something similar, or a different genre/mood.
- **Do NOT call `recommend_titles`.** No cards. Showing unrelated popular
  titles when someone asked about one specific movie is the worst thing you can
  do — it reads as though you ignored them.
- **Do NOT guess from memory**, even for a title you are certain exists. If the
  catalogs don't have it, you don't have it.

Same rule when a summary tool returns nothing: say you don't have details on
that title and offer alternatives, rather than describing it from memory.

## How to answer
1. **Understand before you recommend.** If the request is vague ("suggest
   something good"), gather what's missing using `ask_user` — country/language,
   release year or era, genre, and whether they want a movie, a series or
   animation. Ask ONE round of questions covering the most important gaps, and
   only when the answer would genuinely change your picks. Never interrogate.
   **Never ask for something the user already told you or clearly implied.**
   "Bollywood", "Tollywood" or "Tamil" already mean India; "K-drama" means South
   Korea; "anime"/"manga" means Japan and Animation; "Hollywood" means the US.
   Asking "which country?" after the user said "Bollywood movies" is a bad
   question — skip that field entirely and ask only about what is still open.
2. **Find real titles** with `find_titles`, passing the filters the user gave.
   Search again with different keywords if the first pass is thin.
3. **Write your reply as your normal text content AND call `recommend_titles`
   with the same turn.** Never make a separate follow-up call just to write the
   message — put the whole reply in your text and the picks in the tool call.
4. Recommend **at most 5 titles**, most-wanted first. Say in one short line per
   title why it fits, and mention the rating the tool gave you.
5. Be concise, warm and specific. No walls of text, no lecturing about film.

## Answers about a specific title
Use `get_title_summary` for plot/synopsis questions, `get_title_reviews` for
critics or audience reactions, `get_cast_crew` for who made or starred in it,
and `get_similar_titles` for "movies like this". Ground every claim in the tool
output; if a tool doesn't return something, say you don't have it.

## Ratings
Use the `rating` value from the tools (out of 10) and describe it in words.
Never invent a different score.
"""

# ==================================================================== tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_titles",
            "description": (
                "Find real movies / TV series / anime matching the user's request. This is "
                "the ONLY way to discover titles. Pass whichever filters the user gave you "
                "(genres, country, language, year range, minimum rating); leave the rest "
                "unset. Returns title keys, names, years, ratings and a short overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or a title name."},
                    "media_type": {
                        "type": "string", "enum": ["movie", "tv", "any"],
                        "description": "'tv' covers series and anime.",
                    },
                    "genres": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Genre names, e.g. ['Thriller','Sci-Fi']. Use 'Animation' for anime.",
                    },
                    "country": {"type": "string", "description": "Country name or 2-letter code, e.g. 'India' or 'IN'."},
                    "language": {"type": "string", "description": "Original language name or code, e.g. 'Hindi' or 'hi'."},
                    "year_from": {"type": "integer"},
                    "year_to": {"type": "integer"},
                    "min_rating": {"type": "number", "description": "Minimum rating out of 10."},
                    "sort": {
                        "type": "string", "enum": ["popular", "rating", "newest", "oldest", "revenue"],
                    },
                    "limit": {"type": "integer", "description": "How many titles to return (1-20)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_title_summary",
            "description": (
                "Get a plot summary of a specific title. Use for 'what is X about', "
                "'summarise X', 'tell me the story'. Returns a spoiler-light synopsis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_key": {"type": "string"},
                    "detail": {
                        "type": "string", "enum": ["quick", "full"],
                        "description": "'quick' = 2-3 lines, 'full' = a richer summary.",
                    },
                },
                "required": ["title_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_title_reviews",
            "description": (
                "Get critic/audience reviews and the overall critical reception for a "
                "title. Use for 'is X worth watching', 'what do critics say', 'reviews of X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_key": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["title_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cast_crew",
            "description": "Get the cast and key crew (director, writer, creator) for a title.",
            "parameters": {
                "type": "object",
                "properties": {"title_key": {"type": "string"}},
                "required": ["title_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_titles",
            "description": "Get titles similar to a given one. Use for 'movies like X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_key": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["title_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a structured question when key details are missing and they "
                "would change your recommendation. Build a widget the user can tap: give "
                "one short friendly 'question' line and 1-4 'fields', each with a name, a "
                "label and 2-6 options.\n"
                "CHOOSING THE CONTROL:\n"
                "  checkbox  Use for genre and for multi-select facets (mood, theme, tone) "
                "— tap-to-pick chips read well for these. Keep chips to 5-8 so they stay "
                "on one row.\n"
                "  dropdown  Use for one-choice-from-many: country, language, release "
                "year/era, format, certification. Offer a GENEROUS list (12-20 options) — "
                "a rich scrolling menu is much more useful than a short one.\n"
                "  radio     Use only for 2-4 mutually exclusive choices.\n"
                "Always put an 'Any' option first in a dropdown.\n"
                "Also give 1-4 'quick_options' as one-tap answers to the question itself. "
                "Ask at most one round, then recommend."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "e.g. country, genre, year, language, type"},
                                "label": {"type": "string"},
                                "ui": {"type": "string", "enum": ["dropdown", "radio", "checkbox"]},
                                "options": {
                                    "type": "array",
                                    # Models reach for either shape depending on
                                    # the prompt, and the provider rejects the
                                    # one we didn't predict — taking the whole
                                    # turn down. Accept both; the normaliser
                                    # reduces them to {value, label} anyway.
                                    "items": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "value": {"type": "string"},
                                                    "label": {"type": "string"},
                                                },
                                            },
                                        ]
                                    },
                                    "description": "2-6 option labels, e.g. "
                                                   "['Any','India','Japan']. Put "
                                                   "'Any' first for dropdowns.",
                                },
                            },
                        },
                    },
                    "quick_options": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object",
                                 "properties": {"label": {"type": "string"}}},
                            ]
                        },
                        "description": "Optional 1-4 one-tap answers to the question "
                                       "itself (e.g. 'Surprise me', 'Something light'). "
                                       "Do NOT repeat the option labels you already put "
                                       "in `fields` — those are already tappable there.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_titles",
            "description": (
                "Present your final picks as visual cards. Call this in the SAME turn as "
                "your written reply — the user reads your text and sees the cards. "
                "title_keys must come ONLY from find_titles results, at most 5."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title_keys"],
            },
        },
    },
]

# ==================================================================== tools
def _loosen_nulls(schema: dict) -> dict:
    """Make a tool schema tolerant of the ways models actually write arguments.

    Groq validates streamed tool arguments against this schema and aborts the
    whole turn on the first mismatch:

        APIError: Tool call validation failed: parameters for tool
        find_titles did not match schema: [/country: expected string, but got null]

    Every failure mode we hit in production was the model writing arguments
    the way models naturally write them, not a bug in the tool:

      • an explicit ``null`` to mean "no filter"
      • options as plain strings instead of ``{value, label}`` objects
      • omitting a parameter it had nothing to say about

    The handlers already read arguments defensively and ``_normalise_clarify``
    already accepts string options and fills in a missing ``ui``. So the strict
    schema never guided the model — it only killed conversations. Nulls are
    allowed, and ``required`` is dropped: a tool that receives less than it
    wanted returns nothing, which is far better than losing the turn.
    """
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k != "required"}

    if isinstance(out.get("type"), str):
        out["type"] = [out["type"], "null"]

    enum = out.get("enum")
    if isinstance(enum, list) and None not in enum:
        # A null must be allowed by the enum too, or validation still fails.
        out["enum"] = [*enum, None]

    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _loosen_nulls(v) for k, v in props.items()}

    if isinstance(out.get("items"), dict):
        out["items"] = _loosen_nulls(out["items"])

    # Union branches get the same treatment, otherwise the null-tolerant part
    # is still rejected by a strict sibling branch.
    for key in ("anyOf", "oneOf"):
        branches = out.get(key)
        if isinstance(branches, list):
            out[key] = [_loosen_nulls(b) for b in branches]

    return out


_NULLABLE_TOOLS = [
    {**t, "function": {**t["function"], "parameters": _loosen_nulls(t["function"]["parameters"])}}
    for t in TOOLS
]

_CLARIFY_FIELDS = {"name", "label", "ui", "options"}
_CLARIFY_UI = {"dropdown", "radio", "checkbox"}

# Every argument name any tool accepts. Derived from TOOLS so it can never
# drift out of sync with the schemas.
_TOOL_ARG_NAMES = frozenset(
    prop
    for t in TOOLS
    for prop in t["function"]["parameters"].get("properties", {})
)


def _tools_for(suppress_ask: bool) -> list[dict]:
    """Tool list for this turn.

    Parameters are served from the null-tolerant schema — see _loosen_nulls.

    The tool *set* stays intact for the first round (removing a tool outright
    makes gpt-oss hallucinate the removed call, which Groq then rejects). Only
    after we have intercepted one `ask_user` do we drop it, so it physically
    cannot ask a second round — deterministic, and it keeps the schema small.
    """
    if not suppress_ask:
        return _NULLABLE_TOOLS
    return [t for t in _NULLABLE_TOOLS if t["function"]["name"] != "ask_user"]


# Returned when the model tries to ask again after the user already answered.
_ALREADY_ASKED = {
    "status": "already_collected",
    "instruction": (
        "You already asked the user for their filters and they answered. "
        "Do NOT ask anything else. Use find_titles with their stated preferences "
        "and then call recommend_titles to show your picks now."
    ),
}


# ============================================================ model routing
_COMPLEX_WORDS = re.compile(r"""
\b(recommend|recommendations|suggest|suggestion|compare|comparison|versus|vs\b
|better|why|explain|summarize|summarise|summary|plot|ending|story|twist|critic
|critics|review|reviews|analysis|breakdown|themes|theme|meaning|symbolism|worth
|similar|alternatives|like\s+(this|that|it)|deep|in[- ]depth|detailed|opinion
|masterpiece|underrated|overrated|best\s+\d|top\s+\d|sequence|similar\s+to)
""", re.I | re.X)


def route_task(message: str, history_len: int = 0, force_large: bool = False) -> str:
    """Pick 'small' or 'large' with a heuristic — no extra model call."""
    text = (message or "").strip()
    if force_large:
        return "large"
    if _COMPLEX_WORDS.search(text):
        return "large"
    if len(text) > 200:
        return "large"
    if text.count("?") > 1:
        return "large"
    if history_len > 12:
        return "large"
    return "small"


def _model(task: str) -> str:
    return _settings.model_for(task)


# ======================================================== model failover
# Free tiers meter quota PER MODEL, so the model that just hit its daily
# token cap is almost never the only one left with budget. Walking a chain of
# models turns a hard wall into a slightly different answer.
#
# Only transient conditions are retried. A malformed request or a refused
# prompt will fail identically on every model, so retrying just burns quota.

_FAILOVER_NAMES = ("RateLimitError", "ServiceUnavailableError",
                   "APIConnectionError", "APITimeoutError", "InternalServerError")
_FAILOVER_HINTS = ("rate_limit", "rate limit", "overloaded", "timeout",
                   "timed out", "connection", "502", "503", "529", "temporarily")

# A model can also simply emit unusable arguments — truncated JSON, or the
# `<tool_call><function=…>` markup gpt-oss and qwen both fall back to. That is
# bad luck in sampling, not a malformed request, so the next model genuinely
# has a better chance. Retrying is safe because `tried` blocks re-selecting
# the same one.
_BAD_GENERATION_HINTS = ("failed to parse tool call arguments",
                         "tool call validation failed",
                         "tool_use_failed",
                         "failed to generate json",
                         "failed to validate json")


# Longest we will sit on a provider's back and wait rather than fail. Above
# this the user is better served by an immediate, honest error.
_MAX_RATE_LIMIT_WAIT = 45.0


def _is_bad_generation(exc: BaseException) -> bool:
    """True when the model produced output we could not parse (not a network fault)."""
    text = str(exc).lower()
    return any(h in text for h in _BAD_GENERATION_HINTS)


def _rate_limit_wait(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait, if it is short enough to sit out.

    Groq rate-limits per *organisation*, not per model, so failing over to a
    sibling model does not help: the whole chain 429s in the same breath and
    the turn dies with "every model in the chain failed". The provider tells us
    when the window reopens — `retry-after`, or Groq's own
    `x-ratelimit-reset-*` values, which arrive as "58.774s" — so waiting that
    out turns a hard failure into a slower success.

    Returns None for day/month quotas, which will not clear in a user turn.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None

    body = ""
    try:
        body = (resp.text or "").lower()
    except Exception:
        body = ""

    # A daily/monthly budget will not refill mid-conversation; do not hang.
    for marker in ("tokens per day", "tpd", "requests per day", "rpd"):
        if marker in body:
            return None

    headers = {}
    try:
        headers = {k.lower(): v for k, v in resp.headers.items()}
    except Exception:
        headers = {}

    candidates = [
        headers.get("retry-after"),
        headers.get("x-ratelimit-reset-tokens"),
        headers.get("x-ratelimit-reset-requests"),
    ]
    for raw in candidates:
        if not raw:
            continue
        text = str(raw).strip().lower()
        try:
            # Order matters: "1500ms" also ends with "s", so milliseconds has
            # to be tested first or it parses as 1500 *minutes*-of-seconds.
            if text.endswith("ms"):
                seconds = float(text[:-2]) / 1000.0
            elif text.endswith("s"):
                seconds = float(text[:-1])
            else:
                seconds = float(text)
        except ValueError:
            continue
        if 0 < seconds <= _MAX_RATE_LIMIT_WAIT:
            return seconds
    return None


def _is_failover_error(exc: BaseException) -> bool:
    """True when another model would plausibly succeed where this one failed."""
    if type(exc).__name__ in _FAILOVER_NAMES:
        return True
    text = str(exc).lower()
    if any(h in text for h in _FAILOVER_HINTS):
        return True
    return any(h in text for h in _BAD_GENERATION_HINTS)


def _failover_chain(task: str) -> list[str]:
    """Model ids to try for `task`, routed one first."""
    chain = _settings.failover_chain(task)
    if chain:
        return chain
    primary = _model(task)
    return [primary] if primary else []


def _note_failover(frm: str, to: str, task: str, exc: BaseException) -> None:
    logger.warning("llm failover %s -> %s (%s): %s", frm, to, task, exc)
    metrics.llm_error("failover")


def _complete_with_failover(messages: list[dict], tools: list[dict],
                            chain: list[str], tried: list[str], task: str):
    """Non-streaming completion, moving along the chain on transient errors."""
    last_exc: BaseException | None = None
    for name in chain:
        if name in tried:
            continue
        waited_for = 0.0
        while True:
            try:
                client, model_id = _resolve_target(name)
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.55,
                    **_extra(),
                )
                return resp, model_id, tried
            except Exception as exc:
                wait = _rate_limit_wait(exc)
                if wait and waited_for + wait <= _MAX_RATE_LIMIT_WAIT:
                    waited_for += wait
                    logger.info("rate limited on %s; waiting %.1fs (retry %d)",
                                name, wait, int(waited_for / 5) + 1)
                    metrics.llm_error("rate_limit_wait")
                    time.sleep(wait)
                    continue
                tried.append(name)
                last_exc = exc
                if not _is_failover_error(exc):
                    raise
                nxt = next((c for c in chain if c not in tried), None)
                if not nxt:
                    raise
                _note_failover(name, nxt, task, exc)
                # Hand over to the next model. Without this the `while` retries
                # the same model forever, so a rate-limited provider spins until
                # the process is killed instead of ever reaching the fallback.
                break
    if last_exc:
        raise last_exc
    raise RuntimeError("no model available for this request")


def _stream_with_failover(messages: list[dict], tools: list[dict],
                          chain: list[str], tried: list[str], task: str):
    """Streaming completion. `create` is lazy, so the 429 often only surfaces
    on the first chunk read — we pull one to find out before handing it on."""
    last_exc: BaseException | None = None
    for name in chain:
        if name in tried:
            continue
        # Rate limits are per organisation, so give the provider's own
        # retry window a chance before spending the rest of the chain.
        waited_for = 0.0
        while True:
            try:
                client, model_id = _resolve_target(name)
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.55,
                    stream=True,
                    stream_options={"include_usage": True},
                    **_extra(),
                )
                # Probe the stream so a lazy 429/503 is raised here rather than
                # halfway through the user's reply. The chunk is re-yielded to the
                # caller, so nothing is lost.
                stream_it = iter(stream)
                try:
                    peek = next(stream_it)
                except StopIteration:
                    peek = None
                return _ReplayedStream(stream_it, peek), model_id, tried
            except Exception as exc:
                wait = _rate_limit_wait(exc)
                # Cap the total time spent sleeping so a queue of short
                # windows cannot add up to an invisible multi-minute hang.
                if wait and waited_for + wait <= _MAX_RATE_LIMIT_WAIT:
                    waited_for += wait
                    logger.info("rate limited on %s; waiting %.1fs (retry %d)",
                                name, wait, int(waited_for / 5) + 1)
                    metrics.llm_error("rate_limit_wait")
                    time.sleep(wait)
                    continue
                tried.append(name)
                last_exc = exc
                if not _is_failover_error(exc):
                    raise
                nxt = next((c for c in chain if c not in tried), None)
                if not nxt:
                    raise
                _note_failover(name, nxt, task, exc)
                # Hand over to the next model. Without this the `while` retries
                # the same model forever, so a rate-limited provider spins until
                # the process is killed instead of ever reaching the fallback.
                break
    if last_exc:
        raise last_exc
    raise RuntimeError("no model available for this request")


class _ReplayedStream:
    """Yields the probed chunk first, then the rest of the provider stream.

    Holds the single iterator produced during the probe. Taking a fresh
    `iter()` here would start the provider stream again on some clients and
    replay the first chunk twice.
    """

    def __init__(self, stream_it, first):
        self._it = stream_it
        self._first = first

    def __iter__(self):
        if self._first is not None:
            yield self._first
        yield from self._it


# ================================================== usage / cost accounting
def _record_usage(resp, model: str, task: str, started: float,
                  session_id: str | None, visitor_id: str | None) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    pp, pc = _settings.prices_for(model)
    cost = (pt * pp + ct * pc) / 1_000_000
    latency = int((time.time() - started) * 1000)
    metrics.llm_call(model, task)
    metrics.llm_tokens(model, pt, ct)
    metrics.llm_cost(model, cost)
    metrics.llm_latency(model, latency / 1000)
    try:
        db.log_llm_usage(
            task=task, model=model, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=cost, latency_ms=latency, cached=False,
            session_id=session_id, visitor_id=visitor_id,
        )
    except Exception:
        logger.warning("could not persist llm usage", exc_info=True)
    return pt, ct


def _record_cache_hit(task: str, session_id: str | None, visitor_id: str | None) -> None:
    """A cache hit that avoided a model call still shows up in the dashboard."""
    try:
        db.log_llm_usage(
            task=f"{task}:cached", model="cache", prompt_tokens=0, completion_tokens=0,
            cost_usd=0.0, latency_ms=0, cached=True,
            session_id=session_id, visitor_id=visitor_id,
        )
    except Exception:
        pass


def log_feedback_score(session_id: str | None, rating: str,
                       trace_id: str | None = None,
                       app_rating: int | None = None) -> None:
    """Attach the user's verdict (and star rating) to the answer's trace."""
    if not trace_id:
        return
    obs.score(
        "user_satisfaction",
        _FEEDBACK_SCORE.get(rating, 0.0),
        trace_id=trace_id,
        session_id=session_id,
        comment=rating,
    )
    if app_rating:
        obs.score(
            "app_rating",
            float(app_rating),
            trace_id=trace_id,
            session_id=session_id,
            comment=f"{app_rating}/5",
            source="user-feedback",
        )


# ======================================================== simple primitives
def complete_json(model: str, prompt: str, task: str, max_tokens: int = 400,
                  temperature: float = 0.1,
                  session_id: str | None = None, visitor_id: str | None = None) -> dict | None:
    """A tiny single-shot JSON call (scope checks, name extraction, titles).

    Walks the failover chain like the chat path: this used to fail silently on
    every turn whenever the routed model was rate-limited, which meant burning
    a doomed call on every single message.
    """
    if not has_key():
        return None
    started = time.time()
    messages = [{"role": "user", "content": prompt}]
    tried: list[str] = []
    chain = _failover_chain("small") or [model]
    last_exc: Exception | None = None

    for name in chain:
        try:
            with obs.observation(
                f"generate-{task.replace('_', '-')}", as_type="generation", model=name,
                input=messages, session_id=session_id, user_id=visitor_id,
            ) as gen:
                resp = _client_instance().chat.completions.create(
                    model=name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    **_extra(),
                )
            pt, ct = _record_usage(resp, name, task, started, session_id, visitor_id)
            raw = resp.choices[0].message.content or ""
            # Some reasoning models return the answer in `reasoning` instead.
            if not raw.strip():
                raw = getattr(resp.choices[0].message, "reasoning", None) or ""
            obs.record_generation(gen, output=raw, input_tokens=pt,
                                  output_tokens=ct, model=name)
            return _extract_json(raw)
        except Exception as exc:
            last_exc = exc
            tried.append(name)
            logger.warning("%s call failed on %s", task, name, exc_info=True)
            if not _is_failover_error(exc):
                break
            nxt = next((c for c in chain if c not in tried), None)
            if not nxt:
                break
            _note_failover(name, nxt, task, exc)

    logger.warning("%s produced no usable JSON", task)
    metrics.llm_error(task)
    return None


def complete_text(model: str, system: str, prompt: str, task: str,
                  max_tokens: int = 500, temperature: float = 0.6,
                  session_id: str | None = None, visitor_id: str | None = None) -> str:
    if not has_key():
        return ""
    started = time.time()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    try:
        with obs.observation(
            f"generate-{task.replace('_', '-')}", as_type="generation", model=model,
            input=messages, session_id=session_id, user_id=visitor_id,
        ) as gen:
            resp = _client_instance().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **_extra(),
            )
            pt, ct = _record_usage(resp, model, task, started, session_id, visitor_id)
            msg = resp.choices[0].message
            text = (msg.content or "").strip()
            if not text:
                text = (getattr(msg, "reasoning", None) or "").strip()
            obs.record_generation(gen, output=text, input_tokens=pt,
                                  output_tokens=ct, model=model)
            return text
    except Exception:
        logger.warning("%s call failed", task, exc_info=True)
        metrics.llm_error(task)
        return ""


def _extract_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ================================================================ greeting
_GREETING_PROMPT = """You are Smart Entertainment BOT 🎬, a friendly movie, series and
animation assistant.

Write the opening shown when someone opens the app. Return STRICT JSON only:
{
  "greeting": "...",        // for a brand-new visitor
  "returning": "...",       // for someone who has chatted before; MUST contain {name}
  "suggestions": ["...", "...", "...", "..."]
}

Rules: 1-2 short sentences each, warm and human, not robotic. Introduce yourself
as Smart Entertainment BOT. Mention that they can share their name (never
require it). 'greeting' should read naturally without a name. 'returning' must
greet {name} by that exact placeholder. Suggestion prompts are things a viewer
might tap to start (max ~7 words each), e.g. "Suggest a Korean thriller series".
Give exactly 4 suggestions."""

_DEFAULT_GREETING = {
    "greeting": (
        "Hi, I'm **Smart Entertainment BOT** 🎬 — what are we watching today? "
        "Tell me your name if you'd like me to remember it, or just ask away."
    ),
    "returning": "Welcome back, {name}! 🎬 Ready to find your next great watch?",
    "suggestions": [
        "Recommend a thriller web series",
        "Best Hindi movies of 2024",
        "Something like Money Heist",
        "A feel-good animated film",
    ],
}


def greeting_templates(session_id: str | None = None) -> dict:
    """Generate the greeting ONCE and cache it for everyone (24h).

    Personalisation is pure string formatting, so a returning user is greeted by
    name without spending a single token.
    """
    cached = cache.get_json("greeting:v3")
    if cached:
        return cached
    if not has_key():
        return _DEFAULT_GREETING
    data = complete_json(
        _model("small"), _GREETING_PROMPT, task="greeting", max_tokens=600,
        session_id=session_id,
    )
    out = {
        "greeting": (data or {}).get("greeting") or _DEFAULT_GREETING["greeting"],
        "returning": (data or {}).get("returning") or _DEFAULT_GREETING["returning"],
        "suggestions": [str(s).strip() for s in ((data or {}).get("suggestions") or [])
                        if str(s).strip()][:5] or _DEFAULT_GREETING["suggestions"],
    }
    cache.set_json("greeting:v3", out)
    return out


def personalise_greeting(name: str | None, returning: bool) -> dict:
    templates = greeting_templates()
    if returning and name:
        text = templates.get("returning") or templates["greeting"]
    else:
        text = templates["greeting"]
    text = (text or "").replace("{name}", name) if name else (text or "").replace("{name}", "there")
    return {"greeting": text.strip(), "suggestions": templates["suggestions"]}


# ============================================================ session title
def _clamp_title(t: str) -> str:
    t = " ".join((t or "").split())
    return t[:69].rstrip() + "…" if len(t) > 70 else t


_TITLE_PROMPT = """Write ONE short, specific chat title (50-70 characters, no quotes) for a
conversation about movies, series or anime. Name the actual subject — the genre,
title or country — so it is distinguishable from other chats. Never use generic
phrasing like "Movie recommendations".

Conversation:
%s

Return STRICT JSON only: {"title": "..."}"""


def generate_title(history: list[dict], session_id: str | None = None,
                   visitor_id: str | None = None) -> str:
    users = [m.get("content") for m in history if m.get("role") == "user" and m.get("content")]
    fallback = _clamp_title((users[0] if users else "New entertainment chat").strip())
    if not has_key() or not history:
        return fallback
    transcript = "\n".join(
        f"{m['role']}: {str(m['content'])[:180]}" for m in history[-8:] if m.get("content")
    )
    if len(transcript) > 1200:
        transcript = transcript[:1200]
    data = complete_json(
        _model("small"), _TITLE_PROMPT % transcript, task="title", max_tokens=120,
        temperature=0.5, session_id=session_id, visitor_id=visitor_id,
    )
    title = str((data or {}).get("title") or "").strip().strip('"').strip()
    return _clamp_title(title or fallback)


# ============================================================ title summary
def title_summary(title_key: str, detail: str = "quick",
                  session_id: str | None = None) -> dict:
    """A generated, cached plot summary for one title (small model)."""
    ck = f"summary:{title_key}:{detail}"
    cached = cache.get_json(ck)
    if cached:
        return cached

    card = entertainment.title_details(title_key) or {}
    if not card:
        return {"key": title_key, "summary": "", "title": ""}
    entertainment.remember_card(card)

    kind = "series" if card.get("media_type") == "tv" else "film"
    facts = {
        "title": card.get("title"),
        "year": card.get("year"),
        "kind": kind,
        "genres": (card.get("genres") or [])[:4],
        "runtime": card.get("runtime") or card.get("episode_runtime"),
        "seasons": card.get("seasons"),
        "cast": [c.get("name") for c in (card.get("cast") or [])[:5]],
        "creators": (card.get("creators") or [])[:3] or [c.get("name") for c in (card.get("crew") or [])[:3]],
        "tagline": card.get("tagline") or "",
        "overview": (card.get("overview") or card.get("synopsis") or "")[:900],
    }
    length = "2-3 sentences" if detail == "quick" else "a detailed 4-6 sentence plot summary"

    text = complete_text(
        _model("small"),
        "You summarise films and series accurately and concisely. Never spoil the "
        "ending unless asked. If the facts are thin, say so plainly.",
        f"Summarise this {kind} in {length} using only the facts given.\n\n{facts}",
        task="summary", max_tokens=400 if detail == "quick" else 700,
        session_id=session_id,
    )
    out = {
        "key": title_key,
        "title": card.get("title"),
        "summary": text or facts["overview"] or "I couldn't find a detailed summary for this one.",
    }
    cache.set_json(ck, out)
    return out


# ============================================================== tool layer
def _tool_result(slot_id, payload: dict) -> dict:
    return {"role": "tool", "tool_call_id": slot_id, "content": json.dumps(payload, default=str)}


# A dropdown can hold a long scrolling menu; a chip row cannot. The small
# model tends to offer only ~5 values, so dropdowns are topped up from the
# catalog's own reference lists to make the menu genuinely useful.
_TARGET_DROPDOWN_OPTIONS = 16


def _dropdown_pool(name: str) -> list[str]:
    key = name.strip().lower()
    if key in ("country", "origin", "nation"):
        return list(entertainment.COUNTRIES.values())
    if key in ("language", "dubbing", "dub", "audio"):
        return list(entertainment.LANGUAGES.values())
    if key in ("year", "era", "decade", "release_year", "released"):
        return list(entertainment.YEAR_BUCKETS)
    if key in ("format", "type", "media", "media_type"):
        return list(entertainment.FORMAT_OPTIONS)
    if key in ("genre", "category", "genres"):
        return list(entertainment.CURATED_GENRES)
    return []


def _pad_dropdown(name: str, options: list[dict], ui: str) -> list[dict]:
    """Top a dropdown up to a usable size, preserving the model's own choices."""
    if ui != "dropdown":
        return options
    pool = _dropdown_pool(name)
    if not pool or len(options) >= _TARGET_DROPDOWN_OPTIONS:
        return options

    have = {o["label"].strip().lower() for o in options}
    for label in pool:
        if len(options) >= _TARGET_DROPDOWN_OPTIONS:
            break
        if label.lower() in have:
            continue
        options.append({"value": label, "label": label})
        have.add(label.lower())

    # Split only AFTER padding, otherwise the newly added options are lost.
    # "Any" stays first; the model's own picks keep their relative order.
    unspecified = ("any", "any country", "any language", "any year",
                   "any genre", "any format", "any type")
    # De-duplicate near-identical "Any" variants the pool may also contain.
    seen_any = False
    head: list[dict] = []
    for o in options:
        if o["label"].strip().lower() in unspecified:
            if seen_any:
                continue
            seen_any = True
        head.append(o)
    tail = [o for o in options if o not in head]
    return head + tail


# Words that already pin down a filter. If the user's own message implies one,
# asking about it is a pointless question — so those fields are dropped.
_IMPLIED_COUNTRY = {
    "india": ("bollywood", "hindi cinema", "hindi film", "tollywood", "kollywood",
              "mollywood", "tamil", "telugu", "kannada", "malayalam", "marathi",
              "bengali", "punjabi", "gujarati", "indian film", "indian cinema",
              "rajinikanth", "vijay", "shah rukh", "amitabh", "rajkummar"),
    "South Korea": ("k-drama", "kdrama", "k-movie", "kmovie", "korean drama", "korean film"),
    "Japan": ("anime", "manga", "studio ghibli", "japanese film", "yakusoku"),
    "United States": ("hollywood", "american film", "american movie"),
    "United Kingdom": ("british film", "bbc drama"),
    "France": ("french film", "cinema francais"),
    "Italy": ("italian film", "cinema italiano"),
}
_IMPLIED_LANGUAGE = {
    "hindi": ("bollywood", "hindi"),
    "japanese": ("anime", "manga"),
    "korean": ("k-drama", "kdrama", "korean"),
    "english": ("hollywood", "english"),
}
_IMPLIED_GENRE = {
    "Animation": ("anime", "animated", "cartoon", "ghibli"),
}


def _already_implied(message: str) -> tuple[set[str], set[str], set[str]]:
    """(countries, languages, genres) the user's wording has already settled."""
    text = (message or "").lower()
    countries = {c for c, words in _IMPLIED_COUNTRY.items() if any(w in text for w in words)}
    languages = {l for l, words in _IMPLIED_LANGUAGE.items() if any(w in text for w in words)}
    genres = {g for g, words in _IMPLIED_GENRE.items() if any(w in text for w in words)}
    return countries, languages, genres


def _drop_implied_fields(fields: list[dict], user_text: str) -> list[dict]:
    """Remove filter fields the user's own question already answered."""
    countries, languages, genres = _already_implied(user_text)
    if not (countries or languages or genres):
        return fields

    keep: list[dict] = []
    for f in fields:
        name = f["name"].strip().lower().rstrip("s")
        labels = {o["label"].strip().lower() for o in f["options"]}
        if name in ("country", "origin", "nation") and labels & {c.lower() for c in countries}:
            continue
        if name in ("language", "dubbing", "dub", "audio") and labels & {l.lower() for l in languages}:
            continue
        if name in ("genre", "category") and labels & {g.lower() for g in genres}:
            continue
        keep.append(f)
    return keep


def _normalise_clarify(args: dict, user_text: str = "") -> dict:
    """Sanitise the ask_user payload so the UI can render it safely."""
    fields: list[dict] = []
    for raw in (args.get("fields") or [])[:4]:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()[:32]
        label = str(raw.get("label") or name).strip()[:60]
        ui = str(raw.get("ui") or "radio").lower()
        if not name or ui not in _CLARIFY_UI:
            continue
        options = []
        for opt in (raw.get("options") or [])[:8]:
            if isinstance(opt, dict):
                value = str(opt.get("value") or "").strip()[:60]
                olabel = str(opt.get("label") or value).strip()[:60]
            else:
                value = olabel = str(opt).strip()[:60]
            if value:
                options.append({"value": value, "label": olabel})
        if options:
            options = _pad_dropdown(name, options, ui)
            fields.append({"name": name, "label": label, "ui": ui, "options": options})

    # Never ask about something the user's own wording already settled.
    fields = _drop_implied_fields(fields, user_text)

    # Quick options must NOT restate what the fields already offer — otherwise
    # the same chips render twice and look like a bug to the user.
    offered = {o["label"].strip().lower() for f in fields for o in f["options"]}
    quick: list[str] = []
    for raw in args.get("quick_options") or []:
        # The schema accepts both a bare string and {label: "..."}; models pick
        # either, so reduce them here rather than rejecting the whole turn.
        if isinstance(raw, dict):
            label = str(raw.get("label") or raw.get("value") or "").strip()[:60]
        else:
            label = str(raw).strip()[:60]
        if not label:
            continue
        key = label.lower()
        # De-duplicate BEFORE capping, or a run of repeats at the front would
        # push every genuinely distinct option past the limit.
        if key in offered or key in {q.lower() for q in quick}:
            continue
        quick.append(label)
        if len(quick) >= 4:
            break

    return {
        "question": str(args.get("question") or "").strip()[:600],
        "fields": fields,
        "quick_options": quick,
    }


def _run_tool(name: str, args: dict, session_id: str | None,
              user_text: str = "") -> tuple[dict, dict]:
    """Execute one tool call.

    Returns (tool_result_payload, side_effects) where side_effects may contain
    `cards` (titles to render) or `clarify` (ask the user and stop).
    """
    if name == "find_titles":
        limit = max(1, min(int(args.get("limit") or 8), 20))
        cards = entertainment.find_titles(
            query=args.get("query") or "",
            media_type=args.get("media_type") or "any",
            genres=args.get("genres") or [],
            country=args.get("country"),
            year_from=_as_int(args.get("year_from")),
            year_to=_as_int(args.get("year_to")),
            language=args.get("language"),
            min_rating=_as_float(args.get("min_rating")),
            sort=args.get("sort"),
            limit=limit,
        )
        for c in cards:
            entertainment.remember_card(c)
        if cards:
            return (
                {
                    "titles": entertainment.compact_many(cards, 200),
                    "count": len(cards),
                    "hint": "Use these exact title keys in recommend_titles.",
                },
                {"cards": cards},
            )

        # Nothing matched. Say so plainly and let the model ask whether the
        # user wants something else — showing unrelated popular titles here is
        # how "tell me about Obsession" used to return The Odyssey.
        named = bool(args.get("query")) and not any(
            [args.get("genres"), args.get("country"), args.get("year_from"),
             args.get("year_to"), args.get("language"), args.get("min_rating")]
        )
        if named:
            hint = (
                "NO MATCH for that exact title. Do NOT call recommend_titles and do "
                "NOT suggest other titles unless the user asks. Tell them plainly that "
                "you could not find it, and ask whether they would like something "
                "similar instead."
            )
        else:
            hint = (
                "Nothing matched those filters. Do NOT call recommend_titles. Tell the "
                "user you couldn't find anything for that, and ask whether they'd like "
                "to loosen a filter or try a different genre."
            )
        return ({"titles": [], "count": 0, "not_found": True, "hint": hint}, {})

    if name == "get_title_summary":
        out = title_summary(str(args.get("title_key") or ""),
                            args.get("detail") or "quick", session_id)
        return ({"summary": out.get("summary"), "title": out.get("title")}, {})

    if name == "get_title_reviews":
        key = str(args.get("title_key") or "")
        limit = max(1, min(int(args.get("limit") or 5), 10))
        items = entertainment.reviews(key, limit)
        ratings = [r["rating"] for r in items if r.get("rating")]
        avg = round(sum(ratings) / len(ratings), 1) if ratings else None
        return ({
            "title": key, "reviews": items, "average_review_score": avg,
            "count": len(items),
        }, {})

    if name == "get_cast_crew":
        card = entertainment.title_details(str(args.get("title_key") or "")) or {}
        entertainment.remember_card(card)
        return ({
            "title": card.get("title"),
            "cast": [{"name": c.get("name"), "character": c.get("character")}
                     for c in (card.get("cast") or [])[:8]],
            "crew": [{"name": c.get("name"), "job": c.get("job")}
                     for c in (card.get("crew") or [])[:6]],
            "creators": (card.get("creators") or [])[:3],
        }, {})

    if name == "get_similar_titles":
        key = str(args.get("title_key") or "")
        limit = max(1, min(int(args.get("limit") or 6), 10))
        cards = entertainment.similar_titles(key, limit)
        for c in cards:
            entertainment.remember_card(c)
        return ({"titles": entertainment.compact_many(cards, 160)}, {"cards": cards})

    if name == "ask_user":
        return ({"status": "delivered"}, {"clarify": _normalise_clarify(args, user_text)})

    if name == "recommend_titles":
        keys = args.get("title_keys") or []
        cards = entertainment.cards_for_keys(keys)[: _settings.max_recommendations]
        return ({"shown_titles": [c["title"] for c in cards]}, {"cards": cards, "recommend": True})

    return ({"error": "unknown tool"}, {})


def _traced_tool(name: str, args: dict, session_id: str | None,
                 visitor_id: str | None, parent=None,
                 intercept_ask: bool = False,
                 user_text: str = "") -> tuple[dict, dict]:
    """Run a tool inside a typed observation (retriever for lookups, tool for actions).

    `intercept_ask` swallows a repeat `ask_user` after the user already answered,
    returning guidance to recommend instead. Intercepting matters: a reasoning
    model still emits a removed tool call, and the provider then rejects the
    whole request with a validation error.
    """
    if name == "ask_user" and intercept_ask:
        with obs.tool(name, input=args, session_id=session_id,
                      user_id=visitor_id, parent=parent) as t:
            payload, effects = _ALREADY_ASKED, {}
            if t is not None:
                try:
                    t.update(output=payload)
                except Exception:
                    logger.debug("could not record tool output", exc_info=True)
        return payload, effects

    with obs.tool(name, input=args, session_id=session_id, user_id=visitor_id,
                  parent=parent) as t:
        payload, effects = _run_tool(name, args, session_id, user_text)
        if t is not None:
            try:
                t.update(output=payload)
            except Exception:
                logger.debug("could not record tool output", exc_info=True)
    return payload, effects


def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ============================================================ answer writer
_PROSE_SYSTEM = (
    "You are Smart Entertainment BOT 🎬. Write a short, warm reply for the user about "
    "the titles listed. Refer to them by name only. Be concise — a few sentences."
)


def _write_prose(messages: list[dict], cards: list[dict], session_id: str | None) -> str:
    """Fallback when the model called recommend_titles without writing any text.

    Uses the SMALL model — this path is rare, and the expensive model has
    already done the reasoning.
    """
    listing = ", ".join(f"{c.get('title')} ({c.get('year')})" for c in cards[:5])
    prompt = (
        "The user has been shown these titles: "
        f"{listing}. Write your reply to them now (2-4 sentences)."
    )
    return complete_text(
        _model("small"), _PROSE_SYSTEM, prompt, task="prose", max_tokens=300,
        temperature=0.6, session_id=session_id,
    )


def _finalise_text(text: str, cards: list[dict]) -> str:
    # The whole reply is complete here, so it is the first point where a fence
    # can be told apart from real code and cleaned up safely.
    text = _drop_fenced_payloads(
        strip_bare_tool_json(strip_tool_markup((text or "").strip()))
    ).strip()
    if text:
        return text
    titles = ", ".join(c.get("title") or "?" for c in cards[:5])
    return f"Here are {len(cards)} picks for you: {titles}. Tap any card for full details."


# ============================================================== chat loop
_MAX_ROUNDS = 6


def _prepare(history: list[dict], user_name: str | None,
             after_clarify: bool = False) -> list[dict]:
    system = SYSTEM_PROMPT
    if user_name:
        system += (
            f"\n## The user's name\nYou already know their name is **{user_name}**. "
            "Use it naturally now and then — never insist on it, never ask again."
        )
    else:
        system += (
            "\n## Names\nYou do not know the user's name yet. The app has already "
            "offered to learn it, so never ask for it yourself — just be warm."
        )
    if after_clarify:
        # Hard stop on the over-asking loop: the widget already collected the
        # filters, so this turn must produce titles.
        system += (
            "\n## IMPORTANT — the user just answered your filter questions\n"
            "They have now given you their preferences. DO NOT call `ask_user` "
            "again this turn. Call `find_titles` with their filters and then "
            "`recommend_titles`. Make a recommendation now, even if you would "
            "have liked to know one more detail."
        )
    else:
        # Only when a question might still be asked — keeps the token cost off
        # every turn while making the dropdowns genuinely rich.
        system += "\n" + entertainment.filter_reference()
    return [{"role": "system", "content": system}] + history


def run_chat(history: list[dict], session_id: str | None = None,
             visitor_id: str | None = None, user_name: str | None = None,
             parent=None, after_clarify: bool = False) -> dict:
    """Run one assistant turn. `history` ends with the latest user message."""
    if not has_key():
        return {"kind": "answer", "text": _key_missing_msg(), "titles": [], "suggestions": []}
    if not entertainment.any_source_configured():
        return {"kind": "answer", "text": _NO_SOURCE_MSG, "titles": [], "suggestions": []}

    last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    task = route_task(last_user, history_len=len(history))
    chain = _failover_chain(task)
    model = chain[0] if chain else _model(task)
    tried: list[str] = []

    messages = _prepare(history, user_name, after_clarify)
    titles_out: list[dict] = []
    usage = {"prompt": 0, "completion": 0}

    try:
        suppress_ask = False
        for round_no in range(_MAX_ROUNDS):
            started = time.time()
            # One `generation` per model call — never aggregated across rounds,
            # so you can see what the model chose after each tool result.
            with obs.observation(
                "generate-response", as_type="generation", model=model,
                input=messages, session_id=session_id, user_id=visitor_id,
                metadata={"round": round_no + 1, "task": task}, parent=parent,
            ) as gen:
                resp, model, tried = _complete_with_failover(
                    messages, _tools_for(suppress_ask), chain, tried, task,
                )
                pt, ct = _record_usage(resp, model, task, started, session_id, visitor_id)
                usage["prompt"] += pt
                usage["completion"] += ct

                msg = resp.choices[0].message
                tool_calls = msg.tool_calls or []
                content = strip_tool_markup((msg.content or "").strip())
                if not content:
                    content = strip_tool_markup(getattr(msg, "reasoning", None) or "").strip()

                obs.record_generation(
                    gen,
                    output=content or f"(tool calls: {[t.function.name for t in tool_calls]})",
                    input_tokens=pt, output_tokens=ct, model=model,
                )

            if not tool_calls:
                return {
                    "kind": "answer", "text": _finalise_text(content, titles_out),
                    "titles": titles_out, "suggestions": [], "usage": usage,
                }

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in tool_calls],
            })

            stop: dict | None = None
            recommended = False
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                metrics.tool_call(name)

                if name == "ask_user" and after_clarify:
                    suppress_ask = True
                payload, effects = _traced_tool(
                    name, args, session_id, visitor_id, parent,
                    intercept_ask=(name == "ask_user" and after_clarify),
                    user_text=last_user,
                )
                messages.append(_tool_result(tc.id, payload))

                if effects.get("cards"):
                    titles_out = _merge_titles(titles_out, effects["cards"])
                if effects.get("clarify"):
                    stop = effects["clarify"]
                    break
                if effects.get("recommend"):
                    recommended = True

            if stop:
                metrics.clarify(",".join(f["name"] for f in stop["fields"]) or "general")
                return {
                    "kind": "clarify", "text": stop["question"], "titles": [],
                    "suggestions": stop.get("quick_options") or [], "clarify": stop,
                    "usage": usage,
                }

            if recommended:
                return {
                    "kind": "answer",
                    "text": _finalise_text(content, titles_out),
                    "titles": titles_out, "suggestions": [], "usage": usage,
                }

        return {
            "kind": "answer",
            "text": _finalise_text(content, titles_out),
            "titles": titles_out, "suggestions": [], "usage": usage,
        }

    except Exception as exc:
        logger.exception("run_chat failed")
        metrics.llm_error("run_chat")
        return {
            "kind": "answer",
            "text": _model_error_message(exc),
            "titles": titles_out, "suggestions": [], "usage": usage,
        }


def _merge_titles(current: list[dict], incoming: list[dict]) -> list[dict]:
    """Keep the recommend-sized set, favouring what the model actually chose."""
    seen = {c["key"] for c in current}
    merged = list(current)
    for c in incoming:
        if c["key"] not in seen:
            seen.add(c["key"])
            merged.append(c)
    return merged[: _settings.max_recommendations]


# ============================================================== streaming
def stream_chat(history: list[dict], session_id: str | None = None,
                visitor_id: str | None = None, user_name: str | None = None,
                parent=None, after_clarify: bool = False):
    """Streaming variant of `run_chat`.

    Yields:
      {"type": "status",   "text": ...}   tool activity
      {"type": "token",    "text": ...}   assistant text delta
      {"type": "titles",   "titles": [...]} validated cards
      {"type": "clarify",  ...}           structured question widget
      {"type": "final",    kind, text, titles, suggestions, clarify}
    """
    if not has_key():
        yield {"type": "token", "text": _key_missing_msg()}
        yield {"type": "final", "kind": "answer", "text": _key_missing_msg(),
               "titles": [], "suggestions": [], "clarify": None}
        return
    if not entertainment.any_source_configured():
        yield {"type": "token", "text": _NO_SOURCE_MSG}
        yield {"type": "final", "kind": "answer", "text": _NO_SOURCE_MSG,
               "titles": [], "suggestions": [], "clarify": None}
        return
    last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    task = route_task(last_user, history_len=len(history))
    chain = _failover_chain(task)
    model = chain[0] if chain else _model(task)
    tried: list[str] = []

    messages = _prepare(history, user_name, after_clarify)
    titles_out: list[dict] = []
    usage = {"prompt": 0, "completion": 0}
    tool_messages: list[dict] = []

    try:
        content_buf = ""
        suppress_ask = False
        for round_no in range(_MAX_ROUNDS):
          # A round can still fail AFTER the first chunk ("Failed to parse tool
          # call arguments as JSON" arrives mid-stream). Retry it on the next
          # model, but only while nothing has reached the user — replaying a
          # half-printed reply would be worse than failing.
          last_round_exc: Exception | None = None
          # Set when every model answered but none of them actually said
          # anything. Distinguished from a hard failure so the round can be
          # retried instead of ending the turn.
          empty_round = False
          for attempt in range(len(chain) + 1):
            started = time.time()
            content_buf = ""
            reasoning_buf = ""
            tc_acc: dict[int, dict] = {}
            usage_chunk = None
            markup = _MarkupFilter()
            emitted = False
            try:

                # The generation observation spans the whole stream and is closed
                # once the round finishes, with the final text and token counts.
                with obs.observation(
                    "generate-response", as_type="generation", model=model,
                    input=messages, session_id=session_id, user_id=visitor_id,
                    metadata={"round": round_no + 1, "task": task, "streamed": True},
                    parent=parent,
                ) as gen:
                    stream, model, tried = _stream_with_failover(
                        messages, _tools_for(suppress_ask), chain, tried, task,
                    )

                    for chunk in stream:
                        if getattr(chunk, "usage", None):
                            usage_chunk = chunk.usage
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if getattr(delta, "content", None):
                            safe = markup.feed(delta.content)
                            if safe:
                                content_buf += safe
                                emitted = True
                                yield {"type": "token", "text": safe}
                        else:
                            # Reasoning tokens are NOT shown to the user — they are only
                            # kept as a fallback for models that answer inside `reasoning`.
                            rc = getattr(delta, "reasoning_content", None) or \
                                getattr(delta, "reasoning", None)
                            if rc:
                                reasoning_buf += rc
                        for tcd in getattr(delta, "tool_calls", None) or []:
                            # Gemini's OpenAI-compatible endpoint sends `index: null` on every tool-call
                            # delta. Using that as the dict key would merge two
                            # parallel calls into one slot and corrupt both, so a
                            # missing index falls back to "the next free slot".
                            key = tcd.index if tcd.index is not None else len(tc_acc)
                            slot = tc_acc.setdefault(key, {"id": None, "name": "", "args": ""})
                            if tcd.id:
                                slot["id"] = tcd.id
                            if tcd.function:
                                if tcd.function.name:
                                    slot["name"] = tcd.function.name
                                if tcd.function.arguments:
                                    slot["args"] += tcd.function.arguments

                    pt = ct = 0
                    if usage_chunk:
                        pt, ct = _record_usage(
                            type("R", (), {"usage": usage_chunk})(), model, task, started,
                            session_id, visitor_id,
                        )
                        usage["prompt"] += pt
                        usage["completion"] += ct

                    tail = markup.flush()
                    if tail:
                        content_buf += tail

                    obs.record_generation(
                        gen,
                        output=content_buf or f"(tool calls: {[s['name'] for s in tc_acc.values()]})",
                        input_tokens=pt, output_tokens=ct, model=model,
                    )
            except Exception as exc:
                # Mid-stream failure. Safe to retry only while the user hasn't
                # seen anything from this round, or they'd get it twice.
                last_round_exc = exc
                if emitted or not _is_failover_error(exc):
                    raise
                nxt = next((c for c in chain if c not in tried), None)
                if not nxt:
                    raise
                logger.warning("round %s failed mid-stream, retrying on %s",
                               round_no + 1, nxt)
                _note_failover(model, nxt, task, exc)
                metrics.llm_error("stream_round_retry")
                continue   # retry this round on the next model

            # A reasoning model that never produced `content` still has an
            # answer sitting in `reasoning` — surface it instead of nothing.
            if not content_buf.strip() and not tc_acc and reasoning_buf.strip():
                content_buf = strip_tool_markup(reasoning_buf)
                emitted = True
                yield {"type": "token", "text": content_buf}

                break   # round completed — leaves the attempt loop only
          else:
            # Every model in the chain produced unusable arguments.
            if emitted:
                # Something already reached the user; finish gracefully with
                # what we have rather than replacing a visible reply.
                yield {"type": "final", "kind": "answer",
                       "text": _finalise_text(content_buf, titles_out),
                       "titles": titles_out, "suggestions": [], "clarify": None,
                       "usage": usage}
                return
            # No exception, but also no output: every model returned an empty
            # completion. That happens with reasoning models, which sometimes
            # spend the whole round thinking and emit neither content nor a
            # tool call. It is a soft failure — the next round may well answer
            # — so treat it as one instead of killing the turn outright.
            if last_round_exc is None and round_no + 1 < _MAX_ROUNDS:
                # Retrying is only worthwhile if something is left to try.
                # When every model on every provider has already failed, the
                # extra rounds just replay the same wall six times and stretch
                # one honest error into minutes of silence.
                if any(c not in tried for c in chain):
                    logger.warning("round %d: no model produced any output; retrying",
                                   round_no + 1)
                    metrics.llm_error("stream_empty_round")
                    empty_round = True
                else:
                    raise RuntimeError("every model in the chain failed")
            else:
                raise last_round_exc or RuntimeError("every model in the chain failed")

            if empty_round:
                continue

            if not tc_acc:
                yield {"type": "final", "kind": "answer",
                       "text": _finalise_text(content_buf, titles_out),
                       "titles": titles_out, "suggestions": [], "clarify": None,
                       "usage": usage}
                return

            messages.append({
                "role": "assistant",
                "content": content_buf or "",
                "tool_calls": [{
                    "id": s["id"], "type": "function",
                    "function": {"name": s["name"], "arguments": s["args"]},
                } for s in tc_acc.values()],
            })

            stop: dict | None = None
            recommended = False
            for slot in tc_acc.values():
                name = slot["name"]
                try:
                    args = json.loads(slot["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                metrics.tool_call(name)

                if name != "ask_user":
                    yield {"type": "status", "text": _STATUS_TEXT.get(name, "Working on it…")}
                if name == "ask_user" and after_clarify:
                    # Swallow the repeat question and steer it to a recommendation.
                    suppress_ask = True
                payload, effects = _traced_tool(
                    name, args, session_id, visitor_id, parent,
                    intercept_ask=(name == "ask_user" and after_clarify),
                    user_text=last_user,
                )
                messages.append(_tool_result(slot["id"], payload))
                tool_messages.append(_tool_result(slot["id"], payload))

                if effects.get("cards"):
                    titles_out = _merge_titles(titles_out, effects["cards"])
                    yield {"type": "titles", "titles": titles_out}
                if effects.get("clarify"):
                    stop = effects["clarify"]
                    break
                if effects.get("recommend"):
                    recommended = True

            if stop:
                metrics.clarify(",".join(f["name"] for f in stop["fields"]) or "general")
                question = stop["question"]
                yield {"type": "token", "text": question}
                yield {"type": "clarify", "clarify": stop,
                       "suggestions": stop.get("quick_options") or []}
                yield {"type": "final", "kind": "clarify", "text": question, "titles": [],
                       "suggestions": stop.get("quick_options") or [],
                       "clarify": stop, "usage": usage}
                return

            if recommended:
                text = _finalise_text(strip_tool_markup(content_buf or reasoning_buf), titles_out)
                if not (content_buf or reasoning_buf).strip():
                    # Model only made the tool call — write the prose cheaply.
                    text = _write_prose(messages, titles_out, session_id)
                    if text:
                        yield {"type": "token", "text": text}
                yield {"type": "final", "kind": "answer", "text": text,
                       "titles": titles_out, "suggestions": [], "clarify": None,
                       "usage": usage}
                return

        text = _finalise_text(content_buf, titles_out)
        yield {"type": "final", "kind": "answer", "text": text, "titles": titles_out,
               "suggestions": [], "clarify": None, "usage": usage}

    except Exception as exc:
        logger.exception("stream_chat failed", exc_info=exc)
        metrics.llm_error("stream_chat")
        # A quota/rate-limit wall is not a transient glitch — telling the user
        # to "try again" sends them straight back into the same failure.
        msg = _model_error_message(exc, tried)
        yield {"type": "token", "text": msg}
        yield {"type": "final", "kind": "answer", "text": msg, "titles": titles_out,
               "suggestions": [], "clarify": None, "usage": usage}


_STATUS_TEXT = {
    "find_titles": "Searching the catalogs…",
    "get_title_summary": "Reading up on that…",
    "get_title_reviews": "Checking the reviews…",
    "get_cast_crew": "Looking up cast & crew…",
    "get_similar_titles": "Finding similar titles…",
    "recommend_titles": "Putting your picks together…",
}


# ====================================================== tool-markup filter
# Reasoning models (gpt-oss) sometimes echo their tool call into the text
# channel as well as the structured one, e.g.
#     <recommend_titles title_keys=["tmdb:movie:27205"] />
# That must never reach the user. Because tokens arrive one fragment at a time,
# a plain regex is not enough — we hold back anything after a '<' until we can
# tell markup from ordinary text.
_TOOL_NAMES = (
    "find_titles", "get_title_summary", "get_title_reviews", "get_cast_crew",
    "get_similar_titles", "ask_user", "recommend_titles",
    "tool_call", "tool_calls", "function_call", "function_calls",
)
# Some models emit a zero-width space inside the tag name.
_ZWSP = "\u200b"
_TOOL_MARKUP_ANY = re.compile(
    r"<(/?)(?:" + "|".join(_TOOL_NAMES) + r")\b[^>]*>?",
    re.I | re.S,
)


_MD_BLOCK = re.compile(r"```[a-zA-Z0-9_+#-]*\s*\n(.*?)```", re.S)


def _drop_fenced_payloads(text: str) -> str:
    """Remove a fenced block that only ever held a tool payload.

    gpt-oss likes to present its arguments as a ```json block. The streaming
    filter already swallows the JSON itself; this tidies the empty fence left
    behind. Blocks containing anything else are genuine code the user should
    see, so they stay.
    """
    if "```" not in text:
        return text

    def repl(m: re.Match) -> str:
        body = strip_bare_tool_json(m.group(1)).strip()
        return "" if not body else f"```{m.group(0)[3:m.group(0).find(chr(10))]}\n{body}\n```"

    cleaned = _MD_BLOCK.sub(repl, text)
    # An unmatched trailing fence (nothing after it to close the block).
    cleaned = re.sub(r"```[a-zA-Z0-9_+#-]*[ \t]*$", "", cleaned.rstrip())
    return cleaned


def strip_tool_markup(text: str) -> str:
    """Remove leaked tool-call markup.

    Safe to call on a partial stream, so it deliberately does no fence
    cleanup — a half-arrived "```" looks exactly like an orphan one.
    """
    if not text:
        return text
    cleaned = _TOOL_MARKUP_ANY.sub("", text.replace(_ZWSP, ""))
    return re.sub(r"<\|[^|]{0,40}\|>", "", cleaned)


def _is_tool_tag(tag: str) -> bool:
    """True when a complete `<...>` fragment is tool markup we must drop."""
    return bool(_TOOL_MARKUP_ANY.fullmatch(tag.replace(_ZWSP, "")))


# ==================================================== bare-JSON tool payloads
# gpt-oss does not always emit a real tool call. Sometimes it writes the
# arguments straight into its prose, e.g.
#
#     Here are five horror movies you will love …
#     {"title_keys": ["watchmode:movie:121949", …]}
#
# There is no tag to filter, so it renders as a JSON blob in the chat bubble.

# The first key of a JSON object — enough to classify it without parsing.
_FIRST_JSON_KEY = re.compile(r'\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:')
# How much we will hold back waiting to classify a '{' before giving up on it.
_JSON_PROBE_MAX = 400


def _json_object_end(text: str, start: int) -> int | None:
    """Index just past the '}' closing the object at `start`, else None."""
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _model_error_message(exc: BaseException, tried: list[str] | None = None) -> str:
    """Say what actually went wrong, in the user's terms."""
    name = type(exc).__name__
    detail = str(exc)
    exhausted = "every model I can reach is" if (tried and len(tried) > 1) \
        else "my model is"
    if name == "RateLimitError" or "rate_limit" in detail:
        if "tokens per day" in detail or "TPD" in detail:
            return (
                f"Unfortunately {exhausted} out of daily talking allowance, so I "
                "can't answer right now. Please try again later."
            )
        if "requests per day" in detail or "RPD" in detail:
            return f"I'm afraid {exhausted} hit its daily request limit. Please try again later."
        return "I'm getting a lot of requests right now. Please try again in a moment."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectTimeout"):
        return "I can't reach the model service right now. Please try again in a moment."
    if name == "InternalServerError":
        return "The model service had a problem on my end. Please try again."
    return "⚠️ I hit a snag talking to the model service. Please try that again."


def strip_bare_tool_json(text: str) -> str:
    """Drop JSON objects that are really tool arguments.

    Only an object whose *first* key is a tool parameter name is removed, so
    any other JSON the model chooses to show survives untouched.
    """
    if not text or "{" not in text:
        return text or ""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            out.append(text[i])
            i += 1
            continue
        end = _json_object_end(text, i)
        if end is None:
            out.append(text[i:])
            break
        blob = text[i:end]
        m = _FIRST_JSON_KEY.match(blob)
        if m and m.group(1) in _TOOL_ARG_NAMES:
            # Swallow it along with the blank line that introduced it.
            while out and out[-1] in " \t\r\n":
                out.pop()
            i = end
            continue
        out.append(blob)
        i = end
    return "".join(out)


# Tags that wrap a payload, so the body between them must be dropped too.
_TOOL_PAIRED_NAME = re.compile(
    r"<\s*(?:" + "|".join(_TOOL_NAMES) + r")\b", re.I
)
# Closing tag for a paired tool block.
_TOOL_BLOCK_CLOSE = re.compile(
    r"</\s*(?:" + "|".join(_TOOL_NAMES) + r")\s*>", re.I
)


def _could_start_tool_tag(rest: str) -> bool:
    """True when the text from '<' onwards may still grow into tool markup."""
    body = rest[1:].replace(_ZWSP, "")
    if body.startswith("/"):
        body = body[1:]
    if not body:
        return True                      # only '<' so far — wait for more
    if body[0] in " \t\r\n":
        return False                     # '< 6' / '< b' — prose, not a tag
    lowered = body.lower()
    for name in _TOOL_NAMES:
        n = name.lower()
        if lowered.startswith(n):
            after = body[len(n):]
            return not after or after[0] in " \t\r\n/>"
        if n.startswith(lowered):
            return True
    return False


# A trailing ``` or even a lone ` is held: on a fine-grained stream the fence
# arrives a few characters at a time, and releasing "```" the moment a "j"
# follows would defeat the point. Text resumes as soon as anything else lands.
_MD_FENCE_TAIL = re.compile(r"`{1,3}[A-Za-z0-9_+#-]*[ \t]*\r?\n?[ \t]*$")


class _MarkupFilter:
    """Streaming filter that removes tool markup without delaying real text.

    Text is emitted fragment by fragment. A '<' is held only while it could
    still be the start of a tool tag, so ordinary prose ("5 < 6") streams
    through untouched while `<recommend_titles .../>` is swallowed.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_tool_block = False
        self._json_buf = ""
        self._swallowed = False
        self._resolved = False

    def _emit(self, out: list[str], text: str) -> None:
        """Queue text for the user.

        Fenced code blocks are left alone here: deciding whether "```json" is
        an opening fence or the closing fence of a real code block is not
        possible from a partial stream, and guessing corrupts prose. Orphan
        fences are cleaned up once, on the final text.
        """
        if text:
            out.append(text)

    def _take_close_fence(self) -> str:
        """Always inert — fence handling moved to the final-text pass."""
        return "none"

    def _run_probe(self, out: list[str]) -> None:
        """Feed the held '{' buffer to the probe and emit whatever survives."""
        released = self._probe_json()
        self._resolved = False
        self._swallowed = False
        self._emit(out, released)

    def _probe_json(self) -> str:
        """Classify a '{' that may open a bare tool payload.

        Text is buffered only until the object closes — long enough to tell
        `{"title_keys": …}` from ordinary prose, short enough that normal
        writing never visibly stalls.
        """
        self._json_buf += self._buf
        self._buf = ""
        buf = self._json_buf
        start = buf.find("{")
        end = _json_object_end(buf, start)
        # Anchored at the opening brace, so this is genuinely its first key.
        m = _FIRST_JSON_KEY.match(buf)

        if end is not None:
            self._json_buf = ""
            self._resolved = True
            if m and m.group(1) in _TOOL_ARG_NAMES:
                # A real tool payload — swallow it and resume after it.
                self._buf = buf[end:]
                self._swallowed = True
                self._await_close_fence = True
                return ""
            # Ordinary braces in prose; show it and keep scanning after.
            self._buf = buf[end:]
            return buf[:end]

        # Still open. Hold briefly, then give up and let it through as text.
        if len(buf) > _JSON_PROBE_MAX:
            self._json_buf = ""
            self._resolved = True
            return buf
        self._resolved = False
        return ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if _ZWSP in chunk:
            chunk = chunk.replace(_ZWSP, "")
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            # A fenced block we emptied still has its ``` on the way.
            fence_state = self._take_close_fence()
            if fence_state == "pending":
                break
            if fence_state == "done" and not self._buf:
                break

            # While a '{' is being held for classification, every further
            # character must join that buffer — otherwise later text streams
            # past it and arrives out of order.
            if self._json_buf:
                self._run_probe(out)
                continue

            lt = self._buf.find("<")

            # Inside a paired <tool_call>…</tool_call> block: drop everything
            # up to and including the closing tag, so the JSON payload of a
            # tool call never leaks into the user's view.
            if self._in_tool_block:
                close = _TOOL_BLOCK_CLOSE.search(self._buf)
                if close:
                    self._buf = self._buf[close.end():]
                    self._in_tool_block = False
                    continue
                if len(self._buf) > 4096:   # runaway guard
                    self._buf = ""
                break

            if lt == -1:
                # A '{' may be the start of a tool payload written as prose,
                # so hand it to the probe before releasing the tail as text.
                br = self._buf.find("{")
                if br == -1:
                    self._emit(out, strip_tool_markup(self._buf))
                    self._buf = ""
                    break
                self._emit(out, self._buf[:br])
                self._buf = self._buf[br:]
                self._run_probe(out)
                continue

            self._emit(out, self._buf[:lt])
            rest = self._buf[lt:]
            # Consume the prefix we just emitted. Leaving it in the buffer made
            # it come out a second time on the next chunk, duplicating a space
            # or a word whenever a held '<' delayed the rest of the line.
            self._buf = rest
            end = rest.find(">")

            if end != -1:
                tag = rest[:end + 1]
                if _is_tool_tag(tag) or _could_start_tool_tag(tag):
                    self._buf = rest[end + 1:]
                    # Only a *paired, non-self-closing* opening tag means a
                    # payload follows; `<recommend_titles ... />` does not.
                    body = tag.replace(_ZWSP, "").strip()
                    if not body.startswith("</") and not body.endswith("/>"):
                        self._in_tool_block = _TOOL_PAIRED_NAME.match(body) is not None
                    continue
                self._emit(out, "<")                     # prose '<' — keep it
                self._buf = rest[1:]
                continue

            # No '>' yet: hold only if this could still become a tool tag.
            if _could_start_tool_tag(rest):
                break
            self._emit(out, "<")
            self._buf = rest[1:]

        return "".join(out)

    def flush(self) -> str:
        # Anything still held back by the '{' probe is real text — the stream
        # ended, so there is nothing left to wait for.
        held, self._json_buf = self._json_buf, ""
        rest, self._buf = self._buf, ""
        self._in_tool_block = False
        return strip_tool_markup(rest + held)


# ================================================================ answers
def cached_answer(question: str, session_id: str | None = None,
                  visitor_id: str | None = None) -> dict | None:
    """A finished answer for this question, from any earlier user."""
    key = cache.answer_key(question)
    if not key:
        return None
    hit = cache.get_json(key)
    if not isinstance(hit, dict) or hit.get("kind") != "answer":
        return None
    if not (hit.get("text") or "").strip():
        return None
    _record_cache_hit("answer", session_id, visitor_id)
    return hit


# Answers that report a failure must never enter the cache — otherwise one bad
# turn poisons the same question for 24 hours, even after the underlying source
# is fixed.
_UNCACHEABLE = re.compile(
    r"(i\s+(?:don'?t|do not|cannot|can'?t|am\s+not\s+able\s+to)\s+(?:have|find|get|"
    r"pull|locate|retrieve)|couldn'?t\s+find|could\s+not\s+find|no\s+(?:summary|"
    r"information|data|results?|matches)\b|i\s+hit\s+a\s+snag|i'?m\s+sorry|"
    r"unable\s+to\s+(?:find|retrieve)|not\s+available\s+in\s+my\s+database|"
    r"something\s+went\s+wrong|trouble\s+(?:reaching|talking))",
    re.I,
)


def _is_cacheable(text: str) -> bool:
    """True when an answer is a real answer worth reusing."""
    t = (text or "").strip()
    if len(t) < 15:
        return False
    # Normalise smart quotes/dashes so "don’t" and "don't" both match.
    t = (t.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"')
          .replace("—", "-").replace("–", "-"))
    return not _UNCACHEABLE.search(t)


def store_answer(question: str, result: dict) -> None:
    """Cache a self-contained answer so a similar question costs nothing later."""
    key = cache.answer_key(question)
    if not key or result.get("kind") != "answer":
        return
    text = (result.get("text") or "").strip()
    if not _is_cacheable(text):
        logger.debug("skipping answer cache: failure-style response")
        return
    cache.set_json(key, {
        "kind": "answer",
        "text": text,
        "titles": result.get("titles") or [],
        "suggestions": result.get("suggestions") or [],
    })


# ======================================================= poster recognition
# Only images are ever accepted — see `main.validate_upload`.
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/webp": "webp", "image/gif": "gif",
}

_VISION_PROMPT = """You recognise posters, cover art and stills from films, TV series and anime.

Look at the image and identify what it is. Return STRICT JSON only:
{
  "found": true,
  "title": "Exact Title",
  "year": 2020,
  "type": "movie" | "tv" | "unknown",
  "confidence": 0.0,
  "alternatives": ["Another possible title"]
}

Rules:
- Only name a title you can actually read or clearly recognise from the artwork.
- If the image is not a poster/frame from a film, series or anime (a photo, a
  meme, a logo, a person, a screenshot of unrelated software), set "found": false
  and leave title empty. Do not guess.
- confidence is 0.0-1.0. Below 0.35, set "found": false."""


def _vision_client():
    """Vision may use different credentials from the main chat model."""
    # Both must be declared global: assigning _vision_key below made it local
    # to this function, so reading it raised UnboundLocalError. Python's `or`
    # short-circuited past that read whenever the client was still None — which
    # is why the FIRST upload worked and every one after it failed.
    global _vision, _vision_key
    key = _settings.llm_vision_api_key.strip() or _settings.llm_api_key
    base = _settings.llm_vision_base_url.strip() or _settings.llm_base_url
    if _vision is None or _vision_key != (key, base):
        _vision = OpenAI(api_key=key or "missing-key", base_url=base,
                         timeout=90.0, max_retries=2)
        _vision_key = (key, base)
    return _vision


def vision_available() -> bool:
    """True only when a vision model is genuinely reachable.

    Falling back to the main key made this report "configured" while pointing
    at a model the provider has never heard of, so every upload failed with a
    vague "I couldn't read that image". An explicit vision key is the only
    configuration we trust without asking the provider.
    """
    return bool(_settings.llm_vision_model.strip()) and bool(
        _settings.llm_vision_api_key.strip()
    )


def identify_poster(image_bytes: bytes, mime: str, session_id: str | None = None,
                    visitor_id: str | None = None, parent=None,
                    hint: str = "") -> dict:
    """Identify a poster/frame, then ground it against the real catalog.

    The result is cached against a hash of the image bytes, so re-uploading the
    same poster is free for 24h.
    """
    import base64
    import hashlib

    digest = hashlib.sha256(image_bytes).hexdigest()[:32]
    ck = f"vision:{digest}"
    hit = cache.get_json(ck)
    if isinstance(hit, dict):
        _record_cache_hit("vision", session_id, visitor_id)
        return hit

    if not vision_available():
        return {"ok": False, "code": "no_vision",
                "message": "Image recognition isn't configured yet. Add a vision "
                           "model to `LLM_VISION_MODEL` in backend/.env."}

    encoded = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime};base64,{encoded}"
    model = _settings.llm_vision_model.strip()

    # The generation's traced input carries the image as media (so Langfuse
    # renders it), never the raw base64 blob we send to the provider.
    traced_input = [{
        "role": "user",
        "content": [
            {"type": "text", "text": _VISION_PROMPT},
            {"type": "media", "media": obs.media(data_url, mime)},
        ],
    }]
    model_input = [{"role": "user", "content": [
        {"type": "text", "text": _VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]

    started = time.time()
    try:
        with obs.observation(
            "identify-entertainment-poster", as_type="generation", model=model,
            input=traced_input, session_id=session_id, user_id=visitor_id,
            parent=parent,
        ) as gen:
            resp = _vision_client().chat.completions.create(
                model=model,
                messages=model_input,
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"},
                **_extra(),
            )
            pt, ct = _record_usage(resp, model, "vision", started, session_id, visitor_id)
            raw = resp.choices[0].message.content or ""
            guess = _extract_json(raw) or {}
            obs.record_generation(gen, output=raw, input_tokens=pt,
                                  output_tokens=ct, model=model)
    except Exception as exc:
        logger.warning("vision identification failed", exc_info=True)
        metrics.llm_error("vision")
        # "Couldn't read that image" sent people hunting for a bad upload. If
        # the model itself is missing or unknown, say so and name the fix.
        detail = str(exc).lower()
        if not vision_available() or "does not exist" in detail \
                or "not found" in detail or "model_not_found" in detail:
            return {
                "ok": False, "code": "vision_not_configured",
                "message": (
                    "I can't look at images yet — no vision model is configured on "
                    "this setup. Tell me the title and I'll happily tell you all "
                    "about it."
                ),
            }
        return {"ok": False, "code": "vision_error",
                "message": "I couldn't read that image just now. Could you try again?"}

    title = str(guess.get("title") or "").strip()
    confidence = float(guess.get("confidence") or 0)
    if not guess.get("found") or not title or confidence < 0.35:
        out = {
            "ok": False, "code": "not_recognised",
            "message": "I couldn't confidently match that image to a movie, series "
                       "or anime. Try a clearer poster, or just tell me the title!",
        }
        cache.set_json(ck, out)
        return out

    # Ground the guess: never trust the model's memory, look it up.
    kind = "tv" if str(guess.get("type") or "").lower() == "tv" else "movie"
    candidates = entertainment.find_titles(query=title, media_type="any", limit=6)
    if not candidates:
        out = {"ok": False, "code": "not_found",
               "message": f"I recognised **{title}**, but couldn't find it in my "
                          "catalog. Could you check the spelling?"}
        cache.set_json(ck, out)
        return out

    best = candidates[0]
    # A typed hint ("is this the 2017 one?") disambiguates close matches.
    # It only reorders candidates the vision model actually proposed, so it
    # can never introduce a title that isn't in the grounded result set.
    hint_l = (hint or "").strip().lower()
    if hint_l:
        for c in candidates:
            blob = f"{c.get('title','')} {c.get('year','')}".lower()
            if hint_l in blob or any(
                    w in blob for w in hint_l.split() if len(w) > 3):
                best = c
                break
    for c in candidates:
        if str(c.get("title", "")).lower() == title.lower():
            best = c
            break
    entertainment.remember_card(best)

    details = entertainment.title_details(best["key"]) or best
    entertainment.remember_card(details)
    summary = title_summary(best["key"], "quick", session_id)

    others = []
    for alt in (guess.get("alternatives") or [])[:2]:
        alt_t = str(alt).strip()
        if not alt_t:
            continue
        for c in entertainment.find_titles(query=alt_t, media_type="any", limit=1):
            if c["key"] != best["key"]:
                entertainment.remember_card(c)
                others.append(c)

    # Prefer the model's summary, but fall back to the catalog overview so the
    # reply is never bare — the summary is an extra Groq call and will be
    # missing whenever the chat model is rate-limited or exhausted.
    narration = (summary.get("summary") or "").strip() \
        or (details.get("overview") or "").strip()
    year = details.get("year")
    headline = f"That's **{details.get('title') or title}**" + (f" ({year})" if year else "")
    why = ""
    if details.get("genres"):
        why = f" A {'/'.join(details['genres'][:2])} title."

    out = {
        "ok": True,
        "confidence": round(confidence, 2),
        "detected_title": title,
        "title": details,
        "text": f"{headline}.{why}\n\n{narration}" if narration else f"{headline}.{why}",
        "summary": summary.get("summary") or narration,
        "alternatives": others,
    }
    cache.set_json(ck, out)
    return out
