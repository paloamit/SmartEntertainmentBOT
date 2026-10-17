"""Smart Entertainment BOT — FastAPI backend.

A clean REST/SSE API so any client (this React app, a mobile app, …) can use it.

Key endpoints
  GET    /api/bootstrap            identity + sessions + greeting, in one call
  POST   /api/identity/unlock      restore this visitor on another device
  POST   /api/chat                 one turn, full JSON (non-streaming)
  POST   /api/chat/stream          one turn as an SSE stream
  POST   /api/sessions             create a chat (respects the session cap)
  GET    /api/sessions             this visitor's recent chats
  GET    /api/sessions/{id}        full transcript
  DELETE /api/sessions/{id}        delete a chat
  POST   /api/vision/identify      recognise an uploaded poster (images only)
  GET    /api/titles/{key}         full title record, incl. where to watch
  POST   /api/feedback             satisfied / dissatisfied + review text
  GET    /api/observability/usage  token + cost + question dashboard
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import cache, db, entertainment, guardrails, llm, metrics
from . import observability as obs
from .config import get_settings
from .schemas import ChatRequest, FeedbackRequest, UnlockRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sebot")
settings = get_settings()

# Keep auto-updating the title for the first few turns so it tracks the
# conversation, then leave it stable to avoid churn and cost.
_TITLE_REFRESH_UNTIL_MESSAGES = 6


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


# ------------------------------------------------------------- limits
class LimitError(HTTPException):
    """A friendly, structured "you've hit the limit" response."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(status_code=429, detail={"code": code, "message": message, **extra})


def _session_limit_error(visitor_id: str) -> LimitError:
    limit = settings.max_sessions
    existing = [
        {"id": s["id"], "title": s["title"], "updated_at": s["updated_at"]}
        for s in db.list_sessions(visitor_id, limit)
    ]
    return LimitError(
        "session_limit",
        f"You've reached your limit of {limit} saved chats 🎬. "
        f"Delete one and I'll start a fresh conversation for you right away.",
        limit=limit, sessions=existing,
    )


def _query_limit_error(session: dict) -> LimitError:
    limit = settings.max_queries_per_session
    used = int(session.get("query_count") or 0)
    return LimitError(
        "query_limit",
        f"You've used all {limit} questions in this chat 🎬. "
        f"Start a new chat and we can keep exploring!",
        limit=limit, used=used, session_id=session["id"],
    )


def _new_session(visitor_id: str) -> str:
    if db.session_count(visitor_id) >= settings.max_sessions:
        raise _session_limit_error(visitor_id)
    return db.create_session(visitor_id, "New chat")


# ------------------------------------------------------------- identity
def current_visitor(x_visitor_id: str = Header(default="")) -> str:
    """The browser's stable visitor id; registered on first sight."""
    return db.ensure_visitor(x_visitor_id)


# ------------------------------------------------------------- lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    if obs.enabled():
        logger.info("langfuse: tracing ON -> %s (env=%s)",
                    settings.langfuse_host, settings.langfuse_environment)
    if settings.enable_identity_unlock:
        logger.info("identity: browser visitor id + optional sync passphrase")
    if entertainment.tmdb_configured():
        try:
            entertainment.genre_map("movie")
            logger.info("entertainment source: TMDB ready")
        except Exception as exc:  # non-fatal; retried lazily on first search
            logger.warning("TMDB warm-up failed: %s", exc)
    else:
        logger.warning("TMDB_API_KEY missing — set it in backend/.env")
    logger.info("cache: %s | sessions max %d | questions max %d",
                cache.backend_name(), settings.max_sessions, settings.max_queries_per_session)
    yield
    # Never lose buffered observations on shutdown (Render restarts, Ctrl-C…).
    obs.flush()


app = FastAPI(title="Smart Entertainment BOT API", version="1.0.0", lifespan=lifespan)

_origins = ["*"] if settings.cors_origins.strip() == "*" else [
    o.strip() for o in settings.cors_origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.enable_metrics:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception as exc:  # never let metrics break the app
        logger.warning("metrics disabled: %s", exc)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    metrics.unhandled_error(request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/api/diagnostics")
def diagnostics():
    """Live per-source reachability. Cheap enough to call on demand.

    Separate from /api/health because it makes outbound network calls and can
    take a few seconds on a cold free-tier instance.
    """
    return {
        "configured": {
            "llm": llm.has_key(),
            "vision": llm.vision_available(),
            "watchmode": entertainment.watchmode_configured(),
            "omdb": bool(settings.omdb_api_key.strip()),
        },
        "live": entertainment.probe_sources(),
    }


# ================================================================ bootstrap
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": {"large": settings.llm_model_large, "small": settings.llm_model_small,
                   "vision": settings.llm_vision_model},
        "llm_configured": llm.has_key(),
        "tmdb_configured": entertainment.tmdb_configured(),
        # The failover chain, resolved to real provider endpoints. "Groq
        # configured" is not the same as "Groq usable" -- once its daily budget
        # is spent every model on it fails at once, and this is what the
        # secondary provider is there to absorb.
        "llm_chain": llm.chain_report(),
        # Which catalog sources actually hold a credential. This used to be
        # invisible, so a missing WATCHMODE_API_KEY looked identical to
        # "the search found nothing" from the outside — the bot just said it
        # could not find a title that was sitting in the database.
        "sources": {
            "watchmode": entertainment.watchmode_configured(),
            "omdb": bool(settings.omdb_api_key.strip()),
            "anilist": settings.enable_anilist_fallback,
            "jikan": settings.enable_jikan_fallback,
            "tvmaze": settings.enable_tvmaze_fallback,
        },
        "fallback_sources": [
            name for name, on in (
                ("jikan", settings.enable_jikan_fallback),
                ("tvmaze", settings.enable_tvmaze_fallback),
            ) if on
        ],
        "vision_configured": llm.vision_available(),
        "cache": cache.health(),
        "limits": {
            "max_sessions": settings.max_sessions,
            "max_queries_per_session": settings.max_queries_per_session,
            "max_recommendations": settings.max_recommendations,
            "max_upload_mb": settings.max_upload_mb,
        },
    }


@app.get("/api/bootstrap")
def bootstrap(resume: bool = True, visitor_id: str = Depends(current_visitor)):
    """Everything the UI needs on load: who you are, your chats, and the greeting.

    A returning visitor gets their most recent chat back (with a personalised
    greeting prepended by the client), so nothing is ever lost between visits.
    """
    visitor = db.get_visitor(visitor_id) or {}
    name = visitor.get("display_name")
    sessions = db.list_sessions(visitor_id, settings.max_sessions)

    active = None
    if resume and sessions:
        latest = db.get_session(sessions[0]["id"], visitor_id)
        if latest:
            active = _serialise_session(latest)

    greeting = llm.personalise_greeting(name, returning=bool(sessions))
    return {
        "visitor_id": visitor_id,
        "display_name": name,
        "returning": bool(sessions),
        "has_sync_code": bool(visitor.get("unlock_hash")),
        "session_count": len(sessions),
        "sessions": [
            {"id": s["id"], "title": s["title"], "updated_at": s["updated_at"],
             "query_count": s["query_count"], "user_name": s["user_name"]}
            for s in sessions
        ],
        "active_session": active,
        "greeting": greeting,
        # The client greys out the attach button rather than letting someone
        # upload an image only to be told it can't be read.
        "vision_available": llm.vision_available(),
        "limits": {
            "max_sessions": settings.max_sessions,
            "max_queries_per_session": settings.max_queries_per_session,
            "max_recommendations": settings.max_recommendations,
        },
        "filters": entertainment.reference_data(),
    }


@app.post("/api/identity/unlock")
def unlock(req: UnlockRequest, visitor_id: str = Depends(current_visitor)):
    """Restore an existing visitor's chats on a new device using their sync code."""
    if not settings.enable_identity_unlock:
        raise HTTPException(status_code=400, detail="sync codes are disabled")
    phrase = (req.passphrase or "").strip()
    if len(phrase) < 4:
        raise HTTPException(status_code=400, detail="sync code must be at least 4 characters")
    if not db.set_unlock_passphrase(visitor_id, phrase):
        raise HTTPException(status_code=409, detail="that sync code is already in use")
    return {"ok": True, "visitor_id": visitor_id}


@app.post("/api/identity/unlock/restore")
def unlock_restore(req: UnlockRequest):
    """Swap this browser onto the visitor that owns the given sync code."""
    if not settings.enable_identity_unlock:
        raise HTTPException(status_code=400, detail="sync codes are disabled")
    owner = db.visitor_for_passphrase((req.passphrase or "").strip())
    if not owner:
        raise HTTPException(status_code=404, detail="no chats found for that sync code")
    return {"visitor_id": owner, "sessions": db.list_sessions(owner, settings.max_sessions)}


@app.delete("/api/identity/unlock")
def unlock_clear(visitor_id: str = Depends(current_visitor)):
    db.clear_unlock_passphrase(visitor_id)
    return {"ok": True}


# =================================================================== chat
def _serialise_session(payload: dict) -> dict:
    s = payload["session"]
    return {
        "id": s["id"],
        "title": s["title"],
        "query_count": s["query_count"],
        "user_name": s["user_name"],
        "messages": [
            {
                "id": m["id"],
                "role": m["role"],
                "kind": m["kind"],
                "text": m["content"],
                "titles": m["titles"],
                "suggestions": m["suggestions"],
                "clarify": m["clarify"],
            }            for m in payload["messages"]
            if m["role"] in ("user", "assistant")
        ],
    }


def _resolve_session(visitor_id: str, sid: str | None) -> tuple[str, dict | None]:
    """Return (session_id, session_row). Creates a session on first use."""
    if sid:
        existing = db.get_session(sid, visitor_id)
        if existing:
            return sid, existing["session"]
    return _new_session(visitor_id), None


def _check_query_budget(session_row: dict | None) -> None:
    if session_row and int(session_row.get("query_count") or 0) >= settings.max_queries_per_session:
        raise _query_limit_error(session_row)


def _maybe_learn_name(visitor_id: str, sid: str, session_row: dict | None,
                      message: str) -> str | None:
    """Pick up a volunteered name without ever requiring one."""
    if session_row and session_row.get("user_name"):
        return session_row["user_name"]
    name = guardrails.extract_name(message)
    if name:
        db.set_user_name(visitor_id, sid, name)
        return name
    return (session_row or {}).get("user_name")


def _maybe_title(sid: str, visitor_id: str, history_len: int) -> str | None:
    if not (0 < history_len <= _TITLE_REFRESH_UNTIL_MESSAGES):
        return None
    history = db.history_for_llm(sid, settings.max_history_messages)
    # A turn that was only a volunteered name says nothing about the subject,
    # so don't let it become the chat's title.
    if not _has_substantive_question(history):
        return None
    title = llm.generate_title(history, session_id=sid, visitor_id=visitor_id)
    db.set_title(sid, title)
    return title


# Words that appear in "I am Amit" / "my name is Amit" but never in a real
# question. A message made only of these is an introduction, not a subject.
_IDENTITY_VOCAB = {
    "i", "im", "am", "my", "mine", "name", "is", "call", "called", "me", "hi",
    "hello", "hey", "thanks", "thank", "you", "yes", "ok", "okay", "about",
    "it's", "its", "here", "and",
}


def _has_substantive_question(history: list[dict]) -> bool:
    """True once the user has asked something beyond just introducing themselves.

    An introduction is identity vocabulary plus at most one name ("I am Amit").
    This is a deterministic check rather than a model call, which keeps a whole
    LLM request off the critical path for a chat that opens with a name.
    """
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        words = [w.strip(".,!?'\"").lower() for w in text.split()]
        if not words:
            continue
        content_words = [w for w in words if w not in _IDENTITY_VOCAB]
        return len(content_words) > 1 or len(words) > 5
    return False


def _resolve_or_decline(original: str, enriched: str, turn=None,
                        from_widget: bool = False,
                        after_clarify: bool = False) -> str | None:
    """Return a decline message when the question is off-topic, else None.

    Always judged on the user's OWN words (`original`). The clarify widget's
    structured answers ("genre: Action, era: 2010s") carry no topic signal and
    would otherwise be rejected by the classifier — and since those answers can
    only follow a question we already accepted, they skip the check entirely.
    """
    from .guardrails import check_scope_traced

    if from_widget:
        return None
    probe = (original or "").strip() or (enriched or "").strip()
    if check_scope_traced(probe, parent=turn, after_clarify=after_clarify):
        return None
    return guardrails.OFF_TOPIC_REPLY


@app.post("/api/chat")
def chat(req: ChatRequest, visitor_id: str = Depends(current_visitor)):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    sid, session_row = _resolve_session(visitor_id, req.session_id)
    _check_query_budget(session_row)
    message = _with_answers(message, req.answers)

    db.add_message(sid, visitor_id, "user", message, kind="user")
    db.bump_query_count(sid)
    user_name = _maybe_learn_name(visitor_id, sid, session_row, message)
    history = db.history_for_llm(sid, settings.max_history_messages)
    # Did our last reply ask a filter question the user is now answering?
    after_clarify = db.last_assistant_kind(sid) == "clarify"

    # One trace per turn: root span input/output become the trace's, so the
    # tracing table shows the question and the answer at a glance.
    turn = obs.start_turn(
        "respond-to-entertainment-request",
        input={"user_message": message},
        session_id=sid, user_id=visitor_id,
        tags=["chat"],
        metadata={"model_tier": llm.route_task(message, history_len=len(history)),
                  "history_len": len(history)},
    )
    try:
        decline = _resolve_or_decline(req.message or message, message, turn,
                                      from_widget=bool(req.answers),
                                      after_clarify=after_clarify)
        cached = None
        if decline:
            metrics.chat_outcome("offtopic")
            obs.event("declined-offtopic-question", session_id=sid, user_id=visitor_id)
            result = {"kind": "answer", "text": decline, "titles": [], "suggestions": []}
        else:
            cached = llm.cached_answer(req.message or message, sid, visitor_id)
            if cached:
                obs.event("served-from-answer-cache", session_id=sid, user_id=visitor_id)
                result = {**cached, "cached": True}
            else:
                result = llm.run_chat(history, session_id=sid, visitor_id=visitor_id,
                                      user_name=user_name, parent=turn,
                                      after_clarify=after_clarify)
                if result.get("kind") == "answer":
                    llm.store_answer(req.message or message, result)

        turn.set_output({
            "reply": result.get("text", ""),
            "titles": [t.get("title") for t in (result.get("titles") or [])],
        })
    except Exception:
        turn.set_error("chat turn failed")
        raise
    finally:
        turn.end()

    db.add_message(sid, visitor_id, "assistant", result.get("text", ""),
                   kind=result.get("kind", "answer"), titles=result.get("titles"),
                   suggestions=result.get("suggestions"), clarify=result.get("clarify"),
                   trace_id=turn.trace_id)

    metrics.chat_outcome(result.get("kind", "answer"))
    title = _maybe_title(sid, visitor_id, len(history))
    return {
        "session_id": sid,
        "kind": result.get("kind", "answer"),
        "text": result.get("text", ""),
        "titles": result.get("titles", []),
        "suggestions": result.get("suggestions", []),
        "clarify": result.get("clarify"),
        "title": title,
        "user_name": user_name,
        "cached": bool(cached),
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, visitor_id: str = Depends(current_visitor)):
    """Streaming chat over Server-Sent Events.

    Emits `meta`, `status`, `token`, `titles`, `clarify`, then a final `done`.
    """
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    sid, session_row = _resolve_session(visitor_id, req.session_id)
    _check_query_budget(session_row)
    enriched = _with_answers(message, req.answers)

    db.add_message(sid, visitor_id, "user", enriched, kind="user")
    db.bump_query_count(sid)
    user_name = _maybe_learn_name(visitor_id, sid, session_row, message)
    history = db.history_for_llm(sid, settings.max_history_messages)
    # Did our last reply ask a filter question the user is now answering?
    after_clarify = db.last_assistant_kind(sid) == "clarify"

    def gen():
        yield _sse({"type": "meta", "session_id": sid,
                    "is_new": session_row is None,
                    "user_name": user_name,
                    "query_count": db.get_session(sid, visitor_id)["session"]["query_count"]})
        final: dict = {"kind": "answer", "text": "", "titles": [], "suggestions": [],
                       "clarify": None}
        # The SSE generator yields between steps and resumes in a fresh OTel
        # context, so the turn handle is passed explicitly to every child.
        turn = obs.start_turn(
            "respond-to-entertainment-request",
            input={"user_message": enriched},
            session_id=sid, user_id=visitor_id,
            tags=["chat", "streaming"],
            metadata={"model_tier": llm.route_task(message, history_len=len(history)),
                      "history_len": len(history)},
        )
        try:
            decline = _resolve_or_decline(req.message or message, enriched, turn,
                                          from_widget=bool(req.answers),
                                          after_clarify=after_clarify)
            cached = None if decline else llm.cached_answer(
                req.message or enriched, sid, visitor_id)

            if decline:
                obs.event("declined-offtopic-question", session_id=sid, user_id=visitor_id)
                yield _sse({"type": "token", "text": decline})
                final = {"kind": "answer", "text": decline, "titles": [],
                         "suggestions": [], "clarify": None}
            elif cached:
                obs.event("served-from-answer-cache", session_id=sid, user_id=visitor_id)
                yield _sse({"type": "token", "text": cached["text"]})
                yield _sse({"type": "titles", "titles": cached.get("titles") or []})
                final = {**cached, "clarify": None}
            else:
                for ev in llm.stream_chat(history, session_id=sid, visitor_id=visitor_id,
                                          user_name=user_name, parent=turn,
                                          after_clarify=after_clarify):
                    if ev.get("type") == "final":
                        final = ev
                        continue
                    yield _sse(ev)
                if final.get("kind") == "answer":
                    llm.store_answer(req.message or enriched, final)

            turn.set_output({
                "reply": final.get("text", ""),
                "titles": [t.get("title") for t in (final.get("titles") or [])],
                "kind": final.get("kind", "answer"),
            })

            db.add_message(sid, visitor_id, "assistant", final.get("text", ""),
                           kind=final.get("kind", "answer"), titles=final.get("titles"),
                           suggestions=final.get("suggestions"),
                           clarify=final.get("clarify"), trace_id=turn.trace_id)
            title = _maybe_title(sid, visitor_id, 0 if decline else len(history))
            metrics.chat_outcome(final.get("kind", "answer"))
            yield _sse({"type": "done", "session_id": sid,
                        "kind": final.get("kind", "answer"), "title": title,
                        "user_name": user_name,
                        "cached": bool(cached)})
        except Exception:
            turn.set_error("stream generation failed")
            # Headers are already sent, so surface failures in-stream.
            logger.exception("stream generation failed")
            metrics.unhandled_error("/api/chat/stream")
            msg = "⚠️ Something went wrong while replying. Please try again."
            yield _sse({"type": "token", "text": msg})
            yield _sse({"type": "done", "session_id": sid, "kind": "error", "title": None,
                        "user_name": user_name, "cached": False})
        finally:
            turn.end()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _with_answers(message: str, answers: dict | None) -> str:
    """Fold the clarify widget's selections into a readable user message."""
    if not answers:
        return message
    parts = [message.strip()] if message.strip() else []
    for key, value in answers.items():
        if value in (None, "", [], {}):
            continue
        pretty = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        parts.append(f"{key}: {pretty}")
    return "\n".join(parts) if parts else message


# ============================================================== sessions
@app.get("/api/sessions")
def list_sessions(visitor_id: str = Depends(current_visitor)):
    return [
        {"id": s["id"], "title": s["title"], "updated_at": s["updated_at"],
         "query_count": s["query_count"], "user_name": s["user_name"]}
        for s in db.list_sessions(visitor_id, settings.max_sessions)
    ]


@app.post("/api/sessions")
def create_session(visitor_id: str = Depends(current_visitor)):
    """Start a new chat. Fails with a friendly 429 when the cap is reached."""
    sid = _new_session(visitor_id)
    return {"id": sid, "title": "New chat"}


@app.get("/api/sessions/{sid}")
def session_detail(sid: str, visitor_id: str = Depends(current_visitor)):
    payload = db.get_session(sid, visitor_id)
    if not payload:
        raise HTTPException(status_code=404, detail="session not found")
    return _serialise_session(payload)


@app.delete("/api/sessions/{sid}")
def session_delete(sid: str, visitor_id: str = Depends(current_visitor)):
    db.delete_session(sid, visitor_id)
    return {"deleted": sid}


# ============================================================ vision upload
@app.post("/api/vision/identify")
async def vision_identify(
    file: UploadFile = File(...),
    session_id: str = None,
    hint: str = "",
    visitor_id: str = Depends(current_visitor),
):
    """Identify an uploaded image (images only, never video) as a movie/series/anime.

    `hint` is whatever the user typed alongside the image. It is only used to
    disambiguate between close matches — never to name the title outright, so
    a wrong guess still gets caught by the catalog lookup.
    """
    mime = (file.content_type or "").lower().split(";")[0].strip()
    if mime not in llm.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_type",
                    "message": "Only images are supported (JPG, PNG, WEBP or GIF)."},
        )

    data = await file.read()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"code": "too_large",
                    "message": f"That image is over {settings.max_upload_mb:.0f} MB. "
                               "Please upload a smaller one."},
        )

    # Only persist into the chat when the caller owns a real session.
    existing = db.get_session(session_id, visitor_id) if session_id else None
    sid = existing["session"]["id"] if existing else None

    turn = obs.start_turn(
        "identify-uploaded-poster",
        input={"filename": file.filename, "content_type": mime, "bytes": len(data)},
        session_id=sid, user_id=visitor_id,
        tags=["vision"],
        metadata={"source": "user-upload"},
    )
    try:
        result = llm.identify_poster(data, mime, session_id=sid, visitor_id=visitor_id,
                                     parent=turn, hint=(hint or "").strip())
    finally:
        turn.end()

    if not result.get("ok"):
        return {"ok": False, "code": result.get("code"), "message": result.get("message")}

    title = result["title"]
    if sid:
        db.bump_query_count(sid)
        db.add_message(sid, visitor_id, "user", f"📷 Identified a poster: {title['title']}",
                       kind="user")
        text = (
            f"That's **{title['title']}**"
            + (f" ({title['year']})" if title.get("year") else "")
            + ". "
            + (result.get("summary") or title.get("overview") or "")
        ).strip()
        db.add_message(sid, visitor_id, "assistant", text, kind="answer",
                       titles=[title] + (result.get("alternatives") or []))

    return {
        "ok": True,
        "session_id": sid,
        "confidence": result.get("confidence"),
        "detected_title": result.get("detected_title"),
        "title": title,
        "summary": result.get("summary") or "",
        "alternatives": result.get("alternatives") or [],
        "text": (
            f"That's **{title['title']}**"
            + (f" ({title['year']})" if title.get("year") else "")
            + ". " + (result.get("summary") or title.get("overview") or "")
        ).strip(),
    }


# ============================================================ title detail
@app.get("/api/titles/{key}")
def title_detail(key: str):
    """Full record for the expanded card — cast, crew, trailers, where to watch."""
    card = entertainment.title_details(key)
    if not card:
        raise HTTPException(status_code=404, detail="title not found")
    entertainment.remember_card(card)
    return card


@app.get("/api/browse/trending")
def browse_trending(media_type: str = "all", window: str = "week", limit: int = 10):
    return entertainment.trending(media_type, window, max(1, min(limit, 20)))


# ================================================================ feedback
@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, visitor_id: str = Depends(current_visitor)):
    rating = (req.rating or "").lower().strip()
    if rating not in ("satisfied", "dissatisfied"):
        raise HTTPException(status_code=400,
                            detail="rating must be 'satisfied' or 'dissatisfied'")
    stars = req.app_rating
    if stars is not None and not (1 <= int(stars) <= 5):
        raise HTTPException(status_code=400, detail="app_rating must be between 1 and 5")
    db.add_feedback(
        rating=rating, comment=(req.comment or "").strip() or None,
        session_id=req.session_id, message_id=req.message_id, visitor_id=visitor_id,
        question=req.question, answer=req.answer, titles=req.titles or [],
        app_rating=int(stars) if stars is not None else None,
    )
    metrics.feedback(rating)
    if stars is not None:
        metrics.app_rating(int(stars))
    # Attach the verdict to the Langfuse trace that produced this answer, so you
    # can filter low-scoring turns and see exactly what users disliked.
    trace_id = db.trace_id_for_message(req.message_id, visitor_id) if req.message_id else None
    llm.log_feedback_score(req.session_id, rating, trace_id, app_rating=int(stars) if stars else None)
    return {"status": "recorded", "rating": rating, "app_rating": stars}


@app.get("/api/feedback")
def list_feedback():
    return db.list_feedback()


# ============================================================ observability
@app.get("/api/observability/usage")
def usage(hours: int = 24):
    """Token spend, estimated USD cost, cache hits and the questions asked."""
    summary = db.usage_summary(max(1, min(hours, 24 * 30)))
    summary["cache"] = cache.health()
    return summary
