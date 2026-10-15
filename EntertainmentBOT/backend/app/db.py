"""Persistence for visitors, sessions, messages, feedback and LLM usage.

Storage is SQLite by default and Postgres when `DATABASE_URL` is set — the SQL
is written once with SQLAlchemy Core so both work without changes.

Identity model
--------------
Every browser generates a random `visitor_id` (kept in localStorage) and sends
it with each request. Sessions belong to that visitor, which is what lets a
returning user find their chats again. A visitor may also set a passphrase
("sync code") so the same history can be restored on another device.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.pool import StaticPool

from .config import get_settings

_settings = get_settings()
_SEEN_WRITE_INTERVAL = 300.0
metadata = MetaData()

visitors = Table(
    "visitors",
    metadata,
    Column("id", String, primary_key=True),
    Column("display_name", String, nullable=True),
    # sha256 of the sync passphrase (or None) — never stored in plain text.
    Column("unlock_hash", String, nullable=True),
    Column("created_at", Float, nullable=False),
    Column("last_seen_at", Float, nullable=False),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("visitor_id", String, nullable=False, index=True),
    Column("title", String, nullable=False),
    # Name the user volunteered in chat, used for a warmer tone.
    Column("user_name", String, nullable=True),
    Column("query_count", Integer, nullable=False, default=0),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Index("ix_sessions_visitor_updated", "visitor_id", "updated_at"),
)

messages = Table(
    "messages",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=False, index=True),
    Column("visitor_id", String, nullable=True),
    Column("role", String, nullable=False),
    # user | greeting | answer | clarify
    Column("kind", String, nullable=False, default="answer"),
    Column("content", Text, nullable=False, default=""),
    Column("titles", Text, nullable=False, default="[]"),
    Column("suggestions", Text, nullable=False, default="[]"),
    # The structured question shown by the clarify widget (fields/options).
    Column("clarify", Text, nullable=True),
    # Langfuse trace id for this turn, so feedback scores land on the right trace.
    Column("trace_id", String, nullable=True),
    Column("created_at", Float, nullable=False),
)

feedback = Table(
    "feedback",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=True),
    Column("message_id", String, nullable=True),
    Column("visitor_id", String, nullable=True),
    Column("rating", String, nullable=False),  # satisfied | dissatisfied
    Column("app_rating", Integer, nullable=True),  # 1-5 stars from the user
    Column("comment", Text, nullable=True),
    Column("question", Text, nullable=True),
    Column("answer", Text, nullable=True),
    Column("titles", Text, nullable=False, default="[]"),
    Column("created_at", Float, nullable=False),
)

llm_usage = Table(
    "llm_usage",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=True),
    Column("visitor_id", String, nullable=True),
    Column("task", String, nullable=False),      # small | large | ...
    Column("model", String, nullable=False),
    Column("prompt_tokens", Integer, nullable=False, default=0),
    Column("completion_tokens", Integer, nullable=False, default=0),
    Column("total_tokens", Integer, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("latency_ms", Integer, nullable=False, default=0),
    Column("cached", Integer, nullable=False, default=0),
    Column("created_at", Float, nullable=False),
)


def _make_engine():
    url = _settings.url_for()
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # FastAPI serves requests from a thread pool, so the connection must
        # not be pinned to its creating thread. Also keep a single connection
        # in memory for ":memory:" databases.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url.endswith("sqlite://"):
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
        kwargs["pool_recycle"] = 1800
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        # WAL keeps reads fast while a write is in flight.
        with eng.connect() as c:
            c.exec_driver_sql("PRAGMA journal_mode=WAL")
            c.exec_driver_sql("PRAGMA synchronous=NORMAL")
    return eng


engine = _make_engine()


def init_db() -> None:
    metadata.create_all(engine)


# --------------------------------------------------------------- helpers
def _j(value) -> str:
    return json.dumps(value if value is not None else [], default=str)


def _uj(raw) -> list:
    try:
        out = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return out if isinstance(out, list) else []


def _hash_passphrase(passphrase: str) -> str:
    return hashlib.sha256(f"sebot::{passphrase}".encode()).hexdigest()


# ---------------------------------------------------------------- visitor
def ensure_visitor(visitor_id: str | None) -> str:
    """Return a valid visitor id, registering the row if we haven't seen it.

    `last_seen_at` is only written when it's more than 5 minutes stale, so this
    stays cheap on the hot path of every single request.
    """
    vid = (visitor_id or "").strip() or str(uuid.uuid4())
    now = time.time()
    with engine.begin() as c:
        row = c.execute(
            select(visitors.c.last_seen_at).where(visitors.c.id == vid)
        ).first()
        if row is None:
            c.execute(visitors.insert().values(
                id=vid, created_at=now, last_seen_at=now,
                display_name=None, unlock_hash=None,
            ))
        elif now - (row[0] or 0) > _SEEN_WRITE_INTERVAL:
            c.execute(
                visitors.update().where(visitors.c.id == vid).values(last_seen_at=now)
            )
    return vid


def get_visitor(visitor_id: str) -> dict | None:
    with engine.connect() as c:
        row = c.execute(
            select(visitors).where(visitors.c.id == visitor_id)
        ).mappings().first()
    return dict(row) if row else None


def touch_visitor(visitor_id: str) -> None:
    if not visitor_id:
        return
    with engine.begin() as c:
        c.execute(
            visitors.update()
            .where(visitors.c.id == visitor_id)
            .values(last_seen_at=time.time())
        )


def set_unlock_passphrase(visitor_id: str, passphrase: str) -> bool:
    """Attach a sync code. Fails if another visitor already owns it."""
    digest = _hash_passphrase(passphrase)
    with engine.begin() as c:
        taken = c.execute(
            select(visitors.c.id).where(
                visitors.c.unlock_hash == digest, visitors.c.id != visitor_id
            )
        ).first()
        if taken:
            return False
        c.execute(
            visitors.update()
            .where(visitors.c.id == visitor_id)
            .values(unlock_hash=digest)
        )
    return True


def clear_unlock_passphrase(visitor_id: str) -> None:
    with engine.begin() as c:
        c.execute(
            visitors.update().where(visitors.c.id == visitor_id).values(unlock_hash=None)
        )


def visitor_for_passphrase(passphrase: str) -> str | None:
    digest = _hash_passphrase(passphrase)
    with engine.connect() as c:
        row = c.execute(
            select(visitors.c.id).where(visitors.c.unlock_hash == digest)
        ).first()
    return row[0] if row else None


# --------------------------------------------------------------- sessions
def session_count(visitor_id: str) -> int:
    with engine.connect() as c:
        return c.execute(
            select(sessions.c.id).where(sessions.c.visitor_id == visitor_id)
        ).all().__len__()


def create_session(visitor_id: str, title: str = "New chat") -> str:
    sid = str(uuid.uuid4())
    now = time.time()
    with engine.begin() as c:
        c.execute(sessions.insert().values(
            id=sid, visitor_id=visitor_id, title=title, user_name=None,
            query_count=0, created_at=now, updated_at=now,
        ))
    return sid


def get_session(sid: str, visitor_id: str | None = None) -> dict | None:
    """A session plus its decoded messages (scoped to the visitor when given)."""
    with engine.connect() as c:
        stmt = select(sessions).where(sessions.c.id == sid)
        if visitor_id:
            stmt = stmt.where(sessions.c.visitor_id == visitor_id)
        s = c.execute(stmt).mappings().first()
        if not s:
            return None
        rows = c.execute(
            select(messages).where(messages.c.session_id == sid)
            .order_by(messages.c.created_at.asc())
        ).mappings().all()
    return {"session": dict(s), "messages": [_decode_message(r) for r in rows]}


def list_sessions(visitor_id: str, limit: int) -> list[dict]:
    with engine.connect() as c:
        rows = c.execute(
            select(sessions).where(sessions.c.visitor_id == visitor_id)
            .order_by(sessions.c.updated_at.desc())
            .limit(limit)
        ).mappings().all()
    return [dict(r) for r in rows]


def set_title(sid: str, title: str) -> None:
    with engine.begin() as c:
        c.execute(sessions.update().where(sessions.c.id == sid).values(title=title))


def set_user_name(visitor_id: str, sid: str, name: str) -> None:
    """Remember the name the user gave, for the rest of the conversation."""
    with engine.begin() as c:
        c.execute(
            sessions.update().where(sessions.c.id == sid).values(user_name=name)
        )
        # Fall back to the visitor's name for any other session without one.
        c.execute(
            visitors.update().where(visitors.c.id == visitor_id).values(display_name=name)
        )


def bump_query_count(sid: str) -> int:
    """Increment and return this session's user-query count."""
    with engine.begin() as c:
        c.execute(
            sessions.update().where(sessions.c.id == sid)
            .values(query_count=sessions.c.query_count + 1)
        )
        row = c.execute(
            select(sessions.c.query_count).where(sessions.c.id == sid)
        ).first()
    return int(row[0]) if row else 0


def last_assistant_kind(sid: str) -> str | None:
    """Kind of the most recent assistant reply, e.g. 'clarify' — used to detect
    that the user is answering a clarifying widget and must get a recommendation."""
    with engine.connect() as c:
        row = c.execute(
            select(messages.c.kind).where(
                messages.c.session_id == sid, messages.c.role == "assistant"
            ).order_by(messages.c.created_at.desc()).limit(1)
        ).first()
    return row[0] if row else None


def trace_id_for_message(message_id: str | None, visitor_id: str | None = None) -> str | None:
    """Langfuse trace id for an assistant message, so scores land on the right trace."""
    if not message_id:
        return None
    with engine.connect() as c:
        stmt = select(messages.c.trace_id).where(messages.c.id == message_id)
        if visitor_id:
            stmt = stmt.where(messages.c.visitor_id == visitor_id)
        row = c.execute(stmt).first()
    return row[0] if row and row[0] else None


def delete_session(sid: str, visitor_id: str | None = None) -> bool:
    stmt = messages.delete().where(messages.c.session_id == sid)
    with engine.begin() as c:
        sdel = sessions.delete().where(sessions.c.id == sid)
        if visitor_id:
            sdel = sdel.where(sessions.c.visitor_id == visitor_id)
            stmt = stmt.where(messages.c.visitor_id == visitor_id)
        deleted = c.execute(sdel).rowcount
        if deleted:
            c.execute(stmt)
    return bool(deleted)


# --------------------------------------------------------------- messages
def add_message(
    sid: str,
    visitor_id: str | None,
    role: str,
    content: str,
    kind: str = "answer",
    titles: list | None = None,
    suggestions: list | None = None,
    clarify: dict | None = None,
    trace_id: str | None = None,
) -> str:
    mid = str(uuid.uuid4())
    now = time.time()
    with engine.begin() as c:
        c.execute(messages.insert().values(
            id=mid, session_id=sid, visitor_id=visitor_id, role=role, kind=kind,
            content=content or "", titles=_j(titles), suggestions=_j(suggestions),
            clarify=json.dumps(clarify) if clarify else None,
            trace_id=trace_id, created_at=now,
        ))
        c.execute(sessions.update().where(sessions.c.id == sid).values(updated_at=now))
    return mid


def _decode_message(row) -> dict:
    d = dict(row)
    d["titles"] = _uj(d.get("titles"))
    d["suggestions"] = _uj(d.get("suggestions"))
    if d.get("clarify"):
        try:
            d["clarify"] = json.loads(d["clarify"])
        except (json.JSONDecodeError, TypeError):
            d["clarify"] = None
    else:
        d["clarify"] = None
    return d


def history_for_llm(sid: str, max_messages: int = 14) -> list[dict]:
    """Recent turns as plain {role, content} for the LLM context."""
    with engine.connect() as c:
        rows = c.execute(
            select(messages.c.role, messages.c.content, messages.c.created_at)
            .where(messages.c.session_id == sid)
            .order_by(messages.c.created_at.asc())
        ).all()
    turns = [{"role": r[0], "content": r[1]} for r in rows if r[1]]
    return turns[-max_messages:]


# --------------------------------------------------------------- feedback
def add_feedback(
    rating: str,
    comment: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    visitor_id: str | None = None,
    question: str | None = None,
    answer: str | None = None,
    titles: list | None = None,
    app_rating: int | None = None,
) -> str:
    fid = str(uuid.uuid4())
    with engine.begin() as c:
        c.execute(feedback.insert().values(
            id=fid, session_id=session_id, message_id=message_id,
            visitor_id=visitor_id, rating=rating, app_rating=app_rating,
            comment=comment, question=question, answer=answer, titles=_j(titles),
            created_at=time.time(),
        ))
    return fid


def list_feedback(limit: int = 500) -> list[dict]:
    with engine.connect() as c:
        rows = c.execute(
            select(feedback).order_by(feedback.c.created_at.desc()).limit(limit)
        ).mappings().all()
    return [{**dict(r), "titles": _uj(r["titles"])} for r in rows]


# ------------------------------------------------------------ LLM usage
def log_llm_usage(
    task: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    cached: bool = False,
    session_id: str | None = None,
    visitor_id: str | None = None,
) -> None:
    with engine.begin() as c:
        c.execute(llm_usage.insert().values(
            id=str(uuid.uuid4()), session_id=session_id, visitor_id=visitor_id,
            task=task, model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd, latency_ms=latency_ms,
            cached=1 if cached else 0, created_at=time.time(),
        ))


def usage_summary(hours: int = 24) -> dict:
    """Cost + token roll-up for the observability dashboard."""
    since = time.time() - hours * 3600
    with engine.connect() as c:
        rows = c.execute(
            select(
                llm_usage.c.task, llm_usage.c.model,
                llm_usage.c.cached,
                llm_usage.c.prompt_tokens, llm_usage.c.completion_tokens,
                llm_usage.c.cost_usd, llm_usage.c.latency_ms,
            ).where(llm_usage.c.created_at >= since)
        ).all()
        recent = c.execute(
            select(
                llm_usage.c.created_at, llm_usage.c.task, llm_usage.c.model,
                llm_usage.c.prompt_tokens, llm_usage.c.completion_tokens,
                llm_usage.c.cost_usd, llm_usage.c.cached,
            ).where(llm_usage.c.created_at >= since)
            .order_by(llm_usage.c.created_at.desc()).limit(50)
        ).all()
        asked = c.execute(
            select(messages.c.content, messages.c.created_at, messages.c.session_id)
            .where(messages.c.role == "user")
            .order_by(messages.c.created_at.desc()).limit(50)
        ).all()

    by_model: dict[str, dict] = {}
    total_cost = total_tok = 0
    for task, model, cached, pt, ct, cost, _lat in rows:
        key = model or "unknown"
        b = by_model.setdefault(key, {
            "model": key, "calls": 0, "cached_calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
        })
        b["calls"] += 1
        if cached:
            b["cached_calls"] += 1
        b["prompt_tokens"] += pt or 0
        b["completion_tokens"] += ct or 0
        b["total_tokens"] += (pt or 0) + (ct or 0)
        b["cost_usd"] = round(b["cost_usd"] + (cost or 0.0), 6)
        total_cost += cost or 0.0
        total_tok += (pt or 0) + (ct or 0)

    return {
        "window_hours": hours,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tok,
        "call_count": len(rows),
        "by_model": sorted(by_model.values(), key=lambda x: -x["cost_usd"]),
        "recent_calls": [
            {"at": r[0], "task": r[1], "model": r[2], "prompt_tokens": r[3],
             "completion_tokens": r[4], "cost_usd": round(r[5] or 0.0, 6),
             "cached": bool(r[6])}
            for r in recent
        ],
        "recent_questions": [
            {"text": r[0], "at": r[1], "session_id": r[2]} for r in asked
        ],
    }


# Make sure the parent directory exists for a file-backed SQLite database.
def _ensure_sqlite_dir() -> None:
    url = _settings.url_for()
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


_ensure_sqlite_dir()
