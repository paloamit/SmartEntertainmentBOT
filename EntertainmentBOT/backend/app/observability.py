"""Langfuse tracing, wired to the current SDK's best practices.

Trace shape (one trace per chat turn, one session per conversation):

    respond-to-entertainment-request        span  (root: user msg -> reply)
      +- check-entertainment-scope          guardrail
      +- generate-response                  generation   <- one per model call,
      |    +- search-titles                 retriever       interleaved with the
      |    +- ask-user-for-filters          tool            tool calls they asked for
      |    +- generate-response             generation
      +- record-user-satisfaction           event

Design notes
------------
* **Not aggregated.** Each model invocation is its own `generation`, so you can
  see what the model decided after every tool result instead of one opaque total.
* **Tools are typed.** Pure lookups are `retriever`; actions are `tool`. That
  keeps the Agent Graph meaningful and lets evaluators target them by type.
* **Names are an API.** They are verb-first and carry no dynamic values, so
  dashboards and LLM-as-a-judge filters keep working as the app evolves.
* **No secrets.** Only the user's message and our reply are traced. Uploaded
  images go in as `LangfuseMedia`, never as a raw base64 blob.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterator, Optional

from .config import get_settings

logger = logging.getLogger("sebot.observability")
_settings = get_settings()

_client = None

# Tools that only look something up get `retriever`; everything else is a `tool`.
TOOL_OBSERVATION_TYPES = {
    "find_titles": "retriever",
    "get_similar_titles": "retriever",
    "get_cast_crew": "retriever",
    "get_title_reviews": "retriever",
    "get_title_summary": "tool",
    "ask_user": "tool",
    "recommend_titles": "tool",
}


def enabled() -> bool:
    return bool(_settings.langfuse_public_key and _settings.langfuse_secret_key)


def _ensure_env() -> None:
    """The SDK reads credentials from the environment at construction time."""
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", _settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", _settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", _settings.langfuse_host)


def client():
    global _client
    if _client is None:
        _ensure_env()
        from langfuse import Langfuse

        _client = Langfuse(
            host=_settings.langfuse_host,
            environment=_settings.langfuse_environment,
            release=_settings.langfuse_release,
        )
    return _client


@contextlib.contextmanager
def _noop() -> Iterator[None]:
    yield None


class Turn:
    """Root span for one request — its input/output become the trace's.

    Deliberately NOT a context manager: the SSE generator yields between steps
    and resumes in a different OpenTelemetry context, which makes an ambient
    `with` block fail to detach. Instead we start the span explicitly and pass
    `parent` (a trace_context dict) to each child observation, so nesting is
    restored by hand and works across yields and threads.
    """

    __slots__ = ("session_id", "user_id", "tags", "trace_id", "parent", "_obs")

    def __init__(self, session_id, user_id, tags):
        self.session_id = session_id
        self.user_id = user_id
        self.tags = tags
        self.trace_id: str | None = None
        self.parent: dict | None = None
        self._obs = None

    def set_output(self, output: Any) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(output=output)
        except Exception:
            logger.debug("could not set turn output", exc_info=True)

    def set_error(self, message: str) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(level="ERROR", status_message=message[:500])
        except Exception:
            pass

    def end(self) -> None:
        if self._obs is None:
            return
        try:
            self._obs.end()
        except Exception:
            logger.debug("could not end turn observation", exc_info=True)
        finally:
            self._obs = None


def start_turn(name: str, input: Any, session_id: str | None = None,
               user_id: str | None = None, tags: list[str] | None = None,
               metadata: dict | None = None) -> Turn:
    """Open the root span for one turn and return a handle for its children."""
    turn = Turn(session_id, user_id, tags)
    if not enabled():
        return turn
    try:
        from langfuse import propagate_attributes

        lf = client()
        attrs = {}
        if session_id:
            attrs["session_id"] = session_id
        if user_id:
            attrs["user_id"] = user_id
        if tags:
            attrs["tags"] = tags
        with propagate_attributes(**attrs):
            obs = lf.start_observation(
                as_type="span", name=name, input=input, metadata=metadata
            )
        turn._obs = obs
        turn.trace_id = getattr(obs, "trace_id", None)
        turn.parent = {"trace_id": obs.trace_id, "parent_span_id": obs.id}
    except Exception:
        logger.debug("langfuse start_turn failed", exc_info=True)
    return turn


@contextlib.contextmanager
def observation(name: str, as_type: str = "span", input: Any = None,
                model: str | None = None, session_id: str | None = None,
                user_id: str | None = None, tags: list[str] | None = None,
                metadata: dict | None = None, parent: "Turn | dict | None" = None):
    """A nested observation: generation / retriever / tool / guardrail / span.

    `parent` links this observation to a specific trace instead of relying on
    ambient context, which is what makes streaming (yield-bound) turns work.
    """
    if not enabled():
        yield None
        return

    # --- set-up phase: on failure we yield None and exit, never raise ---
    try:
        lf = client()
        ctx = _parent_context(parent, session_id, user_id, tags)
        params: dict[str, Any] = {"as_type": as_type, "name": name}
        if ctx:
            params["trace_context"] = ctx
        if input is not None:
            params["input"] = input
        if model:
            params["model"] = model
        if metadata:
            params["metadata"] = metadata
        manager = lf.start_as_current_observation(**params)
    except Exception:
        logger.debug("langfuse observation %s could not start", name, exc_info=True)
        yield None
        return

    # --- body phase: an error here belongs to the caller, so it must propagate.
    #     Critically we must NOT yield again in the except branch — a context
    #     manager that yields during unwinding raises
    #     "generator didn't stop after throw()" and masks the real error.
    try:
        with manager as o:
            yield o
    except Exception:
        logger.debug("langfuse observation %s raised", name, exc_info=True)
        raise


def _parent_context(parent, session_id, user_id, tags) -> dict | None:
    """Either an explicit parent link, or attributes alone for a fresh trace."""
    if isinstance(parent, Turn):
        return parent.parent
    if isinstance(parent, dict) and parent.get("trace_id"):
        return parent
    attrs = {}
    if session_id:
        attrs["session_id"] = session_id
    if user_id:
        attrs["user_id"] = user_id
    if tags:
        attrs["tags"] = tags
    if attrs:
        from langfuse import propagate_attributes

        return propagate_attributes(**attrs)
    return None


def tool(name: str, input: Any = None, **kw):
    """Typed automatically: lookups become retrievers, actions become tools."""
    as_type = TOOL_OBSERVATION_TYPES.get(name, "tool")
    return observation(f"observe-{name.replace('_', '-')}", as_type=as_type,
                       input=input, **kw)


def current_trace_id() -> str | None:
    if not enabled():
        return None
    try:
        return client().get_current_trace_id()
    except Exception:
        return None


def record_generation(obs, output: Any = None, input_tokens: int = 0,
                      output_tokens: int = 0, model: str | None = None,
                      level: str | None = None, status_message: str | None = None,
                      metadata: dict | None = None) -> None:
    """Finish a `generation` with its output, token usage and (optional) cost."""
    if obs is None:
        return
    try:
        payload: dict[str, Any] = {}
        if output is not None:
            payload["output"] = output
        if input_tokens or output_tokens:
            payload["usage_details"] = {"input": input_tokens, "output": output_tokens}
        if model:
            payload["model"] = model
        if level:
            payload["level"] = level
        if status_message:
            payload["status_message"] = status_message
        if metadata:
            payload["metadata"] = metadata
        if payload:
            obs.update(**payload)
    except Exception:
        logger.debug("could not update generation observation", exc_info=True)


def media(base64_data_uri: str, content_type: str | None = None):
    """Wrap an uploaded image so Langfuse renders it instead of storing a blob."""
    if not enabled():
        return "<uploaded image>"
    try:
        from langfuse import LangfuseMedia

        return LangfuseMedia(
            base64_data_uri=base64_data_uri,
            content_type=content_type,  # type: ignore[arg-type]
        )
    except Exception:
        return "<uploaded image>"


def score(name: str, value: float, trace_id: str | None = None,
          session_id: str | None = None, comment: str | None = None,
          source: str = "user-feedback") -> None:
    """Attach a score to a trace (used for satisfied/dissatisfied feedback)."""
    if not enabled() or not trace_id:
        return
    try:
        client().create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            session_id=session_id,
            data_type="NUMERIC",
            comment=comment,
            source=source,
        )
    except Exception:
        logger.warning("could not record langfuse score %s", name, exc_info=True)


def event(name: str, metadata: dict | None = None, session_id: str | None = None,
          user_id: str | None = None) -> None:
    """A discrete point in the trace (cache hit, limit reached, decline…)."""
    if not enabled():
        return
    try:
        client().create_event(
            name=name, metadata=metadata, session_id=session_id, user_id=user_id
        )
    except Exception:
        logger.debug("could not create langfuse event %s", name, exc_info=True)


def flush() -> None:
    """Push buffered observations — required before the process exits."""
    if not enabled():
        return
    try:
        client().flush()
    except Exception:
        logger.debug("langfuse flush failed", exc_info=True)


def shutdown() -> None:
    if not enabled():
        return
    try:
        client().shutdown()
    except Exception:
        logger.debug("langfuse shutdown failed", exc_info=True)


def health() -> dict:
    return {
        "enabled": enabled(),
        "host": _settings.langfuse_host,
        "environment": _settings.langfuse_environment,
    }
