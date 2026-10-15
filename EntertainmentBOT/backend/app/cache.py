"""Two-tier cache: Redis when `REDIS_URL` is set, otherwise an in-process TTL map.

Both backends share one interface and one 24h TTL, so the rest of the app never
knows which is active. If Redis is unreachable at any point we log once and
transparently fall back to memory — a cache outage must never break a request.

What gets cached
----------------
* ``answer:<hash>``  a finished assistant reply, so a similar question later is
  served without touching the LLM at all.
* ``api:<hash>``     raw entertainment API responses, which keeps provider
  quotas (and our latency) healthy.
* ``summary:<key>``  generated plot summaries for a single title.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Optional

from .config import get_settings

logger = logging.getLogger("sebot.cache")
_settings = get_settings()

try:
    from . import metrics
except Exception:  # pragma: no cover - metrics must never break the cache
    metrics = None  # type: ignore


def _inc(event: str, **labels) -> None:
    if metrics is not None:
        try:
            getattr(metrics, event)(**labels)
        except Exception:
            pass


class MemoryBackend:
    """Bounded, thread-safe TTL map used when Redis isn't configured."""

    name = "memory"

    def __init__(self, max_entries: int = 2000):
        self._data: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()
        self._max = max_entries

    def _evict_if_needed(self) -> None:
        if len(self._data) <= self._max:
            return
        now = time.time()
        expired = [k for k, (exp, _) in self._data.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
        # Still full: drop the entries closest to expiry.
        while len(self._data) > self._max:
            oldest = min(self._data, key=lambda k: self._data[k][0])
            self._data.pop(oldest, None)

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            expires_at, raw = entry
            if expires_at <= time.time():
                self._data.pop(key, None)
                return None
            return raw

    def set(self, key: str, value: bytes, ttl: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl, value)
            self._evict_if_needed()

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def ping(self) -> bool:
        return True


class RedisBackend:
    """Redis-backed cache. Every call degrades to a miss on failure."""

    name = "redis"

    def __init__(self, url: str):
        import redis  # imported lazily so the dep stays optional

        self._client = redis.Redis.from_url(
            url, decode_responses=False, socket_timeout=3, socket_connect_timeout=3
        )
        self._prefix = _settings.cache_prefix
        self._broken = False

    def _k(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> Optional[bytes]:
        if self._broken:
            return None
        try:
            return self._client.get(self._k(key))
        except Exception:
            self._trip("get")
            return None

    def set(self, key: str, value: bytes, ttl: int) -> None:
        if self._broken:
            return
        try:
            self._client.setex(self._k(key), ttl, value)
        except Exception:
            self._trip("set")

    def delete(self, key: str) -> None:
        if self._broken:
            return
        try:
            self._client.delete(self._k(key))
        except Exception:
            self._trip("delete")

    def ping(self) -> bool:
        if self._broken:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            self._trip("ping")
            return False

    def _trip(self, op: str) -> None:
        """Disable Redis for the process after a failure; memory takes over."""
        if not self._broken:
            self._broken = True
            logger.warning(
                "Redis unavailable during %s — falling back to in-process cache. "
                "Restart to retry Redis.", op,
            )
        _inc("upstream_error", source="redis")


_backend: Any = MemoryBackend()

if _settings.redis_url.strip():
    try:
        _backend = RedisBackend(_settings.redis_url.strip())
        _backend.ping()
        logger.info("cache backend: redis (%s)", _settings.redis_url.split("@")[-1])
    except Exception:
        logger.warning(
            "Redis could not be initialised — using the in-process cache instead.",
            exc_info=True,
        )
        _backend = MemoryBackend()
else:
    logger.info("cache backend: in-process (set REDIS_URL to share a Redis cache)")


def backend_name() -> str:
    return getattr(_backend, "name", "memory")


def get_json(key: str) -> Any | None:
    """Return a cached JSON value, or None on a miss. Counts hits/misses."""
    try:
        raw = _backend.get(key)
    except Exception:
        raw = None
    if raw is None:
        _inc("cache_result", result="miss")
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        _inc("cache_result", result="miss")
        return None
    _inc("cache_result", result="hit")
    return value


def set_json(key: str, value: Any, ttl: int | None = None) -> None:
    try:
        payload = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return
    try:
        _backend.set(key, payload.encode(), ttl or _settings.cache_ttl_seconds)
        _inc("cache_write")
    except Exception:
        pass


def remember(key: str, producer: Callable[[], Any], ttl: int | None = None) -> Any:
    """Cache-aside helper: return the cached value, else call `producer`."""
    hit = get_json(key)
    if hit is not None:
        return hit
    value = producer()
    if value is not None:
        set_json(key, value, ttl)
    return value


def delete(key: str) -> None:
    try:
        _backend.delete(key)
    except Exception:
        pass


def health() -> dict:
    return {"backend": backend_name(), "ttl_seconds": _settings.cache_ttl_seconds,
            "reachable": bool(_backend.ping())}


# ------------------------------------------------------- answer cache keys
# Function words carry no meaning here, so dropping them lets genuinely similar
# questions ("can you suggest a horror movie" / "horror movie") share one cached
# answer — across different users too. Digits are kept, because "5 movies" and
# "3 movies" must never collide.
_FILLER = frozenset("""
a an the of for to me my please can could you i am im do does have has want need
find show suggest recommend give some any whats what is are me there that this
""".split())

_LEAD_INS = ("hi", "hello", "hey", "yo", "ok", "okay", "hii")


def normalize_question(text: str) -> str:
    """Reduce a question to its comparable core (used as the answer cache key)."""
    s = (text or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    tokens = [t for t in s.split() if t not in _FILLER]
    while tokens and tokens[0] in _LEAD_INS:
        tokens.pop(0)
    return " ".join(tokens).strip()


def answer_key(text: str) -> str:
    """Cache key for a question's finished answer. Empty when unusable."""
    norm = normalize_question(text)
    if len(norm) < 3:      # too short to compare safely — never serve a guess
        return ""
    import hashlib

    return f"answer:{hashlib.sha256(norm.encode()).hexdigest()[:32]}"
