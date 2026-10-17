"""Scope guardrail and lightweight user-name detection.

The bot only answers entertainment questions. A classifier decides whether a
message is on-topic, and it is deliberately cheap:

1. Keyword fast-path — an obvious "in scope" or obviously "off topic" message
   never reaches the model at all (this covers most traffic, so costs ~0).
2. Otherwise the *small* model classifies it, and the verdict is cached for
   24h. The same recurring question is therefore classified once, ever.

Name detection works the same way and only fires on very short messages, so a
one-word reply like "Inception" can't be mistaken for a person's name.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import cache, metrics

logger = logging.getLogger("sebot.guardrails")

_ENT_KEYWORDS = re.compile(r"""
\b(movie|movies|film|films|cinema|series|show|shows|episode|episodes|season|seasons
|anime|manga|animation|animated|cartoon|actor|actress|cast|director|producer
|genre|genres|plot|story|ending|synopsis|summary|summarize|summarise|review|reviews
|critics?|rating|ratings|imdb|rottentomatoes|netflix|prime\s?video|hotstar|spotify
|disney|hulu|bbc|amc|hbo|watch|watching|rewatch|binge|streaming|stream|where\s+to\s+watch
|trending|popular|top\s+\d+|best|similar|like\s+.+\s+(movie|film|series|show)
|bollywood|hollywood|korean|k-drama|japanese|holiwood|tamil|telugu|malayalam|kannada
|marathi|bengali|punjabi|actress|performance|character|scene|sequel|prequel|franchise
|oscar|award|box\s+office|release\s+date|upcoming|new\s+series|binge[- ]watch)
\b""", re.I | re.X)

_OFFTOPIC_KEYWORDS = re.compile(r"""
\b(write|code|coding|program|programme|function|bug|compile|python|javascript|java
|sql|regex|debug|api\s+key|algorithm|database\s+schema|homework|essay|poem|summarize\s+this\s+article
|weather|stock\s+price|crypto|bitcoin|cricket\s+score|football\s+score|recipe|calories\s+for\s+dinner
|medical|diagnos|symptom\s+of|legal\s+advice|tax\s+return|resume\s+for|cover\s+letter
|translate\s+into|solve\s+this\s+equation|derivative|integral\s+of)
\b""", re.I | re.X)

_SCOPE_PROMPT = """You classify user messages for an entertainment assistant.
Answer with STRICT JSON only, nothing else.

Reply {"entertainment": true} if the message is about movies, TV series, films,
anime, actors, directors, genres, plots, ratings, streaming services, or finding
something to watch.
Reply {"entertainment": false} for EVERYTHING else (coding, math, general
knowledge, news, health, sports, advice, small talk that is not about a show or
film).

Message: %s"""


def _small_model() -> Optional[str]:
    from .config import get_settings

    return get_settings().llm_model_small or None


def _classify(prompt: str, cache_key: str) -> dict | None:
    """One tiny, cached LLM classification. Returns None if unavailable."""
    from . import llm

    cached = cache.get_json(cache_key)
    if cached is not None:
        return cached
    out = llm.complete_json(_small_model(), prompt, task="scope", max_tokens=50)
    if out:
        cache.set_json(cache_key, out)
    return out


def check_scope(message: str) -> bool:
    """True when the message is an entertainment question we should answer."""
    text = (message or "").strip()
    if not text:
        return False

    # 1. Fast paths — no model call at all.
    if _ENT_KEYWORDS.search(text):
        return True

    if _OFFTOPIC_KEYWORDS.search(text):
        return False

    # 2. Ambiguous — let the small model decide (cached for 24h).
    data = _classify(_SCOPE_PROMPT % text[:400], f"scope:{cache.normalize_question(text)}")
    if data is None:
        # The classifier could not be reached. That is an outage, not a
        # verdict. Reading it as "off topic" made the bot refuse every
        # keyword-free question — including "tell me about Pulp Fiction" —
        # for as long as the model provider was rate-limited. Failing open is
        # the safe direction: grounding already prevents invented titles, so
        # a slightly off-topic answer beats refusing a valid one.
        #
        # Deliberately NOT using the title-recovery heuristic as a fast path
        # here: its "what is ..." prefix happily matches "what is the weather
        # in London", which would wave through exactly the questions the
        # classifier exists to catch. Fail-open keeps precision intact when
        # the classifier is healthy, and only relaxes during an outage.
        metrics.scope_unavailable()
        logger.warning("scope classifier unavailable; treating %r as in scope",
                       text[:80])
        return True
    verdict = bool(data.get("entertainment"))
    if not verdict:
        metrics.scope_rejected()
    return verdict


_AFFIRMATIVE = re.compile(
    r"^(y|ya|yeah|yes|yep|yup|yuppers|sure|ok|okay|k|kk|haan|ha|haa|ji|haanji|"
    r"bilkul|sh|shu|hai|thik\s*hai|fine|good|great|perfect|absolutely|definitely|"
    r"exactly|indeed|sounds?\s*(good|great|perfect|right)|go\s*ahead|please\s*do|"
    r"do\s*it|why\s*not|alright|cool|nice|awesome|love\s*that|"
    r"no|nope|nah|not\s*really|maybe|perhaps|depends|surprise\s*me|anything|"
    r"whatever|anything\s*(is\s*)?(fine|good)|thanks?|thank\s*you|cheers|"
    r"got\s*it|sounds?\s*good|make\s*it\s*so|show\s*me|hit\s*me)\b[\s!.?]*$",
    re.I,
)

# Bare follow-ups that only make sense inside an entertainment conversation.
# "what is this" / "check now" carry no topic signal on their own, so the
# classifier reads them as off-topic and declines a user who is mid-chat.
_CONTINUATION = re.compile(
    r"^(?:"
    r"what(?:'s| is| was| are)?\s*(?:this|that|it|he|she|they)?"
    r"|who(?:'s| is| was| are)?\s*(?:this|that|it|he|she|they)?"
    r"|identify\s*(?:this|that|it)?|recogni[sz]e\s*(?:this|that|it)?"
    r"|check(?:\s*(?:this|that|it|now))?|try\s*(?:again|now)?|go\s*on"
    r"|continue|next|more|and\s*\??|then\s*\??|why\s*\??|how\s*\??"
    r"|really|seriously|are\s*you\s*sure"
    r")\b[\s!.?]*$",
    re.I,
)


def is_affirmative(text: str) -> bool:
    """True for a short yes/no/ack reply — "Yes", "sure", "sounds good", …

    These are always continuations of the current conversation, never a new
    topic, so they must never be classified as out-of-scope.
    """
    t = (text or "").strip()
    return bool(_AFFIRMATIVE.match(t) or _CONTINUATION.match(t))


def check_scope_traced(message: str, parent=None,
                       after_clarify: bool = False) -> bool:
    """`check_scope` recorded as a `guardrail` observation in the trace.

    A short acknowledgement is accepted unconditionally: it cannot be an
    off-topic request. After a clarifying question, any very short reply counts
    too, because the topic is already known from context.
    """
    from . import observability as obs

    text = (message or "").strip()
    if is_affirmative(text) or (after_clarify and len(text.split()) <= 3):
        return True

    with obs.observation(
        "check-entertainment-scope", as_type="guardrail",
        input={"message": text[:400]}, parent=parent,
    ) as g:
        verdict = check_scope(message)
        if g is not None:
            try:
                g.update(
                    output={"entertainment": verdict},
                    level="DEFAULT" if verdict else "WARNING",
                    status_message=None if verdict else "declined: outside entertainment scope",
                )
            except Exception:
                pass
    return verdict


_NAME_PROMPT = """A user was optionally asked for their name in a chat app.
Extract their personal first name from the message below.

Return STRICT JSON only:
{"is_name": true, "name": "Amit"}   when the message is just a person's name
{"is_name": false, "name": ""}      for anything else (questions, movie titles,
                                     sentences, greetings like "hi", numbers)

Only treat it as a name if it looks like a human personal name.
Message: %s"""


def extract_name(message: str) -> Optional[str]:
    """Pull a volunteered personal name out of a short reply, or None."""
    text = (message or "").strip()
    # Long messages are certainly not a bare name — skip the model call.
    if not text or len(text) > 40 or len(text.split()) > 4:
        return None

    data = _classify(_NAME_PROMPT % text, f"name:{text.lower()}")
    if not (data or {}).get("is_name"):
        return None
    name = re.sub(r"[^A-Za-z\s'-]", "", str(data.get("name") or "")).strip()
    if not name or len(name) > 30 or len(name.split()) > 3:
        return None
    return name


OFF_TOPIC_REPLY = (
    "That's outside my beat — I'm **Smart Entertainment BOT** 🎬, so I stick to "
    "movies, web series and animation. Want me to find something great to watch?"
)
