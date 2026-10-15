"""Prometheus counters for errors, tool use, cache behaviour and LLM spend.

These register on the default registry, so they appear at `/metrics` alongside
the HTTP metrics. Every helper is a safe no-op if prometheus_client is missing,
and each one is wrapped so a metrics failure can never break a request.
"""
try:
    from prometheus_client import Counter, Histogram

    _ENABLED = True
except Exception:  # pragma: no cover
    _ENABLED = False

if _ENABLED:
    LLM_CALLS = Counter(
        "sebot_llm_calls_total", "LLM calls by model and task", ["model", "task", "status"]
    )
    LLM_TOKENS = Counter(
        "sebot_llm_tokens_total", "Tokens consumed", ["model", "direction"]
    )
    LLM_COST = Counter(
        "sebot_llm_cost_usd_total", "Estimated USD spend per model", ["model"]
    )
    LLM_LATENCY = Histogram(
        "sebot_llm_latency_seconds", "LLM call latency", ["model"],
        buckets=(0.2, 0.5, 1, 2, 4, 8, 16, 32, 64),
    )
    CACHE_RESULT = Counter(
        "sebot_cache_total", "Cache lookups by result", ["result"]
    )
    CACHE_WRITE = Counter("sebot_cache_writes_total", "Values written to the cache")
    TOOL_CALLS = Counter("sebot_tool_calls_total", "LLM tool invocations", ["tool"])
    CHAT_OUTCOMES = Counter(
        "sebot_chat_outcomes_total", "Assistant turn outcomes", ["kind"]
    )
    LLM_ERRORS = Counter("sebot_llm_errors_total", "LLM failures", ["stage"])
    UPSTREAM_ERRORS = Counter(
        "sebot_upstream_errors_total", "Upstream provider failures", ["source"]
    )
    UPSTREAM_CALLS = Counter(
        "sebot_upstream_calls_total", "Upstream provider requests", ["source"]
    )
    UNHANDLED_ERRORS = Counter(
        "sebot_unhandled_errors_total", "Unhandled request errors", ["path"]
    )
    FEEDBACK = Counter("sebot_feedback_total", "User feedback", ["rating"])
    APP_RATING = Counter("sebot_app_ratings_total", "App star ratings 1-5", ["stars"])
    SCOPE_REJECTIONS = Counter(
        "sebot_scope_rejections_total", "Off-topic questions declined"
    )
    CLARIFY = Counter("sebot_clarify_total", "Clarifying rounds asked", ["slot"])


def _safe(fn):
    try:
        fn()
    except Exception:
        pass


def llm_call(model: str, task: str, status: str = "ok") -> None:
    if _ENABLED:
        _safe(lambda: LLM_CALLS.labels(model=model, task=task, status=status).inc())


def llm_tokens(model: str, prompt: int = 0, completion: int = 0) -> None:
    if not _ENABLED:
        return
    if prompt:
        _safe(lambda: LLM_TOKENS.labels(model=model, direction="prompt").inc(prompt))
    if completion:
        _safe(lambda: LLM_TOKENS.labels(model=model, direction="completion").inc(completion))


def llm_cost(model: str, usd: float) -> None:
    if _ENABLED and usd:
        _safe(lambda: LLM_COST.labels(model=model).inc(usd))


def llm_latency(model: str, seconds: float) -> None:
    if _ENABLED:
        _safe(lambda: LLM_LATENCY.labels(model=model).observe(seconds))


def cache_result(result: str) -> None:
    if _ENABLED:
        _safe(lambda: CACHE_RESULT.labels(result=result).inc())


def cache_write() -> None:
    if _ENABLED:
        _safe(lambda: CACHE_WRITE.inc())


def tool_call(tool: str) -> None:
    if _ENABLED:
        _safe(lambda: TOOL_CALLS.labels(tool=tool).inc())


def chat_outcome(kind: str) -> None:
    if _ENABLED:
        _safe(lambda: CHAT_OUTCOMES.labels(kind=kind).inc())


def llm_error(stage: str) -> None:
    if _ENABLED:
        _safe(lambda: LLM_ERRORS.labels(stage=stage).inc())


def upstream_error(source: str) -> None:
    if _ENABLED:
        _safe(lambda: UPSTREAM_ERRORS.labels(source=source).inc())


def upstream_call(source: str) -> None:
    if _ENABLED:
        _safe(lambda: UPSTREAM_CALLS.labels(source=source).inc())


def unhandled_error(path: str) -> None:
    if _ENABLED:
        _safe(lambda: UNHANDLED_ERRORS.labels(path=path).inc())


def feedback(rating: str) -> None:
    if _ENABLED:
        _safe(lambda: FEEDBACK.labels(rating=rating).inc())


def app_rating(stars: int) -> None:
    if _ENABLED:
        _safe(lambda: APP_RATING.labels(stars=str(stars)).inc())


def scope_rejected() -> None:
    if _ENABLED:
        _safe(lambda: SCOPE_REJECTIONS.inc())


def clarify(slot: str) -> None:
    if _ENABLED:
        _safe(lambda: CLARIFY.labels(slot=slot or "general").inc())
