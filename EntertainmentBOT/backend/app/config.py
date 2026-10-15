"""Environment-driven settings for Smart Entertainment BOT.

Every knob — LLM provider, entertainment data sources, cache, limits — is read
from the `.env` file so the backend stays provider-agnostic and deployable
without touching code.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------ LLM
    # Any OpenAI-compatible endpoint. Defaults to Groq's free tier.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"

    # Two-tier routing: `small` handles cheap deterministic work (greeting,
    # session titles, scope checks, cast lists); `large` handles the reasoning
    # heavy work (recommendations, comparisons, plot summaries, criticism).
    # Defaults match the models available on a Groq free-tier key.
    llm_model_large: str = "openai/gpt-oss-120b"
    llm_model_small: str = "openai/gpt-oss-20b"
    # Reasoning models (gpt-oss) answer inside a `reasoning` field. Setting an
    # effort level was the old workaround — but measured against Groq it
    # disables tool calling entirely (no content, no reasoning, no tool
    # calls), which silently breaks every turn. Leave blank.
    llm_reasoning_effort: str = ""

    # Failover: comma-separated model ids to try, in order, when the routed
    # model is rate-limited or overloaded. Free tiers meter per model, so a
    # second model usually has budget left when the first is exhausted.
    # Every candidate must support tool calling — the whole app depends on it.
    llm_failover_models: str = "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b"
    llm_failover_enabled: bool = True

    # Vision model for poster/image recognition. Groq's free tier does NOT
    # currently serve an image-capable model, so set LLM_VISION_API_KEY +
    # LLM_VISION_BASE_URL to a vision provider (Gemini's OpenAI-compatible
    # endpoint works) to enable poster uploads.
    llm_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm_vision_api_key: str = ""
    llm_vision_base_url: str = ""

    # ------------------------------------------------------- Entertainment
    # TMDB is the primary source of truth for every title shown to the user.
    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    # Optional: OMDb supplies real IMDb ratings (TMDB only has its own score).
    omdb_api_key: str = ""
    # Watchmode — the only reachable provider that does BOTH live-action
    # discovery (genre/year/country) and "where to watch". Free key:
    # https://www.watchmode.com/requestApiKey
    watchmode_api_key: str = ""
    # No-key fallbacks, used when TMDB has nothing (or no key is configured).
    enable_anilist_fallback: bool = True  # anime  — https://anilist.co (no key)
    enable_jikan_fallback: bool = True   # anime  — https://jikan.moe
    enable_tvmaze_fallback: bool = True  # series — https://tvmaze.com

    # --------------------------------------------------------------- Cache
    # When REDIS_URL is set we use Redis, otherwise an in-process TTL cache.
    # Both hold answers for `cache_ttl_seconds` (24h by default).
    redis_url: str = ""
    cache_ttl_seconds: int = 86400
    cache_prefix: str = "sebot"

    # ----------------------------------------------------------------- App
    db_path: str = "entertainment.db"
    # Set to a Postgres URL (Render / Neon) to persist outside the container.
    database_url: str = ""
    # Hard cap on the sidebar: creating a new chat fails until one is deleted.
    max_sessions: int = 5
    # Hard cap on user questions per session (assistant replies don't count).
    max_queries_per_session: int = 10
    # Never show more than this many titles in one recommendation.
    max_recommendations: int = 5
    # Poster-image uploads (images only, never video).
    max_upload_mb: float = 8.0
    # How many past turns are replayed to the LLM (token control).
    max_history_messages: int = 14
    cors_origins: str = "*"

    # -------------------------------------------------- Identity (visitors)
    # The browser stores a random visitor id; sessions are scoped to it. Users
    # may additionally set a passphrase ("sync code") to restore their sessions
    # on another device — enable_identity_unlock=false disables that.
    enable_identity_unlock: bool = True

    # ------------------------------------------------------- Observability
    # Langfuse (LLM tracing) — free tier at https://cloud.langfuse.com
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    # Keeps local/test traces out of your production dashboards.
    langfuse_environment: str = "development"
    langfuse_release: str = "entertainment-bot/1.0.0"
    # Prometheus metrics endpoint (GET /metrics)
    enable_metrics: bool = True

    # ----------------------------------------------- Cost accounting (USD)
    # USD per 1M tokens, used for the /api/observability/usage cost report.
    # These are Groq's public rates — override them in .env if they change.
    price_prompt_small: float = 0.05
    price_completion_small: float = 0.08
    price_prompt_large: float = 0.59
    price_completion_large: float = 0.79

    def url_for(self) -> str:
        """SQLAlchemy URL for the session store (Postgres if configured)."""
        if self.database_url.strip():
            return self.database_url.strip()
        return f"sqlite:///{self.db_path}"

    def prices_for(self, model: str) -> tuple[float, float]:
        """(prompt, completion) USD-per-1M-token rates for a model id."""
        m = (model or "").lower()
        small = (self.llm_model_small or "").lower()
        if m and small and m == small:
            return self.price_prompt_small, self.price_completion_small
        cheap_markers = ("8b", "instant", "flash", "mini", "haiku", "20b", "small", "27b")
        if m and any(k in m for k in cheap_markers):
            return self.price_prompt_small, self.price_completion_small
        return self.price_prompt_large, self.price_completion_large

    def model_for(self, task: str) -> str:
        """Resolve a task name ("small"/"large") to a concrete model id."""
        return self.llm_model_small if task == "small" else self.llm_model_large

    def failover_chain(self, task: str) -> list[str]:
        """Models to try in order for `task`, the routed one first.

        Duplicates are collapsed so a fallback never re-selects the model that
        just failed, and anything the operator has not listed is ignored.
        """
        primary = (self.model_for(task) or "").strip()
        chain = [primary] if primary else []
        if not self.llm_failover_enabled:
            return chain
        for raw in (self.llm_failover_models or "").split(","):
            name = raw.strip()
            if name and name not in chain:
                chain.append(name)
        return chain


@lru_cache
def get_settings() -> Settings:
    return Settings()
