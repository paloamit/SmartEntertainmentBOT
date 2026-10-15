#!/usr/bin/env bash
# ============================================================================
#  Deploy Smart Entertainment BOT to Render.
#
#  Reads secrets from your LOCAL backend/.env and sends them straight to
#  Render's encrypted store. Nothing is written to the repo, and no value is
#  ever printed, logged, or committed.
#
#  USAGE
#    1. Render dashboard -> Account Settings -> API Keys -> create one
#    2. export RENDER_API_KEY=rnd_...          (or prefix with ./deploy.sh)
#    3. ./deploy.sh
#
#  OPTIONS
#    --with-db     also create a free Postgres (sessions survive restarts)
#    --dry-run     show the plan without creating anything
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO_ROOT/EntertainmentBOT/backend/.env"
API_KEY="${RENDER_API_KEY:-}"

WITH_DB=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --with-db) WITH_DB=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

if ! command -v render >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v render >/dev/null 2>&1 || {
    echo "Render CLI not found. Install:" >&2
    echo "  brew tap render-oss/tap && brew install render" >&2
    exit 1
  }
fi

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }

# ---- read a key out of the local .env (never echoed) ------------------------
envval() {
  local key="$1"
  sed -n "s/^${key}=//p" "$ENV_FILE" | head -1 | sed 's/^["'\'']//; s/["'\'']$//'
}

need() {
  local key="$1" val; val="$(envval "$key")"
  if [ -z "$val" ]; then
    echo "ERROR: '$key' is empty in backend/.env" >&2
    echo "       add it, then re-run." >&2
    exit 1
  fi
  printf '%s' "$val"
}

echo "==> Validating local .env"
for k in LLM_API_KEY LLM_VISION_API_KEY WATCHMODE_API_KEY LLM_VISION_MODEL; do
  [ -n "$(envval "$k")" ] && echo "    ✓ $k" || echo "    ✗ $k MISSING"
done

if [ -z "$API_KEY" ]; then
  cat <<'EOF'

ERROR: RENDER_API_KEY is not set.

  Render dashboard -> Account Settings -> API Keys -> Create Key
  then:  export RENDER_API_KEY=rnd_xxxxxxxx

(Note: this key is itself a secret. Treat it like a password.)
EOF
  exit 1
fi

WS_NAME="${RENDER_WORKSPACE:-Smart Entertainment BOT}"
API_SVC="entertainmentbot-api"
WEB_SVC="entertainmentbot-web"

echo "==> Workspace: $WS_NAME"

if [ "$DRY_RUN" = "1" ]; then
  cat <<EOF
DRY RUN — would create:
  Web Service : $API_SVC   (rootDir EntertainmentBOT/backend, free)
  Static Site : $WEB_SVC   (rootDir EntertainmentBOT/frontend)
  [optional]  : Postgres instance for DATABASE_URL
Secrets would be uploaded from local .env (values not shown).
EOF
  exit 0
fi

# ---------------------------------------------------------------- database --
DB_URL=""
if [ "$WITH_DB" = "1" ]; then
  echo "==> Creating Postgres"
  DB_OUT=$(render postgres create --name entertainmentbot-db --plan free \
             --database smartent --user smartent 2>&1)
  echo "$DB_OUT" | sed 's/^/    /'
  DB_URL="$(echo "$DB_OUT" | grep -oE 'postgres://[^ ]+' | head -1)"
  [ -z "$DB_URL" ] && { echo "    could not parse DB URL — set DATABASE_URL manually"; DB_URL=""; }
fi

# ------------------------------------------------------------ api service --
echo "==> Creating web service: $API_SVC"
render services create \
  --name "$API_SVC" \
  --type web \
  --repo https://github.com/paloamit/SmartEntertainmentBOT \
  --branch main \
  --runtime python \
  --plan free \
  --root-dir EntertainmentBOT/backend \
  --build-command "pip install -r requirements.txt" \
  --start-command "uvicorn app.main:app --host 0.0.0.0 --port \$PORT" \
  --health-check-path /api/health \
  --env-var "LLM_BASE_URL=https://api.groq.com/openai/v1" \
  --env-var "LLM_MODEL_LARGE=openai/gpt-oss-120b" \
  --env-var "LLM_MODEL_SMALL=openai/gpt-oss-20b" \
  --env-var "LLM_REASONING_EFFORT=" \
  --env-var "LLM_FAILOVER_ENABLED=true" \
  --env-var "LLM_FAILOVER_MODELS=openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b" \
  --env-var "LLM_VISION_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai" \
  --env-var "LLM_VISION_MODEL=$(need LLM_VISION_MODEL)" \
  --env-var "LLM_API_KEY=$(need LLM_API_KEY)" \
  --env-var "LLM_VISION_API_KEY=$(need LLM_VISION_API_KEY)" \
  --env-var "WATCHMODE_API_KEY=$(need WATCHMODE_API_KEY)" \
  --env-var "OMDB_API_KEY=$(envval OMDB_API_KEY)" \
  --env-var "TMDB_API_KEY=$(envval TMDB_API_KEY)" \
  --env-var "LANGFUSE_HOST=https://cloud.langfuse.com" \
  --env-var "LANGFUSE_ENVIRONMENT=production" \
  --env-var "LANGFUSE_PUBLIC_KEY=$(envval LANGFUSE_PUBLIC_KEY)" \
  --env-var "LANGFUSE_SECRET_KEY=$(envval LANGFUSE_SECRET_KEY)" \
  --env-var "ENABLE_ANILIST_FALLBACK=true" \
  --env-var "ENABLE_JIKAN_FALLBACK=true" \
  --env-var "ENABLE_TVMAZE_FALLBACK=true" \
  --env-var "ENABLE_METRICS=true" \
  --env-var "MAX_SESSIONS=5" \
  --env-var "MAX_QUERIES_PER_SESSION=10" \
  --env-var "MAX_RECOMMENDATIONS=5" \
  --env-var "CACHE_TTL_SECONDS=86400" \
  --env-var "CORS_ORIGINS=*" \
  ${DB_URL:+--env-var "DATABASE_URL=$DB_URL"} \
  | sed 's/^/    /'

# ----------------------------------------------------------- static site ----
# The API URL must exist before the frontend builds, since VITE_API_BASE_URL
# is baked in at build time.
echo "==> Resolving API URL"
API_URL=""
for _ in $(seq 1 20); do
  API_URL=$(render services list 2>/dev/null | grep -i "$API_SVC" \
            | grep -oE 'https://[a-z0-9.-]+\.onrender\.com' | head -1)
  [ -n "$API_URL" ] && break
  sleep 5
done
echo "    API URL: ${API_URL:-<not yet assigned — re-run after deploy>}"

echo "==> Creating static site: $WEB_SVC"
render services create \
  --name "$WEB_SVC" \
  --type web \
  --repo https://github.com/paloamit/SmartEntertainmentBOT \
  --branch main \
  --runtime static \
  --root-dir EntertainmentBOT/frontend \
  --build-command "npm ci && npm run build" \
  --publish-path dist \
  --env-var "VITE_API_BASE_URL=${API_URL:-https://entertainmentbot-api.onrender.com}" \
  --env-var "REACT_APP_rewrite=/* /index.html 200" \
  | sed 's/^/    /'

cat <<'EOF'

==> Done.

  Watch both deploys under the Render dashboard. Then:
    1. Set CORS_ORIGINS on the API to the WEB service URL
       (currently "*"), then redeploy the API.
    2. Confirm the web build picked up VITE_API_BASE_URL — if it was
       unresolved, redeploy the static site once the API URL exists.

  Free tier sleeps after ~15 min idle; first request after that takes ~50s.
EOF