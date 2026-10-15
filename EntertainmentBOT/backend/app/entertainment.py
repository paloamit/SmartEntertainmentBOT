"""Entertainment data access: the single source of truth for every title.

TMDB is the primary provider (movies, series and animation, with posters,
ratings, cast, trailers and streaming availability). TVMaze and Jikan act as
free no-key fallbacks so the bot still answers when TMDB is unconfigured or has
no match. IMDb ratings come from OMDb when a key is supplied.

Every response is cached for 24h, and every title is reduced to a stable
`source:type:id` key. The LLM only ever sees titles that came back through these
functions, which is what makes the "never invent a title" guarantee enforceable.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Optional

import httpx

from . import cache, metrics
from .config import get_settings

logger = logging.getLogger("sebot.entertainment")
_settings = get_settings()

_client = httpx.Client(timeout=20, follow_redirects=True)

IMAGE_BASE = "https://image.tmdb.org/t/p/"
POSTER_SIZE = "w500"
BACKDROP_SIZE = "w1280"
PROFILE_SIZE = "w185"
LOGO_SIZE = "w92"

# Curated genre shortcuts offered to the LLM for the clarifying-question widget.
# Names only — the backend resolves each one to a real TMDB genre id.
CURATED_GENRES = [
    "Action", "Adventure", "Animation", "Anime", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western",
    "Music", "Reality", "Talk Show", "Supernatural", "Cyberpunk", "Heist",
    "Road Movie", "Coming of Age", "Dark Comedy", "Psychological", "Zombie",
]

LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi", "bn": "Bengali",
    "pa": "Punjabi", "gu": "Gujarati", "ur": "Urdu", "or": "Odia",
    "as": "Assamese", "ne": "Nepali", "si": "Sinhala", "es": "Spanish",
    "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "th": "Thai",
    "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "tl": "Filipino",
    "tr": "Turkish", "fa": "Persian", "ar": "Arabic", "he": "Hebrew",
    "cs": "Czech", "hu": "Hungarian", "ro": "Romanian", "el": "Greek",
    "is": "Icelandic", "af": "Afrikaans",
}

# Countries with meaningful film/TV output — offered in the country dropdown.
COUNTRIES = {
    "IN": "India", "US": "United States", "GB": "United Kingdom",
    "AU": "Australia", "CA": "Canada", "DE": "Germany", "FR": "France",
    "ES": "Spain", "IT": "Italy", "JP": "Japan", "KR": "South Korea",
    "CN": "China", "BR": "Brazil", "MX": "Mexico", "RU": "Russia",
    "NL": "Netherlands", "SE": "Sweden", "IE": "Ireland", "NZ": "New Zealand",
    "ZA": "South Africa", "AE": "UAE", "SG": "Singapore", "MY": "Malaysia",
    "PK": "Pakistan", "BD": "Bangladesh", "ID": "Indonesia", "PH": "Philippines",
    "TH": "Thailand", "TR": "Turkey", "NG": "Nigeria", "EG": "Egypt",
    "AR": "Argentina", "CL": "Chile", "CO": "Colombia", "PL": "Poland",
    "DK": "Denmark", "NO": "Norway", "FI": "Finland", "CH": "Switzerland",
    "AT": "Austria", "BE": "Belgium", "PT": "Portugal", "GR": "Greece",
    "IL": "Israel", "SA": "Saudi Arabia", "QA": "Qatar", "KW": "Kuwait",
    "NP": "Nepal", "LK": "Sri Lanka", "MV": "Maldives", "BT": "Bhutan",
    "KZ": "Kazakhstan", "UA": "Ukraine", "RS": "Serbia", "HR": "Croatia",
    "CZ": "Czechia", "SK": "Slovakia", "BG": "Bulgaria", "HU": "Hungary",
    "RO": "Romania", "LT": "Lithuania", "LV": "Latvia", "EE": "Estonia",
    "MA": "Morocco", "KE": "Kenya", "GH": "Ghana", "ET": "Ethiopia",
    "PE": "Peru", "VE": "Venezuela", "EC": "Ecuador", "BO": "Bolivia",
    "UY": "Uruguay", "CU": "Cuba", "HN": "Honduras", "PY": "Paraguay",
    "TW": "Taiwan", "HK": "Hong Kong",
}

# Ready-made release-year buckets so the year dropdown is never a bare "Any".
YEAR_BUCKETS = [
    "2026", "2025", "2024", "2023", "2022", "2021",
    "2020s (2020-2026)", "2010s", "2000s", "1990s", "1980s", "1970s",
    "Before 1980",
]

FORMAT_OPTIONS = ["Movie", "TV Series", "Animation / Anime", "Documentary"]

# Countries we surface first in the watch-provider picker.
WATCH_PRIORITY = ["IN", "US", "GB", "AU", "CA"]

ANIME_KEYWORD_ID = 210024  # TMDB keyword for "anime"
ANIMATION_GENRE_ID = 16


def language_label(code: str | None) -> str:
    if not code:
        return ""
    return LANGUAGES.get(code.lower(), code.upper())


def country_label(code: str | None) -> str:
    if not code:
        return ""
    return COUNTRIES.get(code.upper(), code.upper())


def _img(path: str | None, size: str) -> str:
    return f"{IMAGE_BASE}{size}{path}" if path else ""


# --------------------------------------------------------------- HTTP glue
def _cache_key(path: str, params: dict) -> str:
    import hashlib

    clean = {k: v for k, v in sorted((params or {}).items()) if v not in (None, "", [])}
    blob = f"{path}|{clean}"
    return f"api:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


def _get(source: str, url: str, params: dict | None = None, ttl: int | None = None) -> dict | None:
    """GET with a 24h cache. Returns None (never raises) when the source fails."""
    key = _cache_key(f"{source}:{url}", params or {})
    cached = cache.get_json(key)
    if cached is not None:
        return cached
    try:
        metrics.upstream_call(source)
        resp = _client.get(url, params=params or {})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        metrics.upstream_error(source)
        logger.warning("%s request failed: %s", source, exc)
        return None
    cache.set_json(key, data, ttl)
    return data


def _tmdb(path: str, params: dict | None = None) -> dict | None:
    if not _settings.tmdb_api_key.strip():
        return None
    query = dict(params or {})
    query["api_key"] = _settings.tmdb_api_key.strip()
    return _get("tmdb", f"{_settings.tmdb_base_url}{path}", query)


# Jikan is a free community API and is often slow or briefly rate-limited, so
# it gets a longer timeout than the primary provider.
_jikan_client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True)
_anilist_client = httpx.Client(timeout=httpx.Timeout(25.0, connect=10.0))


def _jikan(path: str, params: dict | None = None) -> dict | None:
    if not _settings.enable_jikan_fallback:
        return None
    key = _cache_key(f"jikan:{path}", params or {})
    cached = cache.get_json(key)
    if cached is not None:
        return cached
    try:
        metrics.upstream_call("jikan")
        resp = _jikan_client.get(f"https://api.jikan.moe/v4{path}", params=params or {})
        if resp.status_code == 429:
            logger.warning("jikan rate limited; skipping this fallback")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        metrics.upstream_error("jikan")
        logger.warning("jikan request failed: %s", exc)
        return None
    cache.set_json(key, data, 43200)
    return data


def _tvmaze(path: str, params: dict | None = None) -> dict | None:
    if not _settings.enable_tvmaze_fallback:
        return None
    return _get("tvmaze", f"https://api.tvmaze.com{path}", params or {}, ttl=43200)


def tmdb_configured() -> bool:
    return bool(_settings.tmdb_api_key.strip())


def any_source_configured() -> bool:
    """True when at least one provider can answer, key or no key."""
    return (
        tmdb_configured()
        or watchmode_configured()
        or bool(_settings.omdb_api_key.strip())
        or _settings.enable_anilist_fallback
        or _settings.enable_jikan_fallback
        or _settings.enable_tvmaze_fallback
    )


# ------------------------------------------------------------ reference data
def genre_map(media_type: str = "movie") -> dict[str, int]:
    """Normalised genre name -> TMDB id (fetched once, cached 24h)."""
    path = "/genre/movie/list" if media_type == "movie" else "/genre/tv/list"
    data = _tmdb(path) or {"genres": []}
    return {
        re.sub(r"[^a-z0-9]+", "", (g.get("name") or "").lower()): g["id"]
        for g in data.get("genres", []) if g.get("id")
    }


def _resolve_genres(names, media_type: str = "movie") -> list[int]:
    """Map user/LLM genre names onto TMDB ids; ignore anything unrecognised."""
    gmap = genre_map(media_type)
    ids: list[int] = []
    for raw in names or []:
        key = re.sub(r"[^a-z0-9]+", "", str(raw).lower()).replace("sciencefiction", "scifi")
        gid = gmap.get(key)
        if gid is None:
            # "Sci-Fi" on the movie side is "Science Fiction", on TV "Sci-Fi & Fantasy".
            for name, candidate in gmap.items():
                if key and (key in name or name in key):
                    gid = candidate
                    break
        if gid and gid not in ids:
            ids.append(gid)
    return ids


def _resolve_country(value) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    if len(v) == 2 and v.upper() in COUNTRIES:
        return v.upper()
    up = re.sub(r"[^a-z]", "", v.lower())
    for code, name in COUNTRIES.items():
        if re.sub(r"[^a-z]", "", name.lower()) == up:
            return code
    return v.upper()[:2] if len(v) >= 2 else None


def _resolve_language(value) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if len(v) == 2:
        return v
    for code, name in LANGUAGES.items():
        if name.lower() == v:
            return code
    return v[:2]


# ------------------------------------------------------------- normalisation
def _tmdb_card(d: dict, media_type: str, detailed: bool = False) -> dict:
    """Normalise a TMDB movie/tv payload into our unified card shape."""
    is_tv = media_type == "tv"
    title = (d.get("title") or d.get("name") or "").strip()
    original = (d.get("original_title") or d.get("original_name") or title).strip()
    date = d.get("release_date") or d.get("first_air_date") or d.get("air_date") or ""
    year = int(date[:4]) if date[:4].isdigit() else None

    genres = [g["name"] for g in (d.get("genres") or []) if g.get("name")]
    if not genres and d.get("genre_ids"):
        gmap = {}
        data = _tmdb("/genre/movie/list" if not is_tv else "/genre/tv/list") or {"genres": []}
        gmap = {g["id"]: g["name"] for g in data.get("genres", [])}
        genres = [gmap[i] for i in d["genre_ids"] if i in gmap]

    origin = d.get("origin_country") or []
    if not origin:
        for pd in (d.get("production_countries") or []):
            code = pd.get("iso_3166_1")
            if code:
                origin.append(code)

    rating = round(float(d.get("vote_average") or 0), 1)
    vote_count = int(d.get("vote_count") or 0)
    runtime = d.get("runtime")
    if not runtime and is_tv:
        rt = d.get("episode_run_time") or []
        runtime = rt[0] if rt else None

    card: dict[str, Any] = {
        "key": f"tmdb:{media_type}:{d.get('id')}",
        "source": "tmdb",
        "media_type": media_type,
        "id": d.get("id"),
        "title": title,
        "original_title": original,
        "overview": (d.get("overview") or "").strip(),
        "tagline": d.get("tagline") or "",
        "poster_url": _img(d.get("poster_path"), POSTER_SIZE),
        "backdrop_url": _img(d.get("backdrop_path"), BACKDROP_SIZE),
        "year": year,
        "release_date": date,
        "genres": genres,
        "countries": [country_label(c) for c in origin],
        "country_codes": origin,
        "original_language": d.get("original_language") or "",
        "language": language_label(d.get("original_language")),
        "rating": rating,
        "vote_count": vote_count,
        "imdb_rating": None,
        "imdb_id": None,
        "runtime": runtime,
        "age_rating": _certification(d, media_type),
        "status": d.get("status") or "",
        "popularity": round(float(d.get("popularity") or 0), 2),
    }

    if is_tv:
        card["seasons"] = d.get("number_of_seasons")
        card["episodes"] = d.get("number_of_episodes")
        card["networks"] = [
            n.get("name") for n in ((d.get("networks") or {}).get("results") or []) if n.get("name")
        ]
        card["creators"] = [p.get("name") for p in (d.get("created_by") or []) if p.get("name")]
        card["episode_runtime"] = (d.get("episode_run_time") or [None])[0]
    else:
        card["seasons"] = None
        card["episodes"] = None
        card["networks"] = []
        card["creators"] = [p.get("name") for p in (d.get("production_countries") or []) if p.get("name")]
        card["episode_runtime"] = None

    if not detailed:
        return card

    # ---- detail-only enrichment -------------------------------------------
    card["homepage"] = d.get("homepage") or ""
    card["cast"] = _cast(d)
    card["crew"] = _crew(d)
    card["keywords"] = [k["name"] for k in ((d.get("keywords") or {}).get("results") or [])][:12]
    card["trailers"] = _trailers(d)
    card["watch"] = _watch(d.get("id"), media_type)
    card["budget"] = d.get("budget") or None
    card["revenue"] = d.get("revenue") or None

    ext = d.get("external_ids") or {}
    card["imdb_id"] = ext.get("imdb_id")
    if card["imdb_id"]:
        card["imdb_rating"] = _imdb_rating(card["imdb_id"])
    card["overview"] = card["overview"] or (d.get("overview") or "")
    return card


def _certification(d: dict, media_type: str) -> str:
    """Best-effort age certificate from the US/IN certification appends."""
    key = "release_dates" if media_type == "movie" else "content_ratings"
    results = (d.get(key) or {}).get("results") or []
    for entry in results:
        items = entry.get("iso_3166_1") or []
        if not items:
            continue
        ordered = sorted(items, key=lambda i: (i.get("priority") or 99))
        if ordered:
            return ordered[0].get("certification") or ""
    return ""


def _cast(d: dict, limit: int = 12) -> list[dict]:
    out = []
    for p in ((d.get("credits") or {}).get("cast") or [])[:limit]:
        out.append({
            "name": p.get("name"),
            "character": p.get("character") or "",
            "profile_url": _img(p.get("profile_path"), PROFILE_SIZE),
        })
    return out


def _crew(d: dict, limit: int = 10) -> list[dict]:
    keep = {"Director", "Writer", "Creator", "Producer", "Executive Producer"}
    out = []
    for p in ((d.get("credits") or {}).get("crew") or []):
        job = p.get("job") or ""
        if job in keep:
            out.append({
                "name": p.get("name"), "job": job,
                "profile_url": _img(p.get("profile_path"), PROFILE_SIZE),
            })
        if len(out) >= limit:
            break
    return out


def _trailers(d: dict, limit: int = 4) -> list[dict]:
    out = []
    for v in ((d.get("videos") or {}).get("results") or []):
        site = (v.get("site") or "").lower()
        if site != "youtube" or not v.get("key"):
            continue
        kind = (v.get("type") or "").lower()
        if kind not in ("trailer", "teaser", "clip"):
            continue
        out.append({
            "key": v["key"], "name": v.get("name") or "Trailer", "type": kind,
            "url": f"https://www.youtube.com/watch?v={v['key']}",
            "thumb": f"https://img.youtube.com/vi/{v['key']}/hqdefault.jpg",
        })
        if len(out) >= limit:
            break
    return out


def _watch(title_id, media_type: str) -> list[dict]:
    """Where to watch, per country, with provider names + direct links."""
    data = _tmdb(f"/{'movie' if media_type == 'movie' else 'tv'}/{title_id}/watch/providers")
    results = (data or {}).get("results") or {}
    buckets = ("flatrate", "free", "ads", "rent", "buy")
    labels = {
        "flatrate": "Streaming (subscription)",
        "free": "Free to watch",
        "ads": "With ads",
        "rent": "Rent",
        "buy": "Buy",
    }

    def sort_key(code: str) -> tuple:
        try:
            return (0, WATCH_PRIORITY.index(code))
        except ValueError:
            return (1, code)

    watch: list[dict] = []
    for code in sorted(results, key=sort_key):
        entry = results[code] or {}
        providers: dict[str, list[dict]] = {}
        for bucket in buckets:
            items = entry.get(bucket) or []
            if not items:
                continue
            providers[bucket] = [
                {
                    "name": p.get("provider_name") or "",
                    "logo_url": _img(p.get("logo_path"), LOGO_SIZE),
                    "priority": p.get("display_priority") or 99,
                }
                for p in sorted(items, key=lambda x: x.get("display_priority") or 99)[:6]
                if p.get("provider_name")
            ]
        if not providers:
            continue
        watch.append({
            "country_code": code,
            "country": country_label(code),
            "link": entry.get("link") or "",
            "categories": [
                {"key": bucket, "label": labels[bucket], "providers": providers[bucket]}
                for bucket in buckets if bucket in providers
            ],
        })
        if len(watch) >= 12:
            break
    return watch


def _imdb_rating(imdb_id: str) -> float | None:
    """Real IMDb score via OMDb (optional). Falls back to the TMDB rating."""
    if not _settings.omdb_api_key.strip() or not imdb_id:
        return None
    data = _get(
        "omdb", "https://www.omdbapi.com/",
        {"i": imdb_id, "apikey": _settings.omdb_api_key.strip()},
        ttl=604800,  # IMDb scores barely move — keep them a week.
    )
    raw = (data or {}).get("imdbRating")
    try:
        return round(float(raw), 1)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- fallbacks
def _jikan_cards(data: dict, limit: int = 6) -> list[dict]:
    out = []
    for a in ((data or {}).get("data") or [])[:limit]:
        out.append({
            "key": f"jikan:anime:{a.get('mal_id')}",
            "source": "jikan", "media_type": "tv", "id": a.get("mal_id"),
            "title": a.get("title") or a.get("title_english") or "",
            "original_title": a.get("title_japanese") or "",
            "overview": (a.get("synopsis") or "").strip(),
            "tagline": "",
            "poster_url": a.get("images", {}).get("jpg", {}).get("large_image_url") or "",
            "backdrop_url": "", "year": a.get("year"), "release_date": "",
            "genres": [g.get("name") for g in (a.get("genres") or []) if g.get("name")],
            "countries": ["Japan"], "country_codes": ["JP"],
            "original_language": "ja", "language": "Japanese",
            "rating": round(float(a.get("score") or 0), 1),
            "vote_count": int((a.get("scored_by") or 0)),
            "imdb_rating": None, "imdb_id": None,
            "runtime": a.get("duration"), "age_rating": "",
            "status": a.get("status") or "", "popularity": 0.0,
            "seasons": a.get("episodes"), "episodes": a.get("episodes"),
            "networks": [], "creators": _jikan_staff(a),
            "episode_runtime": a.get("duration"),
            "studios": [s.get("name") for s in (a.get("studios") or []) if s.get("name")],
        })
    return out


def _jikan_staff(a: dict) -> list[str]:
    for key in ("directors", "producers"):
        people = a.get(key) or []
        names = [p.get("person", {}).get("name") for p in people if p.get("person")]
        if names:
            return names[:4]
    return []


def _tvmaze_genres(genres) -> list[str]:
    """Normalise TVMaze genres, which arrive in two different shapes.

    ``/shows/{id}`` returns objects (``{"id": 1, "name": "Drama"}``) while
    ``/search/shows`` returns bare strings (``"Drama"``). The card builder
    assumed the first shape, so every TVMaze search died on
    ``'str' object has no attribute 'get'`` and silently degraded to whatever
    weaker source was left.
    """
    out: list[str] = []
    for g in genres or []:
        name = g.get("name") if isinstance(g, dict) else g
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _tvmaze_cards(items: list[dict], limit: int = 6) -> list[dict]:
    out = []
    for w in items[:limit]:
        show = w.get("show") or {}
        if not show.get("id"):
            continue
        year = None
        if show.get("premiered"):
            year = int(str(show["premiered"])[:4])
        out.append({
            "key": f"tvmaze:show:{show.get('id')}",
            "source": "tvmaze", "media_type": "tv", "id": show.get("id"),
            "title": show.get("name") or "",
            "original_title": show.get("name") or "",
            "overview": (show.get("summary") or "").replace("<p>", "").replace("</p>", "").strip(),
            "tagline": show.get("summary") or "",
            "poster_url": show.get("image", {}).get("original") or "",
            "backdrop_url": show.get("image", {}).get("original") or "",
            "year": year, "release_date": show.get("premiered") or "",
            "genres": _tvmaze_genres(show.get("genres")),
            "countries": [country_label(show.get("country", {}).get("code"))],
            "country_codes": [show.get("country", {}).get("code") or ""],
            "original_language": "en", "language": "English",
            "rating": round(float((show.get("rating") or {}).get("average") or 0), 1),
            "vote_count": 0, "imdb_rating": None, "imdb_id": None,
            "runtime": show.get("averageRuntime"), "age_rating": "",
            "status": show.get("status") or "", "popularity": 0.0,
            "seasons": show.get("seasons"), "episodes": None,
            "networks": [show.get("network", {}).get("name")] if show.get("network") else [],
            "creators": [], "episode_runtime": show.get("averageRuntime"),
        })
    return out


# ------------------------------------------------------------------ search
SORT_OPTIONS = {
    "popular": "popularity.desc",
    "rating": "vote_average.desc",
    "newest": None,      # resolved per media type below
    "oldest": None,
    "revenue": "revenue.desc",
}


def _sort_param(sort: str | None, media_type: str) -> str:
    s = (sort or "popular").lower()
    if s in ("newest", "latest"):
        return "primary_release_date.desc" if media_type == "movie" else "first_air_date.desc"
    if s == "oldest":
        return "primary_release_date.asc" if media_type == "movie" else "first_air_date.asc"
    if s == "revenue" and media_type == "tv":
        return "popularity.desc"
    return SORT_OPTIONS.get(s) or "popularity.desc"


def find_titles(
    query: str = "",
    media_type: str = "any",
    genres: Optional[list[str]] = None,
    country: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    language: Optional[str] = None,
    min_rating: Optional[float] = None,
    sort: Optional[str] = None,
    limit: int = 6,
) -> list[dict]:
    """Search or discover titles. This is the only way titles enter the chat."""
    media_type = (media_type or "any").lower()
    if media_type not in ("movie", "tv", "any"):
        media_type = "any"
    limit = max(1, min(int(limit or 6), 20))

    kinds = ["movie", "tv"] if media_type == "any" else [media_type]
    results: list[dict] = []

    for kind in kinds:
        gids = _resolve_genres(genres, kind)
        cc = _resolve_country(country)
        lang = _resolve_language(language)

        # An animation pick on the TV side is much sharper with the anime keyword.
        if kind == "tv" and ANIMATION_GENRE_ID in gids and not query:
            gids = [g for g in gids if g != ANIMATION_GENRE_ID]

        has_filters = any([gids, cc, year_from, year_to, lang, min_rating])

        if has_filters and not query:
            data = _discover(kind, gids, cc, year_from, year_to, lang, min_rating, sort, limit)
            results.extend(_from_discover(data, kind))
        elif query:
            # Title search. Watchmode indexes live-action properly, so try it
            # first; TMDB is the fallback when a key is configured.
            if watchmode_configured():
                results.extend(_watchmode_search(query, kind, limit))
            if not results:
                data = _tmdb(f"/search/{kind}", {
                    "query": query, "include_adult": "false",
                    "page": 1, "language": "en-US",
                })
                results.extend([_tmdb_card(r, kind) for r in ((data or {}).get("results") or [])])
        else:
            data = _discover(kind, gids, cc, year_from, year_to, lang, min_rating, sort, limit)
            results.extend(_from_discover(data, kind))

    results = _post_filter(
        results, genres=genres, country=country, year_from=year_from,
        year_to=year_to, language=language, min_rating=min_rating,
    )

    discovery_unavailable = False
    if not results:
        # Did the user name a specific title, or describe what they want?
        # A named title that we cannot find must come back empty — answering
        # "tell me about Obsession" with The Odyssey is worse than saying no.
        title_shaped = bool(query) and not any(
            [genres, country, year_from, year_to, language, min_rating]
        )

        if title_shaped:
            # Last chance: a looser search, still scoped to the named title.
            results = _fallback_search(query, media_type, limit)
        else:
            # Discovery cascade. An explicit anime request goes to AniList
            # first; everything else prefers Watchmode, which understands
            # production filters and also supplies "where to watch".
            explicitly_anime = _looks_anime(genres, country, query)

            if explicitly_anime and _settings.enable_anilist_fallback:
                results = _anilist_discover(
                    genres=_anilist_genres(genres), year_from=year_from, year_to=year_to,
                    country=country, sort=sort, limit=limit,
                )

            if not results and watchmode_configured():
                results = _watchmode_discover(
                    genres=genres, country=country, year_from=year_from, year_to=year_to,
                    language=language, min_rating=min_rating, sort=sort, query=query,
                    media_type=media_type, limit=limit,
                )

            if not results and _settings.enable_anilist_fallback:
                results = _anilist_discover(
                    genres=_anilist_genres(genres), year_from=year_from, year_to=year_to,
                    country=country, sort=sort, limit=limit,
                )

            if not results:
                if query:
                    results = _fallback_search(query, media_type, limit)
                else:
                    # Nothing to go on at all — offer something curated
                    # rather than an empty bubble.
                    discovery_unavailable = (
                        not tmdb_configured()
                        and bool(genres or country or year_from
                                 or year_to or language or min_rating)
                    )
                    results = surprise_titles(limit)

    # De-duplicate by title+year, then keep the best rated.
    seen: set[str] = set()
    unique: list[dict] = []
    # A search for one film returns the same name across many years and
    # remakes ("Obsession" 2026 / 1976 / 1943 / …). Showing five cards that
    # all read "Obsession" looks broken, so keep only the best per name.
    name_seen: set[str] = set()
    for r in sorted(results, key=lambda x: (-(x.get("rating") or 0), -(x.get("popularity") or 0))):
        sig = f"{r['title'].lower()}|{r.get('year')}"
        if sig in seen:
            continue
        name = re.sub(r"[^a-z0-9]+", "", r["title"].lower())
        if name in name_seen:
            continue
        seen.add(sig)
        name_seen.add(name)
        unique.append(r)
    chosen = unique[:limit]

    # Every card carries a real IMDb score when we can get one.
    if _settings.omdb_api_key.strip():
        for card in chosen:
            if card.get("source") == "omdb":
                continue
            try:
                _enrich_with_imdb(card)
            except Exception:
                logger.debug("imdb enrichment failed for %s", card.get("key"))

    # Watchmode's own discovery already hydrates its picks (the details call is
    # where the score and the streaming links live). Search results don't, so
    # fill those in here.
    if watchmode_configured():
        pending = [c for c in chosen
                   if c.get("source") == "watchmode" and not c.get("hydrated")]
        if pending:
            def hydrate(card: dict) -> tuple[dict, dict | None]:
                try:
                    return card, _watchmode_details(str(card.get("id")))
                except Exception:
                    return card, None

            with ThreadPoolExecutor(max_workers=6) as ex:
                for card, full in ex.map(hydrate, pending):
                    if not full:
                        continue
                    merged = dict(full)
                    merged.update({k: v for k, v in card.items()
                                   if v not in (None, "", [], 0, 0.0)})
                    merged["hydrated"] = True
                    merged["overview"] = merged.get("overview") or card.get("overview")
                    card.clear()
                    card.update(merged)

    # AniList has no IMDb score, but its 0-10 rating is the best signal we have.
    for card in chosen:
        if card.get("source") == "anilist" and not card.get("rating"):
            full = _anilist_details(str(card.get("id")))
            if full and full.get("rating"):
                card["rating"] = full["rating"]

    for card in chosen:
        card["discovery_unavailable"] = bool(discovery_unavailable)
    return chosen


def _discover(kind, gids, cc, year_from, year_to, lang, min_rating, sort, limit):
    date_key = "primary_release_date" if kind == "movie" else "first_air_date"
    params: dict[str, Any] = {
        "page": 1, "language": "en-US", "sort_by": _sort_param(sort, kind),
        "include_adult": "false", "vote_count.gte": 50 if min_rating else 0,
    }
    if gids:
        params["with_genres"] = ",".join(str(g) for g in gids)
    if cc:
        params["with_origin_country"] = cc
    if lang:
        params["with_original_language"] = lang
    if year_from:
        params[f"{date_key}.gte"] = f"{int(year_from)}-01-01"
    if year_to:
        params[f"{date_key}.lte"] = f"{int(year_to)}-12-31"
    if min_rating:
        params["vote_average.gte"] = float(min_rating)
    return _tmdb(f"/discover/{kind}", params)


def _from_discover(data: dict, kind: str) -> list[dict]:
    return [_tmdb_card(r, kind) for r in ((data or {}).get("results") or [])]


def _post_filter(items, genres=None, country=None, year_from=None, year_to=None,
                 language=None, min_rating=None) -> list[dict]:
    """Apply filters again client-side: TMDB's genre/country OR-combines."""
    out = items
    want_countries = {_resolve_country(country)} if country else set()
    want_langs = {_resolve_language(language)} if language else set()
    want_genres = {
        re.sub(r"[^a-z0-9]+", "", g.lower()) for g in (genres or []) if g
    }
    want_genres = {g.replace("sciencefiction", "scifi") for g in want_genres}

    def keep(t: dict) -> bool:
        if want_countries:
            codes = {c.upper() for c in (t.get("country_codes") or []) if c}
            if not (codes & want_countries):
                # Some records only carry the label, not the code.
                labels = {(c or "").lower() for c in (t.get("countries") or [])}
                if not labels & {country_label(c).lower() for c in want_countries}:
                    return False
        if want_langs and (t.get("original_language") or "") not in want_langs:
            return False
        if year_from and (t.get("year") or 0) < int(year_from):
            return False
        if year_to and (t.get("year") or 9999) > int(year_to):
            return False
        if min_rating and (t.get("rating") or 0) < float(min_rating):
            return False
        if want_genres:
            have = {
                re.sub(r"[^a-z0-9]+", "", g.lower()).replace("sciencefiction", "scifi")
                for g in (t.get("genres") or [])
            }
            if not (have & want_genres):
                return False
        return True

    return [t for t in out if keep(t)]


# ------------------------------------------------------------------- OMDb
# A second source of truth (no discovery API, but real IMDb ratings and solid
# title search). Used as a fallback when TMDB has no key / no match, and to
# attach a genuine IMDb score to every card we show.


def _omdb_cards(data: dict, limit: int = 6) -> list[dict]:
    out = []
    for r in (data.get("Search") or [])[:limit]:
        imdb_id = r.get("imdbID")
        if not imdb_id:
            continue
        year = None
        y = str(r.get("Year") or "")
        if y.isdigit():
            year = int(y)
        kind = "tv" if r.get("Type") == "series" else "movie"
        out.append(_omdb_stub(r, imdb_id, kind, year))
    return out


def _omdb_stub(r: dict, imdb_id: str, kind: str, year: int | None) -> dict:
    poster = r.get("Poster") or ""
    return {
        "key": f"omdb:{kind}:{imdb_id}",
        "source": "omdb",
        "media_type": kind,
        "id": imdb_id,
        "title": r.get("Title") or "",
        "original_title": r.get("Title") or "",
        "overview": "",
        "tagline": "",
        "poster_url": poster if poster and poster != "N/A" else "",
        "backdrop_url": poster if poster and poster != "N/A" else "",
        "year": year,
        "release_date": r.get("Year") or "",
        "genres": [],
        "countries": [],
        "country_codes": [],
        "original_language": "en",
        "language": "",
        "rating": 0.0,
        "vote_count": 0,
        "imdb_rating": None,
        "imdb_id": imdb_id,
        "runtime": None,
        "age_rating": "",
        "status": "",
        "popularity": 0.0,
        "seasons": r.get("totalSeasons"),
        "episodes": r.get("totalEpisodes"),
        "networks": [],
        "creators": [],
        "episode_runtime": None,
        "type_label": r.get("Type") or "",
    }


def _omdb_detail(data: dict) -> dict:
    """Full OMDb record normalised into our card shape (with a real IMDb score)."""
    imdb_id = data.get("imdbID") or ""
    kind = "tv" if data.get("Type") == "series" else "movie"
    released = data.get("Year") or ""
    year = int(released) if str(released).isdigit() else None
    poster = data.get("Poster") or ""
    poster = "" if poster == "N/A" else poster

    def score(v) -> float | None:
        try:
            return round(float(v), 1)
        except (TypeError, ValueError):
            return None

    runtime_raw = data.get("Runtime") or ""
    runtime = None
    if runtime_raw and runtime_raw != "N/A":
        digits = "".join(ch for ch in runtime_raw if ch.isdigit())
        runtime = int(digits) if digits else None

    countries = [c.strip() for c in (data.get("Country") or "").split(",") if c.strip()]
    codes: list[str] = []
    for c in countries:
        m = re.search(r"\b([A-Z]{2})\b", c)
        if m:
            codes.append(m.group(1))

    cast = []
    for a in (data.get("Actors") or "").split(","):
        a = a.strip()
        if a:
            cast.append({"name": a, "character": "", "profile_url": ""})

    writers = [w.strip() for w in (data.get("Writer") or "").split(",") if w.strip()]
    crew = [{"name": w, "job": "Writer", "profile_url": ""} for w in writers[:6]]
    if data.get("Director"):
        crew.insert(0, {"name": data["Director"], "job": "Director", "profile_url": ""})

    return {
        "key": f"omdb:{kind}:{imdb_id}",
        "source": "omdb",
        "media_type": kind,
        "id": imdb_id,
        "title": data.get("Title") or "",
        "original_title": data.get("Title") or "",
        "overview": data.get("Plot") or "",
        "tagline": "",
        "poster_url": poster,
        "backdrop_url": poster,
        "year": year,
        "release_date": data.get("Released") if data.get("Released") != "N/A" else released,
        "genres": [g.strip() for g in (data.get("Genre") or "").split(",") if g.strip()],
        "countries": countries,
        "country_codes": codes,
        "original_language": "en",
        "language": "",
        # OMDb's score IS the IMDb score — exactly what the UI wants to show.
        "rating": score(data.get("imdbRating")) or 0.0,
        "vote_count": int(str(data.get("imdbVotes") or "0").replace(",", "") or 0),
        "imdb_rating": score(data.get("imdbRating")),
        "imdb_id": imdb_id,
        "runtime": runtime,
        "age_rating": data.get("Rated") if data.get("Rated") != "N/A" else "",
        "status": "",
        "popularity": 0.0,
        "seasons": int(data["totalSeasons"]) if str(data.get("totalSeasons", "")).isdigit() else None,
        "episodes": int(data["totalEpisodes"]) if str(data.get("totalEpisodes", "")).isdigit() else None,
        "networks": [],
        "creators": writers[:3],
        "episode_runtime": None,
        "type_label": data.get("Type") or "",
        "cast": cast[:12],
        "crew": crew,
        "keywords": [],
        "trailers": [],
        "watch": [],
        "homepage": "",
        "awards": data.get("Awards") if data.get("Awards") != "N/A" else "",
        "box_office": data.get("BoxOffice") if data.get("BoxOffice") != "N/A" else "",
    }


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _title_score(candidate: str, title: str) -> int:
    """How well an OMDb result actually matches what we searched for."""
    c, t = _norm_title(candidate), _norm_title(title)
    if not c or not t:
        return 0
    if c == t:
        return 100
    if c in t:
        return 60
    if t in c:
        return 30
    return 0


def _omdb_search(query: str, kind: str, limit: int) -> list[dict]:
    data = _get("omdb", "http://www.omdbapi.com/", {
        "apikey": _settings.omdb_api_key.strip(),
        "s": query, "type": "movie" if kind == "movie" else "series", "page": 1,
    })
    if not data or data.get("Response") != "True":
        return []
    cards = _omdb_cards(data, limit)
    # OMDb returns fuzzy matches; surface the one that really is the title.
    cards.sort(key=lambda c: -_title_score(query, c.get("title", "")))
    return cards


def _omdb_by_id(imdb_id: str) -> dict | None:
    data = _get("omdb", "http://www.omdbapi.com/", {
        "apikey": _settings.omdb_api_key.strip(), "i": imdb_id, "plot": "short",
    }, ttl=604800)
    if not data or data.get("Response") != "True":
        return None
    return _omdb_detail(data)


def _enrich_with_imdb(card: dict) -> dict:
    """Attach a real IMDb rating to a TMDB/other card when we know its IMDb id."""
    if not _settings.omdb_api_key.strip() or card.get("imdb_rating"):
        return card
    imdb_id = card.get("imdb_id")
    if not imdb_id:
        # Try to find the title on OMDb and reuse the IMDb id it finds.
        found = _omdb_search(card.get("title", ""), card.get("media_type", "movie"), 1)
        imdb_id = found[0]["imdb_id"] if found else None
        if imdb_id:
            card["imdb_id"] = imdb_id
    if not imdb_id:
        return card
    full = _omdb_by_id(imdb_id)
    if full and full.get("imdb_rating"):
        card["imdb_rating"] = full["imdb_rating"]
        card["imdb_id"] = imdb_id
    return card


_FILLER_PREFIX = re.compile(
    r"^(please\s+)?(can you\s+|could you\s+)?"
    r"(tell me about|tell me|what is|what's|whats|who is|who's|give me info on|"
    r"give me|find me|show me|look up|summari[sz]e|summari[sz]ation of|describe|"
    r"explain|details on|info on|information about|review of|plots of|"
    r"recommend|suggest|give)\s+",
    re.I,
)
_FILLER_WORDS = {
    "the", "a", "an", "movie", "film", "films", "series", "show", "shows",
    "about", "me", "of", "please", "and", "some", "any", "for", "to", "with",
    "is", "are", "was", "tell", "what", "who", "give", "find", "show", "best",
    "good", "top", "list", "like", "similar", "must", "watch",
}


def query_candidates(q: str) -> list[str]:
    """Progressively shorter ways of reading `q` as a *title*.

    OMDb matches titles almost exactly, so "tell me about the movie Inception"
    must be reduced to "Inception" before it will match. Returns candidates
    ordered from the most faithful to the most aggressive.
    """
    q = (q or "").strip()
    if not q:
        return []

    cands: list[str] = []
    stripped = _FILLER_PREFIX.sub("", q).strip()
    # When the query is clearly an instruction ("tell me about X"), the raw
    # string is not a title and only produces fuzzy noise — skip it.
    if not stripped or stripped == q:
        cands.append(q)
    if stripped and stripped != q:
        cands.append(stripped)

    base = stripped or q
    tokens = [t for t in re.findall(r"[\w'&:.!?-]+", base) if t.lower() not in _FILLER_WORDS]
    if tokens:
        joined = " ".join(tokens)
        if joined not in cands:
            cands.append(joined)
        # Contiguous windows, longest first — a title usually survives intact.
        n = len(tokens)
        for size in range(n, 0, -1):
            for i in range(0, n - size + 1):
                window = " ".join(tokens[i:i + size])
                if window and window.lower() not in {c.lower() for c in cands}:
                    cands.append(window)

    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(c.strip())
    return out[:12]


# A curated seed list for "surprise me" / browse-with-no-filters. Without a
# TMDB key there is no discovery endpoint anywhere, so these well-known,
# critically-liked titles (looked up by IMDb id) are what the bot can offer.
# Ids are stable, so this never goes stale or invents a title.
SURPRISE_SEED = [
    ("tt0111161", "movie"), ("tt0068646", "movie"), ("tt0133093", "movie"),
    ("tt1375666", "movie"), ("tt0816692", "movie"), ("tt0107290", "movie"),
    ("tt0114369", "movie"), ("tt0047478", "movie"), ("tt0167260", "movie"),
    ("tt0120737", "movie"), ("tt6751668", "movie"), ("tt15398776", "movie"),
    ("tt1160419", "movie"), ("tt5052448", "movie"), ("tt0910970", "movie"),
    ("tt0903747", "tv"), ("tt2861424", "tv"), ("tt0903747", "tv"),
    ("tt1475582", "tv"), ("tt2085059", "tv"), ("tt4574334", "tv"),
    ("tt1190634", "tv"), ("tt10919420", "tv"), ("tt2442560", "tv"),
    ("tt0386676", "tv"), ("tt4786824", "tv"), ("tt0898266", "tv"),
    ("tt0306414", "tv"), ("tt0417299", "tv"),
]


def surprise_titles(limit: int = 5, offset: int = 0) -> list[dict]:
    """A rotating handful of acclaimed picks for 'surprise me'.

    Prefers TMDB trending when a key is configured; otherwise resolves the
    curated seed through OMDb, so the button always has something to show.
    """
    if tmdb_configured():
        trending_titles = _tmdb("/trending/all/week", {"language": "en-US"})
        cards = [
            _tmdb_card(r, "tv" if r.get("media_type") == "tv" else "movie")
            for r in ((trending_titles or {}).get("results") or [])
        ]
        if cards:
            for c in cards:
                _enrich_with_imdb(c)
            return cards[:limit]

    if not _settings.omdb_api_key.strip():
        return []

    seen: set[str] = set()
    out: list[dict] = []
    for imdb_id, _kind in SURPRISE_SEED[offset:] + SURPRISE_SEED[:offset]:
        if len(out) >= limit:
            break
        if imdb_id in seen:
            continue
        seen.add(imdb_id)
        card = _omdb_by_id(imdb_id)
        if card:
            out.append(card)
    return out


# AniList uses its own genre vocabulary, so map ours onto it.
_ANILIST_GENRES = {
    "action": "Action", "adventure": "Adventure", "comedy": "Comedy",
    "drama": "Drama", "fantasy": "Fantasy", "horror": "Horror",
    "mystery": "Mystery", "romance": "Romance", "scifi": "Sci-Fi",
    "thriller": "Thriller", "sports": "Sports", "supernatural": "Supernatural",
    "sliceoflife": "Slice of Life", "psychological": "Psychological",
    "school": "School", "military": "Military", "historical": "Historical",
    "music": "Music", "mecha": "Mecha", "isekai": "Isekai",
}


def _anilist_genres(genres) -> list[str] | None:
    out = []
    for g in genres or []:
        key = re.sub(r"[^a-z]", "", str(g).lower())
        mapped = _ANILIST_GENRES.get(key)
        if mapped and mapped not in out:
            out.append(mapped)
    return out or None


def _looks_anime(genres=None, country=None, query: str = "") -> bool:
    """True when the request is plausibly about anime/animation."""
    text = " ".join([query or "", " ".join(genres or []), country or ""]).lower()
    anime_words = ("anime", "manga", "ghibli", "animation", "animated", "cartoon",
                   "japanese")
    return any(w in text for w in anime_words)


# ============================================================== Watchmode
# The one reachable provider that does live-action discovery AND streaming
# availability. Free plan: web links work, deep links do not; only some regions
# are enabled, which we discover at runtime rather than assuming.

_WATCHMODE_BASE = "https://api.watchmode.com/v1"
_WM_TYPES = {
    "movie": "movie", "tv_series": "tv", "tv_movie": "tv",
    "tv_miniseries": "tv", "tv_special": "tv", "short_film": "movie",
}
_WM_SORTS = {
    "popular": "popularity_desc", "rating": "popularity_desc",
    "newest": "release_date_desc", "oldest": "release_date_asc",
    "relevance": "relevance_desc",
}
_WM_SOURCE_TYPES = {
    "sub": "Streaming (subscription)", "free": "Free to watch",
    "ads": "With ads", "rent": "Rent", "buy": "Buy",
}

# Regions this plan can actually filter by; filled in on first use.
_wm_regions: set[str] | None = None


def watchmode_configured() -> bool:
    return bool(_settings.watchmode_api_key.strip())


def probe_sources() -> dict[str, dict]:
    """Live reachability check for every catalog source.

    "Configured" only proves a key is present. This proves the provider
    actually answers from *this* machine, which is how we tell a bad key or a
    rate limit apart from a genuine miss — a distinction that is invisible
    from the chat UI and was the reason a dead Watchmode key made the bot
    claim real films did not exist.
    """
    out: dict[str, dict] = {}

    def timed(name: str, fn) -> None:
        started = time.monotonic()
        try:
            count = fn()
            out[name] = {
                "ok": True,
                "results": count,
                "ms": round((time.monotonic() - started) * 1000),
            }
        except Exception as exc:  # a probe must never raise
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:160]}

    if watchmode_configured():
        timed("watchmode", lambda: len(_watchmode_search("Shawshank", "movie", 3)))
    if _settings.omdb_api_key.strip():
        timed("omdb", lambda: len(_omdb_search("Shawshank", "movie", 3)))
    if _settings.enable_anilist_fallback:
        timed("anilist", lambda: len(_anilist_search("Cowboy Bebop", 3)))
    if _settings.enable_jikan_fallback:
        timed("jikan", lambda: len(_jikan_cards(
            _jikan("/anime", {"q": "cowboy bebop", "limit": 3}) or [], 3)))
    if _settings.enable_tvmaze_fallback:
        timed("tvmaze", lambda: len(_tvmaze_cards(
            _tvmaze("/search/shows", {"q": "dark"}) or [], 3)))
    return out


def _watchmode(path: str, params: dict | None = None,
               ttl: int | None = None) -> Any:
    """GET a Watchmode endpoint. Returns None on failure (never raises)."""
    if not watchmode_configured():
        return None
    query = dict(params or {})
    query["apiKey"] = _settings.watchmode_api_key.strip()
    data = _get("watchmode", f"{_WATCHMODE_BASE}{path}", query, ttl)
    if isinstance(data, dict) and data.get("success") is False:
        msg = data.get("statusMessage") or ""
        if "not enabled for your current plan" in msg:
            # Remember which regions this plan supports so we stop retrying.
            country = None
            m = re.search(r"^\s*([A-Z]{2})\s", msg)
            if m:
                country = m.group(1)
            if country:
                globals().setdefault("_wm_blocked", set()).add(country)
            logger.info("watchmode: region %s not on this plan", country)
        metrics.upstream_error("watchmode")
        return None
    return data


def _watchmode_regions() -> set[str]:
    """Regions this plan can filter by (cached 24h)."""
    global _wm_regions
    if _wm_regions is not None:
        return _wm_regions
    data = _watchmode("/regions")
    if isinstance(data, list):
        _wm_regions = {r["country"] for r in data if r.get("plan_enabled")}
    else:
        _wm_regions = set()
    return _wm_regions


def _watchmode_genres() -> dict[str, int]:
    """Normalised genre name -> Watchmode id."""
    cached = cache.get_json("watchmode:genres")
    if cached:
        return cached
    data = _watchmode("/genres")
    if not isinstance(data, list):
        return {}
    out = {}
    for g in data:
        name = g.get("name")
        if name:
            out[re.sub(r"[^a-z0-9]+", "", name.lower())] = g["id"]
    cache.set_json("watchmode:genres", out, 604800)
    return out


def _wm_genre_ids(genres) -> list[int]:
    gmap = _watchmode_genres()
    ids: list[int] = []
    for g in genres or []:
        key = re.sub(r"[^a-z0-9]+", "", str(g).lower())
        gid = gmap.get(key)
        if gid is None:
            for name, candidate in gmap.items():
                if key and (key in name or name in key):
                    gid = candidate
                    break
        if gid and gid not in ids:
            ids.append(gid)
    return ids


def _wm_card(t: dict, detail: bool = False, score_only: bool = False) -> dict:
    """Watchmode title -> our unified card shape.

    ``score_only`` is the middle tier used while ranking candidates: it costs
    one ``/details`` call and yields the plot, poster and IMDb score, but skips
    the cast, sources and "similar" calls that the expanded card needs.
    """
    wm_type = t.get("type") or "movie"
    media_type = _WM_TYPES.get(wm_type, "movie")
    year = t.get("year")
    imdb_id = t.get("imdb_id") or ""
    card: dict[str, Any] = {
        "key": f"watchmode:{media_type}:{t.get('id')}",
        "source": "watchmode",
        "media_type": media_type,
        "id": t.get("id"),
        "title": t.get("title") or "",
        "original_title": t.get("original_title") or t.get("title") or "",
        "overview": (t.get("plot_overview") or "").strip(),
        "tagline": "",
        "poster_url": t.get("posterLarge") or t.get("posterMedium") or t.get("poster") or "",
        "backdrop_url": t.get("backdrop") or "",
        "year": year,
        "release_date": t.get("release_date") or "",
        "genres": t.get("genre_names") or [],
        "countries": [],
        "country_codes": [],
        "original_language": t.get("original_language") or "",
        "language": language_label(t.get("original_language")),
        "rating": round(float(t.get("user_rating") or 0), 1),
        "vote_count": 0,
        # Watchmode's user_rating is the IMDb score, which is what we display.
        "imdb_rating": round(float(t.get("user_rating") or 0), 1) or None,
        "imdb_id": imdb_id or None,
        "runtime": t.get("runtime_minutes"),
        "age_rating": t.get("us_rating") or "",
        "status": "",
        "popularity": round(float(t.get("popularity_percentile") or 0), 2),
        "critic_score": t.get("critic_score"),
        "seasons": t.get("seasons"),
        "episodes": t.get("episodes"),
        "networks": t.get("network_names") or [],
        "creators": [],
        "episode_runtime": None,
        "watch": [],
    }
    if not detail:
        return card

    card["homepage"] = ""
    card["keywords"] = []
    card["review_summary"] = t.get("review_summary") or ""
    card["trailers"] = (
        [{"key": "", "name": "Trailer", "type": "trailer",
          "url": t["trailer"], "thumb": t.get("trailer_thumbnail") or ""}]
        if t.get("trailer") else []
    )
    card["countries"] = [country_label(c) for c in _watchmode_countries(t.get("id"))]
    card["country_codes"] = _watchmode_countries(t.get("id"))
    if score_only:
        card["cast"] = []
        card["crew"] = []
        card["watch"] = []
        card["similar"] = []
        return card

    cast = _watchmode(f"/title/{t.get('id')}/cast-crew", ttl=604800)
    people = cast if isinstance(cast, list) else []
    card["cast"] = [
        {"name": p.get("full_name"), "character": p.get("role") or "",
         "profile_url": ""}
        for p in people if p.get("type") == "Cast"
    ][:12]
    card["crew"] = [
        {"name": p.get("full_name"), "job": p.get("role") or "", "profile_url": ""}
        for p in people if p.get("type") == "Crew"
    ][:8]

    card["watch"] = _watchmode_sources(t.get("id"))
    sims = _watchmode(f"/title/{t.get('id')}/details")
    card["similar"] = []
    if isinstance(sims, dict) and sims.get("similar_titles"):
        ids = [str(s) for s in sims["similar_titles"][:8]]
        if ids:
            listed = _watchmode(f"/list-titles?limit=20", {"types": "movie,tv_series"})
            found = {str(x["id"]): x for x in (listed or {}).get("titles", [])}
            card["similar"] = [
                _wm_card(found[i]) for i in ids if i in found
            ][:8]
    return card


def _watchmode_countries(title_id) -> list[str]:
    if not title_id:
        return []
    cached = cache.get_json(f"watchmode:countries:{title_id}")
    if cached is not None:
        return cached
    # Watchmode has no per-title country field; infer from origin where we can.
    cache.set_json(f"watchmode:countries:{title_id}", [], 604800)
    return []


def _watchmode_sources(title_id, region: str | None = None) -> list[dict]:
    """Where to watch, grouped by country -> subscription/free/rent/buy."""
    if not title_id:
        return []
    enabled = _watchmode_regions()
    region = (region or (next(iter(enabled)) if enabled else "IN")).upper()

    raw = _watchmode(f"/title/{title_id}/sources", {"regions": region}, ttl=21600)
    if not isinstance(raw, list):
        return []

    buckets: dict[str, list[dict]] = {}
    for s in raw:
        # Deep links are a paid-plan feature; only web links work on free.
        url = s.get("web_url")
        stype = (s.get("type") or "sub").lower()
        if not url or stype not in _WM_SOURCE_TYPES:
            continue
        buckets.setdefault(stype, []).append({
            "name": s.get("name") or "Provider",
            "logo_url": f"https://cdn.watchmode.com/provider_logos/{s.get('source_id')}.png",
            "priority": 0,
            "url": url,
            "price": s.get("price"),
            "format": s.get("format"),
        })

    if not buckets:
        return []

    seen = set()
    categories = []
    for key, label in _WM_SOURCE_TYPES.items():
        items = buckets.get(key) or []
        # Collapse the same provider offered for several seasons/formats.
        unique = []
        for it in items:
            sig = (it["name"], key)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(it)
        if unique:
            categories.append({"key": key, "label": label, "providers": unique})

    return [{
        "country_code": region,
        "country": country_label(region),
        "link": "",
        "categories": categories,
    }]


def _watchmode_list(params: dict, limit: int) -> list[dict]:
    params = {k: v for k, v in params.items() if v not in (None, "", [])}
    params["limit"] = max(1, min(limit, 50))
    data = _watchmode("/list-titles", params)
    titles = (data or {}).get("titles") if isinstance(data, dict) else None
    if not titles:
        return []
    return [_wm_card(t) for t in titles]


def _wm_quality(rating: float, popularity: float) -> float:
    """Rank a candidate: quality first, familiarity as a tiebreak.

    Squaring the rating makes a strong opinion dominate, and the popularity
    factor scales it down rather than merely adding to it — so a 9.3 nobody has
    heard of still loses to a well-known 8.0.
    """
    return (rating ** 2) * (0.5 + 0.5 * min(max(popularity, 0.0), 100.0) / 100.0)


def _wm_is_released(card: dict) -> bool:
    """True once the title has actually reached audiences."""
    today = date.today().isoformat()
    released_on = str(card.get("release_date") or "")[:10]
    if released_on:
        return released_on < today
    year = card.get("year")
    # No date at all but an old year is still fine; neither date nor year is not.
    return bool(year) and int(year) <= date.today().year


def _wm_recency(card: dict) -> float:
    """1.0 for this year, decaying to 0 over fifteen."""
    released_on = str(card.get("release_date") or "")[:4]
    if not released_on.isdigit():
        return 0.5
    age = date.today().year - int(released_on)
    return max(0.0, 1.0 - max(age, 0) / 15.0)


# The user has to explicitly ask for something unreleased to see one.
_WM_UNRELEASED_HINT = re.compile(
    r"\b(upcoming|unreleased|not\s+(?:released|out)\s+yet|coming\s+soon|"
    r"premiere|in\s+theaters?\s+(?:soon|next)|next\s+(?:year|month|week))\b",
    re.I,
)


def _watchmode_discover(genres=None, country=None, year_from=None, year_to=None,
                        language=None, min_rating=None, sort=None, query=None,
                        media_type="any", limit: int = 6) -> list[dict]:
    """Discover titles, then re-rank them by quality rather than raw popularity.

    Watchmode's ``/list-titles`` sorts by a global popularity percentile and
    returns no scores, so taking its top N verbatim surfaces 1998 crime shows
    and unreleased 2026 slates. We over-fetch, hydrate (which is where the IMDb
    score and the streaming links live anyway) and rank on the result.
    """
    types = {"movie": "movie", "tv": "tv_series", "any": "movie,tv_series"}[media_type or "any"]
    want_recent = sort and sort.lower() in ("newest", "recent")
    want_unreleased = bool(query and _WM_UNRELEASED_HINT.search(query))
    # A default floor keeps the candidate pool from filling with obscure dreck;
    # an explicit request always wins. Only a deliberate "what's coming out"
    # question is allowed to surface unreleased, under-reviewed titles.
    floor = min_rating or (0.0 if want_unreleased else 7.0)
    params: dict[str, Any] = {
        "types": types,
        "genres": ",".join(str(g) for g in _wm_genre_ids(genres)) or None,
        "release_date_start": int(year_from) * 10000 + 101 if year_from else None,
        "release_date_end": int(year_to) * 10000 + 1231 if year_to else None,
        "user_rating_low": floor,
        "sort_by": _WM_SORTS.get((sort or "popular").lower(), "popularity_desc"),
    }
    lang = _resolve_language(language)
    if lang:
        params["languages"] = lang

    # Region filtering only works where the plan enables it.
    cc = _resolve_country(country)
    if cc and cc in _watchmode_regions():
        params["regions"] = cc
    elif cc:
        logger.debug("watchmode: region %s not on this plan, skipping filter", cc)

    # Over-fetch so re-ranking has something to work with, but keep the pool
    # tight — every extra candidate costs one details call.
    pool = _watchmode_list(params, min(limit * 3, 20))
    if not pool:
        return []

    def hydrate(card: dict) -> dict | None:
        try:
            raw = _watchmode(f"/title/{card.get('id')}/details", ttl=21600)
        except Exception:
            return None
        if not isinstance(raw, dict) or not raw.get("title"):
            return None
        merged = _wm_card(raw, detail=True, score_only=True)
        merged.update({k: v for k, v in card.items() if v not in (None, "", [], 0, 0.0)})
        merged["hydrated"] = True
        merged["overview"] = merged.get("overview") or card.get("overview")
        return merged

    # The title list is cached but the details are not, so fetch them in
    # parallel — a dozen sequential calls would blow the request budget.
    # Watchmode rate-limits aggressively, so keep the fan-out modest.
    with ThreadPoolExecutor(max_workers=4) as pool_exec:
        full_cards = [c for c in pool_exec.map(hydrate, pool) if c]

    # An unreleased title can carry a provisional score but has nothing to watch
    # and no audience reaction yet, so it is a poor suggestion. This bites
    # hardest on "recent": Watchmode sorts by release date, so the top of that
    # list is literally next year's slate.
    if not want_unreleased:
        released = [c for c in full_cards if _wm_is_released(c)]
        full_cards = released or full_cards

    rated = [c for c in full_cards if (c.get("imdb_rating") or 0) >= floor]
    # Never return nothing just because scores were missing.
    candidates = rated or full_cards

    if want_recent:
        # Bias towards newer releases without letting a fresher but worse title
        # outrank a well-reviewed one.
        candidates.sort(
            key=lambda c: -_wm_quality(
                float(c.get("imdb_rating") or 0),
                float(c.get("popularity") or 0),
            ) * (0.45 + 0.55 * _wm_recency(c)),
        )
        return candidates[:limit]

    candidates.sort(
        key=lambda c: -_wm_quality(
            float(c.get("imdb_rating") or 0),
            float(c.get("popularity") or 0),
        )
    )
    return candidates[:limit]


def _watchmode_search(query: str, media_type: str = "any", limit: int = 6) -> list[dict]:
    types = {"movie": "movie", "tv": "tv_series", "any": "movie,tv_series"}[media_type or "any"]
    data = _watchmode(
        "/search",
        # `search_field` accepts "name", not "title", and results come back
        # under "title_results" — anything else returns 400 or an empty list.
        {"search_field": "name", "search_value": query, "types": types},
        ttl=43200,
    )
    if not isinstance(data, dict):
        return []
    # The payload can hold results for titles, people or sources.
    results = [
        r for r in (data.get("title_results") or [])
        if r.get("resultType") == "title" and r.get("id")
    ]
    if not results:
        return []
    # Search rows use `name`; normalise so _wm_card can read them.
    cards = [_wm_card({**r, "title": r.get("name") or r.get("title") or ""})
             for r in results]
    cards.sort(key=lambda c: -_title_score(query, c.get("title", "")))
    return cards[:limit]


def _watchmode_details(title_id: str) -> dict | None:
    if not str(title_id).isdigit():
        return None
    data = _watchmode(f"/title/{title_id}/details", ttl=21600)
    if not isinstance(data, dict) or not data.get("title"):
        return None
    return _wm_card(data, detail=True)


def _watchmode_similar(title_id: str, limit: int = 8) -> list[dict]:
    full = _watchmode_details(title_id)
    return (full or {}).get("similar", [])[:limit]


# ============================================================== AniList
# Free, no API key, GraphQL. Unlike OMDb it supports genuine discovery —
# genre + year + country + sort — so it closes the gap left by a missing TMDB
# key for anime, which is the category users ask for most.


_ANILIST_QUERY = """
query($search: String, $genres: [String], $y1: FuzzyDateInt, $y2: FuzzyDateInt,
      $country: CountryCode, $sort: [MediaSort], $page: Int) {
  Page(page: $page, perPage: 20) {
    media(type: ANIME, search: $search, genre_in: $genres,
          startDate_greater: $y1, startDate_lesser: $y2,
          countryOfOrigin: $country, sort: $sort) {
      id title { romaji english native }
      description(asHtml: false)
      averageScore meanScore popularity
      genres countryOfOrigin
      startDate { year } endDate { year }
      episodes duration format status
      coverImage { extraLarge large color }
      bannerImage
      studios(isMain: true) { nodes { name } }
    }
  }
}
"""

_ANILIST_SORTS = {
    "popular": ["POPULARITY_DESC"],
    "rating": ["SCORE_DESC"],
    "newest": ["START_DATE_DESC"],
    "oldest": ["START_DATE_ASC"],
    "trending": ["TRENDING_DESC"],
}


def _anilist(variables: dict) -> dict:
    if not _settings.enable_anilist_fallback:
        return {}
    key = _cache_key("anilist", variables)
    cached = cache.get_json(key)
    if cached is not None:
        return cached
    # AniList treats an explicitly-passed `null` as a filter value (matching
    # media whose country is null), not as "no filter". Unset keys must be
    # omitted from the variables object entirely.
    payload_vars = {k: v for k, v in variables.items() if v not in (None, [], "")}
    try:
        metrics.upstream_call("anilist")
        resp = _anilist_client.post(
            "https://graphql.anilist.co",
            json={"query": _ANILIST_QUERY, "variables": payload_vars},
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        metrics.upstream_error("anilist")
        logger.warning("anilist request failed: %s", exc)
        return {}
    if payload.get("errors"):
        logger.warning("anilist errors: %s", payload["errors"][0].get("message"))
        return {}
    cache.set_json(key, payload, 43200)
    return payload


def _anilist_cards(payload: dict) -> list[dict]:
    out: list[dict] = []
    for m in (((payload or {}).get("data") or {}).get("Page") or {}).get("media") or []:
        titles = m.get("title") or {}
        title = titles.get("english") or titles.get("romaji") or titles.get("native") or ""
        year = (m.get("startDate") or {}).get("year")
        end = (m.get("endDate") or {}).get("year")
        score = m.get("averageScore") or m.get("meanScore") or 0
        out.append({
            "key": f"anilist:anime:{m.get('id')}",
            "source": "anilist",
            "media_type": "tv",
            "id": m.get("id"),
            "title": title,
            "original_title": titles.get("romaji") or title,
            "overview": (m.get("description") or "").strip(),
            "tagline": "",
            "poster_url": (m.get("coverImage") or {}).get("extraLarge")
            or (m.get("coverImage") or {}).get("large") or "",
            "backdrop_url": m.get("bannerImage") or "",
            "year": year,
            "release_date": str(year or ""),
            "genres": m.get("genres") or [],
            "countries": [country_label(
                {"JP": "Japan", "KR": "South Korea", "CN": "China"}.get(
                    m.get("countryOfOrigin"), m.get("countryOfOrigin") or ""))],
            "country_codes": [m.get("countryOfOrigin") or ""],
            "original_language": "ja",
            "language": "Japanese",
            "rating": round(float(score) / 10, 1),
            "vote_count": 0,
            "imdb_rating": None,
            "imdb_id": None,
            "runtime": m.get("duration"),
            "age_rating": "",
            "status": (m.get("format") or "") + ("" if m.get("status") in (None, "") else f" · {m['status']}"),
            "popularity": round(float(m.get("popularity") or 0), 2),
            "seasons": None,
            "episodes": m.get("episodes"),
            "networks": [n.get("name") for n in ((m.get("studios") or {}).get("nodes") or []) if n.get("name")],
            "creators": [],
            "episode_runtime": None,
            "ended_year": end,
        })
    return out


def _anilist_discover(genres=None, year_from=None, year_to=None,
                      country=None, sort=None, limit: int = 8) -> list[dict]:
    variables: dict[str, Any] = {
        "search": None,
        "genres": _anilist_genres(genres),
        # AniList wants FuzzyDateInt as YYYYMMDD, not a bare year.
        "y1": int(year_from) * 10000 + 101 if year_from else None,
        "y2": int(year_to) * 10000 + 1231 if year_to else None,
        "country": _anilist_country(country),
        "sort": _ANILIST_SORTS.get((sort or "popular").lower(), ["POPULARITY_DESC"]),
        "page": 1,
    }
    return _anilist_cards(_anilist(variables))[:limit]


# AniList country codes differ from ISO for a few places.
_ANILIST_COUNTRY = {"IN": "IN", "US": "US", "JP": "JP", "KR": "KR",
                    "CN": "CN", "TW": "TW", "HK": "HK"}


def _anilist_country(country) -> str | None:
    code = _resolve_country(country)
    return _ANILIST_COUNTRY.get(code) if code else None


def _anilist_search(query: str, limit: int = 6) -> list[dict]:
    variables: dict[str, Any] = {
        "search": query, "genres": None, "y1": None, "y2": None,
        "country": None, "sort": ["SEARCH_MATCH"], "page": 1,
    }
    return _anilist_cards(_anilist(variables))[:limit]


def _anilist_details(media_id: str) -> dict | None:
    key = _cache_key("anilist:detail", {"id": media_id})
    cached = cache.get_json(key)
    if cached:
        return cached
    query = """
    query($id: Int) {
      Media(id: $id) {
        id title { romaji english native }
        description(asHtml: false) averageScore meanScore popularity
        genres countryOfOrigin countryOfOriginLanguage
        startDate { year } endDate { year } episodes duration format status
        coverImage { extraLarge large } bannerImage
        studios(isMain: true) { nodes { name } }
        relations { edges { relationType node { id title { romaji english } } } }
        characters(perPage: 12, sort: [ROLE, RELEVANCE]) {
          edges { role node { name { full } image { large } } }
        }
        staff(perPage: 8) { edges { role node { name { full } } } }
      }
    }
    """
    try:
        metrics.upstream_call("anilist")
        resp = _anilist_client.post(
            "https://graphql.anilist.co",
            json={"query": query, "variables": {"id": int(media_id)}},
        )
        resp.raise_for_status()
        data = (resp.json().get("data") or {}).get("Media")
    except Exception as exc:
        metrics.upstream_error("anilist")
        logger.warning("anilist detail failed: %s", exc)
        return None
    if not data:
        return None

    card = _anilist_cards({"data": {"Page": {"media": [data]}}})[0]
    card["cast"] = [
        {"name": (e.get("node", {}).get("name") or {}).get("full"),
         "character": e.get("role"),
         "profile_url": ((e.get("node") or {}).get("image") or {}).get("large") or ""}
        for e in ((data.get("characters") or {}).get("edges") or [])[:12]
        if (e.get("node") or {}).get("name")
    ]
    card["crew"] = [
        {"name": (e.get("node", {}).get("name") or {}).get("full"),
         "job": (e.get("role") or "").replace("_", " ").title(),
         "profile_url": ""}
        for e in ((data.get("staff") or {}).get("edges") or [])[:8]
        if (e.get("node") or {}).get("name")
    ]
    card["keywords"] = []
    card["trailers"] = []
    card["watch"] = []
    card["homepage"] = f"https://anilist.co/anime/{media_id}"
    card["similar"] = []
    cache.set_json(key, card, 43200)
    return card


def _fallback_search(query: str, media_type: str, limit: int) -> list[dict]:
    """No-key / secondary sources, used when TMDB is unconfigured or empty.

    The query is relaxed progressively because OMDb matches on title text.
    """
    q = (query or "").strip()
    if not q:
        return []

    kinds = ["movie", "tv"] if media_type == "any" else [media_type]
    candidates = query_candidates(q)

    out: list[dict] = []
    # Watchmode has real title search for live-action; prefer it.
    if watchmode_configured():
        out = _watchmode_search(q, media_type if media_type != "any" else "any", limit)

    for index, candidate in enumerate(candidates if not out else []):
        found: list[dict] = []
        # Search every requested kind before judging, otherwise a weak movie
        # hit can mask an exact series match ("Breaking Bad" is a series).
        for kind in kinds:
            found.extend(_omdb_search(candidate, kind, limit))
        if found:
            found.sort(key=lambda c: -_title_score(candidate, c.get("title", "")))
            out = found
            break
        # Only fall through to the no-key sources once title search looks
        # hopeless — those calls are slow and rate-limited.
        if index >= 2:
            break

    if not out and media_type in ("any", "tv"):
        if _settings.enable_anilist_fallback:
            out.extend(_anilist_search(q, limit))
        if _settings.enable_jikan_fallback and len(out) < limit:
            out.extend(_jikan_cards(
                _jikan("/anime", {"q": q, "limit": limit, "order_by": "score"}), limit))
        if _settings.enable_tvmaze_fallback and len(out) < limit:
            out.extend(_tvmaze_cards(_tvmaze("/search/shows", {"q": q}) or [], limit))

    # OMDb stubs have no overview/rating yet — fill them in for the top picks.
    out = out[:limit]
    for card in out:
        if card.get("source") == "omdb" and not card.get("overview"):
            full = _omdb_by_id(card["imdb_id"])
            if full:
                card.update({k: v for k, v in full.items() if v not in (None, "", [], 0)})
    return out


def trending(media_type: str = "all", window: str = "week", limit: int = 10) -> list[dict]:
    data = _tmdb(f"/trending/{media_type}/{window}", {"language": "en-US"}, ttl=3600)
    out = []
    for r in ((data or {}).get("results") or [])[:limit]:
        kind = "tv" if r.get("media_type") == "tv" else "movie"
        out.append(_tmdb_card(r, kind))
    return out


# ------------------------------------------------------------------ detail
_DETAIL_APPEND = "credits,videos,keywords,external_ids,release_dates,content_ratings"


def parse_key(key: str) -> tuple[str, str, str]:
    parts = (key or "").split(":")
    if len(parts) != 3:
        return ("", "", "")
    return parts[0], parts[1], parts[2]


def title_details(key: str) -> dict | None:
    """Full record for one title — powers the expanded card in the UI."""
    source, kind, ident = parse_key(key)
    if not ident:
        return None

    if source == "tmdb":
        if not ident.isdigit():
            return None
        data = _tmdb(f"/{kind}/{ident}", {
            "append_to_response": _DETAIL_APPEND + ",watch/providers",
            "language": "en-US",
        })
        if not data:
            return None
        card = _tmdb_card(data, kind, detailed=True)
        sims = _similar_tmdb(kind, ident)
        card["similar"] = sims
        return card

    if source == "jikan":
        data = _jikan(f"/anime/{ident}/full")
        full = (data or {}).get("data") or {}
        cards = _jikan_cards({"data": [full]}, 1)
        if not cards:
            return None
        card = cards[0]
        card["cast"] = [
            {"name": p.get("name"), "character": "", "profile_url": p.get("image_url") or ""}
            for p in ((full.get("characters") or {}).get("data") or [])[:10]
        ]
        card["crew"] = [
            {"name": p.get("name"), "job": "Creator", "profile_url": p.get("image_url") or ""}
            for p in _jikan_persons(full)
        ]
        card["synopsis"] = full.get("synopsis") or ""
        card["trailers"] = []
        card["watch"] = []
        card["keywords"] = []
        card["homepage"] = full.get("url") or ""
        return card

    if source == "omdb":
        return _omdb_by_id(ident)

    if source == "watchmode":
        return _watchmode_details(ident)

    if source == "anilist":
        return _anilist_details(ident)

    if source == "tvmaze":
        data = _tvmaze(f"/shows/{ident}", {})
        if not data:
            return None
        cards = _tvmaze_cards([{"show": data}], 1)
        if not cards:
            return None
        card = cards[0]
        cast = _tvmaze(f"/shows/{ident}/cast", {}) or {}
        card["cast"] = [
            {"name": (p.get("person") or {}).get("name"),
             "character": (p.get("character") or {}).get("name") or "",
             "profile_url": ((p.get("person") or {}).get("image") or {}).get("medium") or ""}
            for p in cast[:10]
        ]
        card["crew"] = [
            {"name": (p.get("person") or {}).get("name"), "job": "Actor", "profile_url": ""}
            for p in cast[10:20]
        ]
        return card

    return None


def _jikan_persons(full: dict) -> list[dict]:
    people: list[dict] = []
    for key in ("directors", "producers", "staff"):
        block = (full.get(key) or {}).get("data") or []
        for p in block[:6]:
            person = p.get("person") or {}
            if person.get("name"):
                people.append({"name": person["name"], "image_url": person.get("image_url") or ""})
    return people


def _similar_tmdb(kind: str, ident: str, limit: int = 8) -> list[dict]:
    data = _tmdb(f"/{kind}/{ident}/recommendations", {"language": "en-US", "page": 1})
    results = ((data or {}).get("results") or [])[:limit]
    if len(results) < limit:
        data2 = _tmdb(f"/{kind}/{ident}/similar", {"language": "en-US", "page": 1})
        results += ((data2 or {}).get("results") or [])[: limit - len(results)]
    return [_tmdb_card(r, kind) for r in results[:limit]]


def similar_titles(key: str, limit: int = 8) -> list[dict]:
    source, kind, ident = parse_key(key)
    if source == "watchmode":
        return _watchmode_similar(ident, limit)
    if source == "tmdb" and ident.isdigit():
        return _similar_tmdb(kind, ident, limit)
    # Fallbacks: reuse sibling titles by genre.
    card = title_details(key) or {}
    genres = card.get("genres") or []
    kind = kind or "tv"
    return find_titles(genres=genres[:2], media_type=kind, limit=limit)


def reviews(key: str, limit: int = 6) -> list[dict]:
    """Critic/user reviews for a title, newest first."""
    source, kind, ident = parse_key(key)
    out: list[dict] = []
    if source == "omdb":
        # OMDb exposes a single aggregate review blurb rather than a list.
        full = _omdb_by_id(ident)
        if full and full.get("overview"):
            out.append({
                "author": "IMDb / OMDb",
                "rating": full.get("imdb_rating") or 0,
                "comment": full.get("overview"),
                "created_at": full.get("release_date") or "",
            })
            if full.get("awards"):
                out.append({
                    "author": "Awards",
                    "rating": None,
                    "comment": full["awards"],
                    "created_at": "",
                })
    elif source == "tmdb" and ident.isdigit():
        data = _tmdb(f"/{kind}/{ident}/reviews", {"language": "en-US", "page": 1})
        for r in ((data or {}).get("results") or [])[:limit]:
            out.append({
                "author": r.get("author") or "TMDB reviewer",
                "rating": round(float(r.get("rating") or 0), 1),
                "comment": (r.get("content") or "").strip(),
                "created_at": (r.get("created_at") or "")[:10],
            })
    elif source == "jikan":
        data = _jikan(f"/anime/{ident}/reviews")
        for r in ((data or {}).get("data") or [])[:limit]:
            out.append({
                "author": r.get("user", {}).get("username") or "MAL user",
                "rating": round(float((r.get("score") or 0) / 2), 1),
                "comment": r.get("content") or "",
                "created_at": "",
            })
    return out


def cards_for_keys(keys: list[str]) -> list[dict]:
    """Validate keys against our providers — this is the grounding guarantee.

    Any key the model invents resolves to nothing and is silently dropped.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for raw in keys or []:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        card = _compact_cached(key)
        if card:
            out.append(card)
            seen.add(key)
    return out


def _compact_cached(key: str) -> dict | None:
    """Prefer the compact record the LLM already fetched; fall back to detail."""
    cached = cache.get_json(f"card:{key}")
    if cached:
        return cached
    return title_details(key)


def remember_card(card: dict) -> None:
    """Store a title's public card shape so the UI can render it later."""
    if card and card.get("key"):
        cache.set_json(f"card:{card['key']}", card)


# ---------------------------------------------------- payloads for the LLM
def to_compact(card: dict, overview_chars: int = 220) -> dict:
    """Trimmed view handed to the model — keeps prompts small."""
    out = {
        "key": card["key"],
        "title": card.get("title"),
        "type": "movie" if card.get("media_type") == "movie" else "series",
        "year": card.get("year"),
        "rating": card.get("rating") or 0,
        "genres": (card.get("genres") or [])[:4],
        "language": card.get("language") or "",
        "countries": (card.get("countries") or [])[:2],
        "overview": (card.get("overview") or "")[:overview_chars],
    }
    if card.get("media_type") == "tv":
        out["seasons"] = card.get("seasons")
        out["episodes"] = card.get("episodes")
        out["creators"] = (card.get("creators") or [])[:3]
    else:
        out["runtime_min"] = card.get("runtime")
    return out


def compact_many(cards: list[dict], overview_chars: int = 220) -> list[dict]:
    return [to_compact(c, overview_chars) for c in cards if c]


def resolve_title(name: str, media_type: str = "any") -> list[dict]:
    """Find the specific title(s) a user is asking about."""
    return find_titles(query=name, media_type=media_type, limit=5)


def reference_data() -> dict:
    """Options the bot offers in its clarifying-question widget."""
    return {
        "genres": CURATED_GENRES,
        "countries": [{"value": c, "label": n} for c, n in COUNTRIES.items()],
        "languages": [{"value": c, "label": n} for c, n in LANGUAGES.items()],
        "years": YEAR_BUCKETS,
        "formats": FORMAT_OPTIONS,
        "types": [
            {"value": "movie", "label": "Movies"},
            {"value": "tv", "label": "TV Series"},
            {"value": "tv", "label": "Animation / Anime"},
        ],
    }


def filter_reference() -> str:
    """A compact, cached cheat-sheet of valid filter values for the prompt.

    Without this the model invents a handful of options for each dropdown
    ("Any", "2020s", "Comedy"). With it, the country and language dropdowns can
    be as rich as the catalog actually supports.
    """
    key = "filter-reference:v2"
    cached = cache.get_json(key)
    if cached:
        return cached

    text = (
        "## Valid values for `ask_user` fields (use these exact labels)\n"
        "COUNTRY: " + ", ".join(COUNTRIES.values()) + "\n"
        "LANGUAGE / DUBBING: " + ", ".join(LANGUAGES.values()) + "\n"
        "GENRE: " + ", ".join(CURATED_GENRES) + "\n"
        "RELEASE YEAR / ERA: " + ", ".join(YEAR_BUCKETS) + "\n"
        "FORMAT: " + ", ".join(FORMAT_OPTIONS) + "\n"
        "How many options: for a DROPDOWN field offer 12-20 of the most relevant "
        "values (the list scrolls, so a rich menu is welcome). For a CHECKBOX or "
        "RADIO field offer only 5-8 so the chips stay on one row. Always put an "
        "'Any' option first in a dropdown."
    )
    cache.set_json(key, text)
    return text
