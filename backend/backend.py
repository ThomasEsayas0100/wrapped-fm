import json
import os
import re
import string
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from threading import Lock, Thread

import dateutil.tz
import numpy as np
import pandas as pd
import pytz
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression


API_KEY = "160e814675379ba7267c54a1ff597c97"
USER = "th_ma_"
LOCAL_TZ = pytz.timezone("America/New_York")

unmatched_tags_log: list[str] = []
summary = {"countryCounts": {}, "countryTopArtists": {}}
artist_country_map: dict[str, str | None] = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Fast tag lookup & caching
# ---------------------------------------------------------------------------
SESSION = requests.Session()
TAG_MODE = os.getenv("TAG_MODE", "track_album_artist")
PREFETCH_RPS = int(os.getenv("PREFETCH_RPS", "4"))
PREFETCH_WORKERS = int(os.getenv("PREFETCH_WORKERS", "8"))

CACHE_DIR = "cache"
TRACK_CACHE_PATH = os.path.join(CACHE_DIR, "tags_track.json")
ALBUM_CACHE_PATH = os.path.join(CACHE_DIR, "tags_album.json")
ARTIST_CACHE_PATH = os.path.join(CACHE_DIR, "tags_artist.json")

_CACHE_LOCKS: dict[str, Lock] = {
    TRACK_CACHE_PATH: Lock(),
    ALBUM_CACHE_PATH: Lock(),
    ARTIST_CACHE_PATH: Lock(),
}


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_track_cache = _load_json(TRACK_CACHE_PATH)
_album_cache = _load_json(ALBUM_CACHE_PATH)
_artist_cache = _load_json(ARTIST_CACHE_PATH)


def _save_json_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _set_cache(cache: dict, path: str, key: str, value: dict) -> None:
    lock = _CACHE_LOCKS.setdefault(path, Lock())
    with lock:
        cache[key] = value
        _save_json_atomic(path, dict(cache))


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _track_key(artist, album, track) -> str:
    return f"{_norm(artist)}\t{_norm(album)}\t{_norm(track)}"


def _album_key(artist, album) -> str:
    return f"{_norm(artist)}\t{_norm(album)}"


def _artist_key(artist) -> str:
    return _norm(artist)


def _lfm_get(method: str, params: dict) -> dict:
    url = "https://ws.audioscrobbler.com/2.0/"
    payload = {"method": method, "api_key": API_KEY, "format": "json", **params}
    r = SESSION.get(url, params=payload, timeout=12)
    r.raise_for_status()
    return r.json()


def _parse_tags(payload: dict) -> dict[str, int]:
    tags = payload.get("toptags", {}).get("tag", [])
    result = {}
    for tg in sorted(tags, key=lambda z: int(z.get("count", 0)), reverse=True)[:3]:
        name = tg.get("name")
        if name:
            result[name] = int(tg.get("count", 0))
    return result


def _fetch_track_tags(artist: str, track: str) -> dict[str, int]:
    try:
        return _parse_tags(_lfm_get("track.getTopTags", {"artist": artist, "track": track}))
    except Exception:
        return {}


def _fetch_album_tags(artist: str, album: str) -> dict[str, int]:
    try:
        return _parse_tags(_lfm_get("album.getTopTags", {"artist": artist, "album": album}))
    except Exception:
        return {}


def _fetch_artist_tags(artist: str) -> dict[str, int]:
    try:
        return _parse_tags(_lfm_get("artist.getTopTags", {"artist": artist}))
    except Exception:
        return {}


def get_track_tags_fast(artist, album, track, mode=TAG_MODE) -> dict[str, int]:
    """Cached tag lookup with disk persistence."""
    k_t = _track_key(artist, album, track)
    if k_t in _track_cache:
        return _track_cache[k_t]

    if mode == "artist_only":
        k_a = _artist_key(artist)
        if k_a not in _artist_cache:
            _set_cache(_artist_cache, ARTIST_CACHE_PATH, k_a, _fetch_artist_tags(artist))
        _set_cache(_track_cache, TRACK_CACHE_PATH, k_t, _artist_cache[k_a])
        return _track_cache[k_t]

    tags = _fetch_track_tags(artist, track)
    if tags:
        _set_cache(_track_cache, TRACK_CACHE_PATH, k_t, tags)
        return tags

    if mode != "track_then_artist" and album:
        k_alb = _album_key(artist, album)
        if k_alb in _album_cache:
            tags = _album_cache[k_alb]
        else:
            tags = _fetch_album_tags(artist, album)
            if tags:
                _set_cache(_album_cache, ALBUM_CACHE_PATH, k_alb, tags)
        if tags:
            _set_cache(_track_cache, TRACK_CACHE_PATH, k_t, tags)
            return tags

    k_a = _artist_key(artist)
    if k_a not in _artist_cache:
        _set_cache(_artist_cache, ARTIST_CACHE_PATH, k_a, _fetch_artist_tags(artist))
    _set_cache(_track_cache, TRACK_CACHE_PATH, k_t, _artist_cache[k_a])
    return _track_cache[k_t]


def prefetch_tags_for_keys(track_keys, mode=TAG_MODE, rps=PREFETCH_RPS, workers=PREFETCH_WORKERS):
    missing = [(a, al, t) for a, al, t in track_keys if _track_key(a, al, t) not in _track_cache]
    if not missing:
        return 0

    def _do(triple):
        a, al, t = triple
        get_track_tags_fast(a, al, t, mode=mode)

    i = 0
    n = len(missing)
    while i < n:
        batch = missing[i : i + rps]
        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as ex:
            list(ex.map(_do, batch))
        i += len(batch)
        if i < n:
            time.sleep(1.05)
    return n


# ---------------------------------------------------------------------------
# Scrobble fetching / session helpers
# ---------------------------------------------------------------------------
def fetch_scrobble_page(page: int, params: dict, url: str) -> dict:
    params = params.copy()
    params["page"] = page
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_scrobbles_in_timeframe(start: str, end: str, max_workers: int = 5) -> list[dict]:
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    params = {
        "method": "user.getRecentTracks",
        "user": USER,
        "api_key": API_KEY,
        "format": "json",
        "limit": 200,
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
    }
    url = "https://ws.audioscrobbler.com/2.0/"

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    total_pages = int(data.get("recenttracks", {}).get("@attr", {}).get("totalPages", 1))

    scrobbles = [
        {
            "artist": t["artist"]["#text"],
            "track": t["name"],
            "album": t["album"]["#text"],
            "timestamp": int(t["date"]["uts"]),
            "datetime": t["date"]["#text"],
            "url": t["url"],
        }
        for t in data.get("recenttracks", {}).get("track", [])
        if "date" in t
    ]

    pages = list(range(2, total_pages + 1))
    for i in range(0, len(pages), max_workers):
        batch = pages[i : i + max_workers]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_scrobble_page, p, params, url) for p in batch]
            for fut in as_completed(futures):
                for t in fut.result().get("recenttracks", {}).get("track", []):
                    if "date" in t:
                        scrobbles.append(
                            {
                                "artist": t["artist"]["#text"],
                                "track": t["name"],
                                "album": t["album"]["#text"],
                                "timestamp": int(t["date"]["uts"]),
                                "datetime": t["date"]["#text"],
                                "url": t["url"],
                            }
                        )
        time.sleep(0.25)
    return scrobbles


def identify_session_starts(scrobbles: list[dict], inactivity: int = 900) -> tuple[list[dict], list[bool]]:
    if not scrobbles:
        return [], []
    sorted_s = sorted(scrobbles, key=lambda s: s["timestamp"])
    flags = [False] * len(sorted_s)
    flags[0] = True
    for i in range(1, len(sorted_s)):
        if sorted_s[i]["timestamp"] - sorted_s[i - 1]["timestamp"] > inactivity:
            flags[i] = True
    return [s for s, f in zip(sorted_s, flags) if f], flags


def split_into_sessions(scrobbles: list[dict], inactivity: int = 900) -> list[list[dict]]:
    if not scrobbles:
        return []
    sorted_s = sorted(scrobbles, key=lambda s: s["timestamp"])
    sessions, cur = [], [sorted_s[0]]
    for prev, cur_scrob in zip(sorted_s, sorted_s[1:]):
        if cur_scrob["timestamp"] - prev["timestamp"] > inactivity:
            sessions.append(cur)
            cur = [cur_scrob]
        else:
            cur.append(cur_scrob)
    sessions.append(cur)
    return sessions


def UTCtoLocal(utc_dt: datetime) -> datetime:
    return utc_dt.replace(tzinfo=timezone.utc).astimezone()


def longest_listening_streak(scrobbles: list[dict]) -> dict:
    days = {
        UTCtoLocal(datetime.strptime(s["datetime"], "%d %b %Y, %H:%M")).date()
        for s in scrobbles
    }
    if not days:
        return {"streak": 0, "start_day": None, "end_day": None}

    current = streak = 0
    start = best_start = best_end = None
    day = min(days)
    end_day = max(days)

    while day <= end_day:
        if day in days:
            if current == 0:
                start = day
            current += 1
            if current > streak:
                streak = current
                best_start, best_end = start, day
        else:
            current = 0
        day += timedelta(days=1)
    return {"streak": streak, "start_day": best_start, "end_day": best_end}


def session_summary(sessions: list[list[dict]]) -> dict:
    if not sessions:
        return {"total_sessions": 0, "average_session_length": 0, "median_session_length": 0}
    lengths = [(s[-1]["timestamp"] - s[0]["timestamp"]) / 60 for s in sessions]
    lengths.sort()
    return {
        "total_sessions": len(sessions),
        "average_session_length": sum(lengths) / len(lengths),
        "median_session_length": lengths[len(lengths) // 2],
    }


# ---------------------------------------------------------------------------
# Simple stats
# ---------------------------------------------------------------------------
def most_played_songs(scrobbles: list[dict]) -> list[tuple[str, int]]:
    return Counter(s["track"] for s in scrobbles).most_common()


def most_played_artists(scrobbles: list[dict]) -> list[tuple[str, int]]:
    return Counter(s["artist"] for s in scrobbles).most_common()


def most_played_albums(scrobbles: list[dict]) -> list[tuple[str, int]]:
    return Counter(s["album"] for s in scrobbles).most_common()


def all_artists(scrobbles: list[dict]) -> set[str]:
    return {s["artist"] for s in scrobbles}


def artist_counts(scrobbles: list[dict]) -> list[tuple[str, int]]:
    return Counter(s["artist"] for s in scrobbles).most_common()


# ---------------------------------------------------------------------------
# Tag & country utilities
# ---------------------------------------------------------------------------
def get_artist_info(artist_name: str) -> dict:
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {"method": "artist.getInfo", "artist": artist_name, "api_key": API_KEY, "format": "json"}
    return requests.get(url, params=params).json().get("artist", {})


def normalize(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def infer_country_from_tags_and_bio(artist_info: dict) -> str | None:
    with open("backend/country_keywords.json", encoding="utf-8") as f:
        country_keywords = json.load(f)

    raw_tags = artist_info.get("tags", {}).get("tag", [])
    tags = [normalize(t["name"]) for t in raw_tags if "name" in t]
    for country, keywords in country_keywords.items():
        norm_keywords = [normalize(k) for k in keywords]
        if any(any(k in tag for k in norm_keywords) for tag in tags):
            return country
    unmatched_tags_log.extend(tags)

    bio = normalize(artist_info.get("bio", {}).get("summary", ""))
    best_country = None
    earliest = float("inf")
    for country, keywords in country_keywords.items():
        for k in [normalize(k) for k in keywords]:
            m = re.search(rf"\b{re.escape(k)}\b", bio)
            if m and m.start() < earliest:
                earliest, best_country = m.start(), country
    return best_country


def get_artist_country(artist_name: str) -> str | None:
    return infer_country_from_tags_and_bio(get_artist_info(artist_name))


def get_countries_for_artists_parallel(artists: list[str], max_workers: int = 5) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_artist = {ex.submit(get_artist_country, a): a for a in artists}
        for fut in as_completed(future_to_artist, timeout=60):
            artist = future_to_artist[fut]
            try:
                results[artist] = fut.result(timeout=10)
            except Exception as exc:
                print(f"Error fetching country for {artist}: {exc}")
                results[artist] = None
    return results


def build_country_summary(tracks_df: pd.DataFrame, artist_country_map: dict[str, str | None]) -> dict:
    counts = defaultdict(int)
    top_artists = defaultdict(lambda: defaultdict(int))
    for _, row in tracks_df.iterrows():
        artist = row["artist"]
        country = artist_country_map.get(artist)
        if not country:
            continue
        plays = len(row["timestamps"])
        counts[country] += plays
        top_artists[country][artist] += plays
    top3 = {
        c: [a for a, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:3]]
        for c, d in top_artists.items()
    }
    return {"countryCounts": dict(counts), "countryTopArtists": top3}


# ---------------------------------------------------------------------------
# DataFrame helpers / analytics
# ---------------------------------------------------------------------------
def scrobbles_to_df(scrobbles: list[dict]) -> pd.DataFrame:
    track_dict = defaultdict(list)
    for s in scrobbles:
        artist, track, album, ts = s["artist"], s["track"], s["album"], s["timestamp"]
        if artist and track and album and ts:
            track_dict[(artist, track, album)].append(datetime.utcfromtimestamp(ts))
    data = [
        {"artist": a, "track": t, "album": al, "timestamps": sorted(ts)}
        for (a, t, al), ts in track_dict.items()
    ]
    return pd.DataFrame(data)


def compute_burn_scores(
    tracks_df: pd.DataFrame,
    window_days: int = 30,
    early_windows: int = 3,
    late_windows: int = 3,
) -> pd.DataFrame:
    results = []
    for _, row in tracks_df.iterrows():
        ts = row["timestamps"]
        if len(ts) < 2:
            results.append((0, 0, 0, 0))
            continue

        start, end = ts[0], ts[-1]
        total_days = (end - start).days + 1
        n_windows = max((total_days + window_days - 1) // window_days, 1)
        bins = [start + timedelta(days=i * window_days) for i in range(n_windows + 1)]

        counts = np.zeros(n_windows, dtype=int)
        idx = 0
        for i in range(n_windows):
            bin_start, bin_end = bins[i], bins[i + 1]
            while idx < len(ts) and ts[idx] < bin_end:
                if ts[idx] >= bin_start:
                    counts[i] += 1
                idx += 1
        while idx < len(ts):
            counts[-1] += 1
            idx += 1

        early = counts[:early_windows].mean() if len(counts) >= early_windows else counts.mean()
        late = counts[-late_windows:].mean() if len(counts) >= late_windows else counts.mean()
        slope = LinearRegression().fit(np.arange(n_windows).reshape(-1, 1), counts).coef_[0]
        burn = (late / early) * slope if early > 0 else 0
        results.append((early, late, slope, burn))

    df = tracks_df.copy()
    df["early_avg"], df["late_avg"], df["slope"], df["burn_score"] = zip(*results)
    return df


def filter_track_df_by_time(tracks_df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    def in_range(ts: datetime) -> bool:
        h = ts.replace(tzinfo=pytz.UTC).astimezone().hour
        return start_hour <= h <= end_hour if start_hour <= end_hour else h >= start_hour or h <= end_hour

    rows = []
    for _, row in tracks_df.iterrows():
        times = [ts for ts in row["timestamps"] if in_range(ts)]
        if times:
            rows.append({"artist": row["artist"], "track": row["track"], "album": row["album"], "timestamps": times})
    return pd.DataFrame(rows)


def filter_track_df_by_date(tracks_df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    local_tz = dateutil.tz.tzlocal()

    exploded = tracks_df.explode("timestamps")
    local_dates = (
        exploded["timestamps"].dt.tz_localize(pytz.UTC).dt.tz_convert(local_tz).dt.date
    )
    mask = (local_dates >= start_date) & (local_dates <= end_date)
    filtered = exploded[mask]
    if filtered.empty:
        return pd.DataFrame(columns=tracks_df.columns)
    return (
        filtered.groupby(level=0)
        .agg({"artist": "first", "track": "first", "album": "first", "timestamps": list})
        .reset_index(drop=True)
    )


def hour_pie_chart(scrobbles: list[dict]) -> dict[int, int]:
    counts = Counter()
    for s in scrobbles:
        h = (
            datetime.utcfromtimestamp(s["timestamp"]).replace(tzinfo=pytz.UTC).astimezone().hour
        )
        counts[h] += 1
    return dict(counts)


def filtered_tag_scores_by_date(tracks_df: pd.DataFrame, start, end, threshold: float = 0.025) -> dict[str, float]:
    tracks_df = filter_track_df_by_date(tracks_df, start, end)
    return _tag_scores(tracks_df, threshold)


def filtered_tag_scores(tracks_df: pd.DataFrame, start_hour: int, end_hour: int, threshold: float = 0.025) -> dict[str, float]:
    tracks_df = filter_track_df_by_time(tracks_df, start_hour, end_hour)
    return _tag_scores(tracks_df, threshold)


def _tag_scores(tracks_df: pd.DataFrame, threshold: float) -> dict[str, float]:
    totals = defaultdict(float)
    for _, row in tracks_df.iterrows():
        tags = get_track_tags_fast(row["artist"], row["album"], row["track"])
        plays = len(row["timestamps"])
        for tag, weight in tags.items():
            totals[tag] += weight * plays
    total = sum(totals.values())
    if total == 0:
        return {}
    normalized = {t: w / total for t, w in totals.items()}
    filtered = {t: w for t, w in normalized.items() if w >= threshold}
    filtered_total = sum(filtered.values())
    return {t: w / filtered_total for t, w in filtered.items()} if filtered_total else {}


# ---------------------------------------------------------------------------
# Country summary endpoint and background updater
# ---------------------------------------------------------------------------
@app.get("/country-summary")
def get_country_summary():
    return summary


def background_country_updater() -> None:
    global summary, artist_country_map
    unresolved = [a for a in tracks_df["artist"].unique() if a not in artist_country_map]
    print(f"⏳ Starting country resolution for {len(unresolved)} artists")
    for i, artist in enumerate(unresolved):
        try:
            artist_country_map[artist] = get_artist_country(artist)
            print(f"✅ [{i}] {artist} → {artist_country_map[artist]}")
        except Exception as exc:
            print(f"❌ [{i}] {artist} failed: {exc}")
            continue
        summary = build_country_summary(tracks_df, artist_country_map)
        print(f"📊 Summary updated: {len(summary['countryCounts'])} countries so far")
        time.sleep(1.5)


# ---------------------------------------------------------------------------
# Scrobble file utilities & weekly tags
# ---------------------------------------------------------------------------
@lru_cache(maxsize=20000)
def get_cached_track_tags(artist: str, album: str | None, track: str) -> dict[str, int]:
    return get_track_tags_fast(artist, album, track) or {}


def _dedupe_scrobbles(scrobbles: list[dict]) -> list[dict]:
    seen = {}
    for s in scrobbles:
        key = (s.get("artist", ""), s.get("track", ""), s.get("album", ""), int(s.get("timestamp", 0)))
        if key not in seen:
            seen[key] = s
    return sorted(seen.values(), key=lambda x: x["timestamp"])


def update_scrobbles_file(json_path: str = "scrobbles_2025.json", default_start: str = "2025-01-01") -> list[dict]:
    existing = []
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            existing = json.load(f)

    if existing:
        last_ts = max(int(s["timestamp"]) for s in existing)
        start_date = datetime.utcfromtimestamp(last_ts).strftime("%Y-%m-%d")
    else:
        start_date = default_start
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    new_scrobbles = get_scrobbles_in_timeframe(start_date, end_date) if start_date <= end_date else []
    combined = _dedupe_scrobbles(existing + new_scrobbles)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"📝 Updated {json_path}: {len(existing)} → {len(combined)} scrobbles.")
    return combined


def _week_key_local(ts_int: int) -> tuple[str, int]:
    dt_local = datetime.utcfromtimestamp(ts_int).replace(tzinfo=pytz.UTC).astimezone(LOCAL_TZ)
    iso_year, iso_week, _ = dt_local.isocalendar()
    return f"{iso_year}-W{iso_week:02d}", iso_year


def _normalize_and_threshold(tag_totals: dict[str, float], threshold: float = 0.025) -> dict[str, float]:
    total = sum(tag_totals.values())
    if total <= 0:
        return {}
    normalized = {t: v / total for t, v in tag_totals.items()}
    filtered = {t: w for t, w in normalized.items() if w >= threshold}
    ssum = sum(filtered.values())
    if ssum <= 0:
        return {}
    return dict(sorted(((t, w / ssum) for t, w in filtered.items()), key=lambda kv: kv[1], reverse=True))


def compute_weekly_tag_distributions_from_scrobbles(
    scrobbles: list[dict], year_filter: int = 2025, threshold: float = 0.025, progress: bool = True, top_k: int = 5
) -> dict[str, dict]:
    start_all = time.perf_counter()

    df = pd.DataFrame(scrobbles)
    if df.empty:
        return {}
    dt_local = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    iso = dt_local.dt.isocalendar()
    df["iso_year"] = iso["year"]
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df_year = df[df["iso_year"] == year_filter]

    # 0) Gather unique track keys and prefetch tags
    unique_keys = set(zip(df_year["artist"], df_year["album"], df_year["track"]))
    gather_done = time.perf_counter()
    if progress:
        print(f"⏱️ gathered {len(unique_keys)} unique tracks in {gather_done - start_all:.2f}s")

    fetched = prefetch_tags_for_keys(unique_keys)
    prefetch_done = time.perf_counter()
    if progress:
        print(f"⚡ Prefetched tags for {fetched} tracks (mode={TAG_MODE}).")
        print(f"⏱️ prefetch completed in {prefetch_done - gather_done:.2f}s")

    # Build a local cache dict (fast lookups, avoids repeated dict key build)
    tag_cache = {key: get_track_tags_fast(*key) for key in unique_keys}

    # 1) Count scrobbles per week & track
    grouped = (
        df_year.groupby(["week", "artist", "album", "track"])
        .size()
        .reset_index(name="count")
    )
    week_scrobble_counts = grouped.groupby("week")["count"].sum().to_dict()
    count_done = time.perf_counter()
    if progress:
        total_scrobbles = int(sum(week_scrobble_counts.values()))
        print(f"⏱️ counted {total_scrobbles} scrobbles in {count_done - prefetch_done:.2f}s")

    # 2) Accumulate totals with cached lookups
    weekly: dict[str, dict] = {}
    for wk, sub in grouped.groupby("week"):
        step_start = time.perf_counter()
        tag_totals = defaultdict(float)

        for _, row in sub.iterrows():
            weights = tag_cache[(row["artist"], row["album"], row["track"])]
            for tag, w in weights.items():
                tag_totals[tag] += float(w) * row["count"]

        tags_norm = _normalize_and_threshold(tag_totals, threshold=threshold)
        weekly[wk] = {"tags": tags_norm, "scrobbles": int(week_scrobble_counts[wk])}

        if progress:
            try:
                yr = int(wk[:4]); wknum = int(wk.split("-W")[1])
                wk_start = date.fromisocalendar(yr, wknum, 1)
                wk_end   = date.fromisocalendar(yr, wknum, 7)
                top_items = list(tags_norm.items())[:top_k]
                top_str = ", ".join(f"{t} {w*100:.1f}%" for t, w in top_items) if top_items else "no dominant tags"
                print(f"✅ {wk}  {wk_start:%b %d}–{wk_end:%b %d}  —  {top_str}  |  scrobbles: {week_scrobble_counts[wk]}")
            except Exception:
                print(f"✅ {wk} — scrobbles: {week_scrobble_counts[wk]}")

    if progress:
        end_all = time.perf_counter()
        print(f"⏱️ total processing time {end_all - start_all:.2f}s")

    return dict(sorted(weekly.items()))

def export_weekly_tags(
    weekly: dict[str, dict],
    out_json: str = "tag_distributions_weekly_2025.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save the nested weekly dict to JSON, plus:
      - long CSV: week, tag, weight, scrobbles
      - wide Parquet: index=week, columns=tag, values=weight
    Returns (df_long, df_wide) for downstream analysis if you're running as a script.
    """
    # 0) JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(weekly, f, ensure_ascii=False, indent=2)

    # 1) Long tidy DF
    rows = []
    for week, payload in weekly.items():
        sc = int(payload.get("scrobbles", 0))
        for tag, weight in payload.get("tags", {}).items():
            rows.append({"week": week, "tag": tag, "weight": float(weight), "scrobbles": sc})
    df_long = pd.DataFrame(rows).sort_values(["week", "weight"], ascending=[True, False])

    print(f"💾 Saved:\n  • {out_json}\n")
    return df_long


# ---------------------------------------------------------------------------
# Main script & FastAPI startup
# ---------------------------------------------------------------------------
def main() -> None:
    global tracks_df
    scrobbles = update_scrobbles_file("scrobbles_2025.json", default_start="2025-01-01")

    print("📊 Computing weekly tag distributions for 2025 (local ISO weeks)...")
    weekly = compute_weekly_tag_distributions_from_scrobbles(
        scrobbles, threshold=0.025, year_filter=2025
    )
    with open("tag_distributions_weekly_2025.json", "w", encoding="utf-8") as f:
        json.dump(weekly, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(weekly)} weekly distributions → tag_distributions_weekly_2025.json")

    export_weekly_tags(
        weekly,
        out_json="tag_distributions_weekly_2025.json"
    )
    tracks_df = scrobbles_to_df(scrobbles)
    # Thread(target=background_country_updater, daemon=True).start()


if __name__ == "__main__":
    main()


@app.on_event("startup")
def start_up() -> None:
    global tracks_df
    with open("scrobbles_2025.json", encoding="utf-8") as f:
        scrobbles = json.load(f)
    tracks_df = scrobbles_to_df(scrobbles)
    print(f"✅ Loaded {len(scrobbles)} scrobbles into DataFrame")
    Thread(target=background_country_updater, daemon=True).start()
    print("🚀 Background country updater started")

