# ——— Standard library ———
import os
import json
import re
import string
import time
from collections import defaultdict, Counter
from datetime import datetime, date, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Thread

# ——— Third-party ———
import requests
import pytz
import dateutil.tz
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# ==== FAST TAG FETCH WITH DISK CACHE + PARALLEL PREFETCH ===================
import os, json, time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import tempfile
from threading import Lock
from pathlib import Path

SESSION = requests.Session()

# Toggle speed/accuracy: "track_album_artist" (default) or "artist_only" or "track_then_artist"
TAG_MODE = os.getenv("TAG_MODE", "track_album_artist")

# Rate limit (requests/second) for prefetch; tune if you hit 429s
PREFETCH_RPS = int(os.getenv("PREFETCH_RPS", "4"))
PREFETCH_WORKERS = int(os.getenv("PREFETCH_WORKERS", "8"))

CACHE_DIR = "cache"
TRACK_CACHE_PATH  = os.path.join(CACHE_DIR, "tags_track.json")
ALBUM_CACHE_PATH  = os.path.join(CACHE_DIR, "tags_album.json")
ARTIST_CACHE_PATH = os.path.join(CACHE_DIR, "tags_artist.json")

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

_CACHE_LOCKS = {
    TRACK_CACHE_PATH:  Lock(),
    ALBUM_CACHE_PATH:  Lock(),
    ARTIST_CACHE_PATH: Lock(),
}

def _save_json_atomic(path: str, obj: dict) -> None:
    """Atomic write: unique temp in same dir, then os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=os.path.dirname(path),
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tf:
        json.dump(obj, tf, ensure_ascii=False, indent=2)
        tmp_path = tf.name
    os.replace(tmp_path, path)

def _write_cache(path: str, obj: dict) -> None:
    """Thread-safe write for a given cache file."""
    lock = _CACHE_LOCKS.get(path)
    if lock is None:
        # just in case someone refactors paths later
        _CACHE_LOCKS[path] = Lock()
        lock = _CACHE_LOCKS[path]
    with lock:
        _save_json_atomic(path, obj)
def _norm(s): return (s or "").strip().lower()
def _track_key(artist, album, track): return f"{_norm(artist)}\t{_norm(album)}\t{_norm(track)}"
def _album_key(artist, album):       return f"{_norm(artist)}\t{_norm(album)}"
def _artist_key(artist):             return _norm(artist)

# Load caches once
_track_cache  = _load_json(TRACK_CACHE_PATH)
_album_cache  = _load_json(ALBUM_CACHE_PATH)
_artist_cache = _load_json(ARTIST_CACHE_PATH)

def _lfm_get(method, params):
    url = "https://ws.audioscrobbler.com/2.0/"
    p = {"method": method, "api_key": API_KEY, "format": "json", **params}
    r = SESSION.get(url, params=p, timeout=12)
    r.raise_for_status()
    return r.json()

def _parse_tags(d):
    tags = d.get("toptags", {}).get("tag", [])
    out = {}
    for tg in sorted(tags, key=lambda z: int(z.get("count", 0)), reverse=True)[:3]:
        name = tg.get("name")
        if name:
            out[name] = int(tg.get("count", 0))
    return out

def _fetch_track_tags(artist, track):
    try:
        return _parse_tags(_lfm_get("track.getTopTags", {"artist": artist, "track": track}))
    except Exception:
        return {}

def _fetch_album_tags(artist, album):
    try:
        # include artist for better disambiguation
        return _parse_tags(_lfm_get("album.getTopTags", {"artist": artist, "album": album}))
    except Exception:
        return {}

def _fetch_artist_tags(artist):
    try:
        return _parse_tags(_lfm_get("artist.getTopTags", {"artist": artist}))
    except Exception:
        return {}

def get_track_tags_fast(artist, album, track, mode=TAG_MODE):
    """
    Fast, cached tag lookup. Persists to disk so later runs are instant.
    mode:
      - "track_album_artist": try track -> album -> artist (most specific)
      - "track_then_artist":  try track -> artist (skip album to halve calls)
      - "artist_only":        only artist (fastest; 1 call/artist)
    """
    k_t = _track_key(artist, album, track)
    if k_t in _track_cache:
        return _track_cache[k_t]

    if mode == "artist_only":
        k_a = _artist_key(artist)
        if k_a not in _artist_cache:
            _artist_cache[k_a] = _fetch_artist_tags(artist)
            _write_cache(ARTIST_CACHE_PATH, _artist_cache)
        _track_cache[k_t] = _artist_cache[k_a]
        _write_cache(TRACK_CACHE_PATH, _track_cache)
        return _track_cache[k_t]

    # try track
    tags = _fetch_track_tags(artist, track)
    if tags:
        _track_cache[k_t] = tags
        _write_cache(TRACK_CACHE_PATH, _track_cache)
        return tags

    # optional album hop (skip when mode == "track_then_artist")
    if mode != "track_then_artist" and album:
        k_alb = _album_key(artist, album)
        if k_alb in _album_cache:
            _track_cache[k_t] = _album_cache[k_alb]
            _write_cache(TRACK_CACHE_PATH, _track_cache)
            return _album_cache[k_alb]
        tags = _fetch_album_tags(artist, album)
        if tags:
            _album_cache[k_alb] = tags
            _write_cache(ALBUM_CACHE_PATH, _album_cache)
            _track_cache[k_t] = tags
            _write_cache(TRACK_CACHE_PATH, _track_cache)
            return tags

    # fallback artist
    k_a = _artist_key(artist)
    if k_a not in _artist_cache:
        _artist_cache[k_a] = _fetch_artist_tags(artist)
        _write_cache(ARTIST_CACHE_PATH, _artist_cache)
    _track_cache[k_t] = _artist_cache[k_a]
    _write_cache(TRACK_CACHE_PATH, _track_cache)
    return _track_cache[k_t]

def prefetch_tags_for_keys(track_keys, mode=TAG_MODE, rps=PREFETCH_RPS, workers=PREFETCH_WORKERS):
    """
    Prefetch tags for missing (artist, album, track) keys with parallelism + rate limit.
    """
    to_fetch = []
    for (artist, album, track) in track_keys:
        if _track_key(artist, album, track) not in _track_cache:
            to_fetch.append((artist, album, track))
    if not to_fetch:
        return 0

    def _do(triple):
        a, al, t = triple
        # Each call internally may do 1–3 HTTP requests depending on mode.
        get_track_tags_fast(a, al, t, mode=mode)

    i, n = 0, len(to_fetch)
    while i < n:
        batch = to_fetch[i:i + rps]  # ~rps requests/sec
        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as ex:
            list(ex.map(_do, batch))
        i += len(batch)
        if i < n:
            time.sleep(1.05)  # gentle pacing
    return len(to_fetch)


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = '160e814675379ba7267c54a1ff597c97'     
USER = 'th_ma_' 
unmatched_tags_log = []
summary = {"countryCounts": {}, "countryTopArtists": {}}
artist_country_map = {}

def fetch_scrobble_page(page, params, url):
    '''
    Fetches a page of Last.fm scrobbles from the API.

    Args:
        page: Page number to fetch.
        params: Parameters to pass to the API request.
        url: URL of the Last.fm API endpoint.

    Returns:
        Page of scrobbles as a JSON object.
    '''
    params = params.copy()
    params['page'] = page
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def get_scrobbles_in_timeframe(start, end, max_workers=5):
    '''
    Returns all scrobbles (tracks + timestamps) within a specified time frame using Last.fm API.
    Fetches up to `max_workers` pages in parallel per second.
    '''

    # Convert start and end dates to datetime objects and set timezone to UTC
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    # Convert datetime objects to UNIX timestamps
    time_from = int(start_dt.timestamp())
    time_to = int(end_dt.timestamp())

    # Set the Last.fm API endpoint URL
    url = "https://ws.audioscrobbler.com/2.0/"

    # Define parameters for the API request
    params = {
        "method": "user.getRecentTracks",  # API method to get recent tracks
        "user": USER,                      # Last.fm username
        "api_key": API_KEY,                # Last.fm API key
        "format": "json",                  # Response format
        "limit": 200,                      # Maximum number of tracks per page
        "from": time_from,                 # Start timestamp
        "to": time_to                      # End timestamp
    }

    # Make an initial request to determine the total number of pages of results
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error if the request was unsuccessful
    data = response.json()       # Parse the response JSON

    # Extract the total number of pages from the response
    attr = data.get("recenttracks", {}).get("@attr", {})
    total_pages = int(attr.get("totalPages", 1))

    all_scrobbles = []  # Initialize a list to store all scrobbles

    # Extract tracks from the first page and add them to the list of scrobbles
    tracks = data.get("recenttracks", {}).get("track", [])
    for track in tracks:
        if "date" in track:
            scrobble = {
                "artist": track["artist"]["#text"],
                "track": track["name"],
                "album": track["album"]["#text"],
                "timestamp": int(track["date"]["uts"]),
                "datetime": track["date"]["#text"],
                "url": track["url"]
            }
            all_scrobbles.append(scrobble)

    # Prepare to fetch additional pages of scrobbles in parallel
    pages = list(range(2, total_pages + 1))

    # Process pages in batches using a thread pool
    for i in range(0, len(pages), max_workers):
        batch = pages[i:i+max_workers]  # Select a batch of pages to fetch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to fetch each page in the batch
            futures = [executor.submit(fetch_scrobble_page, page, params, url) for page in batch]
            for future in as_completed(futures):
                data = future.result()  # Get the result of the completed task
                tracks = data.get("recenttracks", {}).get("track", [])
                for track in tracks:
                    if "date" in track:
                        scrobble = {
                            "artist": track["artist"]["#text"],
                            "track": track["name"],
                            "album": track["album"]["#text"],
                            "timestamp": int(track["date"]["uts"]),
                            "datetime": track["date"]["#text"],
                            "url": track["url"]
                        }
                        all_scrobbles.append(scrobble)
        time.sleep(0.25)  # Sleep briefly to respect API rate limits

    return all_scrobbles  # Return the list of all scrobbles collected



# start_date = "2024-01-01"
# end_date = "2024-01-31"

# scrobbles = get_scrobbles_in_timeframe(start_date, end_date)
# print([scrobble['track'] for scrobble in scrobbles])
# print(len(scrobbles))

def identify_session_starts(scrobbles, inactivity_threshold_seconds=900):
    """
    Identify the start of listening sessions from Last.fm scrobble data.

    Args:
        scrobbles: List of Last.fm scrobble dicts (from get_scrobbles_in_timeframe)
        inactivity_threshold_seconds: Time gap defining new sessions (default: 900s/15min)

    Returns:
        tuple: (session_starts, is_session_start)
    """
    if not scrobbles:
        return [], []

    # Sort scrobbles chronologically
    sorted_scrobbles = sorted(scrobbles, key=lambda x: x['timestamp'])

    is_session_start = [False] * len(sorted_scrobbles)
    is_session_start[0] = True  # First scrobble always starts a session

    for i in range(1, len(sorted_scrobbles)):
        time_gap = sorted_scrobbles[i]['timestamp'] - sorted_scrobbles[i - 1]['timestamp']
        if time_gap > inactivity_threshold_seconds:
            is_session_start[i] = True

    # Pull the actual scrobbles that are session starts
    session_starts = [s for s, is_start in zip(sorted_scrobbles, is_session_start) if is_start]

    return session_starts, is_session_start



# # Get scrobbles from Last.fm API
# scrobbles = get_scrobbles_in_timeframe("2025-01-01", "2025-06-07")

# # Find listening sessions
# session_starts, is_start = identify_session_starts(scrobbles)

# # Print session starts
# print(f"Found {len(session_starts)} listening sessions:")
# for session in session_starts:
#     print(f"- {session['datetime']}: {session['artist']} - {session['track']}")

def most_frequent_titles(scrobbles):
    titles = [scrobble["track"] for scrobble in scrobbles]
    title_counts = Counter(titles)
    sorted_titles = title_counts.most_common()

    return sorted_titles

def split_into_sessions(scrobbles, inactivity_threshold_seconds=900):
    """
    Splits a list of scrobbles into listening sessions based on inactivity threshold.

    Args:
        scrobbles: List of scrobble dicts, each with a 'timestamp' key.
        inactivity_threshold_seconds: Time gap (in seconds) to separate sessions.

    Returns:
        List of sessions, where each session is a list of scrobbles.
    """
    if not scrobbles:
        return []

    # Sort scrobbles chronologically
    sorted_scrobbles = sorted(scrobbles, key=lambda x: x['timestamp'])

    sessions = []
    current_session = [sorted_scrobbles[0]]

    for i in range(1, len(sorted_scrobbles)):
        time_gap = sorted_scrobbles[i]['timestamp'] - sorted_scrobbles[i-1]['timestamp']
        if time_gap > inactivity_threshold_seconds:
            # Start new session
            sessions.append(current_session)
            current_session = [sorted_scrobbles[i]]
        else:
            # Continue current session
            current_session.append(sorted_scrobbles[i])

    # Add last session
    sessions.append(current_session)

    return sessions

def UTCtoLocal(utc_time):
    '''
    Convert UTC time to local machine time
    '''
    return utc_time.replace(tzinfo=timezone.utc).astimezone(tz=None)

def longest_listening_streak(scrobbles):
    longest_streak = {"streak": 0, "start_day": None, "end_day": None}

    all_listening_days = set(
        UTCtoLocal(datetime.strptime(scrobble["datetime"], "%d %b %Y, %H:%M")).date()
        for scrobble in scrobbles
    )

    earliest_date = min(all_listening_days)
    latest_date = max(all_listening_days)
    
    current_day = earliest_date
    current_streak = 0
    streak_start_day = None
    
    while current_day <= latest_date:
        if current_day in all_listening_days:
            if current_streak == 0:
                streak_start_day = current_day
            current_streak += 1
            if current_streak > longest_streak["streak"]:
                longest_streak["streak"] = current_streak
                longest_streak["start_day"] = streak_start_day
                longest_streak["end_day"] = current_day
        else:
            current_streak = 0
            streak_start_day = None

        current_day += timedelta(days=1)

    return longest_streak
    
def session_summary(sessions):
    return {
        "total_sessions": len(sessions),
        "average_session_length": sum(sess[-1]['timestamp'] - sess[0]['timestamp'] for sess in sessions) / len(sessions) / 60,
        "median_session_length": sorted([(sess[-1]['timestamp'] - sess[0]['timestamp']) / 60 for sess in sessions])[len(sessions)//2]
    }

def most_played_songs(scrobbles):
    titles = [scrobble["track"] for scrobble in scrobbles]
    title_counts = Counter(titles)
    sorted_titles = title_counts.most_common()

    return sorted_titles

def most_played_artists(scrobbles):
    artists = [scrobble["artist"] for scrobble in scrobbles]
    artist_counts = Counter(artists)
    sorted_artists = artist_counts.most_common()

    return sorted_artists

def most_played_albums(scrobbles):
    albums = [scrobble["album"] for scrobble in scrobbles]
    album_counts = Counter(albums)
    sorted_albums = album_counts.most_common()

    return sorted_albums

def all_artists(scrobbles):
    return {scrobble["artist"] for scrobble in scrobbles}

def artist_counts(scrobbles):
    artists = all_artists(scrobbles)
    artist_counts = Counter(artists)
    sorted_artists = artist_counts.most_common()

    return sorted_artists

import requests

unmatched_tags_log = []

def get_artist_info(artist_name):
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getInfo",
        "artist": artist_name,
        "api_key": API_KEY,
        "format": "json"
    }
    response = requests.get(url, params=params)
    return response.json().get("artist", {})

def get_track_tags(artist, album, track):
    def fetch_tags(method, extra_params):
        url = "https://ws.audioscrobbler.com/2.0/"
        params = {
            "method": method,
            "api_key": API_KEY,
            "artist": artist,
            "format": "json",
            **extra_params
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            tags = data.get('toptags', {}).get('tag', [])
            return {tag['name']: int(tag.get('count', 0)) for tag in sorted(tags, key=lambda tag: int(tag.get('count', 0)), reverse=True)[:3] if 'name' in tag}
        except Exception as e:
            print(f"Error fetching tags with {method}: {e}")
            return {}

    # 1. Try track tags with weights
    tags = fetch_tags("track.getTopTags", {"track": track})
    if tags:
        return tags

    # 2. Try album tags with weights if album is provided
    if album:
        tags = fetch_tags("album.getTopTags", {"album": album})
        if tags:
            return tags

    # 3. Fallback to artist tags with weights
    tags = fetch_tags("artist.getTopTags", {})
    return tags

def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def infer_country_from_tags_and_bio(artist_info):
    with open("backend/country_keywords.json", "r", encoding="utf-8") as f:
        country_keywords = json.load(f)

    # Extract and normalize tags
    raw_tags = artist_info.get('tags', {}).get('tag', [])
    tags = [normalize(tag['name']) for tag in raw_tags if 'name' in tag]

    # Check tags
    for country, keywords in country_keywords.items():
        norm_keywords = [normalize(keyword) for keyword in keywords]
        for tag in tags:
            if any(keyword in tag for keyword in norm_keywords):
                return country

    unmatched_tags_log.extend(tags)  # Optional

    earliest_pos = float('inf')
    best_match = None

    # Clean and normalize bio summary
    bio = artist_info.get('bio', {}).get('summary', '')
    norm_bio = normalize(bio)

    # Check bio
    for country, keywords in country_keywords.items():
        norm_keywords = [normalize(keyword) for keyword in keywords]
        for keyword in norm_keywords:
            # Use word-boundary-aware search
            match = re.search(rf'\b{re.escape(keyword)}\b', norm_bio)
            if match:
                pos = match.start()
                if pos < earliest_pos:
                    earliest_pos = pos
                    best_match = country

    return best_match if best_match else None

def get_artist_country(artist_name):
    artist_info = get_artist_info(artist_name)
    country = infer_country_from_tags_and_bio(artist_info)
    return country

def get_countries_for_artists_parallel(artist_names, max_workers=5):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_artist = {executor.submit(get_artist_country, artist): artist for artist in artist_names}
        for future in as_completed(future_to_artist, timeout=60):  # set timeout to 60 sec
            artist = future_to_artist[future]
            try:
                country = future.result(timeout=10)  # timeout per future
            except Exception as e:
                print(f"Error fetching country for {artist}: {e}")
                country = None
            results[artist] = country
    return results

def build_country_summary(tracks_df, artist_country_map):
    country_counts = defaultdict(int)
    country_top_artists = defaultdict(lambda: defaultdict(int))  # nested: country -> artist -> play count

    for _, row in tracks_df.iterrows():
        artist = row["artist"]
        country = artist_country_map.get(artist)
        if not country:
            continue
        play_count = len(row["timestamps"])
        country_counts[country] += play_count
        country_top_artists[country][artist] += play_count

    # Convert top artist dict to top 3 sorted list
    country_top3 = {
        country: sorted(artist_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        for country, artist_dict in country_top_artists.items()
    }

    # Strip play counts from top3 for display
    country_top3 = {
        country: [artist for artist, _ in top3]
        for country, top3 in country_top3.items()
    }

    return {
        "countryCounts": dict(country_counts),
        "countryTopArtists": country_top3
    }

def scrobbles_to_df(scrobbles):
    """
    Converts scrobbles to a DataFrame where each row represents a unique track
    (identified by artist + track name), and includes all scrobble timestamps.
    """
    track_dict = defaultdict(list)

    for s in scrobbles:
        artist = s['artist']
        album = s['album']
        track = s['track']
        timestamp = s['timestamp']
        if artist and track and timestamp and album:
            dt = datetime.utcfromtimestamp(timestamp)
            track_dict[(artist, track, album)].append(dt)

    data = [
        {
            'artist': artist,
            'track': track,
            'album': album,
            'timestamps': sorted(timestamps)
        }
        for (artist, track, album), timestamps in track_dict.items()
    ]

    return pd.DataFrame(data)

def compute_burn_scores(tracks_df, window_days=30, early_windows=3, late_windows=3):
    """
    Compute burn scores for tracks based on play counts over time.

    Parameters:
    - tracks_df: DataFrame with 'timestamps' as sorted list of datetime objects per row
    - window_days: size of each time window in days (default 30)
    - early_windows: number of initial windows to average for early plays
    - late_windows: number of final windows to average for late plays

    Returns:
    - DataFrame with extra columns: 'early_avg', 'late_avg', 'slope', 'burn_score'
    """

    results = []

    for idx, row in tracks_df.iterrows():
        timestamps = row['timestamps']
        if len(timestamps) < 2:
            results.append((0, 0, 0, 0))
            continue

        start = timestamps[0]
        end = timestamps[-1]
        total_days = (end - start).days + 1
        n_windows = max((total_days + window_days - 1) // window_days, 1)

        bins = [start + timedelta(days=i * window_days) for i in range(n_windows + 1)]

        counts = np.zeros(n_windows, dtype=int)
        ts_idx = 0
        for i in range(n_windows):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            count = 0
            while ts_idx < len(timestamps) and timestamps[ts_idx] < bin_end:
                if timestamps[ts_idx] >= bin_start:
                    count += 1
                ts_idx += 1
            counts[i] = count
        while ts_idx < len(timestamps):
            counts[-1] += 1
            ts_idx += 1

        early_avg = counts[:early_windows].mean() if len(counts) >= early_windows else counts.mean()
        late_avg = counts[-late_windows:].mean() if len(counts) >= late_windows else counts.mean()

        X = np.arange(n_windows).reshape(-1, 1)
        y = counts
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]

        burn_score = (late_avg / early_avg) * slope if early_avg > 0 else 0

        results.append((early_avg, late_avg, slope, burn_score))

    tracks_df = tracks_df.copy()
    tracks_df['early_avg'] = [r[0] for r in results]
    tracks_df['late_avg'] = [r[1] for r in results]
    tracks_df['slope'] = [r[2] for r in results]
    tracks_df['burn_score'] = [r[3] for r in results]

    return tracks_df

from datetime import datetime

def filter_track_df_by_time(tracks_df, local_start_hour, local_end_hour):
    """
    Filters tracks to only those with scrobbles between local_start_hour and local_end_hour (inclusive).
    Converts each UTC timestamp to local time using the machine’s timezone.
    Only retains timestamps that fall in that time window.
    """
    def timestamp_in_range(ts):
        ts_utc = ts.replace(tzinfo=pytz.UTC)  # Set timezone to UTC
        local_ts = ts_utc.astimezone()  # Converts to local time using system default
        h = local_ts.hour
        if local_start_hour <= local_end_hour:
            return local_start_hour <= h <= local_end_hour
        else:  # Handles cases like 22 to 2
            return h >= local_start_hour or h <= local_end_hour

    filtered_rows = []
    for _, row in tracks_df.iterrows():
        matching_times = [ts for ts in row['timestamps'] if timestamp_in_range(ts)]
        if matching_times:
            filtered_rows.append({
                'artist': row['artist'],
                'track': row['track'],
                'album': row['album'],
                'timestamps': matching_times
            })

    return pd.DataFrame(filtered_rows)


def filter_track_df_by_date(tracks_df, local_start_date, local_end_date):
    """
    Filter tracks to only those with scrobbles between ``local_start_date`` and
    ``local_end_date`` (inclusive).

    Each timestamp is assumed to be in UTC and is converted to the local
    timezone before the date portion is compared.  Rows with no timestamps in
    the range are dropped and remaining timestamps are trimmed to the range.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame where the ``timestamps`` column contains lists of ``datetime``
        objects in UTC.
    local_start_date, local_end_date : Union[str, datetime.date]
        Date range to filter on.  Strings are parsed as ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with timestamps limited to the specified date range.
    """

    # Ensure date objects
    start_date = pd.to_datetime(local_start_date).date()
    end_date = pd.to_datetime(local_end_date).date()

    local_tz = dateutil.tz.tzlocal()

    # Explode timestamps so each row corresponds to a single play
    exploded = tracks_df.explode("timestamps")

    # Convert to local dates
    local_dates = (
        exploded["timestamps"].dt.tz_localize(pytz.UTC)
        .dt.tz_convert(local_tz)
        .dt.date
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

def hour_pie_chart(scrobbles):
    '''
    Listening hours pie chart data
    
    '''
    hour_counts = {}
    for s in scrobbles:
        timestamp = s['timestamp']
        dt = datetime.utcfromtimestamp(timestamp)
        dt_local = dt.replace(tzinfo=pytz.UTC).astimezone()  # Convert to local time
        hour = dt_local.hour
        if hour in hour_counts:
            hour_counts[hour] += 1
        else:
            hour_counts[hour] = 1
    return hour_counts


def filtered_tag_scores_by_date(tracks_df, start_date, end_date, threshold=0.025):
    """
    Calculates weighted and normalized tag scores for scrobbled tracks in a date range.
    Filters out tags contributing less than the given threshold (default: 1%).
    Returns a dictionary of {tag: normalized_weight}.
    """
    tracks_df = filter_track_df_by_date(tracks_df, start_date, end_date)
    tag_totals = defaultdict(float)

    for _, row in tracks_df.iterrows():
        track = row['track']
        album = row.get('album')  # Allow for None
        artist = row['artist']
        tags = get_track_tags(artist, track, album)  # Returns {tag: weight}
        play_count = len(row['timestamps'])

        for tag, weight in tags.items():
            tag_totals[tag] += weight * play_count

    total_weight = sum(tag_totals.values())
    if total_weight == 0:
        return {}

    # Normalize scores to sum to 1
    normalized_tags = {
        tag: score / total_weight for tag, score in tag_totals.items()
    }

    # Filter out tags below threshold
    filtered_tags = {
        tag: weight for tag, weight in normalized_tags.items()
        if weight >= threshold
    }

    filtered_total = sum(filtered_tags.values())
    if filtered_total == 0:
        return {}

    # Re-normalize
    final_tags = {
        tag: weight / filtered_total for tag, weight in filtered_tags.items()
    }

    return final_tags

def filtered_tag_scores(tracks_df, start_hour, end_hour, threshold=0.025):
    """
    Calculates weighted and normalized tag scores for scrobbled tracks in a time window.
    Filters out tags contributing less than the given threshold (default: 1%).
    Returns a dictionary of {tag: normalized_weight}.
    """
    tracks_df = filter_track_df_by_time(tracks_df, start_hour, end_hour)
    tag_totals = defaultdict(float)

    for _, row in tracks_df.iterrows():
        track = row['track']
        album = row.get('album')  # Allow for None
        artist = row['artist']
        tags = get_track_tags(artist, track, album)  # Returns {tag: weight}
        play_count = len(row['timestamps'])

        for tag, weight in tags.items():
            tag_totals[tag] += weight * play_count

    total_weight = sum(tag_totals.values())
    if total_weight == 0:
        return {}

    # Normalize scores to sum to 1
    normalized_tags = {
        tag: score / total_weight for tag, score in tag_totals.items()
    }

    # Filter out tags below threshold
    filtered_tags = {
        tag: weight for tag, weight in normalized_tags.items()
        if weight >= threshold
    }

    filtered_total = sum(filtered_tags.values())
    if filtered_total == 0:
        return {}

    # Re-normalize
    final_tags = {
        tag: weight / filtered_total for tag, weight in filtered_tags.items()
    }

    return final_tags

LOCAL_TZ = pytz.timezone("America/New_York")

@app.get("/country-summary")
def get_country_summary():
    return summary

def background_country_updater():
    global summary, artist_country_map
    unresolved_artists = [a for a in tracks_df["artist"].unique() if a not in artist_country_map]
    print(f"⏳ Starting country resolution for {len(unresolved_artists)} artists")

    for i, artist in enumerate(unresolved_artists):
        try:
            country = get_artist_country(artist)
            artist_country_map[artist] = country
            print(f"✅ [{i}] {artist} → {country}")
        except Exception as e:
            print(f"❌ [{i}] {artist} failed: {e}")
            continue

        summary = build_country_summary(tracks_df, artist_country_map)
        print(f"📊 Summary updated: {len(summary['countryCounts'])} countries so far")
        time.sleep(1.5)


@lru_cache(maxsize=20000)
def get_cached_track_tags(artist, album, track):
    # IMPORTANT: your get_track_tags signature is (artist, album, track)
    return get_track_tags(artist, album, track) or {}

def _dedupe_scrobbles(scrobbles):
    seen = {}
    for s in scrobbles:
        key = (s.get("artist",""), s.get("track",""), s.get("album",""), int(s.get("timestamp",0)))
        if key not in seen:
            seen[key] = s
    return sorted(seen.values(), key=lambda x: x["timestamp"])

def update_scrobbles_file(json_path="scrobbles_2025.json", default_start="2025-01-01"):
    # Load existing
    existing = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Decide fetch window in UTC (safe for API)
    if existing:
        last_ts = max(int(s["timestamp"]) for s in existing)
        start_date = datetime.utcfromtimestamp(last_ts).strftime("%Y-%m-%d")
    else:
        start_date = default_start
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # Fetch & write
    new_scrobbles = get_scrobbles_in_timeframe(start_date, end_date) if start_date <= end_date else []
    combined = _dedupe_scrobbles(existing + new_scrobbles)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"📝 Updated {json_path}: {len(existing)} → {len(combined)} scrobbles.")
    return combined

def _week_key_local(ts_utc_int):
    """Return ISO week key like '2025-W23' using local time."""
    dt_local = (
        datetime.utcfromtimestamp(int(ts_utc_int))
        .replace(tzinfo=pytz.UTC)
        .astimezone(LOCAL_TZ)
    )
    iso_year, iso_week, _ = dt_local.isocalendar()
    return f"{iso_year}-W{iso_week:02d}", iso_year

def _normalize_and_threshold(tag_totals, threshold=0.025):
    total = sum(tag_totals.values())
    if total <= 0:
        return {}
    normalized = {t: v/total for t, v in tag_totals.items()}
    filtered = {t: w for t, w in normalized.items() if w >= threshold}
    ssum = sum(filtered.values())
    if ssum <= 0:
        return {}
    # sort high→low
    return dict(sorted(((t, w/ssum) for t, w in filtered.items()), key=lambda kv: kv[1], reverse=True))

from datetime import date  # at top of file if not already imported

from datetime import date

def compute_weekly_tag_distributions_from_scrobbles(
    scrobbles, year_filter=2025, threshold=0.025, progress=True, top_k=3
):
    # 0) Gather unique tracks for this year and prefetch tags
    unique_keys = set()
    for s in scrobbles:
        wk, iso_year = _week_key_local(s["timestamp"])
        if iso_year == year_filter:
            unique_keys.add((s.get("artist",""), s.get("album",""), s.get("track","")))
    fetched = prefetch_tags_for_keys(unique_keys)
    if progress:
        print(f"⚡ Prefetched tags for {fetched} tracks (mode={TAG_MODE}).")

    # 1) Count scrobbles per week and per (artist, album, track)
    week_track_counts = defaultdict(Counter)
    week_scrobble_counts = defaultdict(int)
    for s in scrobbles:
        wk, iso_year = _week_key_local(s["timestamp"])
        if iso_year != year_filter:
            continue
        key = (s.get("artist",""), s.get("album",""), s.get("track",""))
        week_track_counts[wk][key] += 1
        week_scrobble_counts[wk] += 1

    # 2) Accumulate totals with cached lookups
    weekly = {}
    for wk, counter in sorted(week_track_counts.items()):
        tag_totals = defaultdict(float)
        for (artist, album, track), count in counter.items():
            weights = get_track_tags_fast(artist, album, track)  # FAST + persistent
            for tag, w in weights.items():
                tag_totals[tag] += float(w) * count

        tags_norm = _normalize_and_threshold(tag_totals, threshold=threshold)
        weekly[wk] = {"tags": tags_norm, "scrobbles": week_scrobble_counts[wk]}

        if progress:
            try:
                yr = int(wk[:4]); wknum = int(wk.split("-W")[1])
                wk_start = date.fromisocalendar(yr, wknum, 1)
                wk_end   = date.fromisocalendar(yr, wknum, 7)
                top_items = list(tags_norm.items())[:top_k]
                top_str = ", ".join(f"{t} {w*100:.1f}%" for t, w in top_items) if top_items else "no dominant tags"
                print(f"✅ {wk}  {wk_start:%b %d}–{wk_end:%b %d}  —  {top_str}  |  scrobbles: {week_scrobble_counts[wk]}", flush=True)
            except Exception:
                print(f"✅ {wk} — scrobbles: {week_scrobble_counts[wk]}", flush=True)

    return weekly


def main():
    """
    1) Update scrobbles_2025.json to current day (UTC).
    2) Compute weekly tag distributions for ISO weeks in 2025 (local time).
    3) Refresh tracks_df and start background country updater.
    """
    global tracks_df

    # 1) Update JSON
    scrobbles = update_scrobbles_file("scrobbles_2025.json", default_start="2025-01-01")

    # 2) Weekly tag distributions (kept in a separate JSON)
    print("📊 Computing weekly tag distributions for 2025 (local ISO weeks)...")
    weekly = compute_weekly_tag_distributions_from_scrobbles(scrobbles, threshold=0.025, year_filter=2025)

    out_path = "tag_distributions_weekly_2025.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(weekly, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(weekly)} weekly distributions → {out_path}")

    # 3) Refresh DataFrame and kick off country updater (optional but keeps your API consistent)
    tracks_df = scrobbles_to_df(scrobbles)
    #Thread(target=background_country_updater, daemon=True).start()
    #print("🚀 Background country updater started")

if __name__ == "__main__":
    main()

   
@app.on_event("startup")
def start_up():
    global tracks_df
    with open("scrobbles_2025.json", "r") as f:
        scrobbles = json.load(f)
    tracks_df = scrobbles_to_df(scrobbles)
    print(f"✅ Loaded {len(scrobbles)} scrobbles into DataFrame")

    Thread(target=background_country_updater, daemon=True).start()
    print("🚀 Background country updater started")
