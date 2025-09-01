import requests
import re
import string
import json
from collections import Counter, defaultdict
import pytz
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import dateutil.tz
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread

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

# TODO: Address UTC -> local time issue

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



def main():
    # with open("scrobbles_2025.json", "r", encoding="utf-8") as f:
    #     scrobbles = json.load(f)

    # with open("scrobbles_2025.json", "w", encoding="utf-8") as f:
    #     json.dump(scrobbles, f, ensure_ascii=False, indent=2)
    # start_date = "2025-06-06"
    # end_date = "2025-06-12"

    # scrobbles = get_scrobbles_in_timeframe(start_date, end_date)
    # # print(f"Found {len(scrobbles)} scrobbles")

    # session_starts, _ = identify_session_starts(scrobbles)

    # sessions = split_into_sessions(scrobbles, inactivity_threshold_seconds=900)
    # print(f"Total sessions found: {len(sessions)}")

    # longest = max(
    #     (sess[-1]['timestamp'] - sess[0]['timestamp'] + 900)
    #     for sess in sessions
    # )
    # print(f"Longest session: {longest / 60:.1f} minutes")

    # streak = longest_listening_streak(scrobbles)
    # print(f"Longest streak: {streak['streak']} days from {streak['start_day']} to {streak['end_day']}")

    # s_summary = session_summary(sessions)
    # print(f"Total sessions: {s_summary['total_sessions']}")
    # print(f"Average session length: {s_summary['average_session_length']:.1f} minutes")
    # print(f"Median session length: {s_summary['median_session_length']:.1f} minutes")

    #tracks_df = scrobbles_to_df(scrobbles)
    # print(compute_burn_scores(tracks_df, window_days=15, early_windows=3, late_windows=3).sort_values(by='burn_score', ascending=False).head(10))
    # print(f'Most Played Song: {most_played_songs(scrobbles)[:1]}')
    # print(f'Most Played Artist: {most_played_artists(scrobbles)[:1]}')
    # print(f'Most Played Album: {most_played_albums(scrobbles)[:1]}')

    global tracks_df
    with open("scrobbles_2025.json", "r") as f:
        scrobbles = json.load(f)

    tracks_df = scrobbles_to_df(scrobbles)

    # Start the background thread
    Thread(target=background_country_updater, daemon=True).start()

   
@app.on_event("startup")
def start_up():
    global tracks_df
    with open("scrobbles_2025.json", "r") as f:
        scrobbles = json.load(f)
    tracks_df = scrobbles_to_df(scrobbles)
    print(f"✅ Loaded {len(scrobbles)} scrobbles into DataFrame")

    Thread(target=background_country_updater, daemon=True).start()
    print("🚀 Background country updater started")
