import pandas as pd
import requests
import time
from tqdm import tqdm

TMDB_API_KEY = "REPLACE THIS WITH YOUR API KEY!"
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"

LINKS_PATH = "data/links.csv"
MOVIES_PATH = "processed/movies_filtered.csv"
OUTPUT_PATH = "processed/movie_metadata.csv"

print("Loading MovieLens links")
links = pd.read_csv(LINKS_PATH)
movies = pd.read_csv(MOVIES_PATH)

data = movies.merge(links, on="movieId", how="left")
data = data.dropna(subset=["tmdbId"])

rows = []

print("Fetching TMDB metadata")

for _, row in tqdm(data.iterrows(), total=len(data)):
    tmdb_id = int(row["tmdbId"])

    url = f"{TMDB_BASE_URL}/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY}

    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200:
            continue

        m = r.json()

        rows.append({
            "movieId": row["movieId"],
            "title": m.get("title"),
            "overview": m.get("overview"),
            "poster_url": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get("poster_path") else None,
            "backdrop_url": f"https://image.tmdb.org/t/p/w780{m.get('backdrop_path')}" if m.get("backdrop_path") else None,
            "release_date": m.get("release_date"),
            "tmdb_popularity": m.get("popularity"),
            "tmdb_vote_avg": m.get("vote_average"),
        })

        time.sleep(0.02)

    except Exception:
        continue

meta_df = pd.DataFrame(rows)
meta_df.to_csv(OUTPUT_PATH, index=False)

print("Saved metadata:", OUTPUT_PATH)
