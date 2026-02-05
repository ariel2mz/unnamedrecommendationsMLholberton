"""
REQUIRES:
- online_user_model.py
- artifacts/movie_embeddings.npy
- artifacts/movie_map.json
- processed/movies_filtered.csv
- processed/movie_metadata.csv
- processed/movie_popularity.json
- config/recommender_policy.yaml

DESCRIPTION:
Interactive movie recommendation system with online learning.
The user selects 4 favorite movies to initialize a personal taste profile.
Recommendations are generated using pretrained movie embeddings, genre affinity,
popularity penalties, and an exploration policy. User feedback (like/dislike/skip)
updates the user embedding in real time, adapting future recommendations during
the same session.
"""



import os
import json
import random
import yaml
import numpy as np
import pandas as pd
from collections import Counter

from online_user_model import OnlineUserModel

# Color definitions
COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "END": "\033[0m"
}

# paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
metadata = pd.read_csv(
    os.path.join(PROCESSED_DIR, "movie_metadata.csv")
)

metadata = metadata.set_index("movieId")

# config
with open(os.path.join(CONFIG_DIR, "recommender_policy.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

# artifacts
movie_embeddings = np.load(
    os.path.join(ARTIFACTS_DIR, "movie_embeddings.npy")
)

with open(os.path.join(ARTIFACTS_DIR, "movie_map.json")) as f:
    movie_map = {int(k): v for k, v in json.load(f).items()}

inv_movie_map = {v: k for k, v in movie_map.items()}

with open(os.path.join(PROCESSED_DIR, "movie_popularity.json")) as f:
    popularity = {int(k): int(v) for k, v in json.load(f).items()}

max_pop = max(popularity.values())

movies = pd.read_csv(
    os.path.join(PROCESSED_DIR, "movies_filtered.csv")
)
movies["genres"] = movies["genres"].fillna("").str.split("|")

# 4 favorite movie start
print(f"\n{COLOR['CYAN']}{COLOR['BOLD']}Enter exactly 4 favorite movies (exact titles):{COLOR['END']}\n")

favorite_titles = []
while len(favorite_titles) < 4:
    title = input(f"{COLOR['YELLOW']}{len(favorite_titles)+1}. {COLOR['END']}").strip()
    if title:
        favorite_titles.append(title)

# make sure program recognizes the titles
favorites = movies[
    movies["title"].str.lower().isin(
        [t.lower() for t in favorite_titles]
    )
]

if favorites.empty or len(favorites) < 4:
    raise ValueError(f"{COLOR['RED']}One or more favorite titles not found. I suggest looking for the exact movie name in the file 'processed/movie_features.csv' (Ctrl + F){COLOR['END']}")

print(f"\n{COLOR['GREEN']}Matched favorites:{COLOR['END']}")
print(favorites[["movieId", "title"]])

# genre profile
genre_counter = Counter()

for genres in favorites["genres"]:
    for g in genres:
        genre_counter[g] += 1

total_favs = len(favorites)

user_genre_weights = {
    g: count / total_favs
    for g, count in genre_counter.items()
}

# init online model
movie_indices = [
    movie_map[mid]
    for mid in favorites["movieId"]
    if mid in movie_map
]

user = OnlineUserModel(
    movie_embeddings=movie_embeddings,
    lr=0.15,
    negative_weight=0.6
)

user.initialize_from_movies(np.array(movie_indices))

# loop for input
print(f"\n{COLOR['CYAN']}{COLOR['BOLD']}--- Interactive Recommendation Session ---{COLOR['END']}")
print(f"{COLOR['YELLOW']}y = like | n = dislike | s = skip | i = info | q = quit{COLOR['END']}")

step = 1

while True:
    raw_scores = user.user_embedding @ movie_embeddings.T
    ranked = []

    for idx, base_score in enumerate(raw_scores):
        if idx in user.seen_movies:
            continue

        movie_id = inv_movie_map[idx]
        row = movies[movies["movieId"] == movie_id]
        if row.empty:
            continue

        movie_genres = row.iloc[0]["genres"]

        affinity = sum(
            user_genre_weights.get(g, 0.0)
            for g in movie_genres
        )

        if movie_genres:
            affinity /= len(movie_genres)

        genre_distance = 1.0 - affinity

        if affinity < CONFIG["min_genre_affinity"]:
            continue

        if genre_distance > CONFIG["max_genre_distance"]:
            continue

        score = base_score * CONFIG["score_weight_model"]
        score -= genre_distance * CONFIG["genre_distance_penalty"]

        pop = popularity.get(movie_id, 1)
        score -= (pop / max_pop) * CONFIG["popularity_penalty"]

        if 0 < affinity < CONFIG["diversity_affinity_cap"]:
            score += CONFIG["diversity_boost"]

        if random.random() < CONFIG["exploration_rate"]:
            score *= 0.9

        ranked.append((idx, movie_id, score))

    if not ranked:
        print(f"\n{COLOR['YELLOW']}No more valid recommendations.{COLOR['END']}")
        break

    ranked.sort(key=lambda x: x[2], reverse=True)

    rec_idx, movie_id, _ = ranked[0]
    row = movies[movies["movieId"] == movie_id].iloc[0]

    title = row["title"]

    print(f"\n{COLOR['CYAN']}{COLOR['BOLD']}Recommendation #{step}{COLOR['END']}")
    print(f"{COLOR['GREEN']} {title}{COLOR['END']}")

    # does the user like the movie
    while True:
        fb = input(f"{COLOR['YELLOW']}Feedback (y/n/s/i/q): {COLOR['END']}").strip().lower()

        if fb == "i":
            print(f"\n{COLOR['BLUE']}{COLOR['BOLD']}MOVIE INFO{COLOR['END']}")

            if movie_id in metadata.index:
                meta = metadata.loc[movie_id]

                print(f"{COLOR['CYAN']}Title:{COLOR['END']}", meta["title"])
                print(f"{COLOR['CYAN']}Release date:{COLOR['END']}", meta["release_date"])
                print(f"{COLOR['CYAN']}Rating:{COLOR['END']}", meta["tmdb_vote_avg"])
                print(f"{COLOR['CYAN']}Popularity:{COLOR['END']}", meta["tmdb_popularity"])

                print(f"\n{COLOR['CYAN']}Overview:{COLOR['END']}")
                print(meta["overview"])
            else:
                print(f"{COLOR['YELLOW']}No metadata available for this movie.{COLOR['END']}")

            print(f"\n{COLOR['BLUE']}---{COLOR['END']}")
            continue  # ask again

        if fb == "q":
            print(f"\n{COLOR['RED']}Program Closed{COLOR['END']}")
            exit()

        if fb in ("y", "n", "s"):
            if fb == "y":
                user.update(rec_idx, feedback=1)
                print(f"{COLOR['GREEN']}:) liked{COLOR['END']}")
            elif fb == "n":
                user.update(rec_idx, feedback=-1)
                print(f"{COLOR['RED']}:( disliked{COLOR['END']}")
            else:
                user.update(rec_idx, feedback=0)
                print(f"{COLOR['YELLOW']}skipped{COLOR['END']}")
            break

        print(f"{COLOR['RED']}Invalid input, try again.{COLOR['END']}")

    step += 1