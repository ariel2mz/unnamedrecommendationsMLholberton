"""
REQUIRES:
- online_user_model.py
- artifacts/movie_embeddings.npy
- artifacts/movie_map.json
- processed/movies_filtered.csv
- processed/movie_metadata.csv

DESCRIPTION:
Interactive movie recommendation system with online learning.
Pure embedding-based recommendations (policy disabled).
User feedback updates the user embedding in real time.
"""

import os
import json
import numpy as np
import pandas as pd

from online_user_model import OnlineUserModel

# Color definitions
COLOR = {
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BLUE": "\033[94m",
    "BOLD": "\033[1m",
    "END": "\033[0m"
}

# paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# load artifacts
movie_embeddings = np.load(
    os.path.join(ARTIFACTS_DIR, "movie_embeddings.npy")
)

with open(os.path.join(ARTIFACTS_DIR, "movie_map.json")) as f:
    movie_map = {int(k): v for k, v in json.load(f).items()}

inv_movie_map = {v: k for k, v in movie_map.items()}

movies = pd.read_csv(
    os.path.join(PROCESSED_DIR, "movies_filtered.csv")
)
movies["genres"] = movies["genres"].fillna("").str.split("|")

metadata = pd.read_csv(
    os.path.join(PROCESSED_DIR, "movie_metadata.csv")
).set_index("movieId")

# ---- FAVORITE MOVIES ----
print(f"\n{COLOR['CYAN']}{COLOR['BOLD']}Enter exactly 4 favorite movies (exact titles):{COLOR['END']}\n")

favorite_titles = []
while len(favorite_titles) < 4:
    title = input(f"{COLOR['YELLOW']}{len(favorite_titles)+1}. {COLOR['END']}").strip()
    if title:
        favorite_titles.append(title)

favorites = movies[
    movies["title"].str.lower().isin(
        [t.lower() for t in favorite_titles]
    )
]

if len(favorites) < 4:
    raise ValueError(
        f"{COLOR['RED']}One or more titles not found. Check movies_filtered.csv{COLOR['END']}"
    )

print(f"\n{COLOR['GREEN']}Matched favorites:{COLOR['END']}")
print(favorites[["movieId", "title"]])

# ---- INIT USER MODEL ----
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

# ---- INTERACTIVE LOOP ----
print(f"\n{COLOR['CYAN']}{COLOR['BOLD']}--- Interactive Recommendation Session ---{COLOR['END']}")
print(f"{COLOR['YELLOW']}y = like | n = dislike | s = skip | i = info | q = quit{COLOR['END']}")

step = 1

while True:
    scores = user.user_embedding @ movie_embeddings.T

    # never recommend seen movies
    for idx in user.seen_movies:
        scores[idx] = -np.inf

    if np.all(np.isneginf(scores)):
        print(f"\n{COLOR['YELLOW']}No more valid recommendations.{COLOR['END']}")
        break

    rec_idx = int(np.argmax(scores))
    movie_id = inv_movie_map[rec_idx]

    row = movies[movies["movieId"] == movie_id].iloc[0]
    title = row["title"]

    print(f"\n{COLOR['CYAN']}{COLOR['BOLD']}Recommendation #{step}{COLOR['END']}")
    print(f"{COLOR['GREEN']} {title}{COLOR['END']}")

    while True:
        fb = input(f"{COLOR['YELLOW']}Feedback (y/n/s/i/q): {COLOR['END']}").strip().lower()

        if fb == "i":
            print(f"\n{COLOR['BLUE']}{COLOR['BOLD']}MOVIE INFO{COLOR['END']}")
            if movie_id in metadata.index:
                meta = metadata.loc[movie_id]
                print(f"{COLOR['CYAN']}Title:{COLOR['END']}", meta["title"])
                print(f"{COLOR['CYAN']}Release date:{COLOR['END']}", meta["release_date"])
                print(f"{COLOR['CYAN']}Rating:{COLOR['END']}", meta["tmdb_vote_avg"])
                print(f"\n{COLOR['CYAN']}Overview:{COLOR['END']}")
                print(meta["overview"])
            else:
                print(f"{COLOR['YELLOW']}No metadata available.{COLOR['END']}")
            print(f"\n{COLOR['BLUE']}---{COLOR['END']}")
            continue

        if fb == "q":
            print(f"\n{COLOR['RED']}Program closed.{COLOR['END']}")
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

        print(f"{COLOR['RED']}Invalid input.{COLOR['END']}")

    step += 1
