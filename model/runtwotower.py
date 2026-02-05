"""
REQUIRES:
- online_user_model.py
- artifacts/movie_map.json
- processed/movies_filtered.csv
- processed/movie_metadata.csv
- processed/movie_popularity.json
- config/recommender_policy.yaml

DESCRIPTION:
Interactive movie recommendation system using a trained two-tower model.
Movie embeddings are extracted from the item tower.
User preferences are updated online during the session.
"""

import os
import json
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
import numpy as np
import pandas as pd
from collections import Counter

from online_user_model import OnlineUserModel

# paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "twotower_model.pt")


DEVICE = "cpu"

# config
with open(os.path.join(CONFIG_DIR, "recommender_policy.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)
º
with open(os.path.join(ARTIFACTS_DIR, "movie_map.json")) as f:
    movie_map = {int(k): v for k, v in json.load(f).items()}

inv_movie_map = {v: k for k, v in movie_map.items()}
num_items = len(inv_movie_map)

# load data
movies = pd.read_csv(
    os.path.join(PROCESSED_DIR, "movies_filtered.csv")
)
movies["genres"] = movies["genres"].fillna("").str.split("|")

metadata = pd.read_csv(
    os.path.join(PROCESSED_DIR, "movie_metadata.csv")
).set_index("movieId")

with open(os.path.join(PROCESSED_DIR, "movie_popularity.json")) as f:
    popularity = {int(k): int(v) for k, v in json.load(f).items()}

max_pop = max(popularity.values())

# loadmodel
print("Loading two-tower model (item tower only)...")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# infer embedding dim and number of items FROM CHECKPOINT
embed_dim = checkpoint["item_embedding.weight"].shape[1]
num_items = checkpoint["item_embedding.weight"].shape[0]

print(f"Checkpoint items: {num_items}, embed_dim: {embed_dim}")

# create a minimal item-tower-only model
class ItemTower(nn.Module):
    def __init__(self, num_items, embed_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, item_ids):
        x = self.item_embedding(item_ids)
        return self.item_tower(x)


item_model = ItemTower(num_items, embed_dim)
item_model.load_state_dict({
    "item_embedding.weight": checkpoint["item_embedding.weight"],
    "item_tower.0.weight": checkpoint["item_tower.0.weight"],
    "item_tower.0.bias": checkpoint["item_tower.0.bias"],
})

item_model.eval()

print("Extracting movie embeddings...")

with torch.no_grad():
    item_ids = torch.arange(num_items, dtype=torch.long)
    movie_embeddings = item_model(item_ids).numpy()

print("Movie embeddings ready:", movie_embeddings.shape)

print("\nEnter exactly 4 favorite movies (exact titles):\n")

favorite_titles = []
while len(favorite_titles) < 4:
    title = input(f"{len(favorite_titles)+1}. ").strip()
    if title:
        favorite_titles.append(title)

favorites = movies[
    movies["title"].str.lower().isin([t.lower() for t in favorite_titles])
]

if len(favorites) < 4:
    raise ValueError("One or more favorite titles not found.")

print("\nMatched favorites:")
print(favorites[["movieId", "title"]])

genre_counter = Counter()

for genres in favorites["genres"]:
    for g in genres:
        genre_counter[g] += 1

total_favs = len(favorites)

user_genre_weights = {
    g: count / total_favs
    for g, count in genre_counter.items()
}

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

# loop
print("\n--- Interactive Recommendation Session ---")
print("y = like | n = dislike | s = skip | i = info | q = quit")

step = 1

while True:
    raw_scores = user.user_embedding @ movie_embeddings.T
    ranked = []

    for idx, base_score in enumerate(raw_scores):

        if idx not in inv_movie_map:
            continue

        if idx in user.seen_movies:
            continue

        movie_id = inv_movie_map[idx]
        row = movies[movies["movieId"] == movie_id]
        if row.empty:
            continue

        movie_genres = row.iloc[0]["genres"]

        affinity = sum(user_genre_weights.get(g, 0.0) for g in movie_genres)
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
        print("\n⚠ No more valid recommendations.")
        break

    ranked.sort(key=lambda x: x[2], reverse=True)

    rec_idx, movie_id, _ = ranked[0]
    row = movies[movies["movieId"] == movie_id].iloc[0]

    print(f"\nRecommendation #{step}")
    print(row["title"])

    while True:
        fb = input("Feedback (y/n/s/i/q): ").strip().lower()

        if fb == "i":
            if movie_id in metadata.index:
                meta = metadata.loc[movie_id]
                print("\nMOVIE INFO")
                print("Title:", meta["title"])
                print("Release date:", meta["release_date"])
                print("Rating:", meta["tmdb_vote_avg"])
                print("Popularity:", meta["tmdb_popularity"])
                print("\nOverview:")
                print(meta["overview"])
            else:
                print("No metadata available.")
            print("\n---")
            continue

        if fb == "q":
            print("\nProgram closed.")
            exit()

        if fb in ("y", "n", "s"):
            if fb == "y":
                user.update(rec_idx, feedback=1)
                print(":) liked")
            elif fb == "n":
                user.update(rec_idx, feedback=-1)
                print(":( disliked")
            else:
                user.update(rec_idx, feedback=0)
                print("skipped")
            break

        print("Invalid input.")

    step += 1
