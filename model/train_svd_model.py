"""
Offline training script for the collaborative filtering recommender.
"""

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import os
import json
import joblib

# paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_transformed.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val_transformed.csv")

# load
print("Loading transformed data")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

# ------------------
# encode IDs (TRAIN ONLY)
# ------------------
print("Encoding IDs")

user_ids = train["userId"].unique()
movie_ids = train["movieId"].unique()

user_map = {int(u): int(i) for i, u in enumerate(user_ids)}
movie_map = {int(m): int(i) for i, m in enumerate(movie_ids)}

train["user_idx"] = train["userId"].map(user_map)
train["movie_idx"] = train["movieId"].map(movie_map)

val = val[
    val["userId"].isin(user_map) &
    val["movieId"].isin(movie_map)
].copy()

val["user_idx"] = val["userId"].map(user_map)
val["movie_idx"] = val["movieId"].map(movie_map)

n_users = len(user_map)
n_movies = len(movie_map)

print(f"Users: {n_users} | Movies: {n_movies}")

# ------------------
# aggregate duplicates
# ------------------
print("Aggregating duplicate interactions")

train_agg = (
    train
    .groupby(["user_idx", "movie_idx"], as_index=False)
    ["final_rating"]
    .mean()
)

# ------------------
# build sparse matrix
# ------------------
print("Building sparse interaction matrix")

R_sparse = coo_matrix(
    (
        train_agg["final_rating"].values,
        (train_agg["user_idx"].values, train_agg["movie_idx"].values)
    ),
    shape=(n_users, n_movies)
).tocsr()

# ------------------
# train SVD
# ------------------
print("Training SVD model")

svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

user_embeddings = svd.fit_transform(R_sparse)
movie_embeddings = svd.components_.T

# ------------------
# normalize embeddings
# ------------------
user_embeddings /= np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-8
movie_embeddings /= np.linalg.norm(movie_embeddings, axis=1, keepdims=True) + 1e-8

# ------------------
# validation
# ------------------
print("Evaluating on validation set")

batch_size = 100_000
preds = np.empty(len(val), dtype=np.float32)

for start in range(0, len(val), batch_size):
    end = start + batch_size

    u_idx = val["user_idx"].values[start:end]
    m_idx = val["movie_idx"].values[start:end]

    preds[start:end] = np.einsum(
        "ij,ij->i",
        user_embeddings[u_idx],
        movie_embeddings[m_idx]
    )

rmse = np.sqrt(mean_squared_error(val["final_rating"].values, preds))
print("Validation RMSE:", rmse)

# ------------------
# save artifacts
# ------------------
print("\nSaving model artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

with open(os.path.join(ARTIFACT_DIR, "user_map.json"), "w") as f:
    json.dump({str(k): v for k, v in user_map.items()}, f)

with open(os.path.join(ARTIFACT_DIR, "movie_map.json"), "w") as f:
    json.dump({str(k): v for k, v in movie_map.items()}, f)

joblib.dump(svd, os.path.join(ARTIFACT_DIR, "svd_model.joblib"))
np.save(os.path.join(ARTIFACT_DIR, "user_embeddings.npy"), user_embeddings)
np.save(os.path.join(ARTIFACT_DIR, "movie_embeddings.npy"), movie_embeddings)

print("Artifacts saved successfully.")
