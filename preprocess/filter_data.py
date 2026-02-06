"""
REQUIRES:
- data/ratings.csv
- data/movies.csv

CREATES:
- processed/ratings_filtered.csv
- processed/movies_filtered.csv

DESCRIPTION:
Loads raw MovieLens data, fixes timestamps, filters inactive users
and unpopular movies, and outputs cleaned datasets for downstream
processing.
"""

import pandas as pd
import os

# ------------------
# paths
# ------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INPUT_RATINGS = os.path.join(DATA_DIR, "ratings.csv")
INPUT_MOVIES = os.path.join(DATA_DIR, "movies.csv")

OUTPUT_RATINGS = os.path.join(PROCESSED_DIR, "ratings_filtered.csv")
OUTPUT_MOVIES = os.path.join(PROCESSED_DIR, "movies_filtered.csv")

# ------------------
# changeable config
# ------------------
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 50

# ------------------
# load raw data
# ------------------
print("Loading raw data")

ratings = pd.read_csv(INPUT_RATINGS)
movies = pd.read_csv(INPUT_MOVIES)

print("Original ratings shape:", ratings.shape)
print("Original movies shape:", movies.shape)

# ------------------
# fixing timestamps
# ------------------
print("Fixing timestamps")

if ratings["timestamp"].dtype != "int64":
    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"],
        errors="coerce"
    )
else:
    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        errors="coerce"
    )

ratings = ratings.dropna(subset=["timestamp"])

# ------------------
# filter inactive users
# ------------------
print("Filtering inactive users")

user_counts = ratings["userId"].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
ratings = ratings[ratings["userId"].isin(active_users)]

print("After user filtering:", ratings.shape)

# ------------------
# filter unpopular movies
# ------------------
print("Filtering unpopular movies")

movie_counts = ratings["movieId"].value_counts()
popular_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
ratings = ratings[ratings["movieId"].isin(popular_movies)]

print("After movie filtering:", ratings.shape)

movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]
print("Filtered movies shape:", movies.shape)

# ------------------
# sanity checks
# ------------------
print("\nSanity checks:")
print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", ratings["movieId"].nunique())
print("Rating range:", ratings["rating"].min(), "-", ratings["rating"].max())
print("Time range:", ratings["timestamp"].min(), "->", ratings["timestamp"].max())

# ------------------
# save outputs
# ------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)

ratings.to_csv(OUTPUT_RATINGS, index=False)
movies.to_csv(OUTPUT_MOVIES, index=False)

print("\nDone.")
print(f"Saved: {OUTPUT_RATINGS}")
print(f"Saved: {OUTPUT_MOVIES}")
