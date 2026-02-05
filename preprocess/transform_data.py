"""
REQUIRES:
- processed/train.csv
- processed/val.csv
- processed/test.csv

CREATES:
- processed/train_transformed.csv
- processed/val_transformed.csv
- processed/test_transformed.csv
- processed/user_means.json
- processed/movie_popularity.json
- processed/train_max_timestamp.txt

DESCRIPTION:
This script prepares the ratings data so it is more meaningful for training
a recommender system.

Instead of using raw star ratings, it transforms them to better represent
real user preference by applying three ideas:
1) User normalization:
   Different users rate differently (some give high scores to everything,
   others are more strict). For each user, the script subtracts their average
   rating from each rating, so the model learns whether a movie was liked
   more or less than that user’s usual taste.
2) Popularity down-weighting:
   Very popular movies get many ratings and can dominate the model.
   To avoid this, ratings for popular movies are reduced in strength,
   while less popular movies keep more influence.
3) Time decay:
   Older ratings are less relevant than recent ones. Ratings are weighted
   so that newer interactions matter more, and very old ones slowly lose
   influence.

The output is a transformed version of the train, validation, and test
datasets with a final weighted rating that is better suited for learning.
"""

import pandas as pd
import numpy as np
import json
import os

# config
TRAIN_PATH = "processed/train.csv"
VAL_PATH = "processed/val.csv"
TEST_PATH = "processed/test.csv"

OUTPUT_TRAIN = "processed/train_transformed.csv"
OUTPUT_VAL = "processed/val_transformed.csv"
OUTPUT_TEST = "processed/test_transformed.csv"

USER_MEANS_PATH = "processed/user_means.json"
MOVIE_POP_PATH = "processed/movie_popularity.json"
MAX_TIME_PATH = "processed/train_max_timestamp.txt"

POPULARITY_ALPHA = 0.5
TIME_DECAY_HALF_LIFE_DAYS = 365

# load
print("Loading split data")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

for df in [train, val, test]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# find the numbers
print("Fitting user mean ratings (train only)")
user_means = train.groupby("userId")["rating"].mean()

print("Fitting movie popularity (train only)")
movie_counts = train["movieId"].value_counts()

print("Getting max train timestamp")
max_train_time = train["timestamp"].max()

# saving user stats for later (i dont think i use them yet but i could)
os.makedirs("processed", exist_ok=True)

with open(USER_MEANS_PATH, "w") as f:
    json.dump({str(k): float(v) for k, v in user_means.items()}, f)

with open(MOVIE_POP_PATH, "w") as f:
    json.dump({str(k): int(v) for k, v in movie_counts.items()}, f)

with open(MAX_TIME_PATH, "w") as f:
    f.write(str(max_train_time))

# transform function
def transform(df):
    df = df.copy()

    # control de entusiasmo
    df["user_mean"] = df["userId"].map(user_means)
    df["user_mean"] = df["user_mean"].fillna(0.0)

    df["rating_norm"] = df["rating"] - df["user_mean"]

    # control de populariddad
    pop = df["movieId"].map(movie_counts)
    pop = pop.fillna(1)

    df["pop_weight"] = 1 / (pop ** POPULARITY_ALPHA)
    df["rating_pop"] = df["rating_norm"] * df["pop_weight"]

    # control de tiempo
    days_diff = (max_train_time - df["timestamp"]).dt.days
    days_diff = days_diff.clip(lower=0)

    df["time_weight"] = np.exp(
        -np.log(2) * days_diff / TIME_DECAY_HALF_LIFE_DAYS
    )

    df["final_rating"] = df["rating_pop"] * df["time_weight"]

    return df

# transform
print("Applying transformations")

train_t = transform(train)
val_t = transform(val)
test_t = transform(test)

# check if everything makes sense
print("\nSanity checks:")
print("Train final rating range:", train_t["final_rating"].min(), "->", train_t["final_rating"].max())
print("Val final rating range:", val_t["final_rating"].min(), "->", val_t["final_rating"].max())
print("Test final rating range:", test_t["final_rating"].min(), "->", test_t["final_rating"].max())

# save
print("\nSaving transformed data...")

train_t.to_csv(OUTPUT_TRAIN, index=False)
val_t.to_csv(OUTPUT_VAL, index=False)
test_t.to_csv(OUTPUT_TEST, index=False)

print("Transformation complete.")
print(f"Saved: {OUTPUT_TRAIN}")
print(f"Saved: {OUTPUT_VAL}")
print(f"Saved: {OUTPUT_TEST}")
