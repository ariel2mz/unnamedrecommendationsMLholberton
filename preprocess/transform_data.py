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

Key properties of this version:
- Preserves absolute user preference (likes stay positive)
- Limits popularity dominance
- Keeps time decay
- Produces stable, bounded training targets
"""

import pandas as pd
import numpy as np
import json
import os

# ------------------
# paths
# ------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

OUTPUT_TRAIN = os.path.join(PROCESSED_DIR, "train_transformed.csv")
OUTPUT_VAL = os.path.join(PROCESSED_DIR, "val_transformed.csv")
OUTPUT_TEST = os.path.join(PROCESSED_DIR, "test_transformed.csv")

USER_MEANS_PATH = os.path.join(PROCESSED_DIR, "user_means.json")
MOVIE_POP_PATH = os.path.join(PROCESSED_DIR, "movie_popularity.json")
MAX_TIME_PATH = os.path.join(PROCESSED_DIR, "train_max_timestamp.txt")

# ------------------
# config
# ------------------
TIME_DECAY_HALF_LIFE_DAYS = 365

# ------------------
# load
# ------------------
print("Loading split data")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

for df in (train, val, test):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------
# fit statistics (TRAIN ONLY)
# ------------------
print("Fitting user mean ratings (train only)")
user_means = train.groupby("userId")["rating"].mean()

print("Fitting movie popularity (train only)")
movie_counts = train["movieId"].value_counts()

print("Getting max train timestamp")
max_train_time = train["timestamp"].max()

# ------------------
# save statistics
# ------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)

with open(USER_MEANS_PATH, "w") as f:
    json.dump({str(k): float(v) for k, v in user_means.items()}, f)

with open(MOVIE_POP_PATH, "w") as f:
    json.dump({str(k): int(v) for k, v in movie_counts.items()}, f)

with open(MAX_TIME_PATH, "w") as f:
    f.write(str(max_train_time))

# ------------------
# transform function
# ------------------
def transform(df):
    df = df.copy()

    # ----
    # absolute preference signal
    # center ratings around neutral (3.0)
    # ----
    df["rating_centered"] = df["rating"] - 3.0

    # ----
    # popularity control (bounded)
    # ----
    pop = df["movieId"].map(movie_counts).fillna(1)
    df["pop_weight"] = 1.0 / np.sqrt(1.0 + pop)

    df["rating_pop"] = df["rating_centered"] * df["pop_weight"]

    # ----
    # time decay
    # ----
    days_diff = (max_train_time - df["timestamp"]).dt.days
    days_diff = days_diff.clip(lower=0)

    df["time_weight"] = np.exp(
        -np.log(2) * days_diff / TIME_DECAY_HALF_LIFE_DAYS
    )

    # ----
    # final training target
    # ----
    df["final_rating"] = df["rating_pop"] * df["time_weight"]

    # stability clamp
    df["final_rating"] = df["final_rating"].clip(-2.0, 2.0)

    return df

# ------------------
# apply transform
# ------------------
print("Applying transformations")

train_t = transform(train)
val_t = transform(val)
test_t = transform(test)

# ------------------
# sanity checks
# ------------------
print("\nSanity checks:")
print(
    "Train final rating range:",
    train_t["final_rating"].min(),
    "->",
    train_t["final_rating"].max()
)
print(
    "Val final rating range:",
    val_t["final_rating"].min(),
    "->",
    val_t["final_rating"].max()
)
print(
    "Test final rating range:",
    test_t["final_rating"].min(),
    "->",
    test_t["final_rating"].max()
)

# ------------------
# save
# ------------------
print("\nSaving transformed data...")

train_t.to_csv(OUTPUT_TRAIN, index=False)
val_t.to_csv(OUTPUT_VAL, index=False)
test_t.to_csv(OUTPUT_TEST, index=False)

print("Transformation complete.")
print(f"Saved: {OUTPUT_TRAIN}")
print(f"Saved: {OUTPUT_VAL}")
print(f"Saved: {OUTPUT_TEST}")
