"""
REQUIRES:
- processed/ratings_filtered.csv
- processed/movies_filtered.csv

CREATES:
- processed/train.csv
- processed/val.csv
- processed/test.csv
- processed/movie_features.csv

DESCRIPTION:
Splits user–movie ratings into train, validation, and test sets
using a time-based split per user.
"""

import pandas as pd
import os

# ------------------
# paths
# ------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INPUT_RATINGS = os.path.join(PROCESSED_DIR, "ratings_filtered.csv")
INPUT_MOVIES = os.path.join(PROCESSED_DIR, "movies_filtered.csv")

OUTPUT_TRAIN = os.path.join(PROCESSED_DIR, "train.csv")
OUTPUT_VAL = os.path.join(PROCESSED_DIR, "val.csv")
OUTPUT_TEST = os.path.join(PROCESSED_DIR, "test.csv")
OUTPUT_MOVIES = os.path.join(PROCESSED_DIR, "movie_features.csv")

# ------------------
# config
# ------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ------------------
# load data
# ------------------
print("Loading filtered data")

ratings = pd.read_csv(INPUT_RATINGS)
movies = pd.read_csv(INPUT_MOVIES)

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"])

# ------------------
# split per user
# ------------------
print("Splitting train / val / test per user")

train_rows = []
val_rows = []
test_rows = []

for user_id, user_data in ratings.groupby("userId"):
    user_data = user_data.sort_values("timestamp")

    n = len(user_data)
    if n < 3:
        continue

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_rows.append(user_data.iloc[:train_end])
    val_rows.append(user_data.iloc[train_end:val_end])
    test_rows.append(user_data.iloc[val_end:])

train_df = pd.concat(train_rows, ignore_index=True)
val_df = pd.concat(val_rows, ignore_index=True)
test_df = pd.concat(test_rows, ignore_index=True)

# ------------------
# sanity checks
# ------------------
print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\nUsers in splits:")
print("Train:", train_df["userId"].nunique())
print("Val:", val_df["userId"].nunique())
print("Test:", test_df["userId"].nunique())

# ------------------
# save outputs
# ------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)

train_df.to_csv(OUTPUT_TRAIN, index=False)
val_df.to_csv(OUTPUT_VAL, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)
movies.to_csv(OUTPUT_MOVIES, index=False)

print("\nSplit complete.")
print(f"Saved: {OUTPUT_TRAIN}")
print(f"Saved: {OUTPUT_VAL}")
print(f"Saved: {OUTPUT_TEST}")
print(f"Saved: {OUTPUT_MOVIES}")
