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
Offline data preparation script for the recommender system.
Splits user–movie ratings into train, validation, and test sets using
a time-based split per user to preserve interaction order.
This ensures realistic evaluation by training on past interactions
and validating/testing on future ones.
"""

import pandas as pd

# config
INPUT_RATINGS = "processed/ratings_filtered.csv"
INPUT_MOVIES = "processed/movies_filtered.csv"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

OUTPUT_TRAIN = "processed/train.csv"
OUTPUT_VAL = "processed/val.csv"
OUTPUT_TEST = "processed/test.csv"
OUTPUT_MOVIES = "processed/movie_features.csv"

# load
print("Loading filtered data")

ratings = pd.read_csv(INPUT_RATINGS)
movies = pd.read_csv(INPUT_MOVIES)

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"])

# split into train val and test
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

train_df = pd.concat(train_rows)
val_df = pd.concat(val_rows)
test_df = pd.concat(test_rows)

# making sure it makes sense
print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\nUsers in splits:")
print("Train:", train_df["userId"].nunique())
print("Val:", val_df["userId"].nunique())
print("Test:", test_df["userId"].nunique())

# saving data
print("\nSaving split data")

train_df.to_csv(OUTPUT_TRAIN, index=False)
val_df.to_csv(OUTPUT_VAL, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

movies.to_csv(OUTPUT_MOVIES, index=False)

print("Split complete.")
print(f"Saved: {OUTPUT_TRAIN}")
print(f"Saved: {OUTPUT_VAL}")
print(f"Saved: {OUTPUT_TEST}")
print(f"Saved: {OUTPUT_MOVIES}")
