import pandas as pd

# changeable
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 50

INPUT_RATINGS = "data/ratings.csv"
INPUT_MOVIES = "data/movies.csv"

OUTPUT_RATINGS = "processed/ratings_filtered.csv"
OUTPUT_MOVIES = "processed/movies_filtered.csv"

# load
print("Loading raw data")
ratings = pd.read_csv(INPUT_RATINGS)
movies = pd.read_csv(INPUT_MOVIES)
print("Original ratings shape:", ratings.shape)
print("Original movies shape:", movies.shape)

# fixing time
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

# user filter
print("Filtering inactive users")
user_counts = ratings["userId"].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
ratings = ratings[ratings["userId"].isin(active_users)]
print("After user filtering:", ratings.shape)

# movie filter
print("Filtering unpopular movies")
movie_counts = ratings["movieId"].value_counts()
popular_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
ratings = ratings[ratings["movieId"].isin(popular_movies)]
print("After movie filtering:", ratings.shape)
movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]
print("Filtered movies shape:", movies.shape)

# just checking if everything makes sense
print("\nSanity checks:")
print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", ratings["movieId"].nunique())
print("Rating range:", ratings["rating"].min(), "-", ratings["rating"].max())
print("Time range:", ratings["timestamp"].min(), "->", ratings["timestamp"].max())

# saving data
print("\nSaving filtered data")
ratings.to_csv(OUTPUT_RATINGS, index=False)
movies.to_csv(OUTPUT_MOVIES, index=False)
print("Done.")
print(f"Saved: {OUTPUT_RATINGS}")
print(f"Saved: {OUTPUT_MOVIES}")
