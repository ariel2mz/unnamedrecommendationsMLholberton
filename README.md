# UnnamedMovieRecommender by Ariel Diaz

If you only want to run the recommender, do this:
python model/runsvd.py

## a. Intro
## b. Installation & Usage
## c. Downloadable Files
## d. File Guide
## e. How to Train Yourself
## f. Model Comparison
## g. Full End to End Project Flow

# Intro

This project implements a personalized movie recommendation system trained on a MovieLens user-review dataset, designed to go beyond static, one-shot collaborative filtering models. The goal is to build a recommender that not only learns from historical user–movie interactions, but can also adapt to user feedback over time, reflecting changes in individual preferences.

The system is based on a latent embedding approach, where both users and movies are represented as vectors in a shared embedding space. During offline training, the model learns item representations and baseline user representations from historical rating data. During online usage, a user’s embedding is dynamically updated in response to explicit or implicit feedback (such as likes or dislikes), allowing recommendations to evolve within a single session without retraining the entire model.

This repository allows running the system with two different offline models: a Two-Tower model and an SVD model. I strongly recommend using the SVD model, for reasons explained in the Model Comparison section.

The repository also contains data preprocessing and training scripts, intended for users who want to retrain the models themselves or experiment with different data-processing choices.

# Installation & Usage
Requirements
Python 3.9+
pip (included with Python)
Git (for cloning the repository)

Clone the repository:
`` git clone https://github.com/ariel2mz/recommendationprojectholberton/
cd recommendationprojectholberton ``

Install the required Python packages:
`` pip install numpy pandas pyyaml ``

Run the recommender:
`` python model/runsvd.py ``

I suggest looking up the "Exact movie names" from the file processed/movie_features.csv using Ctrl + F
A search engine is going to be implemented in the future but for now you need to copy and paste the exact name and year
of the movie, Example: Forrest Gump (1994)

# Downloadable Files
Pretrained Two-Tower model – Required to run the Two-Tower version of the app (too large for GitHub). Not needed if you train the model yourself.
https://drive.google.com/file/d/1hf_hIqv5waztDCYK7oEVeZBudvL6i4CQ/view?usp=sharing

Raw MovieLens dataset – Used for development and training, not required to run the app.
https://grouplens.org/datasets/movielens/

# File Guide

This is a brief overview of what each file does. More detailed explanations appear later in this README.

model/runsvd.py
Runs the application using the SVD offline model combined with the online user model. This is the recommended entry point to try the app.

model/runtwotower.py
Runs the application using the Two-Tower offline model combined with the online user model. Requires downloading the pretrained Two-Tower model.

model/train_svd_model.py
Trains the SVD model from scratch. Data preprocessing must be completed first.

model/train_twotower.py
Trains the Two-Tower model from scratch. Data preprocessing must be completed first.

model/online_user_model.py
Contains the online user adaptation logic used by both execution scripts. This file is not meant to be run directly.

config/recommender_policy.yaml
Configuration file containing customizable recommendation settings. The default values reflect my preferred behavior, but experimentation is encouraged.

preprocess/setup.py
Runs all data preprocessing steps in the correct order. Downloading the raw MovieLens dataset is required.

preprocess/fetch_tmdb_metadata.py
Used during development to fetch movie metadata from the TMDB API. The resulting processed/movie_metadata.csv file is already included, so this script does not need to be run.

preprocess/filter_data.py
First preprocessing step, filters raw data.

preprocess/split_data.py
Second preprocessing step, splits the dataset into train, validation, and test sets.

preprocess/transform_data.py
Third preprocessing step, applies feature engineering and index transformations.

Data Files

Runtime data included in the repository and required to run the application:

processed/movies_filtered.csv – Final filtered list of movies used at runtime.
processed/movie_features.csv – Precomputed movie feature or embedding vectors.
processed/movie_metadata.csv – Human-readable movie metadata such as titles, genres, and release years.
processed/movie_popularity.json – Popularity statistics used for biasing or fallback recommendations.
processed/user_means.json – Baseline user statistics used for normalization or initialization.

Training-only data not included in the repository:

processed/ratings_filtered.csv – Filtered user–movie interaction data.
processed/train.csv – Training split.
processed/val.csv – Validation split.
processed/test.csv – Test split.
processed/train_transformed.csv – Transformed training data.
processed/val_transformed.csv – Transformed validation data.
processed/test_transformed.csv – Transformed test data.
processed/train_max_timestamp.txt – Timestamp used for time-based splitting.

## How to Train Yourself

Download the raw MovieLens dataset from https://grouplens.org/datasets/movielens/
SPlace the files inside the data/ folder.

Run preprocess/setup.py

Run model/train_svd_model.py or model/train_twotower.py

## Model Comparison

The SVD model is simpler and more lightweight, yet it performs better than the Two-Tower model in this specific project.

Both models aim to learn embeddings that capture hidden user preferences and movie characteristics. However, this dataset only contains explicit movie ratings, which makes matrix factorization approaches like SVD sufficient. In this setting, the Two-Tower model tends to overfit and requires significantly more training time.

A Two-Tower approach would be more appropriate if the project scope were expanded to include richer interaction signals such as viewed versus clicked behavior, watch time, completion rate, or time-of-day patterns. With this additional information, the Two-Tower model would likely outperform SVD by learning more expressive representations.

## End to End Project Flow

# 1. Raw dataset (initial state)
    The system starts from the original MovieLens CSV files:

    data/ratings.csv
    userId
    movieId
    rating (raw star rating)
    timestamp (seconds or datetime)
    data/movies.csv
    movieId
    movie metadata (title, genres, etc.)

    At this stage:
    All users are included (even users with very few ratings)
    All movies are included (even movies with only 1 rating)
    Ratings are raw star values
    Timestamps may not be normalized
    These files are never used directly for training or inference.

# 2. filter_data.py — Cleaning and pruning the dataset

    This script removes low-quality signal and standardizes timestamps.

    2.1 Timestamp normalization
        Converts the timestamp column into a proper datetime
        Drops rows with invalid or missing timestamps

        Every interaction now has a valid time ordering
        Time-based splitting becomes possible later

    2.2 Filtering inactive users
        Counts ratings per user
        Keeps only users with at least 20 ratings

        Removes users with insufficient data
        Ensures user embeddings are learnable and meaningful

    2.3 Filtering unpopular movies
        Counts ratings per movie
        Keeps only movies with at least 50 ratings

        Removes extreme sparsity
        Prevents the model from overfitting on movies with no signal

    2.4 Output
        Saved files:
        processed/ratings_filtered.csv
        processed/movies_filtered.csv

    The dataset is denser
    All remaining users and movies have meaningful interaction histories
    Ratings are still raw star ratings

# 3. split_data.py — Time-aware train / validation / test split

    This script prepares the data for realistic offline evaluation.

    3.1 Per-user chronological ordering
        Ratings are grouped by userId
        Each user’s interactions are sorted by timestamp
        This preserves the real order in which preferences were expressed.

    3.2 Time-based splitting per user
        For each user:
        70% -> train
        15% -> validation
        15% -> test

        Past interactions go to train
        Future interactions go to validation/test
        Users with fewer than 3 ratings are skipped

    3.3 Output of this step

        processed/train.csv
        processed/val.csv
        processed/test.csv
        processed/movie_features.csv (filtered movie list, unchanged)

        Data is split temporally
        Ratings are still raw star values

# 4. transform_data.py — Turning ratings into learning signals

    This script transforms raw ratings into weighted preference signals that better represent user intent.

    4.1 User mean normalization (rating bias correction)
        Computes each user’s average rating from the training set
        Subtracts that mean from every rating

        rating_norm = rating − user_mean

        Accounts for personal rating scale differences
        A 4 star from a strict user and a 4 star from a generous user no longer mean the same thing
        The model learns relative preference, not absolute stars

        Saved artifact:
        processed/user_means.json

    4.2 Popularity down-weighting
        Counts how many times each movie appears in training data
        Applies a penalty to very popular movies

        pop_weight = 1 / (movie_count ^ 0.5)
        rating_pop = rating_norm × pop_weight

        Helps niche movies not get buried over more popular ones
    
        Saved artifact:
        processed/movie_popularity.json

    4.3 Time decay weighting
        Finds the most recent training timestamp
        Older ratings are exponentially down-weighted

        time_weight = exp(−ln(2) × days_since / half_life)

        Recent preferences matter more
        Old tastes slowly fade instead of disappearing abruptly

        Saved artifact:
        processed/train_max_timestamp.txt

    4.4 Final transformed rating
        All components are combined:
        final_rating = rating_norm × pop_weight × time_weight
        This is the actual signal used by the offline models.

    4.5 Output of this step
        processed/train_transformed.csv
        processed/val_transformed.csv
        processed/test_transformed.csv

# 5. Offline training — SVD model (train_svd_model.py)
    This script is responsible for producing the static offline embeddings.

    5.1 Input
        processed/train_transformed.csv
        processed/val_transformed.csv

    5.2 ID encoding
        Because MovieLens IDs are sparse, the script:
            Maps original userId → user_idx
            Maps original movieId → movie_idx

        These mappings are saved so runtime can translate back and forth.
            artifacts/user_map.json
            artifacts/movie_map.json

    5.3 Interaction matrix construction
        A sparse matrix R is built:

        Rows → users
        Columns → movies
        Values → final_rating
        This matrix represents how strongly each user prefers each movie, not just raw stars.

    5.4 SVD factorization
        Using TruncatedSVD:
            The interaction matrix is factorized into:
            user_embeddings (users × latent_dim)
            movie_embeddings (movies × latent_dim)
            Latent dimension = 50.

        Meaning:
            Each user is represented as a dense preference vector
            Each movie is represented as a dense characteristic vector
            Similar users and movies are close in embedding space

        artifacts/svd_model.joblib
        artifacts/user_embeddings.npy
        artifacts/movie_embeddings.npy

    5.5 Validation
        Predictions are computed as:
        user_embedding · movie_embedding
        RMSE is calculated against val["final_rating"].

        Evaluates representation quality
        Does not modify embeddings
        Does not affect runtime behavior

    SVD is trying to answer:
    “Can I approximate this big matrix using only 50 underlying patterns, while losing as little information as possible?”
    Those “patterns” are the latent vectors.


# 6. Two-Tower Offline Training

    This script trains a Two-Tower neural recommender model using PyTorch.
    Its goal is to learn user embeddings, movie embeddings, and tower weights such that the dot product between transformed user and movie vectors predicts the user’s rating.

    6.1 Configuration and environment setup

        The script defines training hyperparameters:
        Input data file (train_transformed.csv)
        Batch size (1024)
        Embedding dimension (64)
        Number of epochs (10)
        Learning rate (0.001)
        Device (CPU)

    6.2 Loading and preparing the dataset

        The RatingsDataset class:
            Loads the CSV file into memory
            Extracts three columns:
                userId
                movieId
                rating
            Converts them into PyTorch tensors
            Determines:
                Total number of users
                Total number of movies


    6.3 Creating the DataLoader

        The DataLoader shuffles the dataset,
        Splits it into batches of 1024 samples and
        Feeds (userId, movieId, rating) triples into the model
        This enables mini-batch gradient descent instead of updating on one example at a time.

    6.4 Model initialization (random start)
        The TwoTowerModel initializes:
            Trainable embeddings
                One embedding vector (64 values) per user
                One embedding vector (64 values) per movie
                These are randomly initialized.

            User tower
                Linear layer (64 → 64)
                ReLU activation

            Movie tower
                Linear layer (64 → 64)
                ReLU activation

        At this point:
            User embeddings are meaningless
            Movie embeddings are meaningless
            Tower weights are random

    6.5 Optimizer and loss function setup
        Optimizer
        Adam optimizer
        Learning rate = 0.001
        The optimizer will later perform gradient descent by adjusting parameters.
        Loss function
        Mean Squared Error (MSE)

    6.6 Training loop begins (this is where learning happens)

        The training process runs for 10 epochs.
        Each epoch loops over the entire dataset once.

    6.7 Forward pass (prediction)

    For each batch:
            User IDs are mapped to user embeddings
            Movie IDs are mapped to movie embeddings
            User embeddings pass through the user tower
            Movie embeddings pass through the movie tower
            A dot product between transformed vectors produces a predicted rating


    6.8 Loss computation

    The model compares predicted rating to actual rating from the dataset

    Using MSE:
    Large error -> large loss


    6.9 Backpropagation (gradient computation)

        When loss.backward() is called
        PyTorch computes gradients for:
            User embeddings
            Movie embeddings
            User tower weights
            Movie tower weights

    6.10 Gradient descent (weight updates)

        When optimizer.step() is called:
        Adam applies gradient descent
        All trainable parameters are updated
        User embeddings
        Movie embeddings
        User tower weights
        Movie tower weights
        The update direction reduces future loss.

    6.11 Repetition and convergence

        This process repeats:
        For every batch
        For every epoch

        Over time predictions become less random and loss decreases
        Embeddings encode preference structure and Towers learn useful transformations

    6.12 Model saving

        After training finishes and only the trained weights are saved

            User embeddings
            Movie embeddings
            Tower weights
            Saved as twotower_model.pt.

        This file is later loaded to generate recommendations.

# 7. Online inference and session-based adaptation

    This stage uses pretrained movie embeddings to run an interactive recommender that adapts to a single user in real time.
    No offline retraining occurs here.
    The system operates entirely at inference time, modifying only the active user representation.

    7.1 Runtime artifacts loaded

    At program start, the following artifacts are loaded:
        artifacts/movie_embeddings.npy
            Dense latent vectors for all movies
            Shape: (n_movies, latent_dim)
        artifacts/movie_map.json
            Maps MovieLens movieId → embedding index
        processed/movies_filtered.csv
            Movie titles and genre lists
        processed/movie_metadata.csv
            Descriptive metadata (overview, release date, popularity, etc.)
        processed/movie_popularity.json
            Global movie popularity counts from training data
        config/recommender_policy.yaml
            Scoring weights and behavioral constraints
    
    7.2 User cold-start initialization (session entry)

        The user is prompted to enter exactly 4 favorite movie titles.
        For each selected movie:
            The movieId is resolved
            The corresponding movie embedding vector is retrieved

        The initial user embedding is computed as the average of those vectors:
            user_embedding = mean(movie_embedding_1 … movie_embedding_4)
        This places the user directly inside the same latent space as the movies,
        already aligned with their stated tastes.
        These movies are marked as “seen” and will never be recommended again.

    7.3 Genre preference profiling

        From the same 4 favorite movies:
            Genres are extracted
            A frequency count is computed
            Counts are normalized into weights

        Result:
            user_genre_weights[g] ∈ [0, 1]

        This produces a soft genre preference profile.
        It does not affect embeddings.
        It is used only for filtering and score adjustment.

    7.4 Recommendation loop (interactive phase)
        The system enters an infinite loop where each iteration produces one recommendation.
    
    7.5 Base collaborative scoring
        For the current user embedding:
            score(movie) = user_embedding · movie_embedding

        This is a pure dot-product similarity in latent space.
        All movies already seen by the user are excluded.

    7.6 Candidate filtering using policy constraints

        For each remaining movie:
            Genre affinity is computed as:
                overlap between movie genres and user_genre_weights
            Genre distance = 1 − affinity

        Movies are discarded if:
            Affinity < min_genre_affinity
            Distance > max_genre_distance
        This prevents recommendations that are too far outside the user’s taste envelope.

    7.7 Final score composition
        For valid candidates, the final ranking score is computed as:
            score =
                (embedding_score × score_weight_model)
                − (genre_distance × genre_distance_penalty)
                − (popularity / max_popularity × popularity_penalty)
                + diversity_boost (when applicable)

        Additional behaviors:
            Popular movies are softly penalized
            Genre-adjacent but non-obvious matches can receive a diversity boost
            With probability exploration_rate, scores are dampened to encourage exploration
        All coefficients come from recommender_policy.yaml.
    
    7.8 Recommendation selection

        Candidates are sorted by final score.
        The top-ranked movie is presented to the user.

    7.9 User feedback handling

        The user provides one of four actions:
            y → like
            n → dislike
            s → skip
            i → request additional movie info

        Only like and dislike modify the user embedding.
    
    7.10 Online user embedding update (learning step)

        Let:
            u = current user embedding
            m = embedding of the recommended movie
            α = learning rate

        If the user likes the movie:
            u_new = (1 − α) · u + α · m
        The user embedding moves closer to the movie.

        If the user dislikes the movie:
            u_new = (1 − α) · u − α · negative_weight · m
        The user embedding moves away from the movie.

        If the user skips:
            No update is applied.

        This update:
            Happens immediately
            Requires no gradients
            Does not involve backpropagation
            Alters all future similarity scores in the session

    7.11 State evolution during the session

        Over time:
            The user embedding drifts through latent space
            Recommendations adapt to consistent feedback patterns
            The effective neighborhood of nearby movies changes

        Fixed throughout the session:
            Movie embeddings
            Genre definitions
            Popularity statistics
            Policy parameters

    7.12 Session termination

        The loop continues until:
            The user quits
            No valid recommendations remain

        No data is persisted.
        The learned user embedding exists only for the duration of the session.

    
