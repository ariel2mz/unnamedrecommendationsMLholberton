"""
Online user preference model for a movie recommender system.
Represents a user as a latent embedding vector initialized from liked movies.
The model updates the user embedding in real time based on feedback
(like, dislike, or ignore), moving it closer to or farther from movie embeddings.
Used to adapt recommendations dynamically during an interactive session.
"""
import numpy as np

class OnlineUserModel:
    def __init__(
        self,
        movie_embeddings,
        lr=0.15,
        negative_weight=0.5
    ):
        """
        movie_embeddings: np.array [n_movies, k]
        lr: learning rate for updates
        negative_weight: how strong dislikes push away
        """
        self.movie_embeddings = movie_embeddings
        self.lr = lr
        self.negative_weight = negative_weight

        self.user_embedding = None
        self.seen_movies = set()

    def initialize_from_movies(self, movie_indices, ratings=None):
        """
        Initialize user from a set of liked movies
        """
        if ratings is None:
            ratings = np.ones(len(movie_indices))

        self.user_embedding = (
            ratings @ self.movie_embeddings[movie_indices]
        ) / len(movie_indices)

        self.seen_movies.update(movie_indices)

    def update(self, movie_idx, feedback):
        """
        feedback:
          +1 = like
           0 = ignore
          -1 = dislike
        """
        m_vec = self.movie_embeddings[movie_idx]

        if feedback > 0:
            self.user_embedding = (
                (1 - self.lr) * self.user_embedding
                + self.lr * m_vec
            )
        elif feedback < 0:
            self.user_embedding = (
                (1 - self.lr) * self.user_embedding
                - self.lr * self.negative_weight * m_vec
            )

        self.seen_movies.add(movie_idx)

    def recommend(self, top_n=10):
        scores = self.user_embedding @ self.movie_embeddings.T

        for idx in self.seen_movies:
            scores[idx] = -np.inf

        top_idx = np.argsort(scores)[::-1][:top_n]
        return top_idx
