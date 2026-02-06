"""
Online user preference model for a movie recommender system.
FIXED VERSION:
- Stabilized user embedding
- Proper normalization
- Stronger, symmetric feedback handling
"""

import numpy as np


class OnlineUserModel:
    def __init__(
        self,
        movie_embeddings,
        lr=0.15,
        negative_weight=0.5,
        normalize=True
    ):
        """
        movie_embeddings: np.array [n_movies, k]
        lr: learning rate for updates
        negative_weight: how strong dislikes push away
        normalize: keep user embedding normalized
        """
        self.movie_embeddings = movie_embeddings
        self.lr = lr
        self.negative_weight = negative_weight
        self.normalize = normalize

        self.user_embedding = None
        self.seen_movies = set()

    def _normalize(self, v):
        return v / (np.linalg.norm(v) + 1e-8)

    def initialize_from_movies(self, movie_indices, ratings=None):
        """
        Initialize user from a set of liked movies
        """
        if ratings is None:
            ratings = np.ones(len(movie_indices))

        emb = ratings @ self.movie_embeddings[movie_indices]
        emb = emb / len(movie_indices)

        if self.normalize:
            emb = self._normalize(emb)

        self.user_embedding = emb
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
            delta = m_vec
        elif feedback < 0:
            delta = -self.negative_weight * m_vec
        else:
            # ignore = do nothing except mark seen
            self.seen_movies.add(movie_idx)
            return

        self.user_embedding = (
            (1 - self.lr) * self.user_embedding
            + self.lr * delta
        )

        if self.normalize:
            self.user_embedding = self._normalize(self.user_embedding)

        self.seen_movies.add(movie_idx)

    def recommend(self, top_n=10):
        scores = self.user_embedding @ self.movie_embeddings.T

        for idx in self.seen_movies:
            scores[idx] = -np.inf

        top_idx = np.argsort(scores)[::-1][:top_n]
        return top_idx
