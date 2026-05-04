"""
================================================================================
File: query_processing/recommender.py

Query recommendation module that suggests similar past queries to the user.

At startup the module loads an existing query history JSON file, or seeds one
from SEED_QUERIES if running for the first time.  Every query in history is
stored alongside its dense embedding so that at query time a simple cosine
similarity computation (matrix dot-product) can identify the most similar past
queries in a single forward pass — no approximate-nearest-neighbour index
required at this scale.

New queries submitted by users are appended to history (up to a maximum of 500
entries) and persisted to disk so recommendations improve over time.
================================================================================
"""

import json
import os
import numpy as np
from typing import List



class QueryRecommender:
    """
    Recommends semantically similar past queries using embedding cosine similarity.

    Maintains an in-memory list of {query, embedding} records backed by a JSON
    file on disk.  Recommendations are derived purely from vector similarity, not
    collaborative filtering, so the system works without any usage data beyond
    the seeded queries.
    """

    def __init__(self, embedder, history_path: str = "data/query_history.json"):
        """
        Initialise the recommender with an embedder and the path to the history file.

        Args:
            embedder:     Embedder instance (retrieval/embedder.py) used to encode queries.
            history_path: Path to the JSON file that persists query history across runs.
        """
        self.embedder = embedder
        self.history_path = history_path
        self.history: List[dict] = []
        self._initialize_history()

    def _initialize_history(self):
        """Load history from disk if it exists; otherwise seed from SEED_QUERIES and save."""
        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
        else:
            self._seed_history()

    def _seed_history(self):
        """Encode all SEED_QUERIES and persist them as the initial history so recommendations work immediately."""
        # Lazy import: avoids triggering the module-level DOCUMENTS load in data.documents
        # when this method is never called (history file already exists on subsequent runs).
        from data.documents import SEED_QUERIES
        embeddings = self.embedder.encode(SEED_QUERIES)

        self.history = [
            {
                "query": query,
                "embedding": embedding.tolist(),
            }
            for query, embedding in zip(SEED_QUERIES, embeddings)
        ]

        self._save_history()

    def _save_history(self):
        """Persist the current in-memory history list to the JSON file on disk."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def recommend(self, query: str, top_k: int = 3) -> List[str]:
        """
        Return the top_k most similar past queries to the given query.

        Encodes the query, computes cosine similarity against all history
        embeddings in a single matrix multiplication, then returns the highest-
        scoring candidates excluding near-duplicates (similarity > 0.95) and
        low-relevance matches (similarity < 0.3).
        """
        if not self.history:
            return []

        query_embedding = self.embedder.encode([query])[0]

        history_embeddings = np.array([item["embedding"] for item in self.history])
        history_queries = [item["query"] for item in self.history]

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        history_norms = history_embeddings / (
            np.linalg.norm(history_embeddings, axis=1, keepdims=True) + 1e-10
        )

        similarities = history_norms @ query_norm

        sorted_indices = np.argsort(similarities)[::-1]

        recommendations = []

        for idx in sorted_indices:
            candidate = history_queries[idx]
            similarity = similarities[idx]

            if similarity > 0.95:
                continue
            if similarity < 0.3:
                continue

            recommendations.append((candidate, float(similarity)))

            if len(recommendations) >= top_k:
                break

        return [rec[0] for rec in recommendations]

    def add_query(self, query: str):
        """
        Append a new query to history and persist to disk, skipping exact duplicates.

        Trims history to at most 500 entries by discarding the oldest records when
        the limit is reached.
        """
        existing_queries = [item["query"].lower() for item in self.history]
        if query.lower() in existing_queries:
            return

        embedding = self.embedder.encode([query])[0]
        self.history.append({
            "query": query,
            "embedding": embedding.tolist(),
        })

        MAX_HISTORY = 500
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        self._save_history()
