"""
================================================================================
File: retrieval/reranker.py

Implements Stage 2 of the two-stage retrieval pipeline: cross-encoder reranking.

Stage 1 (fast, approximate) uses a bi-encoder (nomic-embed-text-v1.5) combined
with FAISS and BM25S to quickly retrieve the top-20 candidate documents in
milliseconds.  Stage 2 (this module) uses a cross-encoder to re-score those 20
candidates and surface the true top-5.

A cross-encoder processes the (query, document) pair jointly through a
transformer, allowing every query token to attend to every document token.
This deep interaction captures nuanced relevance signals that independent
bi-encoder encoding misses, resulting in significantly higher ranking quality
at the cost of slower inference.

Model: mixedbread-ai/mxbai-rerank-xsmall-v1
  ~130 MB, optimised for fast CPU inference while outperforming the older
  cross-encoder/ms-marco-MiniLM-L-6-v2 baseline.
================================================================================
"""

from sentence_transformers import CrossEncoder
from typing import List


class Reranker:
    """
    Cross-encoder reranker that re-scores a list of candidate documents for a query.

    Takes the top-k results from the federated hybrid search and returns them
    reordered by cross-encoder relevance score, most relevant document first.
    """

    # cross-encoder/ms-marco-MiniLM-L-12-v2: 12-layer model with significantly better
    # ranking precision than the 6-layer variant. Directly improves NDCG by ~2-3% on
    # BEIR benchmarks. Trade-off: ~2x slower inference on CPU vs L-6.
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def __init__(self):
        """Load the cross-encoder model from HuggingFace (downloads on first run, then cached)."""
        self.model = CrossEncoder(
            self.MODEL_NAME,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Rerank documents by cross-encoder relevance score and return the top_k results.

        Forms (query, document_text) pairs, scores them in one batch, sorts by
        score descending, attaches the score as 'rerank_score' on each dict, and
        returns the top_k highest-scoring documents.
        """
        if not documents:
            return []

        sentence_pairs = [
            (query, doc["content"][:1500])
            for doc in documents
        ]

        scores = self.model.predict(sentence_pairs, batch_size=16, show_progress_bar=False)

        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]
