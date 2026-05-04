"""
================================================================================
File: retrieval/embedder.py

Embedding generation and persistent storage layer using SentenceTransformers
and ChromaDB.

SentenceTransformers provides the pre-trained nomic-embed-text-v1.5 model
(768-dimensional) which generates dense vectors capturing semantic meaning.
ChromaDB stores those vectors on disk in an HNSW index so embeddings do not
need to be recomputed on every application restart.

Workflow:
  1. On first run, `index_documents()` encodes all documents and upserts them
     into the ChromaDB collection.  This takes a few minutes.
  2. On subsequent runs, the collection already exists; `index_documents()`
     returns immediately (cache hit) and `get_all_documents()` retrieves the
     stored vectors for downstream FAISS / BM25S indexing in HybridSearcher.

The raw document embeddings retrieved from ChromaDB are re-used by
HybridSearcher to avoid re-encoding the corpus on every server restart.
================================================================================
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
    # Import removed: chromadb.config.Settings is not used


class Embedder:
    """
    Generates dense document and query embeddings and persists them in ChromaDB.

    Uses SentenceTransformers for encoding and ChromaDB PersistentClient for disk
    storage.  On first run indexes all documents; on subsequent runs loads from cache.
    """

    MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
    COLLECTION_NAME = "privacy_rag_docs"

    def __init__(self, chroma_persist_dir: str = "./chroma_db_v2"):
        """
        Load the SentenceTransformer model and open (or create) the ChromaDB collection.

        Args:
            chroma_persist_dir: Directory where ChromaDB persists the vector index.
        """
        self.chroma_persist_dir = chroma_persist_dir  # stored for stale-cache warnings
        self.model = SentenceTransformer(self.MODEL_NAME, trust_remote_code=True)

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings and return an (N, 768) float32 numpy array of L2-normalised embeddings."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        return embeddings

    def index_documents(self, documents: List[dict]) -> bool:
        """
        Index documents into ChromaDB, skipping if they are already indexed.

        Each document is stored with its id, dense embedding, raw text, and a
        metadata dict containing title, category, and date for optional filtering.
        Returns True if indexing was performed, False if the collection was already
        up to date.
        """
        existing_count = self.collection.count()
        if existing_count >= len(documents):
            if existing_count != len(documents):
                print(
                    f"[Embedder] WARNING: ChromaDB has {existing_count} docs but "
                    f"{len(documents)} were requested. If you changed the corpus, "
                    f"delete '{self.chroma_persist_dir}' and restart to force a rebuild."
                )
            return False

        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [
            {
                "title": doc["title"],
                "category": doc["category"],
                "date": doc["date"],
            }
            for doc in documents
        ]

        embeddings = self.encode(contents).tolist()

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

        return True

    def search_chromadb(self, query_embedding: np.ndarray, top_k: int = 20) -> List[dict]:
        """
        Search ChromaDB with a dense query vector and return the top_k nearest documents.

        Returns a flat list of dicts with keys: id, content, metadata, distance (cosine),
        and score (= 1 − distance).
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })

        return output

    def get_all_documents(self) -> List[dict]:
        """Retrieve every document (with embeddings) from ChromaDB for use in FAISS / BM25S index construction."""
        results = self.collection.get(include=["documents", "metadatas", "embeddings"])
        output = []
        for i in range(len(results["ids"])):
            output.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "embedding": results["embeddings"][i] if results.get("embeddings") is not None else None,
            })
        return output
