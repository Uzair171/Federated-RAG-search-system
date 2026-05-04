"""
================================================================================
File: retrieval/hybrid_search.py

Hybrid retrieval engine combining dense FAISS search with sparse BM25S ranking,
fused via Reciprocal Rank Fusion (RRF).

Dense retrieval uses nomic-embed-text-v1.5 (768-dimensional, ~270 MB) via
SentenceTransformers to encode documents and queries into semantic vectors.
These vectors are stored in a FAISS IndexFlatIP (inner-product index) which
is equivalent to cosine similarity on L2-normalised vectors.

Sparse retrieval uses BM25S (2024 fast reimplementation of BM25) for keyword
matching.  When BM25S is unavailable, rank_bm25 is used as a fallback.

RRF fusion (k=60, from the original 2009 Cormack et al. paper) combines the
per-document ranks from both retrievers into a single unified score without
requiring score normalisation across retrieval modalities.

Metadata filtering is supported as a post-retrieval step: when filters are
active the candidate pool is expanded to 100 results before filtering so that
enough documents survive to fill the requested top_k quota.
================================================================================
"""

import numpy as np
from typing import List, Dict, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    try:
        from rank_bm25 import BM25Okapi
        BM25_AVAILABLE = True
    except ImportError:
        BM25_AVAILABLE = False


class HybridSearcher:
    """
    Hybrid dense + sparse retriever using nomic-embed-text-v1.5 (FAISS) + BM25S with RRF fusion.

    Instantiate once per shard, call build_index() with the shard's documents,
    then call search() for every query.  Pre-computed query embeddings from the
    main thread can be passed in to avoid redundant encoding in multi-threaded
    federated search.
    """

    RRF_K = 60  # Standard Cormack et al. (2009) value — matches FederatedSearcher

    # SOTA dense model configurations — one mode is assigned per shard.
    RETRIEVAL_MODES = {
        "bge": {
            "model_name": "BAAI/bge-base-en-v1.5",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "doc_prefix": "",
            "embedding_dim": 768,
            "trust_remote_code": False,
        },
        "e5": {
            "model_name": "intfloat/e5-base-v2",
            "query_prefix": "query: ",
            "doc_prefix": "passage: ",
            "embedding_dim": 768,
            "trust_remote_code": False,
        },
        "nomic": {
            "model_name": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "doc_prefix": "search_document: ",
            "embedding_dim": 768,
            "trust_remote_code": True,
        },
    }

    # RRF weighting factors.
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3

    def __init__(self, use_fp16: bool = False, embedder_model=None, retrieval_mode: str = "nomic"):
        """
        Initialise a shard-level dense retriever for the specified retrieval_mode.

        Args:
            use_fp16:        Use half-precision (GPU only); disabled by default for CPU.
            embedder_model:  Pre-loaded SentenceTransformer model to reuse across shards.
            retrieval_mode:  One of 'bge', 'e5', or 'nomic'. Determines which SOTA dense
                             model is used for retrieval on this shard.
        """
        self.retrieval_mode = retrieval_mode
        mode_cfg = self.RETRIEVAL_MODES.get(retrieval_mode, self.RETRIEVAL_MODES["nomic"])
        self.MODEL_NAME = mode_cfg["model_name"]
        self.EMBEDDING_DIM = mode_cfg["embedding_dim"]
        self.doc_instruction = mode_cfg["doc_prefix"]

        self.documents = []
        self.faiss_index = None
        self.doc_embeddings = None
        self.embedding_dim = self.EMBEDDING_DIM
        self.bm25_index = None
        self.use_bm25s = False
        self.contents = []

        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            )

        if embedder_model is not None:
            self.model = embedder_model
        else:
            print(f"[HybridSearcher] Loading {self.MODEL_NAME}...")
            print("[HybridSearcher] First run will download model weights.")
            self.model = SentenceTransformer(
                self.MODEL_NAME,
                trust_remote_code=mode_cfg.get("trust_remote_code", False),
            )

        self.query_instruction = mode_cfg["query_prefix"]
        print(f"[HybridSearcher] Ready — mode={retrieval_mode!r} · {self.MODEL_NAME}")

    def build_index(self, documents: List[dict]):
        """
        Build both the FAISS dense index and the BM25S sparse index from the provided documents.

        If documents already contain pre-computed embedding vectors (from ChromaDB)
        those are loaded directly into FAISS to avoid re-encoding.

        Documents are normalised on ingestion: flat-structured documents (e.g. those
        loaded from the SciFact disk cache) are given a nested ``metadata`` dict so
        downstream result consumers always see a consistent shape.
        """
        normalised = []
        for doc in documents:
            if "metadata" not in doc:
                doc = dict(doc) 
                doc["metadata"] = {
                    "title":    doc.get("title", ""),
                    "category": doc.get("category", ""),
                    "date":     doc.get("date", ""),
                    "journal":  doc.get("journal", ""),
                }
            normalised.append(doc)

        self.documents = normalised
        self.contents = [doc["content"] for doc in self.documents]

        use_precomputed = (
            self.retrieval_mode == "nomic"
            and self.documents
            and self.documents[0].get("embedding") is not None
        )
        if use_precomputed:
            doc_embeddings = np.array([doc["embedding"] for doc in self.documents], dtype=np.float32)
            self.embedding_dim = doc_embeddings.shape[1]
            self.doc_embeddings = doc_embeddings

            if not FAISS_AVAILABLE:
                raise ImportError("faiss-cpu required. Run: pip install faiss-cpu")

            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(doc_embeddings)
            print(f"[HybridSearcher] FAISS loaded from precomputed nomic embeddings — {len(self.documents)} docs @ {self.embedding_dim}-dim")
        else:
            self._build_dense_index(self.contents)

        self._build_sparse_index(self.contents)

    def _build_dense_index(self, contents: List[str]):
        """Encode all document texts with the nomic-embed model and build a FAISS IndexFlatIP."""
        print(f"[HybridSearcher] Encoding {len(contents)} documents with {self.MODEL_NAME}...")

        prefixed_contents = [self.doc_instruction + c for c in contents]
        doc_embeddings = self.model.encode(
            prefixed_contents,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        self.embedding_dim = doc_embeddings.shape[1]
        self.doc_embeddings = doc_embeddings

        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu required. Run: pip install faiss-cpu")

        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(doc_embeddings)

        print(f"[HybridSearcher] FAISS index built — {len(contents)} docs @ {self.embedding_dim}-dim")

    def _build_sparse_index(self, contents: List[str]):
        """Build a BM25S sparse index (or rank_bm25 fallback) over the provided document texts."""
        if BM25S_AVAILABLE:
            corpus_tokens = bm25s.tokenize(contents, stopwords="en")
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(corpus_tokens)
            self.use_bm25s = True
            print("[HybridSearcher] BM25S sparse index built")
        elif BM25_AVAILABLE:
            tokenized = [c.lower().split() for c in contents]
            self.bm25_index = BM25Okapi(tokenized)
            self.use_bm25s = False
            print("[HybridSearcher] BM25 (rank_bm25) sparse index built")
        else:
            print("[HybridSearcher] WARNING: No sparse library found. Install bm25s: pip install bm25s")
            self.bm25_index = None

    def save_index(self, cache_dir: str):
        """
        Persist the built index to disk so it can be reloaded instantly on next run.

        Saves:
          - FAISS binary index  → <cache_dir>/faiss.index
          - Document embeddings → <cache_dir>/embeddings.npy
          - BM25S model         → <cache_dir>/bm25s_index/
          - Document metadata   → <cache_dir>/documents.json
          - Contents list       → <cache_dir>/contents.json
        """
        import os, json
        os.makedirs(cache_dir, exist_ok=True)

        if self.faiss_index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.faiss_index, os.path.join(cache_dir, "faiss.index"))
            print(f"[HybridSearcher] FAISS index saved ✓")

        if self.doc_embeddings is not None:
            np.save(os.path.join(cache_dir, "embeddings.npy"), self.doc_embeddings)
            print(f"[HybridSearcher] Embeddings saved ✓")

        if self.bm25_index is not None and self.use_bm25s and BM25S_AVAILABLE:
            bm25_path = os.path.join(cache_dir, "bm25s_index")
            self.bm25_index.save(bm25_path)
            print(f"[HybridSearcher] BM25S index saved ✓")

        with open(os.path.join(cache_dir, "documents.json"), "w", encoding="utf-8") as f:
            clean_docs = [{k: v for k, v in doc.items() if k != "embedding"} for doc in self.documents]
            json.dump(clean_docs, f, ensure_ascii=False)
        with open(os.path.join(cache_dir, "contents.json"), "w", encoding="utf-8") as f:
            json.dump(self.contents, f, ensure_ascii=False)

        print(f"[HybridSearcher] Index cache saved to '{cache_dir}' ✓")

    def load_index(self, cache_dir: str) -> bool:
        """
        Load a previously saved index from disk. Returns True on success, False if cache is missing.

        Call this before build_index() — if it returns True, skip build_index() entirely.
        """
        import os, json
        faiss_path = os.path.join(cache_dir, "faiss.index")
        emb_path   = os.path.join(cache_dir, "embeddings.npy")
        docs_path  = os.path.join(cache_dir, "documents.json")
        cont_path  = os.path.join(cache_dir, "contents.json")

        if not all(os.path.exists(p) for p in [faiss_path, emb_path, docs_path, cont_path]):
            return False

        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            with open(cont_path, "r", encoding="utf-8") as f:
                self.contents = json.load(f)

            self.doc_embeddings = np.load(emb_path)
            self.embedding_dim  = self.doc_embeddings.shape[1]

            if FAISS_AVAILABLE:
                self.faiss_index = faiss.read_index(faiss_path)
                print(f"[HybridSearcher] FAISS index loaded from cache — {len(self.documents)} docs ✓")

            bm25_path = os.path.join(cache_dir, "bm25s_index")
            if BM25S_AVAILABLE and os.path.exists(bm25_path):
                self.bm25_index = bm25s.BM25.load(bm25_path, mmap=False)
                self.use_bm25s  = True
                print(f"[HybridSearcher] BM25S index loaded from cache ✓")
            else:
                self._build_sparse_index(self.contents)

            return True

        except Exception as e:
            print(f"[HybridSearcher] Cache load failed ({e}); will rebuild from scratch.")
            return False

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode a query string into a 768-dim L2-normalised float32 vector using nomic-embed-text-v1.5.

        Applies the required 'search_query: ' prefix before encoding.
        """
        prefixed = self.query_instruction + query_text
        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    def _matches_filters(self, doc: dict, filters: dict) -> bool:
        """Return True if the document passes all active metadata filters; categories/journals use OR logic across selections."""
        if not filters:
            return True

        if "categories" in filters:
            doc_cat = (doc.get("category") or doc.get("metadata", {}).get("category", "")).lower()
            if not any(sel in doc_cat for sel in filters["categories"]):
                return False

        if "journals" in filters:
            doc_journal = (doc.get("journal") or doc.get("metadata", {}).get("journal", "")).lower()
            if not any(sel in doc_journal for sel in filters["journals"]):
                return False

        if "start_date" in filters:
            doc_date = doc.get("date") or doc.get("metadata", {}).get("date", "")
            if doc_date < filters["start_date"]:
                return False

        if "end_date" in filters:
            doc_date = doc.get("date") or doc.get("metadata", {}).get("date", "")
            if doc_date > filters["end_date"]:
                return False

        return True

    def search(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        metadata_filters: dict = None,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> List[dict]:
        """
        Execute hybrid dense + sparse search with RRF fusion and optional metadata post-filtering.

        ``use_dense`` and ``use_sparse`` allow each retrieval path to be toggled
        independently for ablation studies.  When a path is disabled its
        contribution to the RRF scores is zero; results are drawn solely from the
        enabled path(s).

        When a pre-computed query_embedding is supplied it is validated against the
        index dimension and re-encoded if mismatched.  Metadata filters expand the
        internal candidate pool to 100 to ensure sufficient documents survive trimming.
        Returns a list of result dicts with keys: id, content, metadata, rrf_score,
        in_dense, in_sparse.
        """
        if self.faiss_index is None:
            raise RuntimeError("Call build_index() before searching.")

        base_candidates = 100 if metadata_filters else 20
        n_candidates = min(base_candidates, len(self.documents))

        # ── Dense retrieval (FAISS) ──────────────────────────────────────────
        dense_ranking = {}
        if use_dense:
            if query_embedding is not None:
                if query_embedding.shape[0] != self.embedding_dim:
                    print(
                        f"[HybridSearcher] WARNING: query_embedding dim "
                        f"({query_embedding.shape[0]}) != index dim ({self.embedding_dim}). "
                        f"Re-encoding with {self.MODEL_NAME}."
                    )
                    dense_vec = self.encode_query(query_text).reshape(1, -1)
                else:
                    dense_vec = query_embedding.reshape(1, -1).astype(np.float32)
            else:
                dense_vec = self.encode_query(query_text).reshape(1, -1)

            faiss_scores, faiss_indices = self.faiss_index.search(dense_vec, n_candidates)
            rank = 0
            for idx in faiss_indices[0]:
                if idx == -1:
                    continue
                doc = self.documents[idx]
                if self._matches_filters(doc, metadata_filters):
                    dense_ranking[doc["id"]] = rank
                    rank += 1

        # ── Sparse retrieval (BM25S) ─────────────────────────────────────────
        sparse_ranking = {}
        if use_sparse and self.bm25_index is not None:
            if self.use_bm25s:
                try:
                    query_tokens = bm25s.tokenize([query_text], stopwords="en")
                    results_bm25s, _ = self.bm25_index.retrieve(
                        query_tokens, k=n_candidates
                    )
                    sparse_top_indices = [int(i) for i in results_bm25s[0]]
                except Exception as e:
                    print(f"[HybridSearcher] BM25S error '{e}' bypassed — falling back to Dense ranking for this shard")
                    sparse_top_indices = []
            else:
                tokenized_query = query_text.lower().split()
                bm25_scores = self.bm25_index.get_scores(tokenized_query)
                sparse_top_indices = np.argsort(bm25_scores)[::-1][:n_candidates].tolist()

            rank = 0
            for idx in sparse_top_indices:
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    if self._matches_filters(doc, metadata_filters):
                        sparse_ranking[doc["id"]] = rank
                        rank += 1

        # ── RRF fusion ───────────────────────────────────────────────────────
        all_doc_ids = set(dense_ranking.keys()) | set(sparse_ranking.keys())

        rrf_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in dense_ranking:
                dense_contribution = 1.0 / (self.RRF_K + dense_ranking[doc_id] + 1)
                score += self.DENSE_WEIGHT * dense_contribution

            if doc_id in sparse_ranking:
                sparse_contribution = 1.0 / (self.RRF_K + sparse_ranking[doc_id] + 1)
                score += self.SPARSE_WEIGHT * sparse_contribution

            rrf_scores[doc_id] = score

        sorted_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        doc_lookup = {doc["id"]: doc for doc in self.documents}
        results = []
        for doc_id in sorted_doc_ids:
            doc = doc_lookup.get(doc_id, {})
            results.append({
                "id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "rrf_score": rrf_scores[doc_id],
                "in_dense": doc_id in dense_ranking,
                "in_sparse": doc_id in sparse_ranking,
                "retrieval_model": self.retrieval_mode,
            })

        return results