"""
================================================================================
File: retrieval/federated_search.py

Federated Search coordinator for the Privacy-Enhanced RAG System.

This module implements a federated information retrieval architecture in which
the full document corpus is partitioned into independent shards, each shard
maintains its own HybridSearcher (FAISS dense + BM25S sparse + RRF fusion), and
queries fan out to all shards in parallel via Python threads.

Architecture overview:
  1. FederatedSearcher.build_index() partitions documents into N shards and
     builds a HybridSearcher index for each shard sequentially.
  2. FederatedSearcher.search() pre-computes the query embedding once on the
     main thread (preventing HuggingFace tensor-sharing collisions) then spawns
     one thread per shard.
  3. Each Shard.search() calls its HybridSearcher and tags results with the
     shard's identifier for UI provenance tracing.
  4. The central aggregator applies a global RRF merge across all per-shard
     results to produce a single unified ranked list.

Shard strategies:
  "category" – one shard per document category (Epidemiology, Virology, …).
               Best for demonstrating semantically independent source retrieval.
  "random"   – random assignment across num_shards buckets.
  "size"     – equal-size sequential chunks.

Default strategy is "category" to match the TREC-COVID topic structure.
================================================================================
"""

import itertools
import threading
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from retrieval.hybrid_search import HybridSearcher


class Shard:
    """
    A single independent retrieval unit in the federated architecture.

    Holds a subset of the full corpus and its own HybridSearcher.  Documents
    retrieved from a shard are tagged with the shard's identifier so the UI
    can display document provenance.
    """

    def __init__(self, shard_id: str, documents: List[dict], embedder_model=None, retrieval_mode: str = "nomic"):
        """
        Initialise the shard with an identifier, its document subset, and an optional shared model.

        Args:
            shard_id:      Human-readable label (e.g. 'Epidemiology', 'shard_0').
            documents:     The subset of documents assigned to this shard.
            embedder_model: Pre-loaded SentenceTransformer model passed through to HybridSearcher.
        """
        self.shard_id = shard_id
        self.documents = documents
        self.embedder_model = embedder_model
        self.retrieval_mode = retrieval_mode
        self.searcher: Optional[HybridSearcher] = None
        self._indexed = False

    def build_index(self, cache_dir: str = None):
        """Build or load from cache a HybridSearcher index over this shard's documents."""
        if not self.documents:
            print(f"[Shard:{self.shard_id}] WARNING: no documents — skipping index.")
            return

        self.searcher = HybridSearcher(
            embedder_model=self.embedder_model,
            retrieval_mode=self.retrieval_mode,
        )

        if cache_dir and self.searcher.load_index(cache_dir):
            if len(self.searcher.documents) == len(self.documents):
                self._indexed = True
                print(f"[Shard:{self.shard_id}] Loaded from cache ({self.retrieval_mode}) ✓")
                return
            else:
                print(f"[Shard:{self.shard_id}] Cache stale (doc count mismatch) — rebuilding.")

        print(f"[Shard:{self.shard_id}] Building index over {len(self.documents)} docs...")
        self.searcher.build_index(self.documents)
        self._indexed = True
        print(f"[Shard:{self.shard_id}] Index ready.")

        if cache_dir:
            self.searcher.save_index(cache_dir)
            print(f"[Shard:{self.shard_id}] Index cached to '{cache_dir}' ✓")

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
        Search this shard and return local top-k results, each tagged with 'shard_id'.

        Returns an empty list if the shard has no index or contains no documents.
        """
        if not self._indexed or self.searcher is None:
            return []

        results = self.searcher.search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filters=metadata_filters,
            use_dense=use_dense,
            use_sparse=use_sparse,
        )

        for r in results:
            r["shard_id"] = self.shard_id

        return results

    def __repr__(self):
        """Return a concise string representation showing shard identity and state."""
        return f"<Shard id={self.shard_id!r} mode={self.retrieval_mode!r} docs={len(self.documents)} indexed={self._indexed}>"


class FederatedSearcher:
    """
    Central coordinator that fans out queries to all shards in parallel and merges results with global RRF.

    Usage:
        fed = FederatedSearcher(num_shards=5, strategy='category')
        fed.build_index(documents, embedder)
        results, shard_raw_data = fed.search('what causes COVID fever', top_k=10)
    """

    RRF_K = 60

    RETRIEVAL_CYCLE = ["bge", "e5", "nomic"]

    def __init__(self, num_shards: int = 5, strategy: str = "category", retrieval_cycle: List[str] = None):
        """
        Configure the federated searcher.

        Args:
            num_shards: Target number of shards.  For 'category' strategy the actual
                        count equals the number of unique categories in the corpus.
            strategy:   Partitioning strategy: 'category', 'random', or 'size'.
            retrieval_cycle: Optional list of mode strings to cycle through.
        """
        self.num_shards = num_shards
        self.strategy = strategy
        self.retrieval_cycle = retrieval_cycle or ["bge", "e5", "nomic"]
        self.shards: List[Shard] = []
        self._built = False

    def _partition(self, documents: List[dict]) -> Dict[str, List[dict]]:
        """Dispatch to the appropriate partitioning helper based on self.strategy."""
        if self.strategy == "category":
            return self._partition_by_category(documents)
        elif self.strategy == "random":
            return self._partition_random(documents)
        else:
            return self._partition_by_size(documents)

    def _partition_by_category(self, documents: List[dict]) -> Dict[str, List[dict]]:
        """Assign each document to a shard named after its 'category' metadata field; uncategorised documents go to 'General'."""
        buckets: Dict[str, List[dict]] = defaultdict(list)
        for doc in documents:
            cat = doc.get("category") or doc.get("metadata", {}).get("category") or "General"
            buckets[cat].append(doc)
        return dict(buckets)

    def _partition_random(self, documents: List[dict]) -> Dict[str, List[dict]]:
        """Randomly assign documents to num_shards buckets using a fixed seed for reproducibility."""
        buckets: Dict[str, List[dict]] = {f"shard_{i}": [] for i in range(self.num_shards)}
        rng = np.random.default_rng(seed=42)
        assignments = rng.integers(0, self.num_shards, size=len(documents))
        for doc, shard_idx in zip(documents, assignments):
            buckets[f"shard_{shard_idx}"].append(doc)
        return buckets

    def _partition_by_size(self, documents: List[dict]) -> Dict[str, List[dict]]:
        """Split the corpus into num_shards equal-size sequential chunks."""
        chunk_size = max(1, len(documents) // self.num_shards)
        buckets = {}
        for i in range(self.num_shards):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_shards - 1 else len(documents)
            buckets[f"shard_{i}"] = documents[start:end]
        return buckets

    def build_index(self, documents: List[dict], embedder=None, model_registry: dict = None, cache_root: str = None):
        """
        Partition the corpus and build each shard's HybridSearcher index sequentially.

        Each shard is assigned one of 3 SOTA retrieval modes (bge, e5, nomic) in round-robin
        order. The model_registry dict maps mode names to pre-loaded SentenceTransformer
        instances so each model is loaded only once regardless of shard count.

        Args:
            documents:       Full document corpus to partition and index.
            embedder:        Embedder instance (fallback model source if no registry).
            model_registry:  Dict mapping mode -> SentenceTransformer instance.
                             e.g. {"bge": bge_model, "e5": e5_model, "nomic": nomic_model}
            cache_root:      Optional custom cache directory (useful for evaluating multiple datasets).
        """
        print(f"[FederatedSearcher] Partitioning {len(documents)} docs "
              f"into shards (strategy={self.strategy!r})...")

        buckets = self._partition(documents)

        print(f"[FederatedSearcher] Created {len(buckets)} shards:")
        for shard_id, docs in sorted(buckets.items()):
            print(f"  • {shard_id}: {len(docs)} docs")

        self.shards = []
        mode_cycle = itertools.cycle(self.retrieval_cycle)
        import os
        if cache_root is None:
            cache_root = os.path.join("data", "shard_cache")

        for shard_id, docs in sorted(buckets.items()):
            mode = next(mode_cycle)

            if model_registry and mode in model_registry:
                shard_model = model_registry[mode]
            elif embedder:
                shard_model = embedder.model
            else:
                shard_model = None

            safe_id = shard_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
            cache_dir = os.path.join(cache_root, f"{safe_id}_{mode}")

            print(f"[FederatedSearcher] Shard {shard_id!r} → retrieval_mode={mode!r}")
            shard = Shard(
                shard_id=shard_id,
                documents=docs,
                embedder_model=shard_model,
                retrieval_mode=mode,
            )
            shard.build_index(cache_dir=cache_dir)
            self.shards.append(shard)

        self._built = True
        print(f"[FederatedSearcher] All {len(self.shards)} shards indexed and ready.")

    def _search_shard(
        self,
        shard: Shard,
        query_text: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        metadata_filters: dict,
        use_dense: bool,
        use_sparse: bool,
        output: list,
        idx: int,
    ):
        """Thread worker that searches one shard and writes its results into output[idx]."""
        try:
            results = shard.search(query_text, query_embedding, top_k, metadata_filters,
                                   use_dense=use_dense, use_sparse=use_sparse)
            output[idx] = results
        except Exception as e:
            print(f"[FederatedSearcher] Shard {shard.shard_id} error: {e}")
            output[idx] = []

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
        Fan out the query to all shards in parallel, then merge results with global RRF.

        Pre-computes the query embedding once on the calling thread to prevent
        multi-thread tensor-sharing conflicts.  Each result in the returned list
        includes id, content, metadata, global_score, shard_id, shard_rank, in_dense,
        and in_sparse fields.

        Global fusion uses Z-Score Normalization + CombSUM: each shard's per-shard
        RRF scores are Z-score normalised independently to account for score scale
        differences between BGE, E5, and Nomic shards, then all candidates are merged
        and sorted by their normalised global score.  This is more principled than a
        second RRF pass for partitioned corpora where each document appears in exactly
        one shard (making rank-consensus accumulation impossible).

        Also returns a shard_raw_data dict mapping shard_id to its top-5 doc titles
        for the UI provenance diagram.
        """
        if not self._built or not self.shards:
            raise RuntimeError("Call build_index() before searching.")

        if not hasattr(self, "_query_cache"):
            self._query_cache = {}

        if query_text not in self._query_cache:
            if len(self._query_cache) >= 200:
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            self._query_cache[query_text] = {}

        query_embeddings_by_mode: Dict[str, Optional[np.ndarray]] = {}
        for shard in self.shards:
            mode = shard.retrieval_mode
            if mode not in query_embeddings_by_mode and shard.searcher is not None:
                if mode in self._query_cache[query_text]:
                    query_embeddings_by_mode[mode] = self._query_cache[query_text][mode]
                else:
                    emb = shard.searcher.encode_query(query_text)
                    query_embeddings_by_mode[mode] = emb
                    self._query_cache[query_text][mode] = emb

        shard_results_list: List[List[dict]] = [[] for _ in self.shards]
        threads = []

        for i, shard in enumerate(self.shards):
            mode_qemb = query_embeddings_by_mode.get(shard.retrieval_mode)
            t = threading.Thread(
                target=self._search_shard,
                args=(shard, query_text, mode_qemb, top_k, metadata_filters,
                      use_dense, use_sparse, shard_results_list, i),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)

        candidate_docs: Dict[str, dict] = {}

        for shard, results in zip(self.shards, shard_results_list):
            for rank, result in enumerate(results):
                doc_id = result["id"]
                result["shard_rank"] = rank
                if doc_id not in candidate_docs:
                    candidate_docs[doc_id] = result

        
        global_scores: Dict[str, float] = {}

        for shard, results in zip(self.shards, shard_results_list):
            if not results:
                continue
            scores = np.array([r["rrf_score"] for r in results], dtype=np.float64)
            mean = float(scores.mean())
            std  = float(scores.std())
            if std < 1e-9:
                std = 1e-9 
            for r in results:
                doc_id = r["id"]
                global_scores[doc_id] = (r["rrf_score"] - mean) / std

        sorted_ids = sorted(global_scores, key=global_scores.get, reverse=True)[:top_k]

        results_final = []
        for doc_id in sorted_ids:
            doc = candidate_docs[doc_id]
            results_final.append({
                "id":           doc_id,
                "content":      doc.get("content", ""),
                "metadata":     doc.get("metadata", {}),
                "rrf_score":    doc.get("rrf_score", 0.0),   
                "global_score": global_scores[doc_id],        
                "shard_id":     doc.get("shard_id", "unknown"),
                "shard_rank":   doc.get("shard_rank", -1),
                "in_dense":     doc.get("in_dense", False),
                "in_sparse":    doc.get("in_sparse", False),
            })

        shard_raw_data = {}
        for shard, results in zip(self.shards, shard_results_list):
            if results:
                shard_raw_data[shard.shard_id] = [
                    {"title": r.get("metadata", {}).get("title", r["id"])}
                    for r in results[:5]
                ]

        return results_final, shard_raw_data

    def get_shard_info(self) -> List[dict]:
        """Return a list of dicts describing each shard (id, doc count, indexed status, retrieval model) for the UI."""
        return [
            {
                "shard_id": s.shard_id,
                "num_docs": len(s.documents),
                "indexed": s._indexed,
                "retrieval_model": s.retrieval_mode,
            }
            for s in self.shards
        ]

    def __repr__(self):
        """Return a concise string representation of the federated searcher state."""
        return (
            f"<FederatedSearcher shards={len(self.shards)} "
            f"strategy={self.strategy!r} built={self._built}>"
        )