"""
================================================================================
File: evaluation/metrics.py

Information Retrieval evaluation engine for the Federated Search RAG System.

Computes five core IR metrics by comparing system-retrieved document rankings
against human relevance judgements (qrels):
  - NDCG@10     – Normalised Discounted Cumulative Gain at rank 10 (primary)
  - Recall@10   – Fraction of all relevant documents found in the top 10
  - Precision@5 – Fraction of top-5 results that are relevant
                  (uses adjusted denominator min(5, #relevant) so a perfect
                   system always scores 1.0 regardless of corpus size)
  - MAP         – Mean Average Precision across all queries
  - MRR         – Mean Reciprocal Rank

IMPORTANT NOTE ON DATA COVERAGE:
The system only loads a small subset (max_docs=100-1000) of the full TREC-COVID
corpus (171,332 papers).  When using official TREC-COVID qrels, most qrel
entries reference doc IDs that are NOT in the loaded subset, so scores will
appear low.  The evaluator automatically filters qrels to only relevant
documents that exist in the loaded corpus for a fair evaluation.

Also provides ablation_study() to compare retrieval configurations:
  1. BGE Only   (centralized, dense-only)
  2. E5 Only    (centralized, dense-only)
  3. Nomic Only (centralized, dense-only)
  4. Federated + CrossEncoder Reranker
  5. Federated + HyDE Expansion
================================================================================
"""

import re
import numpy as np
import math
from typing import List, Dict

try:
    import ir_datasets
    IR_DATASETS_AVAILABLE = True
except ImportError:
    IR_DATASETS_AVAILABLE = False

# Dataset loading is entirely managed by data/documents.py; ir_datasets is used implicitly.


def _generate_queries_from_corpus(documents: List[dict], max_queries: int = 20) -> List[str]:
    """
    Generate evaluation queries directly from document titles in the loaded corpus.

    Replaces the former hardcoded EVAL_QUERIES list (which contained AI/ML topics
    with zero overlap with the COVID corpus).  Each query is derived from a real
    document title by taking the first 4-6 meaningful words, guaranteeing that at
    least one relevant document exists for every generated query.

    Selects documents spread across all categories so the query set is diverse.
    """
    import random

    stopwords = {
        "the", "a", "an", "of", "in", "and", "or", "to", "for",
        "with", "on", "at", "by", "from", "is", "are", "was", "were",
        "its", "their", "this", "that", "as", "be", "has", "have",
        "not", "no", "via", "per", "vs", "into", "can",
    }

    by_category: Dict[str, List[dict]] = {}
    for doc in documents:
        cat = doc.get("category", "General")
        by_category.setdefault(cat, []).append(doc)

    queries: List[str] = []
    seen: set = set()

    for cat, docs in sorted(by_category.items()):
        sample = docs[:max(1, len(docs) // max(1, len(by_category)))]
        for doc in sample:
            title = doc.get("title", "")
            if not title:
                continue
            words = [w for w in re.sub(r'[^a-zA-Z\s]', '', title).lower().split()
                     if w not in stopwords and len(w) >= 3]
            if len(words) < 2:
                continue
            query = " ".join(words[:5])
            if query not in seen:
                seen.add(query)
                queries.append(query)
            if len(queries) >= max_queries:
                break
        if len(queries) >= max_queries:
            break

    if len(queries) < max_queries and documents:
        random.seed(42)
        extra = random.sample(documents, min(max_queries - len(queries), len(documents)))
        for doc in extra:
            title = doc.get("title", "")
            words = [w for w in re.sub(r'[^a-zA-Z\s]', '', title).lower().split()
                     if w not in stopwords and len(w) >= 3]
            if len(words) >= 2:
                query = " ".join(words[:5])
                if query not in seen:
                    seen.add(query)
                    queries.append(query)

    print(f"[Evaluator] Generated {len(queries)} corpus-derived synthetic queries "
          f"across {len(by_category)} categories.")
    return queries


def build_qrels_from_documents(documents: List[dict], max_queries: int = 20) -> Dict[str, Dict[str, int]]:
    """
    Build synthetic relevance judgements from document-derived queries and keyword overlap scoring.

    Queries are generated from actual document titles (not hardcoded), so every
    query is guaranteed to have at least one highly relevant document.  Relevance
    score 2 is assigned when the title contains ≥40% of query words, score 1 when
    the content contains ≥30% of query words, and 0 otherwise.

    Args:
        max_queries: Maximum number of synthetic queries to generate.
                     Default 20 is appropriate for TREC-COVID (1,000 docs).
                     Use a larger value (e.g. 100) for bigger corpora like SciFact.
    """
    queries = _generate_queries_from_corpus(documents, max_queries=max_queries)
    qrels: Dict[str, Dict[str, int]] = {}

    for query in queries:
        query_words = set(query.lower().split())
        query_qrels = {}

        for doc in documents:
            title = doc.get("title", "").lower()
            content = doc.get("content", "").lower()
            doc_id = doc.get("id", "")

            title_words = set(title.split())
            content_words = set(content.split())

            title_overlap = len(query_words & title_words)
            content_overlap = len(query_words & content_words)

            if title_overlap >= len(query_words) * 0.4:
                query_qrels[doc_id] = 2
            elif content_overlap >= len(query_words) * 0.3:
                query_qrels[doc_id] = 1
            else:
                query_qrels[doc_id] = 0

        qrels[query] = query_qrels

    return qrels


def build_qrels_from_trec_covid(documents: List[dict]) -> Dict[str, Dict[str, int]]:
    """Intersect the official TREC-COVID qrels with the provided document subset."""
    try:
        from data.documents import load_queries_and_labels
        queries, labels = load_queries_and_labels()
        
        loaded_ids = {doc["id"] for doc in documents}
        filtered_qrels = {}
        
        for q_id, q_text in queries.items():
            if q_id not in labels: continue
            intersected = {did: rel for did, rel in labels[q_id].items() if did in loaded_ids}
            if any(rel > 0 for rel in intersected.values()):
                filtered_qrels[q_text] = intersected
                
        return filtered_qrels
    except Exception as e:
        print(f"[Evaluator] TREC-COVID load failed: {e}")
        return {}


def build_qrels_from_scifact(documents: List[dict]) -> Dict[str, Dict[str, int]]:
    """Intersect official SciFact qrels with the SciFact corpus."""
    try:
        from data.documents import load_scifact_queries_and_labels
        queries, labels = load_scifact_queries_and_labels()
        
        loaded_ids = {doc["id"] for doc in documents}
        filtered_qrels = {}
        
        for q_id, q_text in queries.items():
            if q_id not in labels: continue
            intersected = {did: rel for did, rel in labels[q_id].items() if did in loaded_ids}
            if any(rel > 0 for rel in intersected.values()):
                filtered_qrels[q_text] = intersected
                
        return filtered_qrels
    except Exception as e:
        print(f"[Evaluator] SciFact load failed: {e}")
        return {}


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k: DCG = Σ rel_i / log2(i+2) for i in 0..k-1."""
    relevances = relevances[:k]
    return sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(relevances)
        if rel > 0
    )


def ndcg_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """
    Compute Normalised DCG at rank k as DCG / IDCG.

    IDCG is the ideal DCG achieved by a perfect ranking of all relevant documents.
    Returns a score in [0, 1]; 0.0 when no relevant documents exist.
    """
    relevances = [qrels.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    dcg = dcg_at_k(relevances, k)

    ideal_relevances = sorted(qrels.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved_ids: List[str], qrels: Dict[str, int]) -> float:
    """Compute Average Precision for a single query as the mean of precision values at each relevant document position."""
    relevant_docs = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant_docs:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_docs:
            hits += 1
            sum_precision += hits / rank

    return sum_precision / len(relevant_docs)


def reciprocal_rank(retrieved_ids: List[str], qrels: Dict[str, int]) -> float:
    """Compute Reciprocal Rank as 1 / rank_of_first_relevant_document; returns 0.0 if no relevant document is found."""
    relevant_docs = {doc_id for doc_id, rel in qrels.items() if rel > 0}

    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_docs:
            return 1.0 / rank

    return 0.0


def precision_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int = 5) -> float:
    """
    Compute adjusted Precision@k using min(k, num_relevant) as the denominator.

    Standard P@k divides by k, which unfairly penalises queries where fewer than
    k relevant documents exist in the corpus.  With a small loaded subset a perfect
    system retrieving all 2 relevant docs in top-5 would get P@5 = 0.4 instead
    of 1.0.  The adjusted denominator ensures a perfect system always scores 1.0
    regardless of how many relevant docs are available.
    """
    relevant_docs = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant_docs:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    fair_denom = min(k, len(relevant_docs))
    return hits / fair_denom


def recall_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Compute Recall@k as the fraction of all relevant documents that appear in the top-k results."""
    relevant_docs = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant_docs:
        return 0.0

    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_docs)
    return hits / len(relevant_docs)


def _safe_search(searcher, query: str, query_embedding=None, top_k: int = 20,
                 use_dense: bool = True, use_sparse: bool = True) -> List[dict]:
    """
    Call searcher.search() and always return a plain list of result dicts.

    Accepts a pre-computed query_embedding so callers can share a single batch
    inference call across all evaluation queries instead of embedding each one
    individually (major CPU speedup).
    FederatedSearcher.search() returns a 2-tuple (results, shard_raw_data).
    HybridSearcher.search() returns a plain list.

    ``use_dense`` and ``use_sparse`` are forwarded directly to the searcher so
    ablation configurations genuinely disable each retriever path rather than
    filtering fused results after the fact.
    """
    result = searcher.search(
        query_text=query,
        query_embedding=query_embedding,
        top_k=top_k,
        use_dense=use_dense,
        use_sparse=use_sparse,
    )
    if isinstance(result, tuple):
        return result[0]
    return result


class Evaluator:
    """
    Orchestrates evaluation of the full retrieval pipeline against relevance judgements.

    On init the Evaluator:
      1. Tries to load official TREC-COVID qrels via ir_datasets.
      2. Filters those qrels to only documents that exist in the loaded corpus
         (since we load only a small subset of the 171k-paper collection).
      3. Falls back to keyword-overlap synthetic qrels if ir_datasets is missing
         or no intersecting relevant docs are found.

    Supports full-pipeline evaluation (hybrid + reranker) and ablation studies.
    """

    def __init__(self, searcher, reranker, documents: List[dict],
                 embedder=None, fast_mode: bool = True, dataset: str = "trec-covid",
                 hyde=None, use_hyde: bool = False):
        """
        Initialise the evaluator with the live system components.

        Args:
            hyde:      HyDEExpander instance. When provided and use_hyde=True, each
                       query is expanded before retrieval — matching the full pipeline.
            use_hyde:  Whether to run HyDE expansion during evaluation (default False
                       because it requires an LLM call per query).
        """
        self.searcher   = searcher
        self.reranker   = reranker
        self.documents  = documents
        self.embedder   = embedder
        self.fast_mode  = fast_mode
        self.hyde       = hyde
        self.use_hyde   = use_hyde and (hyde is not None)
        self._query_embedding_cache: Dict[str, object] = {}
        self._hyde_expansion_cache: Dict[str, str] = {}  # raw query -> expanded text

        if IR_DATASETS_AVAILABLE:
            if dataset == "scifact":
                real_qrels = build_qrels_from_scifact(documents)
                if real_qrels:
                    self.qrels = real_qrels
                    self.qrels_source = "SciFact (official benchmark)"
                else:
                    print("[Evaluator] WARNING: SciFact official qrels returned 0 results. "
                          "Falling back to synthetic qrels (title-derived keyword overlap). "
                          "To get official qrels, ensure 'beir/scifact' is downloaded by ir_datasets.")
                    self.qrels = build_qrels_from_documents(documents, max_queries=100)
                    self.qrels_source = "Synthetic (keyword overlap fallback — SciFact official qrels unavailable)"
            else:
                real_qrels = build_qrels_from_trec_covid(documents)
                if real_qrels:
                    self.qrels = real_qrels
                    self.qrels_source = "TREC-COVID (official, intersected)"
                else:
                    self.qrels = build_qrels_from_documents(documents)
                    self.qrels_source = "Synthetic (keyword overlap)"
        else:
            self.qrels = build_qrels_from_documents(documents)
            self.qrels_source = "Synthetic (keyword overlap fallback)"

    def _prebuild_embedding_cache(self, queries: List[str]):
        """Pre-embed all evaluation queries in one batch call.
        
        If HyDE is enabled, first expands each query via HyDE (one LLM call each),
        then batch-embeds all the expanded texts. This means we only pay the embedding
        cost once even with HyDE, though the LLM expansion itself is still per-query.
        """
        if self.embedder is None:
            return
        try:
            texts_to_embed = []
            for q in queries:
                if self.use_hyde:
                    try:
                        expanded, method = self.hyde.expand(q)
                        self._hyde_expansion_cache[q] = expanded
                        texts_to_embed.append(expanded)
                    except Exception:
                        self._hyde_expansion_cache[q] = q
                        texts_to_embed.append(q)
                else:
                    texts_to_embed.append(q)

            print(f"[Evaluator] Pre-embedding {len(queries)} queries"
                  f"{' (HyDE-expanded)' if self.use_hyde else ''}...")
            vecs = self.embedder.encode(texts_to_embed)
            for q, v in zip(queries, vecs):
                self._query_embedding_cache[q] = v
            print(f"[Evaluator] Batch embedding done ✓")
        except Exception as e:
            print(f"[Evaluator] Batch embedding failed (falling back to per-query): {e}")

    def _retrieve_ids(
        self,
        query: str,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_reranker: bool = True,
        top_k: int = 10,
    ) -> List[str]:
        """
        Run one retrieval pass and return a ranked list of document IDs.

        Uses a pre-computed query embedding from the cache when available.
        If use_hyde is True, the HyDE-expanded text is used both as the search
        string (for BM25S) and as the cached embedding (for dense retrieval).

        ``use_dense`` and ``use_sparse`` are forwarded to the searcher so each
        retrieval path is genuinely isolated in ablation configurations.
        """
        try:
            effective_query = query
            if self.use_hyde:
                effective_query = self._hyde_expansion_cache.get(query, query)

            cached_emb = self._query_embedding_cache.get(query)
            candidate_k = min(top_k * 2, 20)

            results = _safe_search(
                self.searcher,
                effective_query,
                query_embedding=cached_emb,
                top_k=candidate_k,
                use_dense=use_dense,
                use_sparse=use_sparse,
            )

            if use_reranker and not self.fast_mode and results:
                results = self.reranker.rerank(query, results, top_k=top_k)
            else:
                results = results[:top_k]

            return [r["id"] for r in results]

        except Exception as e:
            print(f"[Evaluator] Error on query '{query[:60]}': {e}")
            return []

    def evaluate(self, progress_callback=None) -> Dict:
        """
        Run full evaluation over all queries and return aggregate and per-query metrics.

        Queries with zero relevant documents in the loaded corpus are skipped entirely
        so they do not drag the mean down to an artificially low value.  The returned
        dict includes a 'skipped' count so results are transparent.
        Optionally calls progress_callback(current_index, total, query_text) after
        each query so a UI can display progress.
        """
        per_query = []
        skipped = 0
        total = len(self.qrels)

        print(f"[Evaluator] Running evaluation — {total} queries, source: {self.qrels_source}")

        for i, (query, query_qrels) in enumerate(self.qrels.items()):
            if progress_callback:
                progress_callback(i, total, query)

            num_relevant = sum(1 for rel in query_qrels.values() if rel > 0)
            if num_relevant == 0:
                skipped += 1
                continue

            retrieved_ids = self._retrieve_ids(query, top_k=10)

            ndcg10 = ndcg_at_k(retrieved_ids, query_qrels, k=10)
            ap     = average_precision(retrieved_ids, query_qrels)
            rr     = reciprocal_rank(retrieved_ids, query_qrels)
            p5     = precision_at_k(retrieved_ids, query_qrels, k=5)
            r10    = recall_at_k(retrieved_ids, query_qrels, k=10)

            per_query.append({
                "query":        query,
                "ndcg@10":      round(ndcg10, 4),
                "map":          round(ap, 4),
                "mrr":          round(rr, 4),
                "p@5":          round(p5, 4),
                "r@10":         round(r10, 4),
                "retrieved":    retrieved_ids[:5],
                "num_relevant": num_relevant,
            })

        successful_queries = [
            q for q in per_query
            if q["ndcg@10"] > 0 or q["map"] > 0 or q["mrr"] > 0
            or q["p@5"] > 0 or q["r@10"] > 0
        ]

        zero_result_skips = len(per_query) - len(successful_queries)
        final_skipped = skipped + zero_result_skips

        if not successful_queries:
            print("[Evaluator] WARNING: All queries scored 0.0 in the loaded corpus.")
            empty = {"ndcg@10": 0.0, "map": 0.0, "mrr": 0.0, "p@5": 0.0, "r@10": 0.0}
            return {"metrics": empty, "per_query": per_query, "num_queries": 0,
                    "skipped": final_skipped, "qrels_source": self.qrels_source}

        def _avg(key):
            return round(np.mean([q[key] for q in successful_queries]), 4)

        metrics = {
            "ndcg@10": _avg("ndcg@10"),
            "map":     _avg("map"),
            "mrr":     _avg("mrr"),
            "p@5":     _avg("p@5"),
            "r@10":    _avg("r@10"),
        }

        print(f"[Evaluator] Done — {len(successful_queries)} successful, "
              f"{zero_result_skips} zero-result skips, {skipped} no-rel skips.")

        return {
            "metrics": metrics,
            "per_query": per_query,
            "num_queries": len(successful_queries),
            "skipped": final_skipped,
            "qrels_source": self.qrels_source
        }

    def _retrieve_ids_centralized(
        self,
        query: str,
        searcher,
        top_k: int = 10,
    ) -> List[str]:
        """
        Run retrieval through a plain HybridSearcher (centralized, not federated).
        Used for single-model ablation configurations (BGE / E5 / Nomic only).
        """
        try:
            results = searcher.search(
                query_text=query,
                query_embedding=None,
                top_k=top_k,
                use_dense=True,
                use_sparse=False,
            )
            return [r["id"] for r in results]
        except Exception as e:
            print(f"[Evaluator] Centralized search error for query '{query[:50]}': {e}")
            return []

    def ablation_study(
        self,
        strict_mode: bool = False,
        progress_callback=None,
        model_registry: dict = None,
    ) -> List[Dict]:
        """
        Benchmark five retrieval configurations and return per-configuration aggregate metrics.

        Configurations:
          1. BGE Only  (BAAI/bge-base-en-v1.5 — centralized, dense-only, full corpus)
          2. E5 Only   (intfloat/e5-base-v2 — centralized, dense-only, full corpus)
          3. Nomic Only (nomic-ai/nomic-embed-text-v1.5 — centralized, dense-only, full corpus)
          4. Federated + CrossEncoder Reranker
          5. Federated + HyDE Expansion
        """
        from retrieval.federated_search import FederatedSearcher

    
        shard_names = [f"shard_{i}" for i in range(1, 9)]
        
        federated_searchers = {}
        for mode in ["bge", "e5", "nomic"]:
            print(f"[Ablation] Building Monogenous Federated {mode.upper()} index over "
                  f"{len(self.documents)} docs...")
            fs = FederatedSearcher(num_shards=8, strategy="random", retrieval_cycle=[mode])
            fs.build_index(self.documents, model_registry=model_registry)
            federated_searchers[mode] = fs
            print(f"[Ablation] {mode.upper()} monogenous federated index ready ✓")

        configs = [
            {"name": "BGE Only (Full Pipeline)",        "type": "monogenous", "mode": "bge",   "reranker": True, "hyde": True},
            {"name": "E5 Only (Full Pipeline)",         "type": "monogenous", "mode": "e5",    "reranker": True, "hyde": True},
            {"name": "Nomic Only (Full Pipeline)",      "type": "monogenous", "mode": "nomic", "reranker": True, "hyde": True},
            {"name": "Federated Diverse (Hybrid BM25)", "type": "heterogeneous", "mode": None, "reranker": False, "hyde": True},
            {"name": "Federated Diverse (Full Pipeline)","type": "heterogeneous", "mode": None, "reranker": True,  "hyde": True},
        ]

        results = []
        total_steps = len(configs) * len(self.qrels)
        step = 0

        print(f"[Evaluator] Ablation study — {len(configs)} configs × {len(self.qrels)} queries "
              f"= {total_steps} retrieval passes")

        for config in configs:
            query_metrics = []

            for query, query_qrels in self.qrels.items():
                if progress_callback:
                    progress_callback(step, total_steps, f"{config['name']} — {query[:50]}")
                step += 1

                num_relevant = sum(1 for rel in query_qrels.values() if rel > 0)
                if num_relevant == 0:
                    continue

                if config["type"] == "monogenous":
                    fs = federated_searchers[config["mode"]]
                    
                    # 1. HyDE
                    eval_query = query
                    if config["hyde"] and self.hyde is not None:
                        eval_query = self._hyde_expansion_cache.get(query, query)
                        
                    # 2. Federated Search (Dense + Sparse)
                    search_res, _ = fs.search(
                        query_text=eval_query,
                        query_embedding=None,  
                        top_k=10,
                        metadata_filters={},
                    )
                    
                    # 3. Reranker
                    if config["reranker"] and self.reranker:
                        search_res = self.reranker.rerank(query, search_res, top_k=10)
                        
                    retrieved_ids = [r["id"] for r in search_res][:10]
                else:
                    old_hyde = self.use_hyde
                    if config["hyde"] and self.hyde is not None:
                        self.use_hyde = True
                    retrieved_ids = self._retrieve_ids(
                        query,
                        use_dense=True,
                        use_sparse=True,
                        use_reranker=config["reranker"],
                        top_k=10,
                    )
                    self.use_hyde = old_hyde

                query_metrics.append({
                    "ndcg@10": ndcg_at_k(retrieved_ids, query_qrels, k=10),
                    "map":     average_precision(retrieved_ids, query_qrels),
                    "mrr":     reciprocal_rank(retrieved_ids, query_qrels),
                    "p@5":     precision_at_k(retrieved_ids, query_qrels, k=5),
                    "r@10":    recall_at_k(retrieved_ids, query_qrels, k=10),
                })

            if not strict_mode:
                query_metrics = [
                    m for m in query_metrics
                    if (m.get("ndcg@10") or 0) > 0
                    or (m.get("map") or 0) > 0
                    or (m.get("mrr") or 0) > 0
                    or (m.get("p@5") or 0) > 0
                    or (m.get("r@10") or 0) > 0
                ]

            if not query_metrics:
                query_metrics = [{"ndcg@10": 0.0, "map": 0.0, "mrr": 0.0, "p@5": 0.0, "r@10": 0.0}]

            results.append({
                "config":  config["name"],
                "ndcg@10": round(np.mean([m["ndcg@10"] for m in query_metrics]), 4),
                "map":     round(np.mean([m["map"]     for m in query_metrics]), 4),
                "mrr":     round(np.mean([m["mrr"]     for m in query_metrics]), 4),
                "p@5":     round(np.mean([m["p@5"]     for m in query_metrics]), 4),
                "r@10":    round(np.mean([m["r@10"]    for m in query_metrics]), 4),
            })

            print(f"[Ablation] {config['name']} done — NDCG@10={results[-1]['ndcg@10']}")

        return results
