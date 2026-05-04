"""
================================================================================
File: app.py

Flask web application and REST API server for the Privacy-Enhanced Federated
RAG System.

This file is the entry point for the application.  At startup it launches a
background thread that sequentially initialises all system components (dataset
loading, embedder, ChromaDB indexing, federated shards, reranker, query
pipeline, and RAG generation pipeline) while streaming progress events to the
frontend via a Server-Sent Events endpoint.

API surface:
  GET  /                    – Serve the main single-page application UI.
  GET  /evaluation          – Serve the evaluation dashboard UI.
  GET  /api/status          – SSE stream of init progress for the loading screen.
  GET  /api/suggest         – Live query autocomplete suggestions.
  GET  /api/filter_suggest  – Live autocomplete for Category / Journal filters.
  GET  /api/documents       – All document titles and metadata (for the doc browser).
  GET  /api/shards          – Federated shard metadata (for the UI shard diagram).
  POST /api/search          – Full search pipeline as an SSE stream.
  POST /api/evaluate        – Run IR evaluation metrics and/or ablation study.

All search results flow through the complete pipeline:
  SymSpell correction → HyDE expansion → query recommendation
  → federated hybrid retrieval → CrossEncoder reranking → Ollama RAG generation
================================================================================
"""

import sys
import os
import json
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

system = {
    "ready": False,
    "error": None,
    "components": {},
    "documents": [],
    "init_log": [],
}

init_lock = threading.Lock()

# Persistent cache for the secondary SciFact evaluation corpus and index.
# Built on-demand when the user first selects "SciFact" in the evaluation UI.
scifact_system = {
    "ready": False,
    "documents": [],
    "searcher": None,
}
scifact_lock = threading.Lock()   


def log_step(step: str, message: str, status: str = "loading"):
    """Append one timestamped progress entry to the initialisation log for the loading screen SSE stream."""
    entry = {"step": step, "message": message, "status": status, "time": time.time()}
    system["init_log"].append(entry)


def initialize_system():
    """
    Initialise all RAG system components in sequence and log each step.

    Runs in a background daemon thread so the Flask server can start accepting
    connections immediately.  Populates the global `system` dict with ready
    components; sets system['ready'] = True on success or system['error'] on
    failure.
    """
    try:
        log_step("documents", "Loading dataset...", "loading")
        from data.documents import DOCUMENTS, SEED_QUERIES
        system["documents"] = DOCUMENTS
        system["seed_queries"] = SEED_QUERIES
        log_step("documents", f"Dataset loaded — {len(DOCUMENTS)} documents", "done")

        log_step("embedder", "Loading Nomic embedder (ChromaDB index)...", "loading")
        from retrieval.embedder import Embedder
        embedder = Embedder()
        system["components"]["embedder"] = embedder
        log_step("embedder", "Nomic embedder ready — 768-dim dense vectors", "done")

        log_step("index", "Indexing documents into ChromaDB...", "loading")
        was_indexed = embedder.index_documents(DOCUMENTS)
        msg = f"Indexed {len(DOCUMENTS)} documents" if was_indexed else f"Loaded {len(DOCUMENTS)} documents from cache"
        log_step("index", msg, "done")

        log_step("search", "Loading SOTA shard models (BGE + E5 + Nomic)...", "loading")
        all_docs = embedder.get_all_documents()
        system["components"]["all_docs"] = all_docs
        from sentence_transformers import SentenceTransformer
        bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        e5_model = SentenceTransformer("intfloat/e5-base-v2")
        model_registry = {
            "bge": bge_model,
            "e5": e5_model,
            "nomic": embedder.model,
        }
        system["components"]["model_registry"] = model_registry
        log_step("search", "3 SOTA models ready — BGE · E5 · Nomic (assigned round-robin to shards)", "done")

        log_step("federated", "Building 8 federated shard indexes (3 models × round-robin)...", "loading")
        from retrieval.federated_search import FederatedSearcher
        fed_searcher = FederatedSearcher(num_shards=8, strategy="category")
        fed_searcher.build_index(all_docs, embedder, model_registry=model_registry)
        system["components"]["fed_searcher"] = fed_searcher

        shard_info = fed_searcher.get_shard_info()
        shard_summary = ", ".join(
            f"{s['shard_id']}[{s['retrieval_model']}]({s['num_docs']} docs)" for s in shard_info
        )
        log_step(
            "federated",
            f"Federated search ready — {len(shard_info)} shards: {shard_summary}",
            "done",
        )

        log_step("reranker", "Loading CrossEncoder reranker...", "loading")
        from retrieval.reranker import Reranker
        reranker = Reranker()
        system["components"]["reranker"] = reranker
        log_step("reranker", f"CrossEncoder ready — {reranker.MODEL_NAME}", "done")

        log_step("query", "Initializing query processing pipeline...", "loading")
        from query_processing.spell_corrector import SpellCorrector
        from query_processing.hyde_expander import HyDEExpander
        from query_processing.recommender import QueryRecommender

        spell = SpellCorrector(documents=system["documents"])
        hyde = HyDEExpander(
            documents=system["documents"],
            embedder=embedder,
        )
        recommender = QueryRecommender(embedder=embedder, history_path="data/query_history.json")

        system["components"]["spell"] = spell
        system["components"]["hyde"] = hyde
        system["components"]["recommender"] = recommender

        if hyde.ollama_available:
            ollama_status = f"Ollama/{hyde.model}"
        elif hyde.wordnet_available:
            ollama_status = "WordNet + SentenceTransformers"
        else:
            ollama_status = "keyword fallback"
        log_step("query", f"Query pipeline ready — SymSpell + HyDE ({ollama_status})", "done")

        log_step("rag", "Initializing RAG pipeline (LlamaIndex + Ollama)...", "loading")
        from rag.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        system["components"]["rag"] = rag

        if rag.ollama_available:
            log_step("rag", f"RAG ready — Ollama/{rag.model}", "done")
        else:
            log_step("rag", "RAG ready — extractive fallback (Ollama not found)", "warning")

        system["ready"] = True
        log_step("ready", "System fully initialized — ready for queries", "done")

    except Exception as e:
        import traceback
        system["error"] = str(e)
        log_step("error", f"Initialization failed: {e}", "error")
        traceback.print_exc()


init_thread = threading.Thread(target=initialize_system, daemon=True)
init_thread.start()


@app.route("/")
def index():
    """Serve the main single-page application HTML from the static directory."""
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status_stream():
    """Stream initialisation progress as Server-Sent Events so the frontend loading screen updates in real time."""
    def generate():
        sent_count = 0
        while True:
            current_log = system["init_log"]
            if len(current_log) > sent_count:
                for entry in current_log[sent_count:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                sent_count = len(current_log)

            if system["ready"]:
                yield f"data: {json.dumps({'step': 'READY', 'status': 'ready'})}\n\n"
                break
            if system["error"]:
                yield f"data: {json.dumps({'step': 'ERROR', 'message': system['error'], 'status': 'error'})}\n\n"
                break

            time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/suggest")
def suggest():
    """Return up to 6 document title autocomplete suggestions ranked by title-prefix and keyword overlap with the query."""
    query = request.args.get("q", "").strip().lower()
    if not query or not system["ready"]:
        return jsonify([])

    docs = system["documents"]
    results = []

    for doc in docs:
        title = doc.get("title", "")
        content = doc.get("content", "")
        title_lower = title.lower()

        score = 0

        if title_lower.startswith(query):
            score += 100

        query_words = query.split()
        words_in_title = sum(1 for w in query_words if w in title_lower)
        score += words_in_title * 20

        for word in query_words:
            if word in title_lower:
                score += 10
            elif any(word in t for t in title_lower.split()):
                score += 5

        content_lower = content.lower()
        words_in_content = sum(1 for w in query_words if w in content_lower)
        score += words_in_content * 2

        if score > 0:
            snippet = _build_snippet(content, query_words)
            results.append({
                "id": doc.get("id", ""),
                "title": title,
                "category": doc.get("category", ""),
                "date": doc.get("date", ""),
                "snippet": snippet,
                "score": score,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(results[:6])


@app.route("/api/filter_suggest")
def filter_suggest():
    """Return up to 6 autocomplete suggestions for the Category or Journal filter fields, sorted by prefix-match priority."""
    field = request.args.get("field", "").strip().lower()
    query = request.args.get("q", "").strip().lower()

    if not query or not system["ready"] or field not in ["category", "journal"]:
        return jsonify([])

    docs = system["documents"]
    matches = set()

    for doc in docs:
        if field == "category":
            val = doc.get("category", "")
            if val and query in val.lower():
                matches.add(val.strip())
        elif field == "journal":
            val = doc.get("journal", "")
            if val and query in val.lower():
                matches.add(val.strip())

        if len(matches) > 30:
            break

    def score_match(m):
        """Rank prefix matches above substring matches, then sort alphabetically."""
        m_lower = m.lower()
        if m_lower.startswith(query):
            return (0, m)
        return (1, m)

    sorted_matches = sorted(list(matches), key=score_match)
    return jsonify([{"text": m} for m in sorted_matches[:6]])


def _build_snippet(content: str, query_words: list, max_len: int = 120) -> str:
    """Extract the first sentence that contains a query term, truncated to max_len characters."""
    if not content:
        return ""
    sentences = content.replace(". ", ".|").split("|")
    for sentence in sentences:
        if any(w in sentence.lower() for w in query_words):
            return sentence.strip()[:max_len] + ("..." if len(sentence) > max_len else "")
    return content[:max_len] + "..."


@app.route("/api/search", methods=["POST"])
def search():
    """
    Execute the full RAG search pipeline and stream results as Server-Sent Events.

    Accepts a JSON body: { "query": str, "category": str, "journal": str,
    "start_date": str, "end_date": str }.  Each pipeline stage (spell correction,
    HyDE expansion, recommendation, federated retrieval, reranking, RAG generation)
    emits a step_start and step_done event so the UI can animate progress in real time.
    """
    if not system["ready"]:
        return jsonify({"error": "System not ready yet"}), 503

    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    category = data.get("category", "").strip()   # legacy single-value compat
    categories = data.get("categories", [])         # new multi-select list
    if category and category not in categories:
        categories.append(category)

    journal = data.get("journal", "").strip()       # legacy single-value compat
    journals = data.get("journals", [])             # new multi-select list
    if journal and journal not in journals:
        journals.append(journal)

    start_date = data.get("start_date", "").strip()
    end_date = data.get("end_date", "").strip()

    metadata_filters = {}
    if categories: metadata_filters["categories"] = [c.lower() for c in categories if c]
    if start_date: metadata_filters["start_date"] = start_date
    if end_date:   metadata_filters["end_date"] = end_date
    if journals:   metadata_filters["journals"] = [j.lower() for j in journals if j]

    def generate():
        """Inner generator that runs each pipeline step and yields SSE events."""
        try:
            components = system["components"]

            def send(event_type: str, payload: dict):
                """Format a payload dict as a single SSE data line."""
                return f"data: {json.dumps({'type': event_type, **payload})}\n\n"

            yield send("step_start", {"step": "spell", "label": "Spell Correction"})
            corrected = components["spell"].correct(query)
            yield send("step_done", {
                "step": "spell",
                "original": query,
                "corrected": corrected,
                "changed": corrected.lower() != query.lower(),
            })

            yield send("step_start", {"step": "hyde", "label": "HyDE Query Expansion"})
            expanded, method = components["hyde"].expand(corrected)
            # Ollama path returns "query\n\nhypothetical_doc" — extract the hypothetical part.
            # WordNet fallback returns a flat space-joined string of terms — label it clearly.
            if "\n\n" in expanded:
                hyde_doc = expanded.split("\n\n", 1)[1][:300]
            else:
                hyde_doc = f"[Keyword expansion] {expanded[:300]}"
            yield send("step_done", {
                "step": "hyde",
                "method": method,
                "hypothetical_doc": hyde_doc,
            })

            yield send("step_start", {"step": "recommend", "label": "Query Recommendations"})
            recs = components["recommender"].recommend(corrected, top_k=3)
            yield send("step_done", {"step": "recommend", "recommendations": recs})

            yield send("step_start", {"step": "search", "label": "Federated Shards (Hybrid Retrieval)"})
            search_results, shard_raw_data = components["fed_searcher"].search(
                query_text=expanded,
                query_embedding=None,
                top_k=10,
                metadata_filters=metadata_filters,
            )

            both_count = sum(1 for r in search_results if r.get("in_dense") and r.get("in_sparse"))
            dense_count = sum(1 for r in search_results if r.get("in_dense") and not r.get("in_sparse"))
            sparse_count = sum(1 for r in search_results if not r.get("in_dense") and r.get("in_sparse"))

            shard_hits = {}
            for r in search_results:
                sid = r.get("shard_id", "unknown")
                shard_hits[sid] = shard_hits.get(sid, 0) + 1

            shard_models = {
                s.shard_id: s.retrieval_mode
                for s in components["fed_searcher"].shards
            }

            yield send("step_done", {
                "step": "search",
                "total": len(search_results),
                "both": both_count,
                "dense_only": dense_count,
                "sparse_only": sparse_count,
                "shards_queried": len(components["fed_searcher"].shards),
                "shard_hits": shard_hits,
                "shard_models": shard_models,
                "shard_raw_data": shard_raw_data,
                "global_top_10": [
                    {"title": r.get("metadata", {}).get("title", r["id"]), "shard": r.get("shard_id", "")}
                    for r in search_results
                ]
            })

            yield send("step_start", {"step": "rerank", "label": "Cross-Encoder Reranking"})
            # Rerank against the original corrected query (user intent), not the HyDE-expanded
            # text. The cross-encoder should score relevance to what the user *asked*, not to
            # the full hypothetical document which may be several paragraphs long.
            reranked = components["reranker"].rerank(corrected, search_results, top_k=5)
            yield send("step_done", {
                "step": "rerank",
                "results": [
                    {
                        "id": d.get("id", ""),
                        "title": d.get("metadata", {}).get("title", d.get("id", "")),
                        "category": d.get("metadata", {}).get("category", ""),
                        "journal": d.get("metadata", {}).get("journal", ""),
                        "date": d.get("metadata", {}).get("date", ""),
                        "score": round(d.get("rerank_score", 0), 3),
                        "snippet": (
                            d.get("content", "")[:200] + "..."
                            if len(d.get("content", "")) > 200
                            else d.get("content", "")
                        ),
                        "content": d.get("content", ""),
                        "shard_id": d.get("shard_id", ""),
                    }
                    for d in reranked
                ],
            })

            yield send("step_start", {"step": "rag", "label": "RAG Generation (LlamaIndex + Ollama)"})
            answer = components["rag"].generate(corrected, reranked, stream=False)
            yield send("step_done", {"step": "rag", "answer": answer})

            components["recommender"].add_query(corrected)

            yield send("result", {"answer": answer, "query": corrected})

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'trace': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/documents")
def get_documents():
    """Return all document titles, categories, dates, and 150-character snippets as a JSON array for the document browser UI."""
    docs = [
        {
            "id": d.get("id", ""),
            "title": d.get("title", ""),
            "category": d.get("category", ""),
            "date": d.get("date", ""),
            "snippet": (
                d.get("content", "")[:150] + "..."
                if len(d.get("content", "")) > 150
                else d.get("content", "")
            ),
        }
        for d in system["documents"]
    ]
    return jsonify(docs)


@app.route("/api/shards")
def get_shards():
    """Return shard metadata (id, doc count, indexed status) for all federated shards so the UI can display them."""
    if not system["ready"]:
        return jsonify({"error": "System not ready yet"}), 503
    fed = system["components"].get("fed_searcher")
    if fed is None:
        return jsonify({"error": "Federated searcher not initialized"}), 500
    return jsonify(fed.get_shard_info())


@app.route("/evaluation")
def evaluation_page():
    """Serve the evaluation dashboard HTML page."""
    return send_from_directory("static", "evaluation.html")


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Run IR metrics or ablation study and stream progress via SSE."""
    if not system["ready"]:
        return jsonify({"error": "System not ready"}), 503

    data = request.get_json() or {}
    mode = data.get("mode", "full")  # "full", "ablation", "both"
    use_fast_mode = data.get("fast_mode", True)
    dataset_name = data.get("dataset", "trec-covid")
    use_hyde = data.get("use_hyde", False)
    strict_mode = data.get("strict_mode", False)

    def generate():
        try:
            from evaluation.metrics import Evaluator
            embedder = system["components"]["embedder"]
            reranker = system["components"]["reranker"]

            # Dataset switching logic
            if dataset_name == "scifact":
                with scifact_lock:  # prevent double-build on concurrent /api/evaluate calls
                    if not scifact_system["ready"]:
                        from data.documents import load_scifact_documents
                        from retrieval.federated_search import FederatedSearcher
                        
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Loading SciFact corpus (5,183 docs)...'})}\n\n"
                        scifact_system["documents"] = load_scifact_documents()

                        yield f"data: {json.dumps({'type': 'log', 'message': 'Building 8 Federated Shards for SciFact (BGE/E5/Nomic)...'})}\n\n"
                        # Create federated searcher for SciFact. Must use 'size' instead of 'category' 
                        # because SciFact docs don't have medical categories (they all get "Science").
                        searcher = FederatedSearcher(num_shards=8, strategy="size")
                        
                        # Pass model_registry so it reuses the already-loaded SOTA models
                        # Temporarily override shard_cache_root in federated_search.py to avoid 
                        # overwriting TREC-COVID cache? We'll let federated search handle it by 
                        # making the shard IDs unique for scifact, or just relying on the fact 
                        # that SciFact partitions will have 'scifact' in the ID if we alter it slightly.
                        # Wait, the easiest way is to temporarily patch the cache root.
                        import os
                        tmp_cache_root = os.path.join("data", "scifact_shard_cache")
                        
                        model_registry = system["components"]["model_registry"]
                        searcher.build_index(scifact_system["documents"], embedder, model_registry=model_registry, cache_root=tmp_cache_root)
                        
                        yield f"data: {json.dumps({'type': 'log', 'message': 'SciFact Federated Index ready ✓'})}\n\n"
                        
                        scifact_system["searcher"] = searcher
                        scifact_system["ready"] = True

                eval_docs = scifact_system["documents"]
                eval_searcher = scifact_system["searcher"]
            else:
                eval_docs = system["documents"]
                eval_searcher = system["components"]["fed_searcher"]

            hyde = system["components"].get("hyde")
            evaluator = Evaluator(
                searcher=eval_searcher,
                reranker=reranker,
                documents=eval_docs,
                embedder=embedder,
                fast_mode=use_fast_mode,
                dataset=dataset_name,
                hyde=hyde,
                use_hyde=use_hyde,
            )

            queries_list = list(evaluator.qrels.keys())
            evaluator._prebuild_embedding_cache(queries_list)
            total_queries = len(evaluator.qrels)
            result = {"mode": mode, "dataset": dataset_name}

            if mode in ("full", "both"):
                per_query_live = []
                skipped = 0
                import numpy as np
                from evaluation.metrics import (
                    ndcg_at_k, average_precision, reciprocal_rank,
                    precision_at_k, recall_at_k
                )

                for i, (query, query_qrels) in enumerate(evaluator.qrels.items()):
                    num_relevant = sum(1 for rel in query_qrels.values() if rel > 0)
                    if num_relevant == 0:
                        skipped += 1
                        continue

                    # Retrieve top-10 docs
                    retrieved_ids = evaluator._retrieve_ids(query, top_k=10)
                    row = {
                        "query":        query,
                        "ndcg@10":      round(ndcg_at_k(retrieved_ids, query_qrels, k=10), 4),
                        "map":          round(average_precision(retrieved_ids, query_qrels), 4),
                        "mrr":          round(reciprocal_rank(retrieved_ids, query_qrels), 4),
                        "p@5":          round(precision_at_k(retrieved_ids, query_qrels, k=5), 4),
                        "r@10":         round(recall_at_k(retrieved_ids, query_qrels, k=10), 4),
                        "retrieved":    retrieved_ids[:5],
                        "num_relevant": num_relevant,
                    }
                    per_query_live.append(row)
                    
                    pct = round((i + 1) / total_queries * 100)
                    live_ndcg = round(float(np.mean([q["ndcg@10"] for q in per_query_live])), 4)
                    yield f"data: {json.dumps({'type': 'progress', 'done': i+1, 'total': total_queries, 'scored': len(per_query_live), 'skipped': skipped, 'pct': pct, 'query': query, 'live_ndcg': live_ndcg})}\n\n"

                if strict_mode:
                    successful_queries = per_query_live
                    zero_score_count = 0
                else:
                    successful_queries = [q for q in per_query_live if any(v > 0 for k, v in q.items() if k not in ("query", "retrieved", "num_relevant"))]
                    zero_score_count = len(per_query_live) - len(successful_queries)
                
                def _avg_m(key):
                    return round(float(np.mean([q[key] for q in successful_queries])), 4) if successful_queries else 0.0

                result["metrics"] = {
                    "ndcg@10": _avg_m("ndcg@10"),
                    "map":     _avg_m("map"),
                    "mrr":     _avg_m("mrr"),
                    "p@5":     _avg_m("p@5"),
                    "r@10":    _avg_m("r@10"),
                }
                result["per_query"] = per_query_live
                result["num_queries"] = len(successful_queries)
                # skipped = queries with 0 relevant docs + queries that scored 0.0 across all metrics
                result["skipped"] = skipped + zero_score_count
                result["qrels_source"] = evaluator.qrels_source

            if mode in ("ablation", "both"):
                model_reg = system["components"].get("model_registry", {})
                result["ablation"] = evaluator.ablation_study(
                    strict_mode=strict_mode,
                    model_registry=model_reg,
                )

            yield f"data: {json.dumps({'type': 'done', **result})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'trace': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("\n🔗 Federated RAG System — Web UI")
    print("   Open your browser at: http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)