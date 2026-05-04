import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.documents import SEED_QUERIES, DOCUMENTS
from retrieval.federated_search import FederatedSearcher
from query_processing.hyde_expander import HyDEExpander
from retrieval.reranker import CrossEncoderReranker
from evaluation.metrics import Evaluator

def main():
    print("\n" + "="*60)
    print("STARTING FULL SYSTEM ABLATION STUDY")
    print("="*60)

    # 1. Initialize the Evaluator
    evaluator = Evaluator()

    # 2. Extract Document dictionaries (to match exactly how app.py does it)
    print("\n[Ablation] Loading dataset...")
    
    # 3. Initialize Shared Components
    print("\n[Ablation] Loading HyDE and CrossEncoder (this takes a few seconds)...")
    hyde = HyDEExpander()
    reranker = CrossEncoderReranker()
    
    # We will use the SEED_QUERIES for evaluation.
    queries = SEED_QUERIES
    print(f"\n[Ablation] Evaluator ready — {len(queries)} queries queued.")

    # -------------------------------------------------------------------------
    # Helper Function to run a configuration
    # -------------------------------------------------------------------------
    def run_config(config_name, searcher, use_bm25=True, use_reranker=True):
        print(f"\n\n--- RUNNING CONFIG: {config_name} ---")
        
        searcher.use_bm25 = use_bm25
        
        scores = []
        
        # The tqdm progress bar with percentage and estimated time!
        for query_dict in tqdm(queries, desc=f"Evaluating {config_name}", unit="q"):
            original_query = query_dict["query"]
            relevant_doc_ids = query_dict["relevant_doc_ids"]
            
            # Step 1: HyDE Expansion
            expanded_query, _ = hyde.expand(original_query)
            
            # Step 2: Federated Retrieval
            search_results, _ = searcher.search(
                query_text=expanded_query, 
                query_embedding=None, 
                top_k=10, 
                metadata_filters={}
            )
            
            # Step 3: CrossEncoder Reranking (Conditional)
            if use_reranker:
                final_results = reranker.rerank(original_query, search_results, top_k=10)
            else:
                final_results = search_results[:10]
            
            # Step 4: Calculate NDCG@10
            retrieved_ids = [r["id"] for r in final_results]
            ndcg = evaluator.calculate_ndcg(retrieved_ids, relevant_doc_ids, k=10)
            scores.append(ndcg)
            
        final_score = sum(scores) / len(scores) if scores else 0
        print(f"\n[RESULT] {config_name} -> NDCG@10 = {final_score:.4f}")
        return final_score

    # -------------------------------------------------------------------------
    # DEFINE THE 4 CONFIGURATIONS
    # -------------------------------------------------------------------------
    
    shard_names = [f"shard_{i}" for i in range(1, 9)]

    bge_config = {s: "bge" for s in shard_names}
    e5_config = {s: "e5" for s in shard_names}
    nomic_config = {s: "nomic" for s in shard_names}
    
    diverse_config = {}
    models = ["bge", "e5", "nomic"]
    for i, s in enumerate(shard_names):
        diverse_config[s] = models[i % 3]

    # -------------------------------------------------------------------------
    # RUN THE EVALUATIONS
    # -------------------------------------------------------------------------
    results = {}
    
    print("\n[Ablation] Initializing Database Shards into RAM (this takes a moment)...")
    searcher_bge = FederatedSearcher(num_shards=8, strategy="random", retrieval_cycle=["bge"])
    searcher_bge.build_index(DOCUMENTS)
    searcher_e5 = FederatedSearcher(num_shards=8, strategy="random", retrieval_cycle=["e5"])
    searcher_e5.build_index(DOCUMENTS)
    searcher_nomic = FederatedSearcher(num_shards=8, strategy="random", retrieval_cycle=["nomic"])
    searcher_nomic.build_index(DOCUMENTS)
    searcher_diverse = FederatedSearcher(num_shards=8, strategy="random")
    searcher_diverse.build_index(DOCUMENTS)
    
    # 1. BGE Only (Full Pipeline)
    results["BGE Only (Full Pipeline)"] = run_config(
        "BGE Only (Full Pipeline)", searcher_bge, use_bm25=True, use_reranker=True
    )
    
    # 2. E5 Only (Full Pipeline)
    results["E5 Only (Full Pipeline)"] = run_config(
        "E5 Only (Full Pipeline)", searcher_e5, use_bm25=True, use_reranker=True
    )
    
    # 3. Nomic Only (Full Pipeline)
    results["Nomic Only (Full Pipeline)"] = run_config(
        "Nomic Only (Full Pipeline)", searcher_nomic, use_bm25=True, use_reranker=True
    )

    # 4. Federated Diverse (Hybrid BM25, No CrossEncoder)
    results["Federated Diverse (Hybrid BM25)"] = run_config(
        "Federated Diverse (Hybrid BM25)", searcher_diverse, use_bm25=True, use_reranker=False
    )
    
    # 5. Federated Diverse (Full Pipeline)
    results["Federated Diverse (Full Pipeline)"] = run_config(
        "Federated Diverse (Full Pipeline)", searcher_diverse, use_bm25=True, use_reranker=True
    )

    # Print Final Summary
    print("\n" + "="*60)
    print("FINAL ABLATION RESULTS (NDCG@10)")
    print("="*60)
    for name, score in results.items():
        print(f"{name:50s}: {score:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
