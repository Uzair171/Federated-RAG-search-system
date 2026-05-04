# retrieval package — lazy imports to avoid hard-failing without ML deps

def __getattr__(name):
    if name == "Embedder":
        from .embedder import Embedder
        return Embedder
    if name == "HybridSearcher":
        from .hybrid_search import HybridSearcher
        return HybridSearcher
    if name == "Reranker":
        from .reranker import Reranker
        return Reranker
    raise AttributeError(f"module 'retrieval' has no attribute {name!r}")

__all__ = ["Embedder", "HybridSearcher", "Reranker"]
