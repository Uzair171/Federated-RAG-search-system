# query_processing package — lazy imports to avoid hard-failing without ML deps

def __getattr__(name):
    if name == "SpellCorrector":
        from .spell_corrector import SpellCorrector
        return SpellCorrector
    if name == "HyDEExpander":
        from .hyde_expander import HyDEExpander
        return HyDEExpander
    if name == "QueryRecommender":
        from .recommender import QueryRecommender
        return QueryRecommender
    raise AttributeError(f"module 'query_processing' has no attribute {name!r}")

__all__ = ["SpellCorrector", "HyDEExpander", "QueryRecommender"]
