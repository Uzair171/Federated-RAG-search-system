"""
================================================================================
File: query_processing/hyde_expander.py

Query expansion module implementing two strategies in priority order:

1. HyDE (Hypothetical Document Embeddings) via Ollama — highest quality.
   Instructs the local LLM to generate a short hypothetical answer document for
   the query.  The expanded text naturally handles abbreviations, synonyms, and
   related concepts without any curated lookup table — because the LLM already
   knows the full vocabulary.  For example a query "RAG systems" will produce a
   paragraph containing "Retrieval-Augmented Generation", "dense retrieval",
   "language model", etc.  This expansion is then embedded and used as the
   search vector, bridging the vocabulary gap between short keyword queries and
   long academic abstracts.

2. WordNet + SentenceTransformers automatic expansion — no LLM required.
   When Ollama is unavailable this path is used as a fallback.
   - NLTK WordNet (80,000+ English synsets) provides full-dictionary synonym
     lookup for every meaningful word in the query with no hardcoded lists.
   - SentenceTransformers cosine similarity against the in-domain document
     vocabulary surfaces corpus-specific related terms.
   Both sources are deduplicated and concatenated to the original query.

Note: A hardcoded abbreviation map was previously used as a third fallback
but has been removed.  It only covered ~30 acronyms and was inconsistent —
why expand "RAG" but not "BERT", "ACE2", or "mRNA"?  HyDE handles all
abbreviations contextually via the LLM, and WordNet handles common English
expansions via its complete 80k-synset dictionary.  No static list is needed.

The two strategies are transparent to the caller: expand() always returns
(expanded_text, method_description).
================================================================================
"""

import json
import os
import re
import requests
from typing import Tuple, List, Set


HYDE_PROMPT_TEMPLATE = """Write a short, dense, factual paragraph (3-5 sentences) that would directly answer the following question.
Write it as if it's an excerpt from a technical document or textbook.
Do NOT say "This paragraph answers..." — just write the content directly.

Question: {query}

Paragraph:"""


class HyDEExpander:
    """
    Query expander with two fallback levels: HyDE via Ollama, then WordNet + SentenceTransformers.

    At init time checks Ollama availability and optionally builds an in-domain
    document vocabulary for SentenceTransformers-based synonym discovery.
    No hardcoded abbreviation lists are used; full-dictionary coverage comes from
    WordNet (80k synsets) or from the LLM when Ollama is available.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        documents: list = None,
        embedder=None,
    ):
        """
        Initialise the expander.

        Args:
            model:      Ollama model name to use for HyDE generation.
            ollama_url: URL of the local Ollama REST API.
            documents:  Document corpus used to build the in-domain vocabulary.
            embedder:   Embedder instance for SentenceTransformers similarity search.
        """
        self.model = model
        self.ollama_url = ollama_url
        self.documents = documents or []
        self.embedder = embedder

        # Persistent disk cache: maps (model, query) -> expanded text.
        # Ollama is only called ONCE per unique query; all subsequent calls
        # (including re-running evaluation) are served from disk instantly.
        self._cache_path = "data/hyde_expansion_cache.json"
        self._disk_cache: dict = self._load_disk_cache()

        self.ollama_available = self._check_ollama()
        self.wordnet_available = self._setup_wordnet()

        self.doc_vocab = []
        self.doc_vocab_embeddings = None
        if self.embedder and self.documents:
            self._build_doc_vocab()

    def _load_disk_cache(self) -> dict:
        """Load the persistent HyDE expansion cache from disk (empty dict if not found)."""
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_disk_cache(self) -> None:
        """Persist the HyDE expansion cache to disk so future runs skip Ollama for known queries."""
        try:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._disk_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Expander] WARNING: could not save HyDE cache: {e}")

    def _check_ollama(self) -> bool:
        """Ping the Ollama API and return True if the configured model (or any fallback) is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = [m.get("name", "").split(":")[0] for m in models]
                if self.model.split(":")[0] in available:
                    return True
                for fallback in ["llama3.2", "mistral", "llama3", "phi3", "gemma", "tinyllama"]:
                    if fallback in available:
                        self.model = fallback
                        return True
                if available:
                    self.model = available[0]
                    return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        return False

    def _setup_wordnet(self) -> bool:
        """Initialise NLTK WordNet (80k+ English synsets), downloading data on first run, and store the module reference."""
        try:
            import nltk
            from nltk.corpus import wordnet

            try:
                wordnet.synsets("test")
            except LookupError:
                print("[Expander] Downloading WordNet (one time, ~10MB)...")
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            self._wordnet = wordnet
            return True

        except ImportError:
            print("[Expander] NLTK not installed — WordNet expansion disabled")
            print("  Run: pip install nltk")
            return False

    def _get_wordnet_synonyms(self, query: str) -> List[str]:
        """
        Return WordNet synonyms for all meaningful words in the query.

        Looks up every non-stopword in the query against NLTK WordNet's 80,000+
        synsets covering the full English vocabulary — no curated list needed.
        Returns unique synonym strings, excluding the original query words.
        """
        if not self.wordnet_available:
            return []

        synonyms: Set[str] = set()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())

        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "her", "was", "one", "our", "out", "day", "get",
            "has", "him", "his", "how", "its", "may", "new", "now",
            "old", "see", "two", "way", "who", "boy", "did", "does",
            "let", "put", "say", "she", "too", "use", "what", "with",
        }

        for word in words:
            if word in stopwords or len(word) < 3:
                continue

            try:
                for syn in self._wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        term = lemma.name().replace("_", " ").lower()
                        if term != word and len(term) > 2:
                            synonyms.add(term)
            except Exception:
                continue

        return list(synonyms)

    def _build_doc_vocab(self):
        """Extract and encode unique meaningful words from the first 500 documents to form an in-domain vocabulary for similarity search."""
        try:
            import numpy as np

            print("[Expander] Building document vocabulary for synonym expansion...")

            vocab_set: Set[str] = set()
            for doc in self.documents[:500]:
                title_words = re.findall(r'\b[a-zA-Z]{4,}\b', doc.get("title", "").lower())
                vocab_set.update(title_words)

                content_words = re.findall(r'\b[a-zA-Z]{4,}\b', doc.get("content", "")[:300].lower())
                vocab_set.update(content_words)

            common_words = {
                "that", "this", "with", "from", "they", "been", "have",
                "were", "said", "each", "which", "their", "time", "will",
                "about", "would", "there", "could", "other", "after",
                "more", "also", "into", "than", "then", "some", "these",
                "when", "make", "like", "most", "over", "such", "even",
                "between", "through", "during", "before", "under", "while",
            }
            vocab_list = [w for w in vocab_set if w not in common_words and len(w) >= 4]

            self.doc_vocab = vocab_list[:3000]

            if self.doc_vocab:
                self.doc_vocab_embeddings = self.embedder.encode(self.doc_vocab)
                print(f"[Expander] Vocab built: {len(self.doc_vocab)} terms encoded ✓")

        except Exception as e:
            print(f"[Expander] Vocab build failed: {e}")
            self.doc_vocab = []
            self.doc_vocab_embeddings = None

    def _get_st_synonyms(self, query: str, topn: int = 8) -> List[str]:
        """Find terms from the document vocabulary that are semantically closest to the query via cosine similarity."""
        if (
            self.embedder is None
            or self.doc_vocab_embeddings is None
            or not self.doc_vocab
        ):
            return []

        try:
            import numpy as np

            query_emb = self.embedder.encode([query])[0]
            similarities = self.doc_vocab_embeddings @ query_emb
            top_indices = np.argsort(similarities)[::-1][:topn]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.4:
                    term = self.doc_vocab[idx]
                    if term.lower() not in query.lower():
                        results.append(term)

            return results

        except Exception:
            return []

    def expand(self, query: str) -> Tuple[str, str]:
        """
        Expand the query using the best available method and return (expanded_text, method_name).

        Path 1 — Disk cache hit: returns the previously generated expansion instantly.
        Path 2 — HyDE via Ollama (preferred): generates a full hypothetical answer
        document, saves result to disk so the same query is never sent to Ollama twice.
        Path 3 — WordNet + SentenceTransformers (fallback): fast, no LLM needed.
        """
        cache_key = f"{self.model}::{query}"

        if cache_key in self._disk_cache:
            print(f"  [HyDE] ✓ Cache Hit: '{query}'")
            return self._disk_cache[cache_key], f"HyDE (cached / {self.model})"

        if self.ollama_available:
            result = self._hyde_with_ollama(query)
            if result:
                self._disk_cache[cache_key] = result
                self._save_disk_cache()
                print(f"  [HyDE] ✓ Ollama Success: '{query}'")
                return result, f"HyDE (Ollama/{self.model})"

        print(f"  [HyDE] ⚠ WordNet Fallback: '{query}'")
        return self._automatic_expansion(query), "Auto-Expansion (WordNet + SentenceTransformers)"

    def _hyde_with_ollama(self, query: str) -> str:
        """Call the Ollama API to generate a hypothetical answer document for the query; return empty string on failure."""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                        "top_p": 0.9,
                    },
                },
                timeout=120,   
            )
            if response.status_code == 200:
                hypothetical_doc = response.json().get("response", "").strip()
                if hypothetical_doc:
                    return f"{query}\n\n{hypothetical_doc}"
            self.ollama_available = False
            return ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            self.ollama_available = False
            return ""

    def _automatic_expansion(self, query: str) -> str:
        """
        Build an expanded query string using WordNet synonyms and SentenceTransformers vocabulary similarity.

        No static abbreviation map is used — WordNet's full English dictionary
        provides synonym coverage for any word in any domain, and the
        SentenceTransformers similarity search surfaces corpus-specific terms.
        Results are deduplicated before joining.
        """
        all_terms = [query]

        wordnet_synonyms = self._get_wordnet_synonyms(query)
        all_terms.extend(wordnet_synonyms[:12])

        st_synonyms = self._get_st_synonyms(query, topn=8)
        all_terms.extend(st_synonyms)

        seen = set()
        unique_terms = []
        for term in all_terms:
            term_clean = term.lower().strip()
            if term_clean and term_clean not in seen:
                seen.add(term_clean)
                unique_terms.append(term)

        return " ".join(unique_terms)

    def set_documents_and_embedder(self, documents: list, embedder):
        """Update the document corpus and embedder after initialisation, then rebuild the in-domain vocabulary."""
        self.documents = documents
        self.embedder = embedder
        if documents and embedder:
            self._build_doc_vocab()