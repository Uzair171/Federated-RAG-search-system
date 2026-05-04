"""
================================================================================
File: query_processing/spell_corrector.py

Provides fast spelling correction for user search queries using the SymSpell
algorithm.  SymSpell pre-generates all delete-variants of every dictionary word
at load time, enabling O(1) lookup at query time — orders of magnitude faster
than BK-tree or edit-distance loop approaches.

The corrector loads the bundled SymSpellPy 82,765-word English frequency
dictionary and then injects every significant word extracted from the loaded
document corpus as a high-frequency protected term.  This prevents SymSpell
from "correcting" legitimate domain vocabulary (e.g. "remdesivir", "seroprevalence",
"immunoglobulin") — regardless of which documents are loaded, the protection
list always reflects the actual dataset rather than a hardcoded partial word list.
================================================================================
"""

import os
import re
from symspellpy import SymSpell, Verbosity


class SpellCorrector:
    """
    Corrects spelling errors in user queries using the SymSpell symmetric-delete algorithm.

    Loads the full 82,765-word English frequency dictionary at init time, then
    augments it with domain terms extracted directly from the loaded document
    corpus so that any word that appears in the dataset is protected from correction.
    Corrections are applied word-by-word while preserving original capitalisation.
    """

    def __init__(self, max_edit_distance: int = 2, documents: list = None):
        """
        Initialise SymSpell with the bundled frequency dictionary and corpus-derived domain terms.

        Args:
            max_edit_distance: Maximum edit distance considered for a correction (default 2).
            documents:         Loaded document corpus; meaningful words are extracted and
                               registered as protected terms so the corrector never
                               alters legitimate vocabulary from the dataset.
        """
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self._load_dictionary()
        if documents:
            self._add_corpus_terms(documents)

    def _load_dictionary(self):
        """Load the SymSpellPy bundled 82,765-word English dictionary; fall back to a minimal set if missing."""
        import symspellpy
        package_dir = os.path.dirname(symspellpy.__file__)
        dict_path = os.path.join(package_dir, "frequency_dictionary_en_82_765.txt")

        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        else:
            self._load_fallback_dictionary()

    def _load_fallback_dictionary(self):
        """Populate a bare-minimum dictionary from the most common English function words when the bundled file is missing."""
        common_words = [
            "the", "is", "are", "what", "how", "does", "work", "with",
            "and", "for", "privacy", "learning", "model", "data", "system",
            "federated", "retrieval", "generation", "embedding", "vector",
        ]
        for word in common_words:
            self.sym_spell.create_dictionary_entry(word, 1000)

    def _add_corpus_terms(self, documents: list):
        """
        Extract every significant word from document titles AND content and register
        it as a protected high-frequency term.

        This ensures that any word appearing in the loaded corpus — whether in a
        title ("remdesivir") or only in an abstract ("thrombocytopenia",
        "seroconversion", "myocarditis") — is automatically protected from
        correction without requiring a manually maintained list.  Words shorter
        than 4 characters and common English stopwords are skipped.
        """
        stopwords = {
            "that", "this", "with", "from", "they", "been", "have",
            "were", "said", "each", "which", "their", "time", "will",
            "about", "would", "there", "could", "other", "after",
            "more", "also", "into", "than", "then", "some", "these",
            "when", "make", "like", "most", "over", "such", "even",
            "between", "through", "during", "before", "under", "while",
            "study", "using", "based", "used", "found", "show", "high",
            "well", "case", "risk", "data", "test", "type", "both",
        }

        seen: set = set()
        protected = 0

        for doc in documents:
            title_words = re.findall(r'\b[a-zA-Z]{4,}\b', doc.get("title", "").lower())
            content_words = re.findall(r'\b[a-zA-Z]{4,}\b', doc.get("content", "")[:500].lower())

            for word in title_words + content_words:
                if word not in stopwords and word not in seen:
                    seen.add(word)
                    self.sym_spell.create_dictionary_entry(word, 200000)
                    protected += 1

        print(f"[SpellCorrector] {protected} corpus terms registered as protected vocabulary ✓")

    def correct(self, query: str) -> str:
        """
        Return a spell-corrected version of the query string.

        Each word is corrected independently.  Short words (≤2 chars), numbers,
        and ALL-CAPS abbreviations are passed through unchanged.  The original
        capitalisation pattern of each word is preserved in the corrected output.
        """
        if not query or not query.strip():
            return query

        words = query.split()
        corrected_words = []

        for word in words:
            leading = re.match(r'^([^a-zA-Z]*)', word).group(1)
            trailing = re.match(r'.*?([^a-zA-Z]*)$', word).group(1)
            core_word = word[len(leading):len(word) - len(trailing) if trailing else len(word)]

            if not core_word:
                corrected_words.append(word)
                continue

            if len(core_word) <= 2 or core_word.isdigit() or core_word.isupper():
                corrected_words.append(word)
                continue

            suggestions = self.sym_spell.lookup(
                core_word.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=2,
                include_unknown=True
            )

            if suggestions:
                corrected = suggestions[0].term
                corrected = self._match_case(core_word, corrected)
                corrected_words.append(leading + corrected + trailing)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def _match_case(self, original: str, corrected: str) -> str:
        """Apply the capitalisation pattern of the original word to the corrected word."""
        if original[0].isupper():
            return corrected.capitalize()
        return corrected
