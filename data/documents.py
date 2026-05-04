"""
================================================================================
File: data/documents.py

This module is responsible for loading the TREC-COVID research paper dataset
into memory.  It uses the `ir_datasets` library to download and iterate over
the official Cord-19 / TREC-COVID corpus (~171,000 papers).  On the first run
the raw data is downloaded and then serialised to a local JSON cache file so
that subsequent restarts are near-instant.

Each loaded document is enriched with an automatically detected medical
sub-category (Epidemiology, Virology, Treatment, etc.) using a keyword-matching
ruleset, and a hex colour code for UI display.  The module also exposes the
50 TREC-COVID expert queries and their graded relevance labels so the
Evaluator can run formal IR metrics (NDCG, MAP, MRR, Precision, Recall).

Key exports:
  DOCUMENTS   - list of enriched document dicts loaded on import
  SEED_QUERIES - curated list of COVID search queries used to seed the
                 query-recommendation history
================================================================================
"""

import os
import json
import re


CATEGORY_RULES = [
    ("Epidemiology",     ["epidemiology", "transmission", "spread", "outbreak",
                          "incidence", "prevalence", "mortality", "case fatality",
                          "reproduction number", "r0", "attack rate"]),
    ("Treatment",        ["treatment", "therapy", "drug", "clinical trial",
                          "remdesivir", "dexamethasone", "hydroxychloroquine",
                          "antiviral", "antibiotic", "vaccine efficacy",
                          "randomized controlled", "placebo"]),
    ("Virology",         ["virus", "viral", "sars-cov", "coronavirus", "spike protein",
                          "rna", "genome", "mutation", "variant", "strain",
                          "replication", "receptor binding", "ace2"]),
    ("Immunology",       ["immune", "antibody", "immunity", "t cell", "b cell",
                          "cytokine", "inflammation", "lymphocyte", "interferon",
                          "innate immunity", "adaptive immunity", "neutralizing"]),
    ("Clinical",         ["patient", "hospital", "icu", "intensive care", "ventilator",
                          "symptom", "diagnosis", "prognosis", "comorbidity",
                          "severity", "pneumonia", "respiratory", "ct scan"]),
    ("Public Health",    ["public health", "lockdown", "quarantine", "social distancing",
                          "mask", "ppe", "contact tracing", "surveillance",
                          "policy", "intervention", "non-pharmaceutical"]),
    ("Mental Health",    ["mental health", "anxiety", "depression", "psychological",
                          "stress", "wellbeing", "psychiatric", "ptsd"]),
    ("Vaccine",          ["vaccine", "vaccination", "immunization", "mrna",
                          "pfizer", "moderna", "astrazeneca", "booster",
                          "herd immunity", "seroprevalence"]),
    ("Genomics",         ["genomic", "sequencing", "phylogenetic", "evolution",
                          "genetic", "bioinformatics", "proteomics"]),
    # "General COVID" is the final fallback — handled by the return at end of guess_category()
]

CATEGORY_COLORS = {
    "Epidemiology":   "#2563eb",
    "Treatment":      "#dc2626",
    "Virology":       "#7c3aed",
    "Immunology":     "#059669",
    "Clinical":       "#d97706",
    "Public Health":  "#0891b2",
    "Mental Health":  "#be185d",
    "Vaccine":        "#16a34a",
    "Genomics":       "#0369a1",
    "General COVID":  "#6b7280",
}


def guess_category(title: str, abstract: str) -> str:
    """Return the best-matching medical sub-category for a document given its title and abstract."""
    combined = (title + " " + title + " " + abstract[:400]).lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in combined for kw in keywords):
            return category
    return "General COVID"  # no keywords matched any rule


def clean_text(text: str, max_chars: int = 1500) -> str:
    """Normalise whitespace and truncate text to at most max_chars characters, breaking on a sentence boundary."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        last_period = cut.rfind('.')
        if last_period > max_chars * 0.7:
            cut = cut[:last_period + 1]
        text = cut
    return text


def load_documents(max_docs: int = 1000) -> list:
    """
    Load up to max_docs documents from the TREC-COVID corpus.

    Checks for a local JSON cache first and returns immediately if found.
    Otherwise downloads the data via ir_datasets, enriches each document with
    category and colour metadata, writes the cache to disk, and returns the list.
    Falls back to a small set of hard-coded dummy documents when ir_datasets is
    unavailable.
    """
    cache_path = f"data/trec_covid_cache_{max_docs}.json"

    if os.path.exists(cache_path):
        print(f"[TREC-COVID] Loading from cache ({max_docs} docs)...")
        with open(cache_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"[TREC-COVID] {len(docs)} documents loaded from cache ✓")
        return docs

    try:
        import ir_datasets

        print(f"[TREC-COVID] Loading dataset via ir_datasets...")
        print("[TREC-COVID] First run downloads ~500MB — please wait...")

        dataset = ir_datasets.load("beir/trec-covid")

        documents = []
        category_counts = {}
        skipped = 0

        print("[TREC-COVID] Processing documents...")

        for doc in dataset.docs_iter():
            if len(documents) >= max_docs:
                break

            doc_id    = str(doc.doc_id)
            title     = (getattr(doc, "title", "") or "").strip()
            abstract  = (getattr(doc, "text", "")  or "").strip()

            if not title and not abstract:
                skipped += 1
                continue

            content = title
            if abstract:
                content = title + ". " + abstract if title else abstract

            category = guess_category(title, abstract)
            category_counts[category] = category_counts.get(category, 0) + 1

            publish_time = getattr(doc, "date", "") or ""
            if not publish_time:
                publish_time = "2020-01-01"

            journal  = getattr(doc, "source_x", "") or \
                       getattr(doc, "journal", "") or "COVID Research"
            authors  = getattr(doc, "authors", "") or ""
            url      = getattr(doc, "url", "") or \
                       f"https://cord-19.apps.allenai.org/paper/{doc_id}"

            documents.append({
                "id":       doc_id,
                "title":    title or f"COVID Paper {doc_id}",
                "category": category,
                "date":     publish_time[:10],
                "content":  clean_text(content, max_chars=1500),
                "color":    CATEGORY_COLORS.get(category, "#6b7280"),
                "url":      url,
                "journal":  journal,
                "authors":  authors[:200] if authors else "",
                "word_count": len(content.split()),
            })

        print(f"\n[TREC-COVID] ✓ Loaded {len(documents)} documents "
              f"(skipped {skipped} empty)")
        print("[TREC-COVID] Category breakdown:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            bar = "▓" * max(1, count // max(1, len(documents) // 30))
            print(f"  {cat:<20} {count:>4}  {bar}")

        os.makedirs("data", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"\n[TREC-COVID] Cached to {cache_path} — next run loads instantly ✓")

        return documents

    except ImportError:
        print("[TREC-COVID] ERROR: ir_datasets not installed")
        print("  Fix: pip install ir_datasets")
        return _dummy_documents()

def load_scifact_documents() -> list:
    """
    Load the full SciFact (Scientific Fact-Checking) corpus (~5,183 documents) via ir_datasets.
    SciFact is a high-quality benchmark for verifying claims against scientific evidence.
    """
    cache_path = "data/scifact_cache.json"

    if os.path.exists(cache_path):
        print(f"[SciFact] Loading from cache (5,183 docs)...")
        with open(cache_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"[SciFact] {len(docs)} documents loaded from cache ✓")
        return docs

    try:
        import ir_datasets
        print(f"[SciFact] Loading dataset via ir_datasets...")
        dataset = ir_datasets.load("beir/scifact")
        
        documents = []
        print("[SciFact] Processing documents...")
        
        for doc in dataset.docs_iter():
            doc_id = str(doc.doc_id)
            title = (getattr(doc, "title", "") or "").strip()
            abstract = (getattr(doc, "text", "") or "").strip()
            
            content = title
            if abstract:
                content = title + ". " + abstract if title else abstract

            documents.append({
                "id": doc_id,
                "title": title or f"SciFact Doc {doc_id}",
                "category": "Science",
                "date": "2020-01-01",
                "content": clean_text(content, max_chars=1500),
                "color": "#4f46e5",  # Primary Indigo
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}",
                "journal": "Scientific Literature",
                "authors": "",
                "word_count": len(content.split()),
            })

        print(f"[SciFact] ✓ Loaded {len(documents)} documents")
        os.makedirs("data", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"[SciFact] Cached to {cache_path} ✓")
        
        return documents

    except Exception as e:
        print(f"[SciFact] ERROR: {e}")
        return []


def load_scifact_queries_and_labels() -> tuple:
    """
    Load the official SciFact test queries and their binary relevance labels (qrels).

    SciFact in ir_datasets is split across two paths:
      - ``beir/scifact``       -> full corpus (5,183 docs), NO qrels
      - ``beir/scifact/test``  -> 300 test queries + 339 qrel entries (correct path)

    The corpus documents come from the top-level dataset; queries and qrels
    must be loaded from the /test sub-split.
    """
    try:
        import ir_datasets
        # Queries + qrels live under the /test sub-split, NOT the top-level dataset.
        # beir/scifact has has_qrels=False; beir/scifact/test has has_qrels=True.
        test_dataset = ir_datasets.load("beir/scifact/test")
        queries = {}
        labels = {}

        for query in test_dataset.queries_iter():
            queries[str(query.query_id)] = query.text

        for qrel in test_dataset.qrels_iter():
            qid = str(qrel.query_id)
            did = str(qrel.doc_id)
            rel = int(qrel.relevance)
            if qid not in labels:
                labels[qid] = {}
            labels[qid][did] = rel

        print(f"[SciFact] Loaded {len(queries)} test queries, "
              f"{sum(len(v) for v in labels.values())} relevance labels \u2713")
        return queries, labels

    except Exception as e:
        print(f"[SciFact] Could not load SciFact queries/labels: {e}")
        return {}, {}


def load_queries_and_labels() -> tuple:
    """
    Load the 50 official TREC-COVID queries and their graded relevance labels (qrels).

    Returns a tuple (queries, labels) where:
      queries – dict mapping query_id (str) to query text (str)
      labels  – dict mapping query_id to {doc_id: relevance_score}
                with relevance_score in {0, 1, 2}
    Returns ({}, {}) if ir_datasets is unavailable.
    """
    try:
        import ir_datasets

        dataset = ir_datasets.load("beir/trec-covid")
        queries = {}
        labels  = {}

        for query in dataset.queries_iter():
            queries[str(query.query_id)] = query.text

        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            did = str(qrel.doc_id)
            rel = int(qrel.relevance)
            if qid not in labels:
                labels[qid] = {}
            labels[qid][did] = rel

        print(f"[TREC-COVID] Loaded {len(queries)} queries, "
              f"{sum(len(v) for v in labels.values())} relevance labels ✓")

        return queries, labels

    except Exception as e:
        print(f"[TREC-COVID] Could not load queries/labels: {e}")
        return {}, {}


def _dummy_documents() -> list:
    """Return a minimal list of hand-crafted COVID documents used as a fallback when ir_datasets is missing."""
    return [
        {
            "id": "dummy_001",
            "title": "COVID-19 Transmission Dynamics",
            "category": "Epidemiology",
            "date": "2020-03-15",
            "content": "COVID-19 spreads primarily through respiratory droplets. "
                       "The basic reproduction number R0 is estimated between 2 and 3.",
            "color": CATEGORY_COLORS["Epidemiology"],
            "url": "", "journal": "Nature Medicine",
            "authors": "Smith et al.", "word_count": 28,
        },
        {
            "id": "dummy_002",
            "title": "SARS-CoV-2 Spike Protein Structure",
            "category": "Virology",
            "date": "2020-04-10",
            "content": "The SARS-CoV-2 spike protein binds to ACE2 receptors "
                       "on human cells to facilitate viral entry and infection.",
            "color": CATEGORY_COLORS["Virology"],
            "url": "", "journal": "Cell",
            "authors": "Zhang et al.", "word_count": 27,
        },
        {
            "id": "dummy_003",
            "title": "COVID-19 Treatment with Remdesivir",
            "category": "Treatment",
            "date": "2020-05-22",
            "content": "Remdesivir showed promise as antiviral treatment "
                       "in randomized controlled trials for COVID-19 patients.",
            "color": CATEGORY_COLORS["Treatment"],
            "url": "", "journal": "NEJM",
            "authors": "Beigel et al.", "word_count": 24,
        },
        {
            "id": "dummy_004",
            "title": "mRNA Vaccine Development for COVID-19",
            "category": "Vaccine",
            "date": "2020-12-01",
            "content": "mRNA vaccines developed by Pfizer-BioNTech and Moderna "
                       "demonstrated over 90% efficacy in phase 3 clinical trials.",
            "color": CATEGORY_COLORS["Vaccine"],
            "url": "", "journal": "NEJM",
            "authors": "Polack et al.", "word_count": 26,
        },
        {
            "id": "dummy_005",
            "title": "Mental Health Impact of COVID-19 Lockdowns",
            "category": "Mental Health",
            "date": "2020-06-15",
            "content": "Lockdown measures during the COVID-19 pandemic significantly "
                       "increased rates of anxiety and depression in the general population.",
            "color": CATEGORY_COLORS["Mental Health"],
            "url": "", "journal": "Lancet Psychiatry",
            "authors": "Daly et al.", "word_count": 27,
        },
    ]


SEED_QUERIES = [
    "what is the origin of COVID-19",
    "how does coronavirus spread between people",
    "what are the symptoms of COVID-19",
    "how effective are masks against COVID transmission",
    "what treatments work for severe COVID-19",
    "how do mRNA vaccines work against COVID",
    "what is the mortality rate of COVID-19",
    "how does SARS-CoV-2 infect human cells",
    "what is the role of ACE2 receptor in COVID infection",
    "how long does COVID immunity last after infection",
    "what are the long term effects of COVID-19",
    "how does COVID affect the immune system",
    "what is the difference between COVID variants",
    "how effective is remdesivir for COVID treatment",
    "what is the impact of COVID on mental health",
    "how does COVID spread in children",
    "what is the incubation period of COVID-19",
    "how does social distancing reduce COVID spread",
    "what are COVID-19 risk factors for severe disease",
    "how does contact tracing work for COVID",
]


DOCUMENTS = load_documents(max_docs=5000)