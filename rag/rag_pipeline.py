"""
================================================================================
File: rag/rag_pipeline.py

RAG generation pipeline that combines retrieved documents with a local LLM.

The pipeline takes the top-5 reranked documents from the federated search stage,
formats them into a structured context prompt, and sends a single request to a
locally running Ollama server (default model: llama3.2).  Generation is entirely
local — no data leaves the machine, satisfying the system's privacy requirement.

If Ollama is not installed or the request times out, the pipeline automatically
degrades to extractive summarisation: it selects the most query-relevant
sentences from the retrieved documents using keyword overlap scoring and returns
them as a readable plain-text answer.

The system prompt instructs the model to:
  1. Format its answer in at least 2 paragraphs.
  2. Explicitly cite which documents were most useful.
  3. Call out documents that contained no relevant information.
================================================================================
"""

import requests
import json
from typing import List

RAG_SYSTEM_PROMPT = """You are a knowledgeable AI assistant specializing in biomedical research and COVID-19.

Answer the user's question comprehensively based ONLY on the provided context documents.

CRITICAL INSTRUCTIONS:
1. FORMAT: You MUST format your answer into at least 2 or 3 clearly separated paragraphs. Do not write a single massive block of text.
2. CITATION & RELEVANCE: You MUST explicitly critique the provided documents. In your final paragraph, explicitly mention which documents were actively useful for your answer (e.g. "This information relies mostly on Document 1 and Document 2"), AND explicitly call out if any of the provided documents were completely irrelevant to the question (e.g. "Documents 4 and 5 contained no relevant information regarding this query.").

Be factual, precise, and highly detailed.
"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline connecting the retrieval layer to a local Ollama LLM.

    Supports direct Ollama generation (streaming or blocking) and falls back to
    extractive sentence-level summarisation when the LLM is unavailable.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        request_timeout: float = 350.0,
    ):
        """
        Initialise the RAG pipeline.

        Args:
            model:           Ollama model name to use for generation.
            ollama_url:      URL of the local Ollama REST API server.
            request_timeout: Maximum seconds to wait for an Ollama response.
        """
        self.model = model
        self.ollama_url = ollama_url
        self.request_timeout = request_timeout
        self.ollama_available = self._check_ollama()

        if self.ollama_available:
            print(f"[RAG] Ollama ready — model: {self.model}")
        else:
            print("[RAG] Ollama not found — using extractive fallback")

    def _check_ollama(self) -> bool:
        """Verify Ollama is running and that the configured model (or a fallback) is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]

                if self.model.split(":")[0] in model_names:
                    return True
                for fallback in ["llama3.2", "mistral", "llama3", "phi3", "gemma", "tinyllama"]:
                    if fallback in model_names:
                        self.model = fallback
                        return True
                if model_names:
                    self.model = model_names[0]
                    return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        return False

    def generate(
        self,
        query: str,
        retrieved_docs: List[dict],
        stream: bool = False,
    ) -> str:
        """
        Generate an answer from the retrieved documents using the configured LLM.

        Routes to the direct Ollama path when available, or the extractive
        fallback otherwise.  Returns a plain-text answer string.
        """
        if not retrieved_docs:
            return "No relevant documents found. Please try a different query."

        if self.ollama_available:
            return self._generate_with_ollama_direct(query, retrieved_docs, stream)
        else:
            return self._extractive_fallback(query, retrieved_docs)

    def _generate_with_ollama_direct(
        self, query: str, retrieved_docs: List[dict], stream: bool
    ) -> str:
        """Build a context prompt from the retrieved documents and call Ollama for generation."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            title = metadata.get("title", f"Document {i}")
            content = doc.get("content", "")[:500]
            context_parts.append(f"[{i}] {title}\n{content}")

        context = "\n\n".join(context_parts)

        prompt = f"""{RAG_SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        if stream:
            return self._stream_ollama(prompt)
        else:
            return self._call_ollama(prompt)

    def _stream_ollama(self, prompt: str) -> str:
        """Send a streaming request to Ollama, printing each token to stdout and returning the full assembled response."""
        full_response = ""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500,
                        "num_ctx": 2048,
                    },
                },
                stream=True,
                timeout=self.request_timeout,
            )

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if chunk.get("done", False):
                        break

            print()
            return full_response

        except Exception as e:
            return self._extractive_fallback_simple(str(e))

    def _call_ollama(self, prompt: str) -> str:
        """Send a blocking (non-streaming) request to Ollama and return the response text."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500,
                        "num_ctx": 2048,
                    },
                },
                timeout=self.request_timeout,
            )
            return response.json().get("response", "").strip()
        except Exception as e:
            return self._extractive_fallback_simple(str(e))

    def _extractive_fallback_simple(self, error: str = "") -> str:
        """Return a brief timeout message suggesting a lighter Ollama model."""
        return f"[Generation timed out — try 'ollama pull llama3.2:1b' for a faster model]"

    def _extractive_fallback(self, query: str, retrieved_docs: List[dict]) -> str:
        """
        Build an answer by extracting and ranking the most query-relevant sentences from the retrieved documents.

        Scores each sentence from the top-3 documents by keyword overlap with the
        query, selects the top 4 sentences, and formats them into a readable answer.
        """
        query_words = set(query.lower().split())
        stopwords = {
            "the", "is", "are", "what", "how", "does", "a", "an",
            "and", "or", "in", "of", "to", "for", "with", "on", "at"
        }
        query_words -= stopwords

        all_sentences = []
        for doc in retrieved_docs[:3]:
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "Document")
            sentences = [
                s.strip() for s in doc["content"].split(". ")
                if len(s.strip()) > 30
            ]
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                all_sentences.append((overlap, title, sent))

        all_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = all_sentences[:4]

        if not top_sentences:
            return retrieved_docs[0]["content"][:500] + "..." if retrieved_docs else "No answer found."

        answer_parts = ["Based on the retrieved documents:\n"]
        seen_titles = set()
        for score, title, sent in top_sentences:
            if title not in seen_titles:
                answer_parts.append(f"\nFrom '{title}':")
                seen_titles.add(title)
            answer_parts.append(f"  {sent}.")

        answer_parts.append(
            "\n\n[Extractive summary — run 'ollama pull llama3.2:1b' for faster LLM generation]"
        )

        return "\n".join(answer_parts)