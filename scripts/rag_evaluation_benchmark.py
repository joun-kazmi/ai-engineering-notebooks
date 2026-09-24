#!/usr/bin/env python
# coding: utf-8
"""RAG evaluation benchmark: BM25 vs dense vs hybrid/RRF vs hybrid+rerank.

The other retrieval notebooks build each piece (chunking, hybrid search,
reranking). This one answers the question none of them ask: which
combination is actually *better*, and by how much?

Runs two ways:

  * No LLM_PROVIDER API key configured -> OFFLINE mode. Dense retrieval uses
    a deterministic hashed bag-of-words embedder (not a real embedding
    model) and the reranker uses a token-overlap heuristic instead of an
    LLM judge. This lets the whole pipeline, and every metric below, run
    end to end with no network access and no cost, which is what produced
    the numbers committed in this repo.
  * A real key configured (see .env.example / src/ai_engineering/config.py)
    -> uses the real embedding model and an LLM judge for reranking and
    faithfulness. Same code path, real numbers. Expect them to differ from
    the offline ones, and to differ again if you swap LLM_PROVIDER/LLM_MODEL.

Metrics: Recall@K, MRR, nDCG@K (retrieval quality), latency per query,
and an approximate token-cost per query for the LLM-touching methods.
"""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ai_engineering.config import NO_THINK, get_settings, make_chat_client

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

try:
    # tiktoken downloads its BPE file on first use; some sandboxed/offline
    # environments block that fetch even though tiktoken itself is installed.
    ENCODING = tiktoken.get_encoding("cl100k_base")
    _count_tokens = lambda text: len(ENCODING.encode(text))
except Exception:
    print("WARNING: tiktoken's encoding data isn't reachable here; falling back "
          "to a chars/4 token estimate for the cost figures below.")
    _count_tokens = lambda text: max(1, len(text) // 4)

# Illustrative only — providers price per-token differently and change prices
# over time. Edit to your provider's actual rate before trusting the dollar
# figures; the token counts themselves are accurate.
ASSUMED_USD_PER_1K_TOKENS = 0.002


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_corpus() -> list[str]:
    raw = (DATA_DIR / "rag_eval_corpus.txt").read_text()
    return [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]


def load_queries() -> list[dict]:
    return json.loads((DATA_DIR / "rag_eval_queries.json").read_text())


# ========== METRICS (pure functions — no network, unit tested) ==========

def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Binary relevance nDCG@k."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ========== EMBEDDERS ==========

class OfflineEmbedder:
    """Deterministic, offline stand-in for a real embedding model.

    Hashes tokens into a fixed-size bag-of-words vector. This captures
    lexical overlap, *not* semantics — paraphrases with little shared
    vocabulary will NOT score well here the way a real embedding model
    would. It exists so the pipeline is runnable and testable without an
    API key; treat its "dense retrieval" numbers as a structural demo, not
    evidence about real embedding-model quality.
    """
    name = "offline-hashed-bow (NOT a real embedding model)"
    dims = 256

    @staticmethod
    def _stable_hash(token: str) -> int:
        # Python's built-in hash() is randomized per-process for strings
        # (PYTHONHASHSEED); md5 keeps this embedder's output reproducible
        # across runs, which matters for numbers committed in this repo.
        return int(hashlib.md5(token.encode()).hexdigest(), 16)

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dims)
            for tok in tokenize(text):
                vec[self._stable_hash(tok) % self.dims] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append((vec / norm if norm > 0 else vec).tolist())
        return vectors


class NimEmbedder:
    """Real embedding model via the configured OpenAI-compatible provider."""
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.client = make_chat_client(self.settings)
        self.name = self.settings.resolved_embedding_model

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=self.settings.resolved_embedding_model,
            input=texts,
            extra_body={"input_type": input_type},
        )
        return [d.embedding for d in resp.data]


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ========== RETRIEVERS ==========

class Retriever:
    name = "base"

    def retrieve(self, query: str, k: int) -> list[int]:
        raise NotImplementedError


class Bm25Retriever(Retriever):
    name = "bm25"

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def retrieve(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1][:k])


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, documents: list[str], embedder):
        self.documents = documents
        self.embedder = embedder
        self.doc_embeddings = embedder.embed(documents, input_type="passage")

    def retrieve(self, query: str, k: int) -> list[int]:
        q_emb = self.embedder.embed([query], input_type="query")[0]
        scores = [cosine_sim(q_emb, e) for e in self.doc_embeddings]
        return list(np.argsort(scores)[::-1][:k])


class HybridRrfRetriever(Retriever):
    """Fuses BM25 + dense rankings by Reciprocal Rank Fusion."""
    name = "hybrid_rrf"

    def __init__(self, bm25: Bm25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def retrieve(self, query: str, k: int) -> list[int]:
        # Over-fetch each side so fusion has enough candidates to reorder.
        fetch_k = max(k * 3, 10)
        bm25_ranked = self.bm25.retrieve(query, fetch_k)
        dense_ranked = self.dense.retrieve(query, fetch_k)
        fused: dict[int, float] = {}
        for rank, doc_id in enumerate(bm25_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (self.rrf_k + rank)
        for rank, doc_id in enumerate(dense_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (self.rrf_k + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]


@dataclass
class RerankOutcome:
    ranked_ids: list[int]
    prompt_tokens: int = 0


class OfflineReranker:
    """Token-overlap heuristic stand-in for an LLM judge (no API key)."""
    name = "offline-token-overlap (NOT an LLM judge)"

    def rerank(self, query: str, candidate_ids: list[int], documents: list[str], top_k: int) -> RerankOutcome:
        q_toks = set(tokenize(query))
        scored = sorted(
            candidate_ids,
            key=lambda i: len(q_toks & set(tokenize(documents[i]))),
            reverse=True,
        )
        return RerankOutcome(ranked_ids=scored[:top_k], prompt_tokens=0)


class LlmReranker:
    name = "llm-judge"

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.client = make_chat_client(self.settings)

    def rerank(self, query: str, candidate_ids: list[int], documents: list[str], top_k: int) -> RerankOutcome:
        numbered = "\n".join(f"[{i}] {documents[doc_id]}" for i, doc_id in enumerate(candidate_ids))
        prompt = (
            "Score the relevance of each numbered document to the query, 0-10.\n"
            'Return ONLY a JSON array like [{"id": 0, "score": 7}, ...].\n\n'
            f"Query: {query}\n\nDocuments:\n{numbered}"
        )
        prompt_tokens = _count_tokens(prompt)
        resp = self.client.chat.completions.create(
            model=self.settings.resolved_model,
            extra_body=NO_THINK,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0,
        )
        raw = re.sub(r"^```(json)?|```$", "", resp.choices[0].message.content.strip(), flags=re.M).strip()
        try:
            scored = json.loads(raw)
        except json.JSONDecodeError:
            return RerankOutcome(ranked_ids=candidate_ids[:top_k], prompt_tokens=prompt_tokens)
        ranked = sorted(scored, key=lambda e: e.get("score", 0), reverse=True)
        ids = [candidate_ids[e["id"]] for e in ranked if e.get("id") is not None and e["id"] < len(candidate_ids)]
        return RerankOutcome(ranked_ids=ids[:top_k] or candidate_ids[:top_k], prompt_tokens=prompt_tokens)


# ========== BENCHMARK ==========

@dataclass
class MethodResult:
    method: str
    recall_at_5: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    ndcg_at_5: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        total_tokens = sum(self.prompt_tokens)
        return {
            "method": self.method,
            "recall@5": round(avg(self.recall_at_5), 3),
            "mrr": round(avg(self.mrr), 3),
            "ndcg@5": round(avg(self.ndcg_at_5), 3),
            "p50_latency_ms": round(float(np.percentile(self.latency_ms, 50)), 2) if self.latency_ms else 0.0,
            "p95_latency_ms": round(float(np.percentile(self.latency_ms, 95)), 2) if self.latency_ms else 0.0,
            "est_cost_usd": round(total_tokens / 1000 * ASSUMED_USD_PER_1K_TOKENS, 6),
        }


def run_benchmark(documents, queries, retrievers: dict, rerank_methods: dict, k=5) -> list[dict]:
    results = {name: MethodResult(name) for name in list(retrievers) + list(rerank_methods)}

    for q in queries:
        relevant = set(q["relevant"])

        for name, retriever in retrievers.items():
            start = time.perf_counter()
            ranked = retriever.retrieve(q["query"], k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            r = results[name]
            r.recall_at_5.append(recall_at_k(ranked, relevant, k))
            r.mrr.append(reciprocal_rank(ranked, relevant))
            r.ndcg_at_5.append(ndcg_at_k(ranked, relevant, k))
            r.latency_ms.append(elapsed_ms)

        # Reranking stages start from the hybrid retriever's shortlist.
        base_candidates = retrievers["hybrid_rrf"].retrieve(q["query"], k=10)
        for name, reranker in rerank_methods.items():
            start = time.perf_counter()
            outcome = reranker.rerank(q["query"], base_candidates, documents, top_k=k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            r = results[name]
            r.recall_at_5.append(recall_at_k(outcome.ranked_ids, relevant, k))
            r.mrr.append(reciprocal_rank(outcome.ranked_ids, relevant))
            r.ndcg_at_5.append(ndcg_at_k(outcome.ranked_ids, relevant, k))
            r.latency_ms.append(elapsed_ms)
            r.prompt_tokens.append(outcome.prompt_tokens)

    return [results[name].summary() for name in results]


def print_table(rows: list[dict]) -> None:
    cols = ["method", "recall@5", "mrr", "ndcg@5", "p50_latency_ms", "p95_latency_ms", "est_cost_usd"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    documents = load_corpus()
    queries = load_queries()
    settings = get_settings()

    offline = not settings.has_llm_credentials
    print(f"Mode: {'OFFLINE (no LLM_PROVIDER API key configured)' if offline else f'LIVE ({settings.llm_provider})'}")
    print(f"Corpus: {len(documents)} docs. Queries: {len(queries)} "
          f"({sum(q['category'] == 'paraphrase' for q in queries)} paraphrase, "
          f"{sum(q['category'] == 'lexical' for q in queries)} lexical).\n")

    embedder = OfflineEmbedder() if offline else NimEmbedder(settings)
    reranker = OfflineReranker() if offline else LlmReranker(settings)
    print(f"Embedder: {embedder.name}")
    print(f"Reranker: {reranker.name}\n")

    bm25 = Bm25Retriever(documents)
    dense = DenseRetriever(documents, embedder)
    hybrid = HybridRrfRetriever(bm25, dense)

    retrievers = {"bm25": bm25, "dense": dense, "hybrid_rrf": hybrid}
    rerank_methods = {"hybrid_rrf+rerank": reranker}

    rows = run_benchmark(documents, queries, retrievers, rerank_methods, k=5)
    print_table(rows)

    if offline:
        print(
            "\nThese numbers are from the offline fallback embedder/reranker, not a real "
            "model — set LLM_PROVIDER + an API key in .env (see .env.example) and re-run "
            "for numbers that reflect an actual embedding model and LLM judge."
        )

    # Per-category breakdown: this is the number that actually argues for
    # dense/hybrid over BM25 — lexical-only queries under-sell it.
    print("\nRecall@5 by query category (hybrid_rrf):")
    for category in ("lexical", "paraphrase"):
        subset = [q for q in queries if q["category"] == category]
        recalls = [
            recall_at_k(hybrid.retrieve(q["query"], 5), set(q["relevant"]), 5) for q in subset
        ]
        avg = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"  {category:10s} (n={len(subset):2d}): {avg:.3f}")
