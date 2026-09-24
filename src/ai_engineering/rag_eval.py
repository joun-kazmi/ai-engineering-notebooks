"""RAG evaluation benchmark: BM25 vs dense vs hybrid/RRF vs hybrid+rerank.

Library code for notebooks/retrieval/rag_evaluation_benchmark.ipynb. Kept
here (not in the notebook) so the metric functions and the offline pipeline
are importable and unit tested.

Runs two ways:

  * No API key configured for LLM_PROVIDER -> OFFLINE mode. Dense retrieval
    uses a deterministic hashed bag-of-words embedder (not a real embedding
    model) and the reranker uses a token-overlap heuristic instead of an LLM
    judge. Useful for tests and zero-setup runs; says nothing about real
    model quality.
  * A real key configured (see .env.example / config.py) -> the real
    embedding model, an LLM reranker, and an LLM faithfulness judge.

Metrics: Recall@K, MRR, nDCG@K (retrieval quality), per-query latency,
token cost for the LLM-touching stages, and answer faithfulness (live only).
"""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ai_engineering.config import NO_THINK, get_settings, make_chat_client

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Illustrative only — providers price per-token differently and change prices
# over time (NIM's hosted trial endpoints are free). Edit to your provider's
# actual rate before trusting the dollar figures; the token counts are real
# usage numbers returned by the API.
ASSUMED_USD_PER_1K_TOKENS = 0.002

# Every live call in a benchmark run goes through one of these; hosted NIM
# returns transient 503s often enough that the SDK default of 2 isn't enough.
EVAL_MAX_RETRIES = 6


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


def rrf_fuse(rankings: list[list[int]], rrf_k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank),
    with rank starting at 1 as in Cormack et al. (2009)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (rrf_k + rank)
    return sorted(fused, key=fused.get, reverse=True)


def parse_rerank_scores(raw: str, n_candidates: int) -> list[tuple[int, float]] | None:
    """Parse an LLM reranker reply into (candidate_index, score) pairs.

    Returns None when the reply isn't a usable JSON array, so the caller can
    count it as a parse failure instead of silently scoring it as a ranking.
    Ids may come back as ints or numeric strings; out-of-range, negative, or
    duplicate ids are dropped.
    """
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    pairs, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx, score = int(item["id"]), float(item.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n_candidates and idx not in seen:
            seen.add(idx)
            pairs.append((idx, score))
    return pairs or None


# ========== EMBEDDERS ==========

class OfflineEmbedder:
    """Deterministic, offline stand-in for a real embedding model.

    Hashes tokens into a fixed-size bag-of-words vector. This captures
    lexical overlap, *not* semantics — paraphrases with little shared
    vocabulary will NOT score well here the way a real embedding model
    would. Treat its "dense retrieval" numbers as a structural demo only.
    """
    name = "offline-hashed-bow (NOT a real embedding model)"
    dims = 256

    @staticmethod
    def _stable_hash(token: str) -> int:
        # Built-in hash() is randomized per process for strings; md5 keeps
        # this embedder reproducible across runs.
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


class LiveEmbedder:
    """Real embedding model via the configured OpenAI-compatible provider."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.client = make_chat_client(self.settings, max_retries=EVAL_MAX_RETRIES)
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

class Bm25Retriever:
    name = "bm25"

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def retrieve(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(scores)[::-1][:k]]


class DenseRetriever:
    name = "dense"

    def __init__(self, documents: list[str], embedder):
        self.documents = documents
        self.embedder = embedder
        self.doc_embeddings = embedder.embed(documents, input_type="passage")

    def retrieve(self, query: str, k: int) -> list[int]:
        q_emb = self.embedder.embed([query], input_type="query")[0]
        scores = [cosine_sim(q_emb, e) for e in self.doc_embeddings]
        return [int(i) for i in np.argsort(scores)[::-1][:k]]


class HybridRrfRetriever:
    """Fuses BM25 + dense rankings by Reciprocal Rank Fusion."""
    name = "hybrid_rrf"

    def __init__(self, bm25: Bm25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def retrieve(self, query: str, k: int) -> list[int]:
        # Over-fetch each side so fusion has enough candidates to reorder.
        fetch_k = max(k * 3, 10)
        return rrf_fuse([self.bm25.retrieve(query, fetch_k), self.dense.retrieve(query, fetch_k)], self.rrf_k)[:k]


# ========== RERANKERS ==========

@dataclass
class RerankOutcome:
    ranked_ids: list[int]
    tokens: int = 0
    parse_failed: bool = False


class OfflineReranker:
    """Token-overlap heuristic stand-in for an LLM judge (no API key)."""
    name = "offline-token-overlap (NOT an LLM judge)"

    def rerank(self, query, candidate_ids, documents, top_k) -> RerankOutcome:
        q_toks = set(tokenize(query))
        scored = sorted(candidate_ids, key=lambda i: len(q_toks & set(tokenize(documents[i]))), reverse=True)
        return RerankOutcome(ranked_ids=scored[:top_k])


class LlmReranker:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.client = make_chat_client(self.settings, max_retries=EVAL_MAX_RETRIES)
        self.name = f"llm-judge ({self.settings.resolved_model})"

    def rerank(self, query, candidate_ids, documents, top_k) -> RerankOutcome:
        numbered = "\n".join(f"[{i}] {documents[doc_id]}" for i, doc_id in enumerate(candidate_ids))
        prompt = (
            "Score the relevance of each numbered document to the query, 0-10.\n"
            'Return ONLY a JSON array like [{"id": 0, "score": 7}, ...] covering every document.\n\n'
            f"Query: {query}\n\nDocuments:\n{numbered}"
        )
        resp = self.client.chat.completions.create(
            model=self.settings.resolved_model,
            extra_body=NO_THINK,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0,
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        pairs = parse_rerank_scores(resp.choices[0].message.content or "", len(candidate_ids))
        if pairs is None:
            # Fall back to the retriever's order, but record it — a reranker
            # that silently fails looks exactly like one that agrees.
            return RerankOutcome(candidate_ids[:top_k], tokens, parse_failed=True)
        # Stable sort: ties keep the retriever's order; unscored candidates go last.
        scores = dict(pairs)
        order = sorted(range(len(candidate_ids)), key=lambda i: -scores.get(i, -1))
        return RerankOutcome([candidate_ids[i] for i in order][:top_k], tokens)


# ========== BENCHMARK ==========

@dataclass
class MethodResult:
    method: str
    recall_at_5: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    ndcg_at_5: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    parse_failures: int = 0

    def add(self, ranked: list[int], relevant: set[int], k: int, elapsed_ms: float) -> None:
        self.recall_at_5.append(recall_at_k(ranked, relevant, k))
        self.mrr.append(reciprocal_rank(ranked, relevant))
        self.ndcg_at_5.append(ndcg_at_k(ranked, relevant, k))
        self.latency_ms.append(elapsed_ms)

    def summary(self) -> dict:
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        pct = lambda p: round(float(np.percentile(self.latency_ms, p)), 1) if self.latency_ms else 0.0
        return {
            "method": self.method,
            "recall@5": round(avg(self.recall_at_5), 3),
            "mrr": round(avg(self.mrr), 3),
            "ndcg@5": round(avg(self.ndcg_at_5), 3),
            "p50_latency_ms": pct(50),
            "p95_latency_ms": pct(95),
            "tokens": sum(self.tokens),
            "est_cost_usd": round(sum(self.tokens) / 1000 * ASSUMED_USD_PER_1K_TOKENS, 4),
            "parse_failures": self.parse_failures,
        }


def run_benchmark(documents, queries, retrievers: dict, reranker, k: int = 5, shortlist: int = 10):
    """Returns (summary rows, per-query rows). The rerank stage reranks the
    hybrid retriever's top-`shortlist`, and its latency includes that
    retrieval, so the latency column is end to end for every method."""
    methods = list(retrievers) + ["hybrid_rrf+rerank"]
    results = {name: MethodResult(name) for name in methods}
    per_query = []

    for q in queries:
        relevant = set(q["relevant"])
        row = {"id": q["id"], "category": q["category"]}

        for name, retriever in retrievers.items():
            start = time.perf_counter()
            ranked = retriever.retrieve(q["query"], k)
            results[name].add(ranked, relevant, k, (time.perf_counter() - start) * 1000)
            row[name] = results[name].recall_at_5[-1]

        start = time.perf_counter()
        candidates = retrievers["hybrid_rrf"].retrieve(q["query"], k=shortlist)
        outcome = reranker.rerank(q["query"], candidates, documents, top_k=k)
        r = results["hybrid_rrf+rerank"]
        r.add(outcome.ranked_ids, relevant, k, (time.perf_counter() - start) * 1000)
        r.tokens.append(outcome.tokens)
        r.parse_failures += outcome.parse_failed
        row["hybrid_rrf+rerank"] = r.recall_at_5[-1]
        row["_reranked"] = outcome.ranked_ids
        per_query.append(row)

    return [results[m].summary() for m in methods], per_query


def recall_by_category(per_query: list[dict], methods: list[str]) -> list[dict]:
    rows = []
    for category in sorted({r["category"] for r in per_query}):
        subset = [r for r in per_query if r["category"] == category]
        rows.append({"category": category, "n": len(subset),
                     **{m: round(sum(r[m] for r in subset) / len(subset), 3) for m in methods}})
    return rows


# ========== FAITHFULNESS (live only) ==========

class FaithfulnessJudge:
    """Generates an answer from the retrieved context, then asks a judge call
    whether every claim in it is supported by that context.

    Same model generates and judges, so treat this as a self-consistency
    check, not an independent audit — a stronger or different judge model
    is the obvious upgrade (set it via a second Settings with LLM_MODEL).
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.client = make_chat_client(self.settings, max_retries=EVAL_MAX_RETRIES)

    def _chat(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        resp = self.client.chat.completions.create(
            model=self.settings.resolved_model, extra_body=NO_THINK,
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0,
        )
        return resp.choices[0].message.content or "", resp.usage.total_tokens if resp.usage else 0

    def evaluate(self, query: str, context: list[str]) -> dict:
        ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context))
        answer, t1 = self._chat(
            f"Answer the question using ONLY the context. If the context is insufficient, say so.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer in 1-3 sentences.", 300)
        verdict_raw, t2 = self._chat(
            "You are checking an answer for faithfulness to its context.\n"
            "List every factual claim in the ANSWER and mark whether the CONTEXT supports it.\n"
            'Return ONLY JSON: {"claims": [{"claim": "...", "supported": true}], "faithful": true}\n'
            "faithful is true only if every claim is supported.\n\n"
            f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}", 600)
        text = re.sub(r"^```(?:json)?|```$", "", verdict_raw.strip(), flags=re.M).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            verdict = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            verdict = None
        claims = verdict.get("claims", []) if isinstance(verdict, dict) else []
        supported = sum(bool(c.get("supported")) for c in claims if isinstance(c, dict))
        return {
            "answer": answer.strip(),
            "faithful": bool(verdict.get("faithful")) if isinstance(verdict, dict) else None,
            "claims": len(claims),
            "supported_claims": supported,
            "tokens": t1 + t2,
        }
