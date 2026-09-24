#!/usr/bin/env python
# coding: utf-8

# # RAG evaluation benchmark
# 
# The other retrieval notebooks build each piece: chunking, dense and sparse retrieval, hybrid fusion, reranking. This one asks the question they don't: **which combination is actually better, and what does it cost?**
# 
# Four pipelines, one labeled query set:
# 
# | Pipeline | What it is |
# |---|---|
# | `bm25` | Sparse keyword retrieval |
# | `dense` | `nemotron-3-embed-1b` embeddings, cosine similarity |
# | `hybrid_rrf` | BM25 + dense fused with Reciprocal Rank Fusion |
# | `hybrid_rrf+rerank` | Hybrid top-10 reranked by an LLM judge (`nemotron-3-super`) |
# 
# Metrics: Recall@5, MRR, nDCG@5, end-to-end p50/p95 latency, query-time tokens (embedding and LLM, separately), and answer faithfulness for the best pipeline.
# 
# The library code (metrics, retrievers, reranker, judge) lives in [`src/ai_engineering/rag_eval.py`](../../src/ai_engineering/rag_eval.py) so it can be unit tested; this notebook runs it and reads the results.

# In[1]:


import sys
from pathlib import Path

# Repo root, whether this runs as a notebook (cwd = notebooks/<topic>/) or as the scripts/ export
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_engineering.config import get_settings
from ai_engineering import rag_eval as re_

settings = get_settings()
OFFLINE = not settings.has_llm_credentials
print("Mode:", "OFFLINE (no API key; stand-in embedder and reranker)" if OFFLINE
      else f"LIVE ({settings.llm_provider}: {settings.resolved_model}, {settings.resolved_embedding_model})")


# ## The eval set
# 
# A 30-passage corpus about a fictional payments platform and LLM-systems concepts, plus 35 labeled queries. Two things make it harder than a toy set:
# 
# - **Distractor passages** (docs 22–29) share vocabulary with the right answer but don't answer the question: a *different* service's connection pool, a *different* kind of rollback, a *different* rate limit.
# - **Paraphrase queries** describe the need without the corpus's words ("the card-charging service" for `payments-api`, "mints login tokens" for the JWT signer). A benchmark made only of keyword-matching queries flatters BM25.

# In[2]:


documents = re_.load_corpus()
queries = re_.load_queries()
by_cat = {c: [q for q in queries if q["category"] == c] for c in ("lexical", "paraphrase")}
print(f"{len(documents)} passages, {len(queries)} queries: "
      f"{len(by_cat['lexical'])} lexical, {len(by_cat['paraphrase'])} paraphrase\n")
for q in (by_cat["lexical"][0], by_cat["paraphrase"][-12]):
    print(f"[{q['category']}] {q['query']}")
    for d in q["relevant"]:
        print(f"   relevant doc {d}: {documents[d][:90]}...")


# ## Build the four pipelines

# In[3]:


embedder = re_.OfflineEmbedder() if OFFLINE else re_.LiveEmbedder(settings)
reranker = re_.OfflineReranker() if OFFLINE else re_.LlmReranker(settings)
print("Embedder:", embedder.name)
print("Reranker:", reranker.name)

bm25 = re_.Bm25Retriever(documents)
dense = re_.DenseRetriever(documents, embedder)   # embeds the corpus once, up front
print(f"Indexing (one-time): {len(documents)} passages, {dense.index_tokens} embedding tokens")
hybrid = re_.HybridRrfRetriever(bm25, dense)
retrievers = {"bm25": bm25, "dense": dense, "hybrid_rrf": hybrid}


# ## Run the benchmark
# 
# Latency and tokens are end to end per query. The dense side includes the query-embedding call, and the rerank row includes the hybrid retrieval that produced its shortlist. Latency excludes the client-side pacing used to stay under the endpoint's rate limit. Tokens are split into `embed_tok/q` (query embeddings) and `llm_tok/q` (reranker calls). Both are real API usage; indexing is reported once above, not per query.
# 
# The reranker's reply only counts as a ranking if it scores **every** candidate exactly once, with scores in 0–10. Partial replies, out-of-range scores and unparseable replies all fall back to the hybrid order and are counted by kind. A silently failing reranker would otherwise look like one that agrees with the retriever.

# In[4]:


def print_table(rows, cols):
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

rows, per_query, rerank_status = re_.run_benchmark(documents, queries, retrievers, reranker, k=5)
print_table(rows, ["method", "recall@5", "mrr", "ndcg@5", "p50_latency_ms", "p95_latency_ms",
                   "embed_tok/q", "llm_tok/q", "est_usd/1k_q"])
print(f"\nreranker replies: {rerank_status}")
print(f"(est_usd/1k_q = cost per 1,000 queries at an assumed ${re_.ASSUMED_USD_PER_1K_TOKENS}/1K tokens; token counts are real API usage)")


# ## Lexical vs paraphrase queries
# 
# The split is where the pipelines actually differ.

# In[5]:


methods = [r["method"] for r in rows]
print_table(re_.recall_by_category(per_query, methods), ["category", "n", *methods])


# ## Where they disagree
# 
# Every query where at least one pipeline missed a relevant passage in its top 5.

# In[6]:


for row in per_query:
    if min(row[m] for m in methods) < 1.0:
        q = next(q for q in queries if q["id"] == row["id"])
        scores = "  ".join(f"{m}={row[m]:.2f}" for m in methods)
        print(f"{row['id']} [{row['category']}] {q['query']}\n    recall@5: {scores}")


# ## Faithfulness
# 
# Retrieval metrics say whether the right passage was found; they say nothing about whether the answer built from it sticks to it. For every query, the reranked top 3 passages are handed to the model to answer, then a judge call lists each claim in the answer and whether the context supports it. An answer counts as faithful only if every claim is supported: this is computed from the claim list, not taken from a summary flag the judge might contradict.
# 
# The same model generates and judges, so this is a self-consistency check rather than an independent audit. Live mode only: there is no offline stand-in for a generator.

# In[7]:


if OFFLINE:
    print("Skipped in OFFLINE mode: needs a real model to generate and judge answers.")
else:
    judge = re_.FaithfulnessJudge(settings)
    faith = []
    for row, q in zip(per_query, queries):
        context = [documents[i] for i in row["_reranked"][:3]]
        faith.append({"id": q["id"], "query": q["query"], **judge.evaluate(q["query"], context)})

    judged = [f for f in faith if f["faithful"] is not None]
    claims = sum(f["claims"] for f in judged)
    supported = sum(f["supported_claims"] for f in judged)
    print(f"faithful answers: {sum(f['faithful'] for f in judged)}/{len(judged)} "
          f"(no usable claims from the judge for {len(faith) - len(judged)})")
    print(f"supported claims: {supported}/{claims} ({supported / max(claims, 1):.1%})")
    print(f"tokens: {sum(f['tokens'] for f in faith)}")
    for f in faith:
        if f["faithful"] is False:
            print(f"\nUNFAITHFUL {f['id']}: {f['query']}\n  answer: {f['answer']}")


# ## Findings
# 
# From the live run above (`nemotron-3-embed-1b` embeddings, `nemotron-3-super-120b-a12b` reranker and judge):
# 
# - **Dense beat hybrid.** Dense alone reached Recall@5 = 1.00 and MRR = 0.971. Fusing in BM25 *lowered* it to 0.914 / 0.887. Hybrid only helps when the sparse side adds hits the dense side misses. Here the embedder already finds everything, so BM25 mostly contributes distractors that share keywords with the query (q07, q21, q24 and q25 above). "Hybrid is always better" is a default to test, not a law.
# - **BM25 collapses on paraphrases.** Recall@5 fell from 0.921 on lexical queries to 0.719 on paraphrases, and it scored 0 on queries like q24/q25/q27/q29 that share no vocabulary with the answer. That is the gap an evaluation set made only of keyword queries would hide.
# - **The reranker is the most precise and by far the most expensive.** It recovered everything hybrid lost and ranked perfectly (MRR = nDCG@5 = 1.00), but it adds about 599 LLM tokens per query on top of about 16 embedding tokens. That is roughly 38× dense-only's token cost per query. Its p50 latency is about 3.2 s against about 0.3 s for dense. The p95 (25.6 s) mostly reflects SDK retry backoff while the endpoint was returning 429/503s during this run, not model latency. Over dense alone, the reranker's gain is +0.029 MRR. On this corpus, dense-only is the better cost/quality point. The reranker earns its place when the first-stage retriever is weaker or the corpus is larger.
# - **All reranker replies were complete.** With strict validation (every candidate scored exactly once, scores 0–10), all 35 replies were usable, so the reranker row isn't flattered by silent fallbacks to the hybrid order.
# - **Answers were mostly faithful, but not entirely.** 32/34 judged answers were faithful and 88/92 claims supported; for one query the judge returned no usable claim list, and it's reported as unjudged rather than counted as faithful. Both flagged answers added plausible outside knowledge to correctly retrieved context ("even at night", "struggles with synonyms"). Correct retrieval did not stop unsupported additions.
# - **Indexing is cheap and one-time.** Embedding all 30 passages took 1,207 tokens, paid once and not per query.
# 
# **Limits.** 30 passages and 35 queries is small: dense retrieval hits the Recall@5 ceiling, so MRR/nDCG carry the comparison. The judge is the same model as the generator. Dollar figures use an assumed $0.002/1K tokens (NIM's hosted trial is free), so compare ratios rather than absolute costs. Latency is from one hosted endpoint on one afternoon.
