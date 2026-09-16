#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, re, json
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tenacity import retry, wait_exponential, stop_after_attempt

# ========== CONFIG ==========
# Don't hardcode API keys in source. Set this in your shell before running:
#   export NVIDIA_API_KEY=nvapi-XXXX

API_KEY = os.environ["NVIDIA_API_KEY"]
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)
MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

RETRY = dict(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))


# ========== EMBEDDING HELPERS (batched, not one call per chunk) ==========
# nv-embedqa-e5-v5 is an *asymmetric* embedding model: it encodes queries and
# documents differently internally, so it requires input_type to tell it which
# side it's embedding. Get this wrong and retrieval quality silently degrades
# even though the code "works."
@retry(**RETRY)
def get_embeddings_batch(texts, batch_size=64, input_type="passage"):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            extra_body={"input_type": input_type},
        )
        out.extend([d.embedding for d in resp.data])
    return out


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ========== CHUNKING (truly recursive, overlap actually applied) ==========
def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def recursive_split(text, chunk_size=500, overlap=50):
    """
    Splits on paragraph breaks first. Any paragraph still longer than
    chunk_size falls back to sentence-level splitting. `overlap` carries
    the tail of the previous chunk into the next, so sentences that land
    on a boundary still appear whole in at least one chunk.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    units = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(split_into_sentences(para))

    chunks, current = [], ""
    for unit in units:
        if len(current) + len(unit) + 1 <= chunk_size:
            current = f"{current} {unit}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


# ========== BM25 (lowercased, punctuation-stripped tokens) ==========
def tokenize(text):
    return re.findall(r"\w+", text.lower())


# ========== RETRIEVAL: RRF fusion instead of min-max normalization ==========
class HybridRetriever:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def retrieve(self, query, k=10, rrf_k=60):
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranked = list(np.argsort(bm25_scores)[::-1])

        q_emb = get_embeddings_batch([query], input_type="query")[0]
        vec_scores = [cosine_sim(q_emb, emb) for emb in self.embeddings]
        vec_ranked = list(np.argsort(vec_scores)[::-1])

        # Reciprocal Rank Fusion: combine by rank position, not raw score,
        # so BM25 and cosine similarity never need to share a scale.
        fused = {}
        for rank, idx in enumerate(bm25_ranked):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank)
        for rank, idx in enumerate(vec_ranked):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank)

        top_indices = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(idx, self.documents[idx]) for idx in top_indices]


# ========== RERANK (one batched call, structured JSON, relevance threshold) ==========
class RerankResult:
    def __init__(self, idx, doc, score):
        self.idx, self.doc, self.score = idx, doc, score


@retry(**RETRY)
def llm_rerank(query, candidates, top_k=3, min_score=5.0):
    """candidates: list of (idx, doc_text). Scores all of them in a single call."""
    numbered = "\n".join(f"[{i}] {doc}" for i, (_, doc) in enumerate(candidates))
    prompt = f"""Score the relevance of each numbered document to the query, 0 (irrelevant) to 10 (directly answers it).
Return ONLY a JSON array like [{{"id": 0, "score": 7}}, ...] — one entry per document, nothing else.

Query: {query}

Documents:
{numbered}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
    )
    raw = re.sub(r"^```(json)?|```$", "", resp.choices[0].message.content.strip(),
                 flags=re.MULTILINE).strip()

    try:
        scored = json.loads(raw)
    except json.JSONDecodeError:
        print("WARNING: rerank response was not valid JSON, returning no results")
        return []

    results = []
    for entry in scored:
        i, score = entry.get("id"), entry.get("score", 0)
        if i is None or i >= len(candidates):
            continue
        idx, doc = candidates[i]
        results.append(RerankResult(idx, doc, float(score)))

    results.sort(key=lambda r: r.score, reverse=True)
    return [r for r in results if r.score >= min_score][:top_k]


# ========== GENERATION ==========
@retry(**RETRY)
def generate_answer(query, context_docs):
    if not context_docs:
        return "I don't know based on the available information."

    context = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(context_docs))
    prompt = f"""Answer the question using ONLY the numbered context below. Cite sources like [0], [1] inline.
If the context does not contain the answer, say "I don't know" — do not guess.

Question: {query}

Context:
{context}

Answer:"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ========== EVALUATION ==========
@retry(**RETRY)
def evaluate_faithfulness(question, answer, context):
    prompt = f"""Is the answer fully supported by the context, with no unsupported claims? Answer only YES or NO.
Question: {question}
Context: {context}
Answer: {answer}"""
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=3, temperature=0,
    )
    return "YES" in resp.choices[0].message.content.upper()


@retry(**RETRY)
def evaluate_context_precision(question, top_doc):
    """Checks whether the retriever's top pick is actually relevant —
    separate signal from generation faithfulness."""
    prompt = f"""Does this document contain information that helps answer the question? Answer only YES or NO.
Question: {question}
Document: {top_doc}"""
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=3, temperature=0,
    )
    return "YES" in resp.choices[0].message.content.upper()


# ========== MAIN PIPELINE ==========
if __name__ == "__main__":
    with open("your_document.txt", "r") as f:
        raw_text = f.read()

    chunks = recursive_split(raw_text, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks")

    chunk_embeddings = get_embeddings_batch(chunks, input_type="passage")  # one batched call, not N calls
    retriever = HybridRetriever(chunks, chunk_embeddings)

    query = "What is the capital of France?"

    candidates = retriever.retrieve(query, k=10)
    print("\nRetrieved (top 3 by RRF):")
    for idx, doc in candidates[:3]:
        print(f"  [{idx}] {doc[:80]}...")

    reranked = llm_rerank(query, candidates, top_k=3, min_score=5.0)
    if not reranked:
        print("\nNo candidates cleared the relevance threshold.")
    else:
        print("\nReranked top:")
        for r in reranked:
            print(f"  [{r.idx}] score={r.score} {r.doc[:80]}...")

    context_texts = [r.doc for r in reranked]
    answer = generate_answer(query, context_texts)
    print("\nAnswer:", answer)

    if context_texts:
        faithful = evaluate_faithfulness(query, answer, "\n\n".join(context_texts))
        print("Faithful:", faithful)

        precise = evaluate_context_precision(query, context_texts[0])
        print("Top result actually relevant:", precise)


# In[ ]:




