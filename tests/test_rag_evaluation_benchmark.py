"""Offline tests for the RAG eval benchmark (src/ai_engineering/rag_eval.py).

No network, no LLM calls — the metrics are pure functions of ranked id
lists, and the OFFLINE embedder/reranker let the whole benchmark run here.
"""
import json
from pathlib import Path

from ai_engineering.rag_eval import (
    Bm25Retriever,
    DenseRetriever,
    HybridRrfRetriever,
    OfflineEmbedder,
    OfflineReranker,
    load_corpus,
    load_queries,
    ndcg_at_k,
    parse_rerank_scores,
    recall_at_k,
    reciprocal_rank,
    rrf_fuse,
    run_benchmark,
)

ROOT = Path(__file__).resolve().parent.parent


def test_recall_at_k_counts_hits_within_k():
    assert recall_at_k([5, 1, 2], {1, 2}, k=3) == 1.0
    assert recall_at_k([5, 1, 2], {1, 2}, k=1) == 0.0
    assert recall_at_k([1, 5, 2], {1, 2}, k=2) == 0.5


def test_recall_at_k_empty_relevant_set_is_zero():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_reciprocal_rank_rewards_earlier_hits():
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0
    assert reciprocal_rank([2, 1, 3], {1}) == 0.5
    assert reciprocal_rank([2, 3, 4], {1}) == 0.0


def test_ndcg_at_k_perfect_ranking_scores_one():
    assert ndcg_at_k([1, 2, 3], {1, 2}, k=3) == 1.0


def test_ndcg_at_k_penalizes_relevant_docs_ranked_lower():
    assert ndcg_at_k([3, 2, 1], {1}, k=3) < ndcg_at_k([1, 2, 3], {1}, k=3)


def test_rrf_ranks_start_at_one_and_reward_agreement():
    # doc 7 is 2nd in both lists; docs 1 and 2 are 1st in one list and absent from the other
    fused = rrf_fuse([[1, 7], [2, 7]], rrf_k=60)
    assert fused[0] == 7
    # rank 1 contributes 1/61, not 1/60
    assert rrf_fuse([[3]], rrf_k=60) == [3]


def test_parse_rerank_scores_handles_fences_strings_and_junk():
    raw = '```json\n[{"id": "1", "score": 9}, {"id": 0, "score": 2}, {"id": 5, "score": 10}, {"id": 1, "score": 1}]\n```'
    assert parse_rerank_scores(raw, n_candidates=3) == [(1, 9.0), (0, 2.0)]
    assert parse_rerank_scores("I think doc 2 is best", n_candidates=3) is None
    assert parse_rerank_scores("[]", n_candidates=3) is None


def test_query_set_is_internally_consistent_with_the_corpus():
    documents = load_corpus()
    queries = load_queries()
    assert documents and queries
    for q in queries:
        assert q["category"] in ("lexical", "paraphrase")
        assert q["relevant"], f"{q['id']} has no labeled relevant docs"
        for doc_id in q["relevant"]:
            assert 0 <= doc_id < len(documents), f"{q['id']} references out-of-range doc {doc_id}"


def test_query_ids_are_unique():
    ids = [q["id"] for q in load_queries()]
    assert len(ids) == len(set(ids))


def test_rag_eval_queries_json_matches_loader():
    raw = json.loads((ROOT / "data" / "rag_eval_queries.json").read_text())
    assert raw == load_queries()


def test_offline_benchmark_runs_end_to_end_and_is_deterministic():
    documents, queries = load_corpus(), load_queries()

    def once():
        bm25 = Bm25Retriever(documents)
        dense = DenseRetriever(documents, OfflineEmbedder())
        retrievers = {"bm25": bm25, "dense": dense, "hybrid_rrf": HybridRrfRetriever(bm25, dense)}
        rows, per_query = run_benchmark(documents, queries, retrievers, OfflineReranker(), k=5)
        return [{k: v for k, v in r.items() if "latency" not in k} for r in rows], per_query

    rows, per_query = once()
    assert [r["method"] for r in rows] == ["bm25", "dense", "hybrid_rrf", "hybrid_rrf+rerank"]
    assert len(per_query) == len(queries)
    assert all(0.0 <= r["recall@5"] <= 1.0 for r in rows)
    assert rows == once()[0]
