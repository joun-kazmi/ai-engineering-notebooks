"""Offline unit tests for the RAG eval benchmark's metric functions.

No network, no LLM calls — recall/MRR/nDCG are pure functions of ranked
id lists, so they're testable directly against hand-worked examples.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_evaluation_benchmark import (  # noqa: E402
    load_corpus,
    load_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


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
    perfect = ndcg_at_k([1, 2, 3], {1}, k=3)
    worse = ndcg_at_k([3, 2, 1], {1}, k=3)
    assert worse < perfect


def test_query_set_is_internally_consistent_with_the_corpus():
    documents = load_corpus()
    queries = load_queries()
    assert len(documents) > 0
    assert len(queries) > 0
    for q in queries:
        assert q["relevant"], f"{q['id']} has no labeled relevant docs"
        for doc_id in q["relevant"]:
            assert 0 <= doc_id < len(documents), f"{q['id']} references out-of-range doc {doc_id}"


def test_query_ids_are_unique():
    queries = load_queries()
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_rag_eval_queries_json_matches_loader():
    raw = json.loads((ROOT / "data" / "rag_eval_queries.json").read_text())
    assert raw == load_queries()
