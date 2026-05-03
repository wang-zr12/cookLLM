from recipe_rag.training.reranker.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ranking_metrics():
    qrels = {"q1": {"a": 3, "b": 1}, "q2": {"c": 3}}
    run = {"q1": ["a", "x", "b"], "q2": ["x", "c"]}
    assert ndcg_at_k(qrels, run, 3) > 0.7
    assert recall_at_k(qrels, run, 3) == 1.0
    assert mrr_at_k(qrels, run, 3) == 0.75
