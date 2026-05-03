from __future__ import annotations

import argparse

from recipe_rag.training.reranker.metrics import load_qrels, load_run, mrr_at_k, ndcg_at_k, recall_at_k
from recipe_rag.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranking with NDCG, Recall and MRR.")
    parser.add_argument("--qrels", required=True, help="JSONL: query_id, chunk_id, relevance(0-3).")
    parser.add_argument("--run", required=True, help="JSONL: query_id, chunk_id or ranked_chunk_ids.")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    qrels = load_qrels(list(read_jsonl(args.qrels)))
    run = load_run(list(read_jsonl(args.run)))
    print(f"NDCG@{args.k}: {ndcg_at_k(qrels, run, args.k):.4f}")
    print(f"Recall@20: {recall_at_k(qrels, run, 20):.4f}")
    print(f"MRR@10: {mrr_at_k(qrels, run, 10):.4f}")


if __name__ == "__main__":
    main()
