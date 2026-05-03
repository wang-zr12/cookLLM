from __future__ import annotations

import math
from collections import defaultdict


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum((2**gain - 1) / math.log2(idx + 2) for idx, gain in enumerate(gains[:k]))


def ndcg_at_k(qrels: dict[str, dict[str, float]], run: dict[str, list[str]], k: int = 3) -> float:
    scores = []
    for qid, rels in qrels.items():
        ranked = run.get(qid, [])
        gains = [rels.get(doc_id, 0.0) for doc_id in ranked[:k]]
        ideal = sorted(rels.values(), reverse=True)
        denom = dcg_at_k(ideal, k)
        scores.append(dcg_at_k(gains, k) / denom if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(qrels: dict[str, dict[str, float]], run: dict[str, list[str]], k: int = 20) -> float:
    scores = []
    for qid, rels in qrels.items():
        relevant = {doc_id for doc_id, rel in rels.items() if rel > 0}
        if not relevant:
            continue
        retrieved = set(run.get(qid, [])[:k])
        scores.append(len(relevant & retrieved) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(qrels: dict[str, dict[str, float]], run: dict[str, list[str]], k: int = 10) -> float:
    reciprocal_ranks = []
    for qid, rels in qrels.items():
        ranked = run.get(qid, [])[:k]
        rr = 0.0
        for idx, doc_id in enumerate(ranked, start=1):
            if rels.get(doc_id, 0.0) > 0:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def load_qrels(rows: list[dict]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        qrels[str(row["query_id"])][str(row["chunk_id"])] = float(row["relevance"])
    return dict(qrels)


def load_run(rows: list[dict]) -> dict[str, list[str]]:
    run: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        qid = str(row["query_id"])
        if "ranked_chunk_ids" in row:
            run[qid].extend(str(x) for x in row["ranked_chunk_ids"])
        else:
            run[qid].append(str(row["chunk_id"]))
    return dict(run)
