from __future__ import annotations

from collections import defaultdict

from recipe_rag.schemas import SearchResult


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = 60, top_k: int = 20) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, SearchResult] = {}
    debug: dict[str, dict] = defaultdict(dict)

    for result_list in result_lists:
        for idx, result in enumerate(result_list):
            rank = result.rank or idx + 1
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += 1.0 / (k + rank)
            best.setdefault(chunk_id, result)
            debug[chunk_id][result.source] = {"rank": rank, "score": result.score}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        base = best[chunk_id]
        fused.append(SearchResult(chunk=base.chunk, score=score, source="rrf", rank=rank, debug=debug[chunk_id]))
    return fused
