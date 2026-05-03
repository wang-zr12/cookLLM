from __future__ import annotations

from recipe_rag.indexing.tokenize import tokenize_zh
from recipe_rag.schemas import SearchResult


class BCEReranker:
    """BCE Reranker 适配器。

    有 sentence-transformers CrossEncoder 时走真实模型；否则用词项重叠作为可测试 fallback。
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._model = None
        if model_name:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name)
            except Exception:
                self._model = None

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 3) -> list[SearchResult]:
        if not candidates:
            return []
        if self._model is not None:
            pairs = [(query, x.chunk.text) for x in candidates]
            scores = [float(x) for x in self._model.predict(pairs)]
        else:
            scores = [self._lexical_score(query, x.chunk.text) for x in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk=item.chunk,
                score=score,
                source="reranker",
                rank=rank + 1,
                debug={**item.debug, "pre_rerank_score": item.score},
            )
            for rank, (item, score) in enumerate(ranked)
        ]

    def _lexical_score(self, query: str, passage: str) -> float:
        q = set(tokenize_zh(query))
        p = set(tokenize_zh(passage))
        if not q or not p:
            return 0.0
        return len(q & p) / len(q)
