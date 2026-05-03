from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from recipe_rag.indexing.bm25 import BM25Index
from recipe_rag.indexing.embeddings import EmbeddingModel
from recipe_rag.indexing.vector_store import VectorStore
from recipe_rag.schemas import ChunkType, Intent, RecipeChunk, RerankerPair


_INTENT_TO_CHUNK = {
    Intent.RECOMMEND.value: ChunkType.SUMMARY,
    Intent.METHOD.value: ChunkType.STEPS,
    Intent.TIP.value: ChunkType.TIPS,
    Intent.INGREDIENT.value: ChunkType.INGREDIENTS,
}


@dataclass
class QuerySpec:
    query_id: str
    query: str
    recipe_id: str
    intent: str

    @classmethod
    def from_dict(cls, row: dict) -> "QuerySpec":
        return cls(
            query_id=str(row.get("query_id") or row.get("id") or row["query"]),
            query=str(row["query"]),
            recipe_id=str(row["recipe_id"]),
            intent=str(row.get("intent") or Intent.RECOMMEND.value),
        )


def build_reranker_pairs(
    queries: Iterable[QuerySpec],
    chunks: list[RecipeChunk],
    negatives_per_positive: int = 7,
    random_seed: int = 42,
    embedding_model_name: str | None = None,
) -> list[RerankerPair]:
    rng = random.Random(random_seed)
    by_recipe: dict[str, list[RecipeChunk]] = {}
    for chunk in chunks:
        by_recipe.setdefault(chunk.recipe_id, []).append(chunk)

    bm25 = BM25Index(chunks)
    embedder = EmbeddingModel(embedding_model_name)
    vectors = embedder.encode([c.text for c in chunks])
    vector_store = VectorStore.build(chunks, vectors, backend="numpy")

    pairs: list[RerankerPair] = []
    for query in queries:
        positives = _select_positives(query, by_recipe)
        if not positives:
            continue
        for positive in positives:
            label = 1.0 if positive.chunk_type == _INTENT_TO_CHUNK.get(query.intent) else 0.5
            pairs.append(_pair(query, positive, label, None))
            negatives = _mine_negatives(query, positive, chunks, bm25, vector_store, embedder, rng, negatives_per_positive)
            pairs.extend(_pair(query, chunk, 0.0, neg_type) for chunk, neg_type in negatives)
    return pairs


def _select_positives(query: QuerySpec, by_recipe: dict[str, list[RecipeChunk]]) -> list[RecipeChunk]:
    recipe_chunks = by_recipe.get(query.recipe_id, [])
    target_type = _INTENT_TO_CHUNK.get(query.intent)
    strong = [x for x in recipe_chunks if x.chunk_type == target_type]
    weak = [x for x in recipe_chunks if x.chunk_type != target_type]
    return strong or weak[:1]


def _mine_negatives(
    query: QuerySpec,
    positive: RecipeChunk,
    chunks: list[RecipeChunk],
    bm25: BM25Index,
    vector_store: VectorStore,
    embedder: EmbeddingModel,
    rng: random.Random,
    total: int,
) -> list[tuple[RecipeChunk, str]]:
    non_positive = [x for x in chunks if x.recipe_id != positive.recipe_id]
    selected: list[tuple[RecipeChunk, str]] = []
    seen = {positive.chunk_id}

    def add(candidates: list[RecipeChunk], label: str, limit: int) -> None:
        for chunk in candidates:
            if len([x for x in selected if x[1] == label]) >= limit:
                break
            if chunk.chunk_id in seen or chunk.recipe_id == positive.recipe_id:
                continue
            selected.append((chunk, label))
            seen.add(chunk.chunk_id)

    random_limit = max(1, round(total * 2 / 7))
    bm25_limit = max(1, round(total * 3 / 7))
    embedding_limit = max(1, total - random_limit - bm25_limit)

    random_candidates = non_positive[:]
    rng.shuffle(random_candidates)
    add(random_candidates, "random", random_limit)

    bm25_candidates = [x.chunk for x in bm25.search(query.query, top_k=50)]
    add(bm25_candidates, "bm25_hard", bm25_limit)

    query_vector = embedder.encode([query.query])[0]
    embedding_candidates = [x.chunk for x in vector_store.search(query_vector, top_k=50)]
    add(embedding_candidates, "embedding_hard", embedding_limit)

    if len(selected) < total:
        add(random_candidates, "random_fill", total - len(selected))
    return selected[:total]


def _pair(query: QuerySpec, chunk: RecipeChunk, label: float, negative_type: str | None) -> RerankerPair:
    return RerankerPair(
        query=query.query,
        passage=chunk.text,
        label=label,
        query_id=query.query_id,
        chunk_id=chunk.chunk_id,
        recipe_id=chunk.recipe_id,
        intent=query.intent,
        negative_type=negative_type,
    )
