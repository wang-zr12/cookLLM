from __future__ import annotations

from recipe_rag.schemas import ChunkType, Intent, QueryUnderstanding, SearchResult


_INTENT_CHUNK_TYPES = {
    Intent.RECOMMEND: {ChunkType.SUMMARY},
    Intent.METHOD: {ChunkType.STEPS},
    Intent.TIP: {ChunkType.TIPS, ChunkType.STEPS},
    Intent.INGREDIENT: {ChunkType.INGREDIENTS},
}


def hard_filter(results: list[SearchResult], query: QueryUnderstanding) -> list[SearchResult]:
    filtered = results
    if query.ingredients:
        required = set(query.ingredients)
        filtered = [x for x in filtered if required.issubset(set(x.chunk.ingredients))]
    allowed_types = _INTENT_CHUNK_TYPES.get(query.intent)
    if allowed_types:
        intent_filtered = [x for x in filtered if x.chunk.chunk_type in allowed_types]
        if intent_filtered:
            filtered = intent_filtered
    if query.constraints:
        constraint_filtered = [
            x
            for x in filtered
            if any(c in x.chunk.text or c in x.chunk.tags for c in query.constraints)
        ]
        if constraint_filtered:
            filtered = constraint_filtered
    return filtered
