from __future__ import annotations

from pathlib import Path

from recipe_rag.indexing import BM25Index, EmbeddingModel, VectorStore
from recipe_rag.preprocess import IngredientNormalizer
from recipe_rag.retrieval.filters import hard_filter
from recipe_rag.retrieval.fusion import reciprocal_rank_fusion
from recipe_rag.retrieval.query_understanding import QueryUnderstandingService
from recipe_rag.retrieval.reranker import BCEReranker
from recipe_rag.schemas import QueryUnderstanding, SearchResult


class RecipeRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: EmbeddingModel,
        query_service: QueryUnderstandingService,
        reranker: BCEReranker,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        fused_top_k: int = 20,
        rerank_top_k: int = 3,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.query_service = query_service
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.fused_top_k = fused_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        self.known_ingredients = {i for c in vector_store.chunks for i in c.ingredients}

    @classmethod
    def from_dir(
        cls,
        index_dir: str | Path,
        alias_path: str | Path | None = None,
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
        vector_backend: str = "auto",
    ) -> "RecipeRAGPipeline":
        index_path = Path(index_dir)
        vector_store = VectorStore.load(index_path, backend=vector_backend)
        bm25 = BM25Index.load(index_path / "bm25.pkl")
        normalizer = IngredientNormalizer.from_file(alias_path)
        return cls(
            vector_store=vector_store,
            bm25_index=bm25,
            embedder=EmbeddingModel(embedding_model_name),
            query_service=QueryUnderstandingService(normalizer),
            reranker=BCEReranker(reranker_model_name),
        )

    def retrieve(self, query: str) -> tuple[QueryUnderstanding, list[SearchResult]]:
        understood = self.query_service.parse(query, self.known_ingredients)
        query_vector = self.embedder.encode([understood.rewritten_query])[0]
        vector_results = self.vector_store.search(query_vector, self.vector_top_k)
        bm25_results = self.bm25_index.search(understood.rewritten_query, self.bm25_top_k)
        fused = reciprocal_rank_fusion([vector_results, bm25_results], k=self.rrf_k, top_k=self.fused_top_k)
        filtered = hard_filter(fused, understood)
        candidates = filtered or fused
        reranked = self.reranker.rerank(understood.rewritten_query, candidates, top_k=self.rerank_top_k)
        return understood, reranked
