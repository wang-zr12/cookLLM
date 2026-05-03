from __future__ import annotations

from pathlib import Path

from recipe_rag.indexing.bm25 import BM25Index
from recipe_rag.indexing.embeddings import EmbeddingModel
from recipe_rag.indexing.vector_store import VectorStore
from recipe_rag.ingest import JsonlRecipeCrawler
from recipe_rag.preprocess import IngredientNormalizer, RecipeChunker, RecipeCleaner
from recipe_rag.utils.io import ensure_dir, write_jsonl


def build_recipe_index(
    input_path: str | Path,
    output_dir: str | Path,
    alias_path: str | Path | None = None,
    embedding_model_name: str | None = None,
    vector_backend: str = "auto",
) -> None:
    output = ensure_dir(output_dir)
    normalizer = IngredientNormalizer.from_file(alias_path)
    cleaner = RecipeCleaner(normalizer)
    chunker = RecipeChunker()

    recipes = [cleaner.clean_recipe(x) for x in JsonlRecipeCrawler(input_path).crawl()]
    chunks = [chunk for recipe in recipes for chunk in chunker.chunk(recipe)]
    write_jsonl(output / "recipes.clean.jsonl", [x.to_dict() for x in recipes])
    write_jsonl(output / "chunks.jsonl", [x.to_dict() for x in chunks])

    embedder = EmbeddingModel(embedding_model_name)
    vectors = embedder.encode([x.text for x in chunks])
    vector_store = VectorStore.build(chunks, vectors, backend=vector_backend)
    vector_store.save(output)

    bm25 = BM25Index(chunks)
    bm25.save(output / "bm25.pkl")
