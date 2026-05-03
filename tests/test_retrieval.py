from recipe_rag.indexing import BM25Index, EmbeddingModel, VectorStore
from recipe_rag.preprocess import IngredientNormalizer, RecipeChunker, RecipeCleaner
from recipe_rag.retrieval import RecipeRAGPipeline, QueryUnderstandingService
from recipe_rag.retrieval.reranker import BCEReranker
from recipe_rag.schemas import Intent, Recipe


def _pipeline():
    recipes = [
        Recipe.from_dict(
            {
                "recipe_id": "tomato_egg",
                "title": "番茄炒蛋",
                "ingredients": [{"name": "西红柿"}, {"name": "鸡蛋"}],
                "steps": ["番茄炒出汁", "加入鸡蛋翻炒"],
                "tags": ["快手"],
            }
        ),
        Recipe.from_dict(
            {
                "recipe_id": "chicken",
                "title": "低脂鸡胸肉",
                "ingredients": [{"name": "鸡胸肉"}],
                "steps": ["少油煎熟"],
                "tags": ["低脂"],
            }
        ),
    ]
    normalizer = IngredientNormalizer({"西红柿": "番茄"})
    cleaner = RecipeCleaner(normalizer)
    chunker = RecipeChunker()
    chunks = [chunk for recipe in recipes for chunk in chunker.chunk(cleaner.clean_recipe(recipe))]
    embedder = EmbeddingModel()
    vectors = embedder.encode([x.text for x in chunks])
    return RecipeRAGPipeline(
        VectorStore.build(chunks, vectors, backend="numpy"),
        BM25Index(chunks),
        embedder,
        QueryUnderstandingService(normalizer),
        BCEReranker(),
    )


def test_query_understanding_extracts_alias():
    service = QueryUnderstandingService(IngredientNormalizer({"西红柿": "番茄"}))
    parsed = service.parse("西红柿怎么做", {"番茄"})
    assert parsed.intent == Intent.METHOD
    assert parsed.ingredients == ["番茄"]


def test_pipeline_retrieves_method_chunk():
    understood, results = _pipeline().retrieve("鸡胸肉低脂做法")
    assert understood.ingredients == ["鸡胸肉"]
    assert results
    assert results[0].chunk.recipe_id == "chicken"
