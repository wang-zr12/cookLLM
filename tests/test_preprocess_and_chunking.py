from recipe_rag.preprocess import IngredientNormalizer, RecipeChunker, RecipeCleaner
from recipe_rag.schemas import ChunkType, Recipe


def test_cleaner_normalizes_alias_and_amount():
    recipe = Recipe.from_dict(
        {
            "recipe_id": "r1",
            "title": "番茄炒蛋",
            "ingredients": [{"name": "西红柿", "amount": "一小勺"}],
            "steps": ["关注公众号 获取更多", "炒熟"],
        }
    )
    cleaner = RecipeCleaner(IngredientNormalizer({"西红柿": "番茄"}))
    clean = cleaner.clean_recipe(recipe)
    assert clean.ingredients[0].normalized_name == "番茄"
    assert clean.ingredients[0].normalized_amount == "5g"
    assert clean.steps == ["炒熟"]


def test_chunker_emits_intent_specific_chunks():
    recipe = Recipe.from_dict(
        {
            "recipe_id": "r1",
            "title": "番茄炒蛋",
            "ingredients": [{"name": "番茄"}, {"name": "鸡蛋"}],
            "steps": ["炒番茄", "炒鸡蛋"],
            "tips": ["先炒出汁"],
        }
    )
    chunks = RecipeChunker().chunk(recipe)
    chunk_types = {x.chunk_type for x in chunks}
    assert {ChunkType.SUMMARY, ChunkType.INGREDIENTS, ChunkType.STEPS, ChunkType.TIPS}.issubset(chunk_types)
