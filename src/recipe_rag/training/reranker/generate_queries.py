from __future__ import annotations

from recipe_rag.schemas import Intent, Recipe


def template_queries(recipe: Recipe) -> list[dict]:
    ingredients = "、".join([x.normalized_name or x.name for x in recipe.ingredients[:3]])
    rows = [
        {
            "query_id": f"{recipe.recipe_id}:recommend",
            "query": f"推荐一道用{ingredients}做的菜",
            "recipe_id": recipe.recipe_id,
            "intent": Intent.RECOMMEND.value,
        },
        {
            "query_id": f"{recipe.recipe_id}:method",
            "query": f"{recipe.title}怎么做",
            "recipe_id": recipe.recipe_id,
            "intent": Intent.METHOD.value,
        },
        {
            "query_id": f"{recipe.recipe_id}:ingredient",
            "query": f"做{recipe.title}需要哪些材料",
            "recipe_id": recipe.recipe_id,
            "intent": Intent.INGREDIENT.value,
        },
    ]
    if recipe.tips:
        rows.append(
            {
                "query_id": f"{recipe.recipe_id}:tip",
                "query": f"{recipe.title}有什么技巧",
                "recipe_id": recipe.recipe_id,
                "intent": Intent.TIP.value,
            }
        )
    return rows
