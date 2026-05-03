from __future__ import annotations

from recipe_rag.schemas import ChunkType, Recipe, RecipeChunk


class RecipeChunker:
    def chunk(self, recipe: Recipe) -> list[RecipeChunk]:
        ingredients = [x.normalized_name or x.name for x in recipe.ingredients]
        tags = list(dict.fromkeys([*recipe.tags, *recipe.taste_tags]))
        chunks: list[RecipeChunk] = []

        summary_parts = [
            f"菜名：{recipe.title}",
            f"食材：{'、'.join(ingredients)}" if ingredients else "",
            f"标签：{'、'.join(tags)}" if tags else "",
            f"简介：这是一道{('、'.join(tags)) if tags else '家常'}菜，主要食材包括{'、'.join(ingredients[:6])}。" if ingredients else "",
        ]
        chunks.append(self._make(recipe, ChunkType.SUMMARY, "\n".join(x for x in summary_parts if x), ingredients, tags))

        ingredient_lines = []
        for item in recipe.ingredients:
            name = item.normalized_name or item.name
            amount = item.normalized_amount or item.amount or ""
            ingredient_lines.append(f"{name} {amount}".strip())
        chunks.append(self._make(recipe, ChunkType.INGREDIENTS, "食材清单：\n" + "\n".join(ingredient_lines), ingredients, tags))

        step_lines = [f"{idx + 1}. {step}" for idx, step in enumerate(recipe.steps)]
        chunks.append(self._make(recipe, ChunkType.STEPS, f"{recipe.title} 做法：\n" + "\n".join(step_lines), ingredients, tags))

        if recipe.tips:
            tip_lines = [f"- {tip}" for tip in recipe.tips]
            chunks.append(self._make(recipe, ChunkType.TIPS, f"{recipe.title} 技巧贴士：\n" + "\n".join(tip_lines), ingredients, tags))

        return chunks

    def _make(
        self,
        recipe: Recipe,
        chunk_type: ChunkType,
        text: str,
        ingredients: list[str],
        tags: list[str],
    ) -> RecipeChunk:
        return RecipeChunk(
            chunk_id=f"{recipe.recipe_id}:{chunk_type.value}",
            recipe_id=recipe.recipe_id,
            title=recipe.title,
            chunk_type=chunk_type,
            text=text,
            ingredients=ingredients,
            tags=tags,
            source=recipe.source,
            meta={"url": recipe.url, **recipe.meta},
        )
