from __future__ import annotations

from recipe_rag.preprocess.normalization import IngredientNormalizer
from recipe_rag.schemas import Intent, QueryUnderstanding


_INTENT_KEYWORDS = {
    Intent.RECOMMEND: ["推荐", "吃什么", "适合", "搭配", "菜谱"],
    Intent.METHOD: ["怎么做", "做法", "步骤", "流程", "教程"],
    Intent.TIP: ["技巧", "窍门", "为什么", "失败", "不粘", "入味", "去腥"],
    Intent.INGREDIENT: ["食材", "材料", "配料", "需要什么", "用量"],
}

_CONSTRAINTS = [
    "低脂",
    "减脂",
    "低卡",
    "高蛋白",
    "快手",
    "下饭",
    "清淡",
    "辣",
    "不辣",
    "少油",
    "早餐",
    "晚餐",
    "家常",
    "儿童",
]


class QueryUnderstandingService:
    def __init__(self, ingredient_normalizer: IngredientNormalizer | None = None):
        self.ingredient_normalizer = ingredient_normalizer or IngredientNormalizer()

    def parse(self, query: str, known_ingredients: set[str] | None = None) -> QueryUnderstanding:
        intent = self._detect_intent(query)
        constraints = [x for x in _CONSTRAINTS if x in query]
        ingredients = self._extract_ingredients(query, known_ingredients or set())
        rewritten = self._rewrite(query, intent, ingredients, constraints)
        return QueryUnderstanding(
            raw_query=query,
            rewritten_query=rewritten,
            intent=intent,
            ingredients=ingredients,
            constraints=constraints,
        )

    def _detect_intent(self, query: str) -> Intent:
        for intent, keywords in _INTENT_KEYWORDS.items():
            if any(x in query for x in keywords):
                return intent
        return Intent.RECOMMEND

    def _extract_ingredients(self, query: str, known_ingredients: set[str]) -> list[str]:
        found = []
        for ingredient in sorted(known_ingredients, key=len, reverse=True):
            if ingredient and ingredient in query:
                found.append(ingredient)
        for alias, canonical in self.ingredient_normalizer.alias_map.items():
            if alias in query or canonical in query:
                found.append(canonical)
        return list(dict.fromkeys(found))

    def _rewrite(self, query: str, intent: Intent, ingredients: list[str], constraints: list[str]) -> str:
        parts = [query]
        if intent != Intent.UNKNOWN:
            parts.append(f"意图:{intent.value}")
        if ingredients:
            parts.append("食材:" + "、".join(ingredients))
        if constraints:
            parts.append("约束:" + "、".join(constraints))
        return " ".join(parts)
