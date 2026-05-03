from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from recipe_rag.schemas import Ingredient, Recipe
from recipe_rag.utils.io import read_json


_AD_PATTERNS = [
    re.compile(r"关注.*?公众号"),
    re.compile(r"VX[:：]?\s*\w+", re.IGNORECASE),
    re.compile(r"微信[:：]?\s*\w+"),
    re.compile(r"淘宝店[:：]?.*"),
]

_UNIT_ALIASES = {
    "一小勺": "5g",
    "1小勺": "5g",
    "一茶匙": "5g",
    "一大勺": "15g",
    "1大勺": "15g",
    "一汤匙": "15g",
    "少许": "适量",
    "适量": "适量",
}


class IngredientNormalizer:
    def __init__(self, alias_map: dict[str, str] | None = None):
        self.alias_map = alias_map or {}

    @classmethod
    def from_file(cls, path: str | Path | None) -> "IngredientNormalizer":
        if not path:
            return cls()
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("alias map must be a JSON object")
        return cls({str(k): str(v) for k, v in data.items()})

    def normalize_name(self, name: str) -> str:
        name = re.sub(r"\s+", "", name.strip())
        return self.alias_map.get(name, name)

    def normalize_amount(self, amount: str | None, unit: str | None = None) -> str | None:
        if not amount and not unit:
            return None
        text = f"{amount or ''}{unit or ''}".strip()
        text = re.sub(r"\s+", "", text)
        return _UNIT_ALIASES.get(text, text)

    def normalize_ingredient(self, ingredient: Ingredient) -> Ingredient:
        return replace(
            ingredient,
            normalized_name=self.normalize_name(ingredient.name),
            normalized_amount=self.normalize_amount(ingredient.amount, ingredient.unit),
        )


class RecipeCleaner:
    def __init__(self, ingredient_normalizer: IngredientNormalizer):
        self.ingredient_normalizer = ingredient_normalizer

    def clean_text(self, text: str) -> str:
        text = text.strip()
        if "关注" in text and "公众号" in text:
            return ""
        for pattern in _AD_PATTERNS:
            text = pattern.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_recipe(self, recipe: Recipe) -> Recipe:
        ingredients = [
            self.ingredient_normalizer.normalize_ingredient(x)
            for x in recipe.ingredients
            if x.name.strip()
        ]
        steps = [self.clean_text(x) for x in recipe.steps]
        tips = [self.clean_text(x) for x in recipe.tips]
        return replace(
            recipe,
            title=self.clean_text(recipe.title),
            ingredients=ingredients,
            steps=[x for x in steps if x],
            tips=[x for x in tips if x],
            tags=list(dict.fromkeys([self.clean_text(x) for x in recipe.tags if x.strip()])),
            taste_tags=list(dict.fromkeys([self.clean_text(x) for x in recipe.taste_tags if x.strip()])),
        )
