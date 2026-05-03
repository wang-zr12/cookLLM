from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Intent(str, Enum):
    RECOMMEND = "recommend"
    METHOD = "method"
    TIP = "tip"
    INGREDIENT = "ingredient"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    SUMMARY = "summary"
    INGREDIENTS = "ingredients"
    STEPS = "steps"
    TIPS = "tips"


@dataclass
class Ingredient:
    name: str
    amount: str | None = None
    unit: str | None = None
    normalized_name: str | None = None
    normalized_amount: str | None = None

    @classmethod
    def from_obj(cls, obj: str | dict[str, Any]) -> "Ingredient":
        if isinstance(obj, str):
            return cls(name=obj)
        return cls(
            name=str(obj.get("name", "")).strip(),
            amount=obj.get("amount"),
            unit=obj.get("unit"),
            normalized_name=obj.get("normalized_name"),
            normalized_amount=obj.get("normalized_amount"),
        )


@dataclass
class Recipe:
    recipe_id: str
    title: str
    ingredients: list[Ingredient]
    steps: list[str]
    tags: list[str] = field(default_factory=list)
    taste_tags: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    source: str | None = None
    url: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Recipe":
        return cls(
            recipe_id=str(row.get("recipe_id") or row.get("id") or row.get("url") or row["title"]),
            title=str(row["title"]).strip(),
            ingredients=[Ingredient.from_obj(x) for x in row.get("ingredients", [])],
            steps=[str(x).strip() for x in row.get("steps", []) if str(x).strip()],
            tags=[str(x).strip() for x in row.get("tags", []) if str(x).strip()],
            taste_tags=[str(x).strip() for x in row.get("taste_tags", []) if str(x).strip()],
            tips=[str(x).strip() for x in row.get("tips", []) if str(x).strip()],
            source=row.get("source"),
            url=row.get("url"),
            meta=dict(row.get("meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ingredients"] = [asdict(x) for x in self.ingredients]
        return data

    @property
    def normalized_ingredient_names(self) -> set[str]:
        return {x.normalized_name or x.name for x in self.ingredients if x.name}


@dataclass
class RecipeChunk:
    chunk_id: str
    recipe_id: str
    title: str
    chunk_type: ChunkType
    text: str
    ingredients: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RecipeChunk":
        return cls(
            chunk_id=str(row["chunk_id"]),
            recipe_id=str(row["recipe_id"]),
            title=str(row["title"]),
            chunk_type=ChunkType(row["chunk_type"]),
            text=str(row["text"]),
            ingredients=[str(x) for x in row.get("ingredients", [])],
            tags=[str(x) for x in row.get("tags", [])],
            source=row.get("source"),
            meta=dict(row.get("meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunk_type"] = self.chunk_type.value
        return data


@dataclass
class QueryUnderstanding:
    raw_query: str
    rewritten_query: str
    intent: Intent
    ingredients: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    chunk: RecipeChunk
    score: float
    source: Literal["vector", "bm25", "rrf", "reranker"]
    rank: int | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerPair:
    query: str
    passage: str
    label: float
    query_id: str | None = None
    chunk_id: str | None = None
    recipe_id: str | None = None
    intent: str | None = None
    negative_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
