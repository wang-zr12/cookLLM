from __future__ import annotations

import argparse

from recipe_rag.schemas import Recipe
from recipe_rag.utils.io import read_jsonl, write_jsonl


def build_sft_examples(recipes: list[Recipe]) -> list[dict]:
    examples = []
    for recipe in recipes:
        ingredients = "、".join(
            f"{x.normalized_name or x.name}{' ' + (x.normalized_amount or x.amount) if (x.normalized_amount or x.amount) else ''}"
            for x in recipe.ingredients
        )
        steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(recipe.steps))
        tips = "\n".join(f"- {tip}" for tip in recipe.tips) if recipe.tips else "无特殊贴士。"
        answer = f"{recipe.title}\n\n食材：{ingredients}\n\n步骤：\n{steps}\n\n贴士：\n{tips}"
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": "你是专业中文菜谱助手，回答要准确、清晰、可执行。"},
                    {"role": "user", "content": f"{recipe.title}怎么做？"},
                    {"role": "assistant", "content": answer},
                ],
                "recipe_id": recipe.recipe_id,
            }
        )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": "你是专业中文菜谱助手，回答要准确、清晰、可执行。"},
                    {"role": "user", "content": f"做{recipe.title}需要哪些食材？"},
                    {"role": "assistant", "content": f"做{recipe.title}需要：{ingredients}。"},
                ],
                "recipe_id": recipe.recipe_id,
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLaMA SFT JSONL data from cleaned recipes.")
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recipes = [Recipe.from_dict(row) for row in read_jsonl(args.recipes)]
    write_jsonl(args.output, build_sft_examples(recipes))


if __name__ == "__main__":
    main()
