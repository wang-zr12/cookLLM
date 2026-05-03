from __future__ import annotations

import argparse

from recipe_rag.generation import build_llama3_chat_prompt
from recipe_rag.retrieval import RecipeRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online hybrid retrieval for a recipe query.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--alias", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    pipeline = RecipeRAGPipeline.from_dir(
        args.index,
        alias_path=args.alias,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
    )
    understood, results = pipeline.retrieve(args.query)
    print(f"intent={understood.intent.value} ingredients={understood.ingredients} constraints={understood.constraints}")
    for item in results:
        print(f"\n#{item.rank} score={item.score:.4f} chunk={item.chunk.chunk_id}")
        print(item.chunk.text[:500])
    if args.show_prompt:
        print("\n--- LLaMA-3 Prompt ---")
        print(build_llama3_chat_prompt(understood, results))


if __name__ == "__main__":
    main()
