from __future__ import annotations

import argparse

from recipe_rag.indexing.build import build_recipe_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline recipe RAG indexes.")
    parser.add_argument("--input", required=True, help="Raw recipe JSONL path.")
    parser.add_argument("--output", required=True, help="Index output directory.")
    parser.add_argument("--alias", default=None, help="Ingredient alias JSON path.")
    parser.add_argument("--embedding-model", default=None, help="BCE embedding model name or local path.")
    parser.add_argument("--vector-backend", default="auto", choices=["auto", "faiss", "numpy"])
    args = parser.parse_args()
    build_recipe_index(
        input_path=args.input,
        output_dir=args.output,
        alias_path=args.alias,
        embedding_model_name=args.embedding_model,
        vector_backend=args.vector_backend,
    )
    print(f"Index written to {args.output}")


if __name__ == "__main__":
    main()
