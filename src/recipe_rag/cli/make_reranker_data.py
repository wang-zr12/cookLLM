from __future__ import annotations

import argparse

from recipe_rag.schemas import RecipeChunk
from recipe_rag.training.reranker.data_builder import QuerySpec, build_reranker_pairs
from recipe_rag.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BCE Reranker fine-tuning pairs.")
    parser.add_argument("--queries", required=True, help="JSONL: query_id, query, recipe_id, intent.")
    parser.add_argument("--chunks", required=True, help="Index chunks JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--negatives-per-positive", type=int, default=7)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    queries = [QuerySpec.from_dict(x) for x in read_jsonl(args.queries)]
    chunks = [RecipeChunk.from_dict(x) for x in read_jsonl(args.chunks)]
    pairs = build_reranker_pairs(
        queries,
        chunks,
        negatives_per_positive=args.negatives_per_positive,
        embedding_model_name=args.embedding_model,
    )
    write_jsonl(args.output, [x.to_dict() for x in pairs])
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
