from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BCE Reranker for recipe-domain relevance.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args)


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
        from sentence_transformers.readers import InputExample

        from recipe_rag.utils.io import read_jsonl
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: pip install -e '.[retrieval,train]'") from exc

    train_samples = [
        InputExample(texts=[row["query"], row["passage"]], label=float(row["label"]))
        for row in read_jsonl(args.train)
    ]
    train_loader = torch.utils.data.DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    model = CrossEncoder(args.model, num_labels=1)

    evaluator = None
    if args.eval:
        eval_rows = list(read_jsonl(args.eval))
        evaluator = CEBinaryClassificationEvaluator(
            sentence_pairs=[[row["query"], row["passage"]] for row in eval_rows],
            labels=[1 if float(row["label"]) > 0 else 0 for row in eval_rows],
            name="recipe-reranker-eval",
        )

    warmup_steps = max(10, int(len(train_loader) * args.epochs * 0.1))
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(Path(args.output)),
    )


if __name__ == "__main__":
    main()
