from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: object) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_pickle(path: str | Path, obj: object) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> T:
    with Path(path).open("rb") as f:
        return pickle.load(f)
