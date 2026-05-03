from __future__ import annotations

from pathlib import Path

import numpy as np

from recipe_rag.schemas import RecipeChunk, SearchResult
from recipe_rag.utils.io import dump_pickle, ensure_dir, load_pickle, read_jsonl, write_jsonl


class VectorStore:
    def __init__(self, chunks: list[RecipeChunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = self._normalize(vectors.astype("float32"))
        self._faiss_index = None

    @classmethod
    def build(cls, chunks: list[RecipeChunk], vectors: np.ndarray, backend: str = "auto") -> "VectorStore":
        store = cls(chunks, vectors)
        if backend in {"auto", "faiss"}:
            try:
                import faiss

                index = faiss.IndexFlatIP(store.vectors.shape[1])
                index.add(store.vectors)
                store._faiss_index = index
            except Exception:
                if backend == "faiss":
                    raise
        return store

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        q = self._normalize(query_vector.reshape(1, -1).astype("float32"))
        top_k = min(top_k, len(self.chunks))
        if top_k <= 0:
            return []
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(q, top_k)
            pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if int(i) >= 0]
        else:
            scores = (self.vectors @ q[0]).astype("float32")
            ids = np.argsort(-scores)[:top_k]
            pairs = [(int(i), float(scores[i])) for i in ids]
        return [
            SearchResult(chunk=self.chunks[idx], score=score, source="vector", rank=rank + 1)
            for rank, (idx, score) in enumerate(pairs)
        ]

    def save(self, directory: str | Path) -> None:
        path = ensure_dir(directory)
        np.save(path / "vectors.npy", self.vectors)
        write_jsonl(path / "chunks.jsonl", [x.to_dict() for x in self.chunks])
        dump_pickle(path / "vector_meta.pkl", {"dim": int(self.vectors.shape[1])})

    @classmethod
    def load(cls, directory: str | Path, backend: str = "auto") -> "VectorStore":
        path = Path(directory)
        chunks = [RecipeChunk.from_dict(x) for x in read_jsonl(path / "chunks.jsonl")]
        vectors = np.load(path / "vectors.npy")
        _ = load_pickle(path / "vector_meta.pkl")
        return cls.build(chunks, vectors, backend=backend)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
