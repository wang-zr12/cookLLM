from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class EmbeddingModel:
    """BCE embedding 适配器。

    安装 sentence-transformers 后会使用真实模型；否则使用确定性的哈希向量，便于本地测试。
    """

    def __init__(self, model_name: str | None = None, dim: int = 768):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                inferred_dim = self._model.get_sentence_embedding_dimension()
                if inferred_dim:
                    self.dim = int(inferred_dim)
            except Exception:
                self._model = None

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        text_list = list(texts)
        if self._model is not None:
            vectors = self._model.encode(
                text_list,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype="float32")
        return np.vstack([self._hash_embedding(x) for x in text_list]).astype("float32")

    def _hash_embedding(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for token in text.split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        if not np.any(vec):
            digest = hashlib.md5(text.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dim] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec
