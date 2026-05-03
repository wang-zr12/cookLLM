from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from recipe_rag.indexing.tokenize import tokenize_zh
from recipe_rag.schemas import RecipeChunk, SearchResult
from recipe_rag.utils.io import dump_pickle, load_pickle


class BM25Index:
    def __init__(self, chunks: list[RecipeChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize_zh(x.text) for x in chunks]
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        for tokens in self.tokenized:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_freq.update(tf.keys())
        self.doc_lens = [len(x) for x in self.tokenized]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_tokens = tokenize_zh(query)
        scored = []
        for idx, tf in enumerate(self.term_freqs):
            score = self._score(query_tokens, tf, self.doc_lens[idx])
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(chunk=self.chunks[idx], score=score, source="bm25", rank=rank + 1)
            for rank, (idx, score) in enumerate(scored[:top_k])
        ]

    def _score(self, query_tokens: list[str], tf: Counter[str], doc_len: int) -> float:
        if not self.chunks:
            return 0.0
        score = 0.0
        n_docs = len(self.chunks)
        for token in query_tokens:
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
            score += idf * (freq * (self.k1 + 1) / denom)
        return float(score)

    def save(self, path: str | Path) -> None:
        dump_pickle(path, self)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        return load_pickle(path)
