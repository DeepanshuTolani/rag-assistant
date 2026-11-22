"""Vector stores. FAISS (exact inner-product index) when installed, else a NumPy fallback.

Both expect L2-normalised vectors, so inner product == cosine similarity.
"""
from __future__ import annotations

import numpy as np


class NumpyStore:
    kind = "numpy"

    def __init__(self, dim: int):
        self.dim = dim
        self._m = np.zeros((0, dim), dtype="float32")

    def __len__(self) -> int:
        return self._m.shape[0]

    def add(self, vecs: np.ndarray) -> None:
        self._m = np.vstack([self._m, vecs.astype("float32")])

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        scores = self._m @ q.astype("float32").reshape(-1)
        k = min(k, len(self))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


class FaissStore:
    kind = "faiss"

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self._n = 0
        self._index = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:
        return self._n

    def add(self, vecs: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vecs, dtype="float32"))
        self._n += vecs.shape[0]

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._n == 0:
            return []
        s, i = self._index.search(np.ascontiguousarray(q, dtype="float32").reshape(1, -1), min(k, self._n))
        return [(int(a), float(b)) for a, b in zip(i[0], s[0], strict=True) if a >= 0]


def make_store(dim: int):
    try:
        return FaissStore(dim)
    except ImportError:
        return NumpyStore(dim)
