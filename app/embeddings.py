"""Embedding backends. Production = local sentence-transformers; HashingEmbedder is for tests only."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .chunking import SpanFn, whitespace_spans


class Embedder:
    name: str = "base"
    max_seq_length: int | None = None
    dim: int = 0

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def span_fn(self) -> SpanFn:
        return whitespace_spans


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed (pip install -r requirements.txt). "
                "For tests/offline use set EMBEDDING_MODEL=hashing."
            ) from e
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.max_seq_length = int(self.model.max_seq_length)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        v = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(v, dtype="float32")

    def span_fn(self) -> SpanFn:
        """Character spans of the model's own sub-word tokens (so chunk sizes are real tokens)."""
        tok = self.model.tokenizer

        def fn(text: str) -> list[tuple[int, int]]:
            spans: list[tuple[int, int]] = []
            # tokenize in slices to stay clear of the tokenizer's max-length warnings
            step = 20000
            for off in range(0, len(text), step):
                piece = text[off : off + step]
                enc = tok(piece, add_special_tokens=False, return_offsets_mapping=True)
                spans.extend((a + off, b + off) for a, b in enc["offset_mapping"] if b > a)
            return spans

        return fn


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder. NOT a quality model: tests/offline smoke only."""

    name = "hashing"
    max_seq_length = None

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


_CACHE: dict[str, Embedder] = {}


def get_embedder(model_name: str) -> Embedder:
    """Process-wide singleton per model (loading a model is slow; it is read-only afterwards)."""
    if model_name not in _CACHE:
        _CACHE[model_name] = (
            HashingEmbedder() if model_name == "hashing" else SentenceTransformerEmbedder(model_name)
        )
    return _CACHE[model_name]
