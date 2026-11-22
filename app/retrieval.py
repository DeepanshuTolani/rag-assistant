"""Per-document-set index + top-k retrieval with optional cross-encoder reranking."""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import Embedder
from .vectorstore import make_store


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity (or reranker score if reranking is on)
    rank: int


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class Index:
    """All chunks of one session's documents. Not shared between sessions."""

    def __init__(self, embedder: Embedder, reranker: Reranker | None = None):
        self.embedder = embedder
        self.reranker = reranker
        self.chunks: list[Chunk] = []
        self.store = make_store(embedder.dim) if embedder.dim else None

    def __len__(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.encode([c.text for c in chunks])
        if self.store is None:  # dim unknown until first encode
            self.store = make_store(vecs.shape[1])
        self.store.add(vecs)
        self.chunks.extend(chunks)

    def remove_file(self, file: str) -> None:
        """Rebuild without ``file`` (re-upload replaces a document)."""
        keep = [c for c in self.chunks if c.file != file]
        if len(keep) == len(self.chunks):
            return
        self.chunks = []
        self.store = make_store(self.embedder.dim) if self.embedder.dim else None
        self.add_chunks(keep)

    def search(self, query: str, k: int, candidates_mult: int = 4) -> list[Hit]:
        if not self.chunks or self.store is None:
            return []
        q = self.embedder.encode([query])[0]
        n = k * candidates_mult if self.reranker else k
        raw = self.store.search(q, n)
        if self.reranker and raw:
            rs = self.reranker.score(query, [self.chunks[i].text for i, _ in raw])
            raw = sorted(((i, s) for (i, _), s in zip(raw, rs, strict=True)), key=lambda t: -t[1])[:k]
        return [Hit(self.chunks[i], s, r + 1) for r, (i, s) in enumerate(raw[:k])]
