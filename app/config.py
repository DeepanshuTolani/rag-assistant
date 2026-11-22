"""Application settings, read exclusively from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "groq"))  # groq|mock
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", ""))

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", ""))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    min_retrieval_score: float = field(default_factory=lambda: _float("MIN_RETRIEVAL_SCORE", 0.0))

    max_upload_bytes: int = field(default_factory=lambda: _int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
    max_pdf_pages: int = field(default_factory=lambda: _int("MAX_PDF_PAGES", 300))
    max_docs_per_session: int = field(default_factory=lambda: _int("MAX_DOCS_PER_SESSION", 10))
    history_turns: int = field(default_factory=lambda: _int("HISTORY_TURNS", 6))
    session_ttl_s: int = field(default_factory=lambda: _int("SESSION_TTL_S", 3600))
    max_sessions: int = field(default_factory=lambda: _int("MAX_SESSIONS", 200))
    max_question_chars: int = field(default_factory=lambda: _int("MAX_QUESTION_CHARS", 2000))

    rate_limit_per_min: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MIN", 30))
    llm_timeout_connect: float = field(default_factory=lambda: _float("LLM_TIMEOUT_CONNECT", 5))
    llm_timeout_read: float = field(default_factory=lambda: _float("LLM_TIMEOUT_READ", 60))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 3))

    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))

    def public(self) -> dict:
        """Non-secret config, exposed at /healthz so benchmark runs can record it."""
        return {
            "llm_backend": self.llm_backend,
            "groq_model": self.groq_model,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model or None,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "min_retrieval_score": self.min_retrieval_score,
            "rate_limit_per_min": self.rate_limit_per_min,
            "max_upload_bytes": self.max_upload_bytes,
        }
