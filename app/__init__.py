"""Flask application factory."""
from __future__ import annotations

import os
import secrets

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from .config import Settings
from .embeddings import get_embedder
from .llm import GroqClient, MockLLM
from .logging_utils import log_event, setup_logging
from .rag import RagPipeline
from .ratelimit import RateLimiter
from .retrieval import Index, Reranker
from .sessions import SessionStore


def create_app(settings: Settings | None = None, llm=None, embedder=None) -> Flask:
    try:  # local dev convenience only; production passes real env vars
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    s = settings or Settings()
    setup_logging(s.log_dir)

    app = Flask(__name__)
    app.config["SETTINGS"] = s
    if not s.secret_key:
        s.secret_key = secrets.token_hex(32)
        log_event("SECRET_KEY not set: using a random per-process key (sessions reset on restart; "
                  "set SECRET_KEY in production)")
    app.secret_key = s.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=s.max_upload_bytes + 64 * 1024,  # headroom for multipart framing
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
    )

    emb = embedder or get_embedder(s.embedding_model)
    reranker = Reranker(s.rerank_model) if s.rerank_model else None
    if llm is None:
        llm = MockLLM() if s.llm_backend == "mock" else GroqClient(
            s.groq_api_key, s.groq_model, s.groq_base_url, s.llm_timeout_connect,
            s.llm_timeout_read, s.llm_max_retries)
        if s.llm_backend != "mock" and not s.groq_api_key:
            log_event("GROQ_API_KEY is not set: chat requests will fail with HTTP 500")

    app.extensions["embedder"] = emb
    app.extensions["sessions"] = SessionStore(lambda: Index(emb, reranker), s.session_ttl_s,
                                              s.max_sessions)
    app.extensions["limiter"] = RateLimiter(s.rate_limit_per_min)
    app.extensions["pipeline"] = RagPipeline(s, llm)

    from .routes import bp

    app.register_blueprint(bp)

    @app.errorhandler(HTTPException)
    def _http_err(e: HTTPException):
        msg = "File too large" if e.code == 413 else e.description
        return jsonify({"error": msg}), e.code

    @app.errorhandler(Exception)
    def _any_err(e: Exception):
        log_event("unhandled_exception", error=repr(e))
        return jsonify({"error": "Internal server error"}), 500

    return app
