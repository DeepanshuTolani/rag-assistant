"""HTTP routes. Session id lives in a signed, HttpOnly cookie; all state is keyed by it."""
from __future__ import annotations

import json
import time
import uuid

from flask import (
    Blueprint,
    Response,
    current_app,
    g,
    jsonify,
    render_template,
    request,
    session,
    stream_with_context,
)

from .chunking import chunk_pages
from .ingest import UploadError, validate_and_extract
from .llm import LLMError
from .logging_utils import log_event

bp = Blueprint("main", __name__)


def _sid() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
        session.permanent = False
    return session["sid"]


def _rl_key() -> str:
    return f"{_sid()}|{request.remote_addr}"


def _error(msg: str, status: int, **extra):
    resp = jsonify({"error": msg, **extra})
    resp.status_code = status
    return resp


def _check_rate():
    ok, retry = current_app.extensions["limiter"].check(_rl_key())
    if ok:
        return None
    g.log["rate_limited"] = True
    resp = _error("Rate limit exceeded", 429, retry_after_s=round(retry, 1))
    resp.headers["Retry-After"] = str(int(retry) + 1)
    return resp


@bp.before_app_request
def _start():
    g.t0 = time.perf_counter()
    g.log = {}
    g.skip_log = False


@bp.after_app_request
def _log(resp):
    if getattr(g, "skip_log", False) or request.path.startswith("/static"):
        return resp
    log_event("request", method=request.method, path=request.path, status=resp.status_code,
              latency_ms=round((time.perf_counter() - g.t0) * 1000, 2),
              sid=session.get("sid", "")[:8], **g.log)
    return resp


@bp.get("/")
def index():
    _sid()
    return render_template("chat.html")


@bp.get("/healthz")
def healthz():
    s = current_app.config["SETTINGS"]
    emb = current_app.extensions["embedder"]
    return jsonify({"status": "ok", "config": s.public(), "embedder": emb.name,
                    "embedder_max_seq_length": emb.max_seq_length})


@bp.post("/api/upload")
def upload():
    s = current_app.config["SETTINGS"]
    sess = current_app.extensions["sessions"].get(_sid())
    if (r := _check_rate()):
        return r
    f = request.files.get("file")
    if f is None:
        return _error("No file part named 'file'", 400)
    data = f.read(s.max_upload_bytes + 1)  # never buffer more than limit+1 bytes
    try:
        name, pages = validate_and_extract(f.filename or "", data, s.max_upload_bytes, s.max_pdf_pages)
    except UploadError as e:
        g.log["upload_error"] = str(e)
        return _error(str(e), e.status)

    emb = current_app.extensions["embedder"]
    with sess.lock:
        if name not in sess.docs and len(sess.docs) >= s.max_docs_per_session:
            return _error(f"At most {s.max_docs_per_session} documents per session", 409)
        chunks = chunk_pages(name, pages, s.chunk_size, s.chunk_overlap, emb.span_fn())
        if not chunks:
            return _error("No extractable text found", 422)
        t0 = time.perf_counter()
        sess.index.remove_file(name)
        sess.index.add_chunks(chunks)
        sess.docs[name] = len(chunks)
        g.log.update(file=name, n_chunks=len(chunks), n_pages=len(pages),
                     embed_ms=round((time.perf_counter() - t0) * 1000, 1))
    return jsonify({"file": name, "chunks": len(chunks), "pages": len(pages),
                    "documents": sess.docs}), 201


@bp.get("/api/documents")
def documents():
    return jsonify({"documents": current_app.extensions["sessions"].get(_sid()).docs})


@bp.post("/api/session/reset")
def reset():
    current_app.extensions["sessions"].clear(_sid())
    return jsonify({"status": "cleared"})


def _parse_question():
    body = request.get_json(silent=True) or {}
    q = body.get("message")
    k = body.get("k")
    s = current_app.config["SETTINGS"]
    if not isinstance(q, str) or not q.strip():
        return None, None, _error("'message' is required", 400)
    if len(q) > s.max_question_chars:
        return None, None, _error(f"Message exceeds {s.max_question_chars} characters", 413)
    if k is not None and (not isinstance(k, int) or not 1 <= k <= 20):
        return None, None, _error("'k' must be an integer 1-20", 400)
    return q.strip(), k, None


def _llm_error_response(e: LLMError):
    status = 504 if e.status in (504, 408) else 429 if e.status == 429 else 502
    return _error("The language model is temporarily unavailable. Please retry.", status,
                  upstream_status=e.status)


@bp.post("/api/chat")
def chat():
    if (r := _check_rate()):
        return r
    q, k, err = _parse_question()
    if err:
        return err
    sess = current_app.extensions["sessions"].get(_sid())
    try:
        out = current_app.extensions["pipeline"].answer(sess, q, k)
    except LLMError as e:
        g.log["llm_error"] = str(e)
        return _llm_error_response(e)
    t = out["timings_ms"]
    g.log.update(retrieval_ms=t["retrieval"], llm_ms=t["llm_total"], refused=out["refused"],
                 n_sources=len(out["sources"]), n_cited=len(out["cited"]), stream=False)
    return jsonify(out)


@bp.post("/api/chat/stream")
def chat_stream():
    if (r := _check_rate()):
        return r
    q, k, err = _parse_question()
    if err:
        return err
    sess = current_app.extensions["sessions"].get(_sid())
    pipeline = current_app.extensions["pipeline"]
    t0, sid = g.t0, session["sid"][:8]
    g.skip_log = True  # logged by the generator once the stream really ends

    def gen():
        fields = {"stream": True}
        status = 200
        try:
            for ev in pipeline.stream(sess, q, k):
                if ev["type"] == "done":
                    t = ev["timings_ms"]
                    fields.update(retrieval_ms=t["retrieval"], ttft_ms=t.get("ttft"),
                                  llm_ms=t["llm_total"], refused=ev["refused"],
                                  n_sources=len(ev["sources"]), n_cited=len(ev["cited"]))
                yield f"data: {json.dumps(ev)}\n\n"
        except LLMError as e:
            status = 502
            fields["llm_error"] = str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': 'The language model is unavailable. Please retry.', 'upstream_status': e.status})}\n\n"  # noqa: E501
        except GeneratorExit:
            status = 499  # client went away
            raise
        finally:
            log_event("request", method="POST", path="/api/chat/stream", status=status,
                      latency_ms=round((time.perf_counter() - t0) * 1000, 2), sid=sid, **fields)

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.get("/api/history")
def history():
    return jsonify({"history": current_app.extensions["sessions"].get(_sid()).history})
