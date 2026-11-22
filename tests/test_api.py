import io
import json

from app.llm import LLMError
from app.rag import NO_DOCS, REFUSAL
from tests.conftest import make_pdf

TXT = b"The warranty period for the router is twenty four months from purchase date. " * 3


def up(client, name="manual.txt", data=TXT):
    return client.post("/api/upload", data={"file": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_upload_and_documents(client):
    r = up(client)
    assert r.status_code == 201 and r.json["chunks"] >= 1
    assert "manual.txt" in client.get("/api/documents").json["documents"]


def test_upload_rejects_type_and_size(client, app):
    assert up(client, "x.exe", b"abc").status_code == 415
    big = b"a" * (5 * 1024 * 1024 + 1)
    r = up(client, "big.txt", big)
    assert r.status_code == 413 and "error" in r.json


def test_upload_pdf_citation_has_page(client):
    up(client, "m.pdf", make_pdf(["intro text here", "warranty is twenty four months"]))
    out = client.post("/api/chat", json={"message": "warranty months?"}).json
    assert any(s["file"] == "m.pdf" and s["page_start"] in (1, 2) for s in out["sources"])


def test_chat_returns_cited_sources(client, fake_llm):
    up(client)
    out = client.post("/api/chat", json={"message": "How long is the warranty?"}).json
    assert out["cited"] == [1] and not out["refused"]
    assert out["sources"][0]["file"] == "manual.txt"
    assert "chunk" in out["sources"][0]["label"]


def test_no_documents_does_not_call_llm(client, fake_llm):
    out = client.post("/api/chat", json={"message": "hi"}).json
    assert out["answer"] == NO_DOCS and fake_llm.calls == []


def test_model_refusal_is_detected(app, client, fake_llm):
    fake_llm.reply = REFUSAL
    up(client)
    out = client.post("/api/chat", json={"message": "what is the CEO's name?"}).json
    assert out["refused"] is True and out["cited"] == []


def test_retrieval_gate_refuses_without_llm_call(app, client, fake_llm):
    app.config["SETTINGS"].min_retrieval_score = 0.99
    up(client)
    out = client.post("/api/chat", json={"message": "completely unrelated zebra"}).json
    assert out["refused"] and out["no_llm_call"] and fake_llm.calls == []


def test_history_is_really_sent_to_model(client, fake_llm):
    up(client)
    client.post("/api/chat", json={"message": "How long is the warranty?"})
    client.post("/api/chat", json={"message": "And what does it cover?"})
    msgs = fake_llm.calls[-1]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert "How long is the warranty?" in msgs[1]["content"]
    assert msgs[2]["content"] == "The warranty is 24 months [1]."
    # stored history holds raw turns only (no bulky context)
    assert "Context passages" not in client.get("/api/history").json["history"][0]["content"]


def test_sessions_are_isolated(app):
    c1, c2 = app.test_client(), app.test_client()
    up(c1)
    assert c1.get("/api/documents").json["documents"]
    assert c2.get("/api/documents").json["documents"] == {}
    assert c2.post("/api/chat", json={"message": "warranty?"}).json["answer"] == NO_DOCS
    c1.post("/api/chat", json={"message": "warranty?"})
    assert c2.get("/api/history").json["history"][-1]["content"] == NO_DOCS
    assert len(c1.get("/api/history").json["history"]) == 2


def test_reset_clears_session(client):
    up(client)
    client.post("/api/session/reset")
    assert client.get("/api/documents").json["documents"] == {}


def test_validation_errors(client):
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 413
    assert client.post("/api/chat", json={"message": "x", "k": 99}).status_code == 400
    assert client.post("/api/upload", data={}).status_code == 400


def test_llm_errors_map_to_http_status(app, client, fake_llm):
    up(client)
    for upstream, expected in [(429, 429), (503, 502), (504, 504), (None, 502)]:
        fake_llm.error = LLMError("boom", status=upstream, retryable=True)
        r = client.post("/api/chat", json={"message": "warranty?"})
        assert r.status_code == expected and "error" in r.json and "boom" not in r.json["error"]
    # failed turns must not pollute history
    assert client.get("/api/history").json["history"] == []


def test_rate_limit_returns_429_with_retry_after(app, client):
    from app.ratelimit import RateLimiter

    app.extensions["limiter"] = RateLimiter(2)
    up(client)  # 1
    assert client.post("/api/chat", json={"message": "warranty?"}).status_code == 200  # 2
    r = client.post("/api/chat", json={"message": "warranty?"})
    assert r.status_code == 429 and "Retry-After" in r.headers


def sse(resp):
    return [json.loads(x[5:]) for x in resp.get_data(as_text=True).split("\n\n") if x.startswith("data:")]


def test_stream_event_order_and_content(client, fake_llm):
    up(client)
    evs = sse(client.post("/api/chat/stream", json={"message": "How long is the warranty?"}))
    types = [e["type"] for e in evs]
    assert types[0] == "sources" and types[-1] == "done" and "token" in types
    assert "".join(e["text"] for e in evs if e["type"] == "token").strip() == fake_llm.reply
    assert evs[-1]["cited"] == [1] and evs[-1]["timings_ms"]["ttft"] >= 0


def test_stream_upstream_failure_emits_error_event(client, fake_llm):
    up(client)
    fake_llm.error = LLMError("x", status=503)
    evs = sse(client.post("/api/chat/stream", json={"message": "warranty?"}))
    assert evs[-1]["type"] == "error"


def test_request_latency_is_logged(client, tmp_path):
    up(client)
    client.post("/api/chat", json={"message": "warranty?"})
    client.post("/api/chat/stream", json={"message": "warranty?"}).get_data()
    lines = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    reqs = [x for x in lines if x["msg"] == "request"]
    assert {x["path"] for x in reqs} >= {"/api/upload", "/api/chat", "/api/chat/stream"}
    assert all(isinstance(x["latency_ms"], float) for x in reqs)
    assert any("ttft_ms" in x for x in reqs if x["path"] == "/api/chat/stream")


def test_healthz_has_no_secret(client):
    body = client.get("/healthz").get_data(as_text=True)
    assert "gsk_" not in body and "api_key" not in body.lower()
