import json

import pytest
import requests

from app.llm import GroqClient, LLMError


class Resp:
    def __init__(self, status=200, body=None, lines=None, headers=None):
        self.status_code, self._body, self._lines = status, body, lines or []
        self.headers = headers or {}
        self.closed = False

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def close(self):
        self.closed = True


class Http:
    def __init__(self, seq):
        self.seq, self.n = list(seq), 0

    def post(self, *a, **kw):
        self.n += 1
        item = self.seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(seq, retries=3, key="k"):
    sleeps = []
    c = GroqClient(key, "m", "http://x/v1", max_retries=retries, session=Http(seq), sleep=sleeps.append)
    return c, sleeps


OK = Resp(200, {"choices": [{"message": {"content": "hi"}}]})


def test_success():
    c, _ = client([OK])
    assert c.chat([{"role": "user", "content": "q"}]) == "hi"


def test_retries_on_429_then_succeeds_and_honours_retry_after():
    c, sleeps = client([Resp(429, headers={"Retry-After": "2"}), Resp(503), OK])
    assert c.chat([]) == "hi" and c.http.n == 3 and sleeps[0] == 2.0


def test_retries_on_timeout_then_gives_up():
    c, sleeps = client([requests.Timeout()] * 4, retries=3)
    with pytest.raises(LLMError) as e:
        c.chat([])
    assert e.value.status == 504 and c.http.n == 4 and len(sleeps) == 3


def test_no_retry_on_4xx():
    c, _ = client([Resp(401)])
    with pytest.raises(LLMError) as e:
        c.chat([])
    assert e.value.status == 401 and c.http.n == 1


def test_missing_key():
    c, _ = client([OK], key="")
    with pytest.raises(LLMError):
        c.chat([])


def test_malformed_body():
    c, _ = client([Resp(200, {"oops": 1})])
    with pytest.raises(LLMError):
        c.chat([])


def test_stream_parses_sse_and_ignores_noise():
    lines = [": keepalive", "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
             "data: " + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
             "data: not-json", "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
             "data: [DONE]", "data: " + json.dumps({"choices": [{"delta": {"content": "NO"}}]})]
    r = Resp(200, lines=lines)
    c, _ = client([r])
    assert "".join(c.stream([])) == "Hello" and r.closed
