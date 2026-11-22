"""Groq chat client (OpenAI-compatible REST) with timeouts, retries/backoff and SSE streaming."""
from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator

import requests

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class GroqClient:
    def __init__(self, api_key: str, model: str, base_url: str, timeout_connect: float = 5,
                 timeout_read: float = 60, max_retries: int = 3, session: requests.Session | None = None,
                 sleep=time.sleep):
        self.api_key, self.model = api_key, model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = (timeout_connect, timeout_read)
        self.max_retries = max_retries
        self.http = session or requests.Session()
        self._sleep = sleep

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 20.0)
            except ValueError:
                pass
        return min(0.5 * 2**attempt, 8.0) * (0.5 + random.random() / 2)

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not configured", status=500)
        last: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                r = self.http.post(self.url, headers=self._headers(), json=payload,
                                   timeout=self.timeout, stream=stream)
            except (requests.Timeout, requests.ConnectionError) as e:
                last = LLMError(f"LLM request failed: {type(e).__name__}", status=504, retryable=True)
            else:
                if r.status_code < 400:
                    return r
                retry_after = r.headers.get("Retry-After")
                retryable = r.status_code in RETRY_STATUS
                last = LLMError(f"LLM returned HTTP {r.status_code}", status=r.status_code,
                                retryable=retryable)
                r.close()
                if not retryable:
                    raise last
            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, retry_after))
        assert last is not None
        raise last

    def _payload(self, messages: list[dict], stream: bool, temperature: float) -> dict:
        return {"model": self.model, "messages": messages, "temperature": temperature, "stream": stream}

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        r = self._post(self._payload(messages, False, temperature), stream=False)
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError("Malformed LLM response", status=502) from e

    def stream(self, messages: list[dict], temperature: float = 0.1) -> Iterator[str]:
        """Yield text deltas. Retries only apply before the first byte is received."""
        r = self._post(self._payload(messages, True, temperature), stream=True)
        r.encoding = "utf-8"  # SSE has no charset header; requests would guess latin-1 and garble non-ASCII
        try:
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    yield delta
        except (requests.Timeout, requests.ConnectionError) as e:
            raise LLMError("LLM stream interrupted", status=504, retryable=True) from e
        finally:
            r.close()


class MockLLM:
    """Offline stand-in (LLM_BACKEND=mock) for UI work and wiring tests. Never use for metrics."""

    model = "mock"

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        return "".join(self.stream(messages))

    def stream(self, messages: list[dict], temperature: float = 0.0) -> Iterator[str]:
        ctx = messages[-1]["content"]
        yield "[mock llm] "
        yield "I don't know based on the provided documents." if "[1]" not in ctx else "See [1]."
