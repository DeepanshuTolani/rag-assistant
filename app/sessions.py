"""Session-scoped state: chat history + a private document index per session id.

In-memory and per-process. Run gunicorn with ONE worker and many threads (see gunicorn.conf.py),
or put this behind sticky routing; see README "Limitations".
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .retrieval import Index


@dataclass
class Session:
    sid: str
    index: Index
    history: list[dict] = field(default_factory=list)  # [{"role","content"}], raw Q/A only
    docs: dict[str, int] = field(default_factory=dict)  # filename -> n_chunks
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self, index_factory, ttl_s: int = 3600, max_sessions: int = 200, clock=time.monotonic):
        self._factory, self._ttl, self._max, self._clock = index_factory, ttl_s, max_sessions, clock
        self._s: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, sid: str) -> Session:
        now = self._clock()
        with self._lock:
            self._evict(now)
            s = self._s.get(sid)
            if s is None:
                s = Session(sid=sid, index=self._factory())
                self._s[sid] = s
                while len(self._s) > self._max:
                    self._s.popitem(last=False)
            s.last_used = now
            self._s.move_to_end(sid)
            return s

    def clear(self, sid: str) -> None:
        with self._lock:
            self._s.pop(sid, None)

    def _evict(self, now: float) -> None:
        dead = [k for k, v in self._s.items() if now - v.last_used > self._ttl]
        for k in dead:
            del self._s[k]

    def __len__(self) -> int:
        return len(self._s)
