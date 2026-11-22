
"""Grounded prompt construction, generation, citation extraction, refusal handling."""
from __future__ import annotations

import re
import time
from collections.abc import Iterator

from .retrieval import Hit
from .sessions import Session

REFUSAL = "I don't know based on the provided documents."
NO_DOCS = "No documents are loaded in this session yet. Upload a PDF or TXT file first."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Answer ONLY using the numbered context passages provided in the user's message.
Rules:
1. Cite every claim with the passage number(s) in plain square brackets, e.g. [1] or [2][3],
   placed at the end of each sentence. Use only ASCII brackets. An answer without citations is invalid.
2. If the passages do not contain the information needed to answer, reply with exactly:
   "{REFUSAL}" and nothing else. Do not guess and do not use outside knowledge.
3. Use earlier conversation turns only to resolve references such as "it" or "that section";
   facts must still come from the passages.
4. Be concise."""

_CITE = re.compile(r"[\[【]\s*(\d+(?:\s*[,，]\s*\d+)*)\s*[\]】]")


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = f", {h.chunk.page_label}" if h.chunk.page_label else ""
        blocks.append(f"[{i}] (file: {h.chunk.file}{loc})\n{h.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(history: list[dict], question: str, hits: list[Hit], max_turns: int = 6) -> list[dict]:
    """System prompt + real prior turns + the new question with retrieved context."""
    recent = history[-2 * max_turns:] if max_turns > 0 else []
    user = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *recent, {"role": "user", "content": user}]


def parse_citations(answer: str, n_hits: int) -> list[int]:
    """Sorted unique 1-based passage numbers cited in the answer that actually exist."""
    nums = {int(n) for grp in _CITE.findall(answer) for n in re.findall(r"\d+", grp)}
    return sorted(n for n in nums if 1 <= n <= n_hits)


def is_refusal(answer: str) -> bool:
    return answer.strip().strip('"').lower().startswith(REFUSAL.lower()[:-1])


def source_dict(i: int, h: Hit) -> dict:
    c = h.chunk
    return {
        "n": i, "file": c.file, "chunk_id": c.chunk_id, "page_start": c.page_start,
        "page_end": c.page_end, "label": f"{c.file}" + (f", {c.page_label}" if c.page_label else "")
        + f", chunk {c.chunk_id}",
        "score": round(h.score, 4), "snippet": c.text[:300],
    }


class RagPipeline:
    def __init__(self, settings, llm):
        self.s, self.llm = settings, llm

    def _retrieve(self, session: Session, question: str, k: int | None):
        t0 = time.perf_counter()
        hits = session.index.search(question, k or self.s.top_k)
        return hits, (time.perf_counter() - t0) * 1000

    def _finish(self, session, question, answer, hits, timings) -> dict:
        refused = is_refusal(answer)
        cited = [] if refused else parse_citations(answer, len(hits))
        session.history += [{"role": "user", "content": question},
                            {"role": "assistant", "content": answer}]
        return {
            "answer": answer, "refused": refused, "cited": cited,
            "sources": [source_dict(i, h) for i, h in enumerate(hits, 1)],
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        }

    def _gate(self, session, hits) -> str | None:
        if len(session.index) == 0:
            return NO_DOCS
        if not hits or hits[0].score < self.s.min_retrieval_score:
            return REFUSAL
        return None

    def answer(self, session: Session, question: str, k: int | None = None) -> dict:
        with session.lock:
            hits, t_ret = self._retrieve(session, question, k)
            early = self._gate(session, hits)
            if early:
                out = self._finish(session, question, early, hits if early == REFUSAL else [],
                                   {"retrieval": t_ret, "llm_total": 0.0})
                out["no_llm_call"] = True
                return out
            t0 = time.perf_counter()
            text = self.llm.chat(build_messages(session.history, question, hits, self.s.history_turns))
            return self._finish(session, question, text.strip(), hits,
                                {"retrieval": t_ret, "llm_total": (time.perf_counter() - t0) * 1000})

    def stream(self, session: Session, question: str, k: int | None = None) -> Iterator[dict]:
        """Yield events: sources -> token* -> done. LLMError propagates to the caller."""
        with session.lock:
            hits, t_ret = self._retrieve(session, question, k)
            early = self._gate(session, hits)
            if early:
                yield {"type": "sources", "sources": [source_dict(i, h) for i, h in enumerate(hits, 1)]
                       if early == REFUSAL else []}
                yield {"type": "token", "text": early}
                out = self._finish(session, question, early, hits if early == REFUSAL else [],
                                   {"retrieval": t_ret, "llm_total": 0.0, "ttft": 0.0})
                out["no_llm_call"] = True
                yield {"type": "done", **out}
                return
            yield {"type": "sources", "sources": [source_dict(i, h) for i, h in enumerate(hits, 1)]}
            t0 = time.perf_counter()
            ttft, parts = None, []
            for delta in self.llm.stream(build_messages(session.history, question, hits,
                                                        self.s.history_turns)):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(delta)
                yield {"type": "token", "text": delta}
            out = self._finish(session, question, "".join(parts).strip(), hits,
                               {"retrieval": t_ret, "ttft": ttft or 0.0,
                                "llm_total": (time.perf_counter() - t0) * 1000})
            yield {"type": "done", **out}
