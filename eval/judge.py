"""LLM-as-judge. The exact prompt below is what produced any reported 'correctness' number and is
copied into results.json by run_eval.py."""
from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are grading an answer to a question about a document set.
You are given the QUESTION, the GROUND TRUTH answer, and the CANDIDATE answer.
Ignore citation markers like [1]. Judge only substance.

Return ONLY a JSON object: {{"verdict": "correct"|"partial"|"incorrect", "refused": true|false}}
- "refused" is true if the candidate declines to answer / says it does not know or that the
  information is not available, instead of giving an answer.
- If GROUND TRUTH is "NOT ANSWERABLE", the question cannot be answered from the documents:
  verdict is "correct" only if the candidate refuses; any substantive answer is "incorrect".
- Otherwise: "correct" = candidate states the same facts as GROUND TRUTH (extra correct detail is
  fine; wording may differ; numbers/names must match). "partial" = some but not all of the required
  facts. "incorrect" = wrong, contradictory, or a refusal.

QUESTION: {question}
GROUND TRUTH: {gold}
CANDIDATE: {answer}
JSON:"""


def parse_verdict(raw: str) -> dict:
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).lower()
            if v in {"correct", "partial", "incorrect"}:
                return {"verdict": v, "refused": bool(d.get("refused", False))}
        except ValueError:
            pass
    return {"verdict": "unparsed", "refused": False}


def judge(llm, question: str, gold: str, answer: str) -> dict:
    raw = llm.chat([{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, gold=gold, answer=answer)}], temperature=0.0)
    return parse_verdict(raw)
