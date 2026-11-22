"""Draft an eval set from your documents with an LLM, for HUMAN REVIEW.

  python -m eval.make_eval_set --per-doc 12 --unanswerable 12

Writes eval/dataset.draft.jsonl with reviewed=false. Open it, fix/delete bad rows, check each gold
quote against the PDF, write the final file to eval/dataset.jsonl and set "reviewed": true per row.
Answerable drafts whose quote is not a verbatim substring of the source chunk are dropped automatically.
Caveat: LLM-drafted questions tend to echo the chunk's wording, which flatters retrieval. Rewrite some
questions in your own words (paraphrase, multi-hop, numeric) before trusting hit-rate numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from app.chunking import chunk_pages
from app.ingest import validate_and_extract
from app.llm import GroqClient
from eval.metrics import containment

GEN = """From the passage below, write ONE question a reader could answer from it alone, its short
answer, and a verbatim quote (<= 40 words, copied exactly) that contains the answer.
Do not mention "the passage" or "the text" in the question. Return ONLY JSON:
{{"question": "...", "answer": "...", "quote": "..."}}

PASSAGE:
{text}"""

UNANS = """Here is the beginning of a document titled "{name}":
{text}

Write {n} questions that are on the same topic and sound natural for this document, but whose answers
are NOT contained in the document (e.g. details a reader might expect but that are absent, or facts
about related things not covered). Return ONLY a JSON list of strings."""


def _json(raw: str):
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    return json.loads(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="eval/docs")
    ap.add_argument("--out", default="eval/dataset.draft.jsonl")
    ap.add_argument("--per-doc", type=int, default=12)
    ap.add_argument("--unanswerable", type=int, default=12, help="total across all docs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set")
    llm = GroqClient(key, os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                     os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"), max_retries=5)
    rnd = random.Random(a.seed)
    rows, docs, errors = [], [], []
    dropped = 0
    for p in sorted(Path(a.docs).iterdir()):
        if p.suffix.lower() not in {".pdf", ".txt"}:
            continue
        name, pages = validate_and_extract(p.name, p.read_bytes(), 5 * 1024 * 1024, 10_000)
        docs.append((name, pages))
        chunks = [c for c in chunk_pages(name, pages, 300, 0) if c.n_tokens > 120]
        print(f"found {name}: {len(pages)} page(s), {len(chunks)} usable chunk(s)")
        for c in rnd.sample(chunks, min(a.per_doc, len(chunks))):
            try:
                d = _json(llm.chat([{"role": "user", "content": GEN.format(text=c.text)}], 0.2))
            except Exception as e:  # noqa: BLE001 - keep going, but report
                errors.append(repr(e))
                continue
            if not isinstance(d, dict) or containment(d.get("quote", ""), c.text) < 0.95:
                dropped += 1  # missing JSON or quote not verbatim -> drop
                continue
            rows.append({"id": f"q{len(rows) + 1:03d}", "question": d["question"], "answerable": True,
                         "answer": d["answer"], "reviewed": False,
                         "gold": [{"file": name, "page": c.page_start, "quote": d["quote"]}]})
    per = max(1, a.unanswerable // max(len(docs), 1))
    for name, pages in docs:
        head = " ".join(t for _, t in pages)[:3000]
        try:
            qs = _json(llm.chat([{"role": "user", "content": UNANS.format(name=name, text=head, n=per)}], 0.5))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
            qs = []
        for q in qs or []:
            rows.append({"id": f"q{len(rows) + 1:03d}", "question": q, "answerable": False,
                         "answer": "NOT ANSWERABLE", "gold": [], "reviewed": False, "source_doc_topic": name})
    if not docs:
        raise SystemExit(f"No .pdf/.txt files found in {a.docs}. Copy your documents there first.")
    if errors:
        print(f"{len(errors)} API/parse error(s); first: {errors[0]}")
    if dropped:
        print(f"{dropped} draft(s) dropped (no valid JSON or quote not verbatim)")
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} draft rows to {a.out}; REVIEW before use (verify unanswerable ones too).")


if __name__ == "__main__":
    main()
