"""Evaluation: retrieval quality, answer correctness, citation accuracy, refusals, no-RAG baseline.

Usage (needs your documents in eval/docs/ and a reviewed eval/dataset.jsonl):
  python -m eval.run_eval --chunk-sizes 500 1000 --ks 1 3 5 8 --gen-ks 5
  python -m eval.run_eval --retrieval-only        # no API key needed (still needs the embedding model)

Dataset row (JSONL):
  {"id": "q01", "question": "...", "answerable": true, "answer": "...",
   "gold": [{"file": "doc.pdf", "page": 3, "quote": "verbatim sentence(s) <= 40 words"}], "reviewed": true}
  Unanswerable rows: "answerable": false, "answer": "NOT ANSWERABLE", "gold": [].
Every number written to results/results.json comes from this script's computation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from app.chunking import chunk_pages
from app.embeddings import get_embedder
from app.ingest import validate_and_extract
from app.llm import GroqClient
from app.rag import REFUSAL, build_messages, is_refusal, parse_citations
from app.retrieval import Index
from eval.common import RESULTS, collect_env, update_results
from eval.judge import JUDGE_PROMPT, judge
from eval.metrics import (
    chunk_supports,
    contains_answer,
    first_hit_rank,
    hit_at_k,
    mean,
    rr_at_k,
    token_f1,
    wilson,
)

BASELINE_SYSTEM = ("Answer the question using only your own general knowledge. You have not been given "
                   "any documents. If you do not know or cannot be sure, reply exactly: I don't know.")


def load_dataset(path: Path, allow_unreviewed: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate question ids in dataset")
    for r in rows:
        if r["answerable"] and not r.get("gold"):
            raise SystemExit(f"{r['id']}: answerable question needs gold quotes")
        if not allow_unreviewed and not r.get("reviewed", False):
            raise SystemExit(f"{r['id']}: not marked reviewed=true (human-verify the draft first, "
                             "or pass --allow-unreviewed and say so when reporting)")
    return rows


def load_corpus(docs_dir: Path, max_bytes: int = 5 * 1024 * 1024) -> dict[str, list]:
    corpus = {}
    for p in sorted(docs_dir.iterdir()):
        if p.suffix.lower() in {".pdf", ".txt"}:
            name, pages = validate_and_extract(p.name, p.read_bytes(), max_bytes, 10_000)
            corpus[name] = pages
    if not corpus:
        raise SystemExit(f"no .pdf/.txt documents found in {docs_dir}")
    return corpus


def truncation_stats(embedder, chunks) -> dict | None:
    """Share of chunks longer than the embedder's max sequence length (silently truncated)."""
    model = getattr(embedder, "model", None)
    if model is None or not embedder.max_seq_length:
        return None
    tok = model.tokenizer
    lens = [len(tok(c.text, add_special_tokens=True)["input_ids"]) for c in chunks]
    over = sum(1 for n in lens if n > embedder.max_seq_length)
    return {"max_seq_length": embedder.max_seq_length, "chunks_over_limit": over,
            "fraction_over_limit": round(over / len(lens), 4) if lens else None}


def build_index(embedder, corpus, size, overlap) -> tuple[Index, list]:
    idx, all_chunks = Index(embedder), []
    for name, pages in corpus.items():
        ch = chunk_pages(name, pages, size, overlap, embedder.span_fn())
        all_chunks += ch
        idx.add_chunks(ch)
    return idx, all_chunks


def retrieval_metrics(idx, rows, ks):
    depth, ans = max(ks), [r for r in rows if r["answerable"]]
    per_q = {}
    for r in ans:
        hits = idx.search(r["question"], depth)
        per_q[r["id"]] = first_hit_rank([(h.chunk.file, h.chunk.text) for h in hits], r["gold"])
    out = {}
    for k in ks:
        hits_k = [hit_at_k(per_q[r["id"]], k) for r in ans]
        out[f"k{k}"] = {"hit_rate": round(mean(hits_k), 4), "hit_rate_ci95": wilson(sum(hits_k), len(ans)),
                        "mrr": round(mean([rr_at_k(per_q[r["id"]], k) for r in ans]), 4), "n": len(ans)}
    return out, per_q


def generation_metrics(idx, rows, k, min_score, llm, judge_llm, sleep, detail_sink, tag):
    recs = []
    for r in rows:
        hits = idx.search(r["question"], k)
        if hits and hits[0].score >= min_score:
            ans, called = llm.chat(build_messages([], r["question"], hits)).strip(), True
        else:
            ans, called = REFUSAL, False
        sleep()
        j = judge(judge_llm, r["question"], r["answer"] if r["answerable"] else "NOT ANSWERABLE", ans)
        sleep()
        cited = [] if is_refusal(ans) else parse_citations(ans, len(hits))
        sup = [chunk_supports(hits[n - 1].chunk.file, hits[n - 1].chunk.text, r["gold"]) for n in cited] \
            if r["answerable"] else []
        rec = {"id": r["id"], "answerable": r["answerable"], "answer": ans, "llm_called": called,
               "verdict": j["verdict"], "judge_refused": j["refused"], "exact_refusal": is_refusal(ans),
               "n_cited": len(cited), "n_cited_supported": sum(sup),
               "contains_gold": contains_answer(ans, r["answer"]) if r["answerable"] else None,
               "f1": token_f1(ans, r["answer"]) if r["answerable"] else None}
        recs.append(rec)
        detail_sink.write(json.dumps({"config": tag, **rec}) + "\n")
    return summarize(recs)


def summarize(recs) -> dict:
    ans = [x for x in recs if x["answerable"]]
    un = [x for x in recs if not x["answerable"]]
    strict = sum(x["verdict"] == "correct" for x in ans)
    lenient = strict + 0.5 * sum(x["verdict"] == "partial" for x in ans)
    answered = [x for x in ans if not x["judge_refused"]]
    with_cit = [x for x in answered if x["n_cited"] > 0]
    n_cit = sum(x["n_cited"] for x in answered)
    out = {
        "n_answerable": len(ans), "n_unanswerable": len(un),
        "unparsed_judgements": sum(x["verdict"] == "unparsed" for x in recs),
        "correctness_judge_strict": round(strict / len(ans), 4) if ans else None,
        "correctness_judge_strict_ci95": wilson(strict, len(ans)),
        "correctness_judge_lenient": round(lenient / len(ans), 4) if ans else None,
        "contains_gold_answer_rate": round(mean([float(x["contains_gold"]) for x in ans]), 4) if ans else None,
        "token_f1_mean": round(mean([x["f1"] for x in ans]), 4) if ans else None,
        "false_refusal_rate_on_answerable": round(sum(x["judge_refused"] for x in ans) / len(ans), 4) if ans else None,
        "refusal_rate_on_unanswerable": round(sum(x["judge_refused"] for x in un) / len(un), 4) if un else None,
        "refusal_rate_on_unanswerable_ci95": wilson(sum(x["judge_refused"] for x in un), len(un)),
        "refusal_rate_on_unanswerable_exact_match": round(sum(x["exact_refusal"] for x in un) / len(un), 4) if un else None,
        "llm_calls_skipped_by_score_gate": sum(not x["llm_called"] for x in recs),
        "citation": {
            "answered_answerable_n": len(answered),
            "answers_with_any_citation_rate": round(len(with_cit) / len(answered), 4) if answered else None,
            "citation_precision": round(sum(x["n_cited_supported"] for x in answered) / n_cit, 4) if n_cit else None,
            "citations_total": n_cit,
            "answers_with_supported_citation_rate": round(
                sum(x["n_cited_supported"] > 0 for x in answered) / len(answered), 4) if answered else None,
        },
    }
    return out


def run(args, llm=None, judge_llm=None, embedder=None, results_path: Path = RESULTS) -> dict:
    rows = load_dataset(Path(args.dataset), args.allow_unreviewed)
    n_ans = sum(r["answerable"] for r in rows)
    notes = []
    if not 40 <= len(rows) <= 60:
        notes.append(f"dataset has {len(rows)} questions (target 40-60)")
    if len(rows) - n_ans < 8:
        notes.append(f"only {len(rows) - n_ans} unanswerable questions: refusal-rate CI will be wide")
    corpus = load_corpus(Path(args.docs))
    embedder = embedder or get_embedder(args.embedding_model)
    sleep = (lambda: time.sleep(args.sleep)) if args.sleep else (lambda: None)

    gen_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    judge_model = os.getenv("EVAL_JUDGE_MODEL", gen_model)
    if not args.retrieval_only and llm is None:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise SystemExit("GROQ_API_KEY is not set (use --retrieval-only for retrieval metrics)")
        base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        llm = GroqClient(key, gen_model, base, max_retries=5)
        judge_llm = GroqClient(key, judge_model, base, max_retries=5)
    judge_llm = judge_llm or llm
    if not args.retrieval_only and judge_model == gen_model:
        notes.append("judge model == generation model (self-preference bias possible); "
                     "set EVAL_JUDGE_MODEL to a different model for an independent judge")

    configs, detail_path = {}, results_path.parent / "eval_details.jsonl"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_path, "w") as detail:
        for size in args.chunk_sizes:
            overlap = round(size * args.overlap_frac)
            idx, chunks = build_index(embedder, corpus, size, overlap)
            ret, _ = retrieval_metrics(idx, rows, args.ks)
            cfg = {"chunk_size_tokens": size, "overlap_tokens": overlap, "n_chunks": len(chunks),
                   "truncation": truncation_stats(embedder, chunks), "retrieval": ret, "generation": {}}
            if not args.retrieval_only:
                for k in args.gen_ks:
                    for ms in args.min_scores:
                        tag = f"cs{size}_k{k}_min{ms}"
                        cfg["generation"][f"k{k}_min_score{ms}"] = generation_metrics(
                            idx, rows, k, ms, llm, judge_llm, sleep, detail, tag)
            configs[f"chunk_{size}"] = cfg

        baseline = None
        if not args.retrieval_only:
            recs = []
            for r in rows:
                ans = llm.chat([{"role": "system", "content": BASELINE_SYSTEM},
                                {"role": "user", "content": r["question"]}], temperature=0.1).strip()
                sleep()
                j = judge(judge_llm, r["question"], r["answer"] if r["answerable"] else "NOT ANSWERABLE", ans)
                sleep()
                rec = {"id": r["id"], "answerable": r["answerable"], "answer": ans, "llm_called": True,
                       "verdict": j["verdict"], "judge_refused": j["refused"], "exact_refusal": False,
                       "n_cited": 0, "n_cited_supported": 0,
                       "contains_gold": contains_answer(ans, r["answer"]) if r["answerable"] else None,
                       "f1": token_f1(ans, r["answer"]) if r["answerable"] else None}
                recs.append(rec)
                detail.write(json.dumps({"config": "no_rag_baseline", **rec}) + "\n")
            baseline = summarize(recs)
            baseline.pop("citation")

    payload = {
        "command": "python -m eval.run_eval " + " ".join(sys.argv[1:]),
        "environment": collect_env(),
        "embedding_model": embedder.name,
        "vector_store": "faiss" if _has_faiss() else "numpy (exact cosine)",
        "reranker": os.getenv("RERANK_MODEL") or None,
        "tokenizer_for_chunking": "embedding-model subword tokenizer" if getattr(embedder, "model", None)
        else "whitespace words",
        "documents": sorted(corpus),
        "n_questions": len(rows), "n_answerable": n_ans, "n_unanswerable": len(rows) - n_ans,
        "dataset_reviewed_by_human": all(r.get("reviewed") for r in rows),
        "hit_definition": "retrieved chunk contains >=80% of the gold quote's word 4-grams (and is from "
                          "the gold file); MRR/hit-rate computed over answerable questions only",
        "generation_model": None if args.retrieval_only else gen_model,
        "judge_model": None if args.retrieval_only else judge_model,
        "judge_prompt": None if args.retrieval_only else JUDGE_PROMPT,
        "judge_temperature": None if args.retrieval_only else 0.0,
        "history_used_in_eval": False,
        "configs": configs, "no_rag_baseline": baseline, "notes": notes,
        "raw_outputs": str(detail_path.name),
    }
    update_results("eval", payload, results_path)
    return payload


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default="eval/docs")
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000])
    ap.add_argument("--overlap-frac", type=float, default=0.1)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--gen-ks", type=int, nargs="+", default=[5], help="k values to run generation for")
    ap.add_argument("--min-scores", type=float, nargs="+", default=[0.0],
                    help="retrieval-score gates to compare (0 = prompt-only refusal)")
    ap.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--allow-unreviewed", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls (rate limits)")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
