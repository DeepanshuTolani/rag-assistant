# Document-grounded RAG assistant (Flask + Groq)

Upload PDFs/TXT, ask questions, get answers **with source citations** (file + page + chunk).
If the retrieved context doesn't support an answer, the bot says it doesn't know.

> **Status of measurements:** the code for every metric below is implemented and unit-tested, but the
> evaluation and load numbers have **not been run yet** (no evaluation documents and no Groq key were
> available when this was built). `results/results.json` therefore contains `"status": "not_run"` and
> no figures. Run the commands in [Reproduce](#reproduce-the-numbers); the tables below are generated from
> that file, not typed by hand.

## Security first: rotate the leaked key

The original repo committed a Groq key in `.env` (two keys, one in a comment) and one in
`Notebook/Notebook-Custom_chatgpt-open-source.ipynb`. **Revoke all of them at console.groq.com and create a
new one.** Removing the file is not enough: if the repo was ever pushed, the keys are in git history and in
any forks/clones. Revoke first, then optionally purge history (`git filter-repo --path .env --invert-paths`
and force-push). This project reads every secret from environment variables; `.env` is git-ignored and
`.env.example` documents the variables.

## Architecture

```mermaid
flowchart LR
  UI[Browser UI<br/>upload + streaming chat] -->|multipart| UP[/POST /api/upload/]
  UI -->|fetch + SSE| CH[/POST /api/chat/stream/]
  UP --> V[Validate: ext, magic bytes,<br/>5 MB, pages] --> X[Extract text<br/>pypdf / UTF-8] --> C[Chunk<br/>N tokens + overlap<br/>page tracking]
  C --> E[Embed locally<br/>sentence-transformers] --> S[(Per-session index<br/>FAISS / NumPy fallback)]
  CH --> RL[Rate limit<br/>per session+IP] --> R[Retrieve top-k<br/>optional cross-encoder rerank]
  S --> R --> G{score gate<br/>+ docs loaded?}
  G -- no support --> IDK["I don't know…" (no LLM call)]
  G -- ok --> P[Grounded prompt:<br/>system + real session history<br/>+ numbered passages] --> L[Groq chat API<br/>timeouts, retry/backoff, SSE] --> A[Answer + [n] citations]
  SESS[(Session store<br/>history + docs, TTL)] <--> R
  SESS <--> P
  CH -. JSON log: latency, TTFT .-> LOG[(logs/requests.jsonl)]
```

| Module | Role |
|---|---|
| `app/ingest.py` | extension + magic-byte + size validation, PDF/TXT extraction |
| `app/chunking.py` | token-window chunking with overlap; chunks carry page range |
| `app/embeddings.py` | local sentence-transformers (default `BAAI/bge-small-en-v1.5`); `hashing` test stub |
| `app/vectorstore.py`, `app/retrieval.py` | FAISS `IndexFlatIP` (NumPy fallback), top-k, optional rerank |
| `app/sessions.py` | **session-scoped** history + document index (replaces the old global `ChatManager`) |
| `app/rag.py` | grounded prompt, history, citation parsing, refusal handling |
| `app/llm.py` | Groq REST client: connect/read timeouts, retries on 408/429/5xx with `Retry-After`, streaming |
| `app/ratelimit.py` | sliding-window limiter (default 30 req/min per session+IP) |
| `app/logging_utils.py` | JSON-lines logs with per-request `latency_ms`, `ttft_ms`, retrieval/LLM timings |
| `eval/`, `bench/` | evaluation harness, load test, log summariser |

### What was fixed from the original app
- Global `ChatManager` shared by all users → per-session state keyed by a signed HttpOnly cookie.
- History was stored but **never sent** to the model → last `HISTORY_TURNS` turns are sent (raw Q/A only, so
  retrieved context doesn't accumulate).
- `innerHTML` with user text (XSS) → `textContent` everywhere in the UI.
- Flask debug server → gunicorn. Debug mode, `filemode="w"` log files and per-instance log files → one JSON logger.
- Secrets → environment variables only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # fill in GROQ_API_KEY (a NEW key) and SECRET_KEY
gunicorn -c gunicorn.conf.py wsgi:app      # http://localhost:8000
# or: docker compose up --build
pytest -q && ruff check .
```
`LLM_BACKEND=mock EMBEDDING_MODEL=hashing` runs the whole UI offline with canned answers (never use for metrics).

### API
| Endpoint | |
|---|---|
| `POST /api/upload` (multipart `file`) | 201 `{file, chunks, pages}`; 413 >5 MB, 415 wrong type, 422 unreadable/no text |
| `POST /api/chat` `{message, k?}` | `{answer, refused, cited, sources[], timings_ms}` |
| `POST /api/chat/stream` | SSE: `sources` → `token`* → `done` (or `error`) |
| `GET /api/documents`, `GET /api/history`, `POST /api/session/reset`, `GET /healthz` | |

## Evaluation methodology

1. **Corpus:** 3–5 real documents you place in `eval/docs/`; all are indexed together (hardest, realistic setting).
2. **Eval set:** 40–60 questions in `eval/dataset.jsonl`, each with a ground-truth answer and a verbatim
   **gold quote** (+file/page). Include ≥ 8–10 *unanswerable* questions. `python -m eval.make_eval_set` drafts
   candidates with an LLM and auto-drops quotes that aren't verbatim, but rows are `reviewed: false` and the harness
   refuses them until a human verifies them. LLM-drafted questions echo the source wording, which flatters
   retrieval; paraphrase some by hand.
3. **Retrieval:** a retrieved chunk is a *hit* if it contains ≥ 80 % of the gold quote's word 4-grams (and is from the
   gold file). Using quote containment rather than chunk IDs makes configurations with different chunk sizes
   comparable. Reported: hit-rate@k and MRR@k over answerable questions, with Wilson 95 % CIs. A quote split across a
   chunk boundary can count as a miss; this biases against small chunks.
4. **Answer correctness:** LLM judge (`eval/judge.py`; the exact prompt, model and temperature 0 are written into
   `results.json`), strict (correct only) and lenient (partial = ½). Non-LLM cross-checks: normalised
   gold-answer containment and token-F1. Set `EVAL_JUDGE_MODEL` to a *different* model from `GROQ_MODEL`; if they
   match, the run records a self-judging warning.
5. **Citation accuracy:** for non-refused answers, *citation precision* = cited chunks that actually contain gold
   evidence / all cited chunks; also the share of answers with ≥ 1 supported citation.
6. **Refusals:** refusal rate on unanswerable questions (higher is better) and false-refusal rate on answerable ones;
   optional retrieval-score gates via `--min-scores`.
7. **No-RAG baseline:** same model and questions, no documents, told to say "I don't know" if unsure; judged identically.
8. Single-turn (no chat history) in eval; generation temperature 0.1, so reruns can differ slightly.

### Reproduce the numbers
```bash
cp <your docs> eval/docs/
python -m eval.make_eval_set --per-doc 12 --unanswerable 12     # then review -> eval/dataset.jsonl
python -m eval.run_eval --chunk-sizes 500 1000 --ks 1 3 5 8 --gen-ks 5 --min-scores 0 0.3 --sleep 0.5
GROQ_API_KEY=... gunicorn -c gunicorn.conf.py wsgi:app &        # RATE_LIMIT_PER_MIN=0 to bench the model path
python -m bench.load_test --url http://127.0.0.1:8000 --doc eval/docs/<doc> --questions eval/dataset.jsonl \
    --users 1 5 20 --duration 60 --server-hardware "<describe machine>"
python -m scripts.resume_numbers --write-readme                  # refreshes the block below
```

### Important caveat on chunk sizes
`all-MiniLM-L6-v2` truncates input at 256 tokens and `bge-small-en-v1.5` at 512. A 1000-token chunk is
therefore **silently truncated for embedding** (the tail can't be retrieved by meaning). `run_eval` records
`truncation.fraction_over_limit` per chunk size so you can see this; if it is large, either pick an embedder with a
longer context (e.g. a 8k-context model) or compare 200/400 instead of 500/1000. Chunk sizes are measured with the
embedder's own sub-word tokenizer when available (whitespace words otherwise).

## Results

<!-- RESUME:START -->
**No measured figures exist yet.** Sections not yet run: eval, load, log_summary.
Nothing below may be quoted until `results/results.json` contains it. Generate with:

```bash
cp your 3-5 PDFs/TXTs into eval/docs/
python -m eval.make_eval_set   # drafts questions; review by hand -> eval/dataset.jsonl (reviewed: true)
python -m eval.run_eval --chunk-sizes 500 1000 --ks 1 3 5 8 --gen-ks 5
GROQ_API_KEY=... gunicorn -c gunicorn.conf.py wsgi:app
python -m bench.load_test --url http://127.0.0.1:8000 --doc eval/docs/<doc> --questions eval/dataset.jsonl --users 1 5 20 --duration 60 --server-hardware '<describe machine>'
```

<!-- RESUME:END -->

Result tables to fill from `results/results.json` once run: RAG vs no-RAG; chunk-size × k comparison; load at 1/5/20
users (p50/p95, error rate, req/s, TTFT, hardware, Groq model).

## RESUME NUMBERS

Rule: only figures present in `results/results.json` may be quoted, each with its reproducing command. The block
between the markers above is generated by `python -m scripts.resume_numbers --write-readme`; if it says "No measured
figures exist yet", **there is nothing to put on a resume yet.** Do not claim real users, request volumes or uptime:
those can only come from logs of a deployment you actually ran
(`python -m bench.summarize_logs logs/requests.jsonl --write-results`).

Things you *can* truthfully say today (they're facts about the code, not metrics): pytest suite with 67 tests incl.
upload validation, chunking, retrieval, API error mapping and the eval metrics; CI runs ruff + pytest; Docker +
gunicorn deployment; per-session isolation; SSE streaming. Re-count tests yourself with `pytest --collect-only -q`.

## Limitations
- **In-memory state:** sessions/indexes live in process memory (TTL 1 h, max 200 sessions) and are lost on restart.
  gunicorn runs **1 worker × 16 threads** because state isn't shared across processes; scale out only with sticky
  routing or externalised state (Redis + persisted FAISS).
- Rate limiter is per process, keyed by session cookie + IP (a client can dodge it by dropping cookies; use a
  reverse proxy limiter in production).
- Scanned/image PDFs are rejected (no OCR); tables and multi-column PDFs extract imperfectly with `pypdf`.
- Refusal relies on the prompt plus an optional score gate (`MIN_RETRIEVAL_SCORE`, default off and uncalibrated);
  the model can still hallucinate a cited answer, which is what citation precision measures.
- Citations are model-generated `[n]` markers mapped to retrieved chunks; the UI shows all retrieved sources and
  highlights the cited ones.
- Eval n≈50 → wide confidence intervals; differences of a few points between configurations are not significant.
- Groq model availability changes; set `GROQ_MODEL` to a model your account lists. Retries cover 408/429/5xx and
  timeouts before the first byte; a stream that dies mid-answer surfaces an `error` event rather than retrying.
