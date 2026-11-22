"""Closed-loop load test against a RUNNING server. Dependency-light (requests + threads).

  # terminal 1 (real deployment config):  GROQ_API_KEY=... gunicorn -c gunicorn.conf.py wsgi:app
  # terminal 2:
  python -m bench.load_test --url http://127.0.0.1:8000 --doc eval/docs/yourdoc.pdf \
      --questions eval/dataset.jsonl --users 1 5 20 --duration 60 --server-hardware "describe the server"

Each simulated user has its own session (cookie), uploads the document once (excluded from latency),
then asks questions back-to-back (no think time) for --duration seconds. Modes:
  stream : POST /api/chat/stream, records time-to-first-token (first SSE 'token' event) and total time
  chat   : POST /api/chat, records total latency
Error = transport failure, HTTP >= 400 (incl. 429), or an SSE 'error' event. 429s are also counted
separately. Server rate limit is read from /healthz and recorded; set RATE_LIMIT_PER_MIN high enough
(or 0) if you want to measure the model path rather than the limiter.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

import requests

from eval.common import collect_env, update_results
from eval.metrics import percentile


def load_questions(path: str | None) -> list[str]:
    if not path:
        return ["Summarize the main topic of the document."]
    qs = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            qs.append(r["question"] if isinstance(r, dict) else str(r))
    return qs


def one_request(sess, base, mode, q, timeout):
    t0 = time.perf_counter()
    try:
        if mode == "chat":
            r = sess.post(f"{base}/api/chat", json={"message": q}, timeout=timeout)
            return {"ok": r.status_code == 200, "status": r.status_code, "total_ms": (time.perf_counter() - t0) * 1000,
                    "ttft_ms": None}
        r = sess.post(f"{base}/api/chat/stream", json={"message": q}, timeout=timeout, stream=True)
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "total_ms": (time.perf_counter() - t0) * 1000, "ttft_ms": None}
        ttft, ok = None, True
        for line in r.iter_lines(decode_unicode=True):
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev["type"] == "token" and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            elif ev["type"] == "error":
                ok = False
        return {"ok": ok, "status": 200, "total_ms": (time.perf_counter() - t0) * 1000, "ttft_ms": ttft}
    except requests.RequestException:
        return {"ok": False, "status": None, "total_ms": (time.perf_counter() - t0) * 1000, "ttft_ms": None}


def user_loop(base, doc, questions, mode, stop_at, timeout, out, setup_errors, seed):
    rnd = random.Random(seed)
    s = requests.Session()
    try:
        with open(doc, "rb") as f:
            r = s.post(f"{base}/api/upload", files={"file": (Path(doc).name, f)}, timeout=120)
        if r.status_code != 201:
            setup_errors.append(r.status_code)
            return
    except requests.RequestException:
        setup_errors.append("transport")
        return
    while time.perf_counter() < stop_at:
        out.append(one_request(s, base, mode, rnd.choice(questions), timeout))


def run_level(base, doc, questions, mode, users, duration, timeout) -> dict:
    out, setup_errors, threads = [], [], []
    t_start = time.perf_counter()
    stop_at = t_start + duration
    for i in range(users):
        t = threading.Thread(target=user_loop, args=(base, doc, questions, mode, stop_at, timeout, out,
                                                     setup_errors, i), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(duration + timeout + 130)
    wall = time.perf_counter() - t_start
    lat_ok = [x["total_ms"] for x in out if x["ok"]]
    ttft = [x["ttft_ms"] for x in out if x["ok"] and x["ttft_ms"] is not None]
    n = len(out)
    r = lambda v: None if v is None else round(v, 1)  # noqa: E731
    return {
        "users": users, "duration_s_requested": duration, "wall_s": round(wall, 2), "mode": mode,
        "requests": n, "errors": sum(not x["ok"] for x in out),
        "error_rate": round(sum(not x["ok"] for x in out) / n, 4) if n else None,
        "rate_limited_429": sum(x["status"] == 429 for x in out),
        "setup_failures": len(setup_errors),
        "rps": round(n / wall, 3) if wall else None,
        "latency_ms_p50": r(percentile(lat_ok, 50)), "latency_ms_p95": r(percentile(lat_ok, 95)),
        "latency_ms_p99": r(percentile(lat_ok, 99)),
        "ttft_ms_p50": r(percentile(ttft, 50)), "ttft_ms_p95": r(percentile(ttft, 95)),
        "latency_percentiles_over": "successful requests only",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--doc", required=True, help="PDF/TXT each simulated user uploads")
    ap.add_argument("--questions", help="JSONL with a 'question' field (e.g. eval/dataset.jsonl)")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--modes", nargs="+", default=["stream", "chat"], choices=["stream", "chat"])
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--server-hardware", default="", help="describe the machine running the server")
    ap.add_argument("--server-command", default="gunicorn -c gunicorn.conf.py wsgi:app")
    ap.add_argument("--out-section", default="load")
    a = ap.parse_args(argv)
    base = a.url.rstrip("/")
    health = requests.get(f"{base}/healthz", timeout=10).json()
    if health["config"]["llm_backend"] == "mock":
        print("WARNING: server uses LLM_BACKEND=mock; these numbers do NOT reflect Groq.")
    qs = load_questions(a.questions)
    results = {m: [] for m in a.modes}
    for mode in a.modes:
        for u in a.users:
            print(f"== {mode}, {u} user(s), {a.duration}s ...", flush=True)
            res = run_level(base, a.doc, qs, mode, u, a.duration, a.timeout)
            print(json.dumps(res), flush=True)
            results[mode].append(res)
    update_results(a.out_section, {
        "command": "python -m bench.load_test " + " ".join(argv if argv is not None else __import__("sys").argv[1:]),
        "load_generator_environment": collect_env(),
        "server_hardware_stated_by_operator": a.server_hardware or None,
        "server_command_stated_by_operator": a.server_command,
        "server_reported_config": health["config"],
        "llm_backend_is_mock": health["config"]["llm_backend"] == "mock",
        "n_distinct_questions": len(qs), "think_time_s": 0,
        "results": results,
    })


if __name__ == "__main__":
    main()
