"""Derive latency stats from the app's own structured logs (logs/requests.jsonl).

  python -m bench.summarize_logs logs/requests.jsonl [--write-results]

Numbers describe whatever traffic actually hit THIS deployment's log file. They only support claims
about usage if the log came from a deployment you really ran.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from eval.common import update_results
from eval.metrics import percentile


def summarize(path: str) -> dict:
    by = defaultdict(list)
    first = last = None
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("msg") != "request" or "latency_ms" not in d:
            continue
        by[d["path"]].append(d)
        first, last = first or d["ts"], d["ts"]
    out = {"log_file": path, "first_ts": first, "last_ts": last, "endpoints": {}}
    for p, rows in sorted(by.items()):
        lat = [r["latency_ms"] for r in rows if r["status"] < 400]
        ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None and r["status"] < 400]
        out["endpoints"][p] = {
            "requests": len(rows), "errors_status_ge_400": sum(r["status"] >= 400 for r in rows),
            "latency_ms_p50": percentile(lat, 50), "latency_ms_p95": percentile(lat, 95),
            **({"ttft_ms_p50": percentile(ttft, 50), "ttft_ms_p95": percentile(ttft, 95)} if ttft else {}),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--write-results", action="store_true")
    a = ap.parse_args()
    res = summarize(a.log)
    print(json.dumps(res, indent=2))
    if a.write_results:
        update_results("log_summary", {"command": f"python -m bench.summarize_logs {a.log} --write-results", **res})
