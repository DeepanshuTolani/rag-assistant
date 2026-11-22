"""Print (or inject into README.md) the ONLY figures that may be quoted, straight from results.json.

  python -m scripts.resume_numbers                 # print markdown
  python -m scripts.resume_numbers --write-readme  # replace block between RESUME markers in README.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- RESUME:START -->", "<!-- RESUME:END -->"


def _fmt(v):
    return "n/a" if v is None else (f"{v:.1%}" if isinstance(v, float) and v <= 1 else str(v))


def build(results: dict) -> str:
    lines = []
    ev = results.get("eval", {})
    if ev.get("status") == "ok":
        cmd = ev["command"]
        tag = "" if ev.get("dataset_reviewed_by_human") else " (dataset NOT human-reviewed: do not quote)"
        lines.append(f"### Evaluation{tag}\n")
        lines.append(f"{ev['n_questions']} questions ({ev['n_answerable']} answerable / {ev['n_unanswerable']} "
                     f"unanswerable) over {len(ev['documents'])} documents; embedding model `{ev['embedding_model']}`; "
                     f"generation `{ev.get('generation_model')}`; judge `{ev.get('judge_model')}`.\n")
        lines.append("| Figure | Value | results.json path | Reproduce |\n|---|---|---|---|")
        for cname, c in ev["configs"].items():
            for kname, r in c["retrieval"].items():
                for metric in ("hit_rate", "mrr"):
                    lines.append(f"| {cname} {kname} {metric} | {_fmt(r[metric]) if metric == 'hit_rate' else r[metric]} "
                                 f"| eval.configs.{cname}.retrieval.{kname}.{metric} | `{cmd}` |")
            for gname, g in c["generation"].items():
                for metric, val in (("correctness_judge_strict", g["correctness_judge_strict"]),
                                    ("citation_precision", g["citation"]["citation_precision"]),
                                    ("refusal_rate_on_unanswerable", g["refusal_rate_on_unanswerable"])):
                    path = f"eval.configs.{cname}.generation.{gname}." + (
                        "citation." + metric if metric == "citation_precision" else metric)
                    lines.append(f"| {cname} {gname} {metric} | {_fmt(val)} | {path} | `{cmd}` |")
        b = ev.get("no_rag_baseline")
        if b:
            for metric in ("correctness_judge_strict", "refusal_rate_on_unanswerable"):
                lines.append(f"| no-RAG baseline {metric} | {_fmt(b[metric])} | eval.no_rag_baseline.{metric} | `{cmd}` |")
        lines.append("")
    ld = results.get("load", {})
    if ld.get("status") == "ok" and not ld.get("llm_backend_is_mock"):
        cmd = ld["command"]
        hw = ld.get("server_hardware_stated_by_operator") or "NOT STATED"
        lines.append("### Load test\n")
        lines.append(f"Model `{ld['server_reported_config']['groq_model']}`; server hardware: {hw}; "
                     f"load generator: {ld['load_generator_environment']['cpu_model']} "
                     f"({ld['load_generator_environment']['logical_cpus']} vCPU).\n")
        lines.append("| Mode | Users | p50 ms | p95 ms | TTFT p50 ms | TTFT p95 ms | Error rate | Req/s | Reproduce |\n"
                     "|---|---|---|---|---|---|---|---|---|")
        for mode, rows in ld["results"].items():
            for r in rows:
                lines.append(f"| {mode} | {r['users']} | {r['latency_ms_p50']} | {r['latency_ms_p95']} | "
                             f"{r['ttft_ms_p50']} | {r['ttft_ms_p95']} | {r['error_rate']} | {r['rps']} | `{cmd}` |")
        lines.append("")
    ls = results.get("log_summary", {})
    if ls.get("status") == "ok":
        lines.append("### From deployment logs\n")
        for p, e in ls["endpoints"].items():
            lines.append(f"- `{p}`: {e['requests']} requests, p50 {e['latency_ms_p50']} ms, p95 {e['latency_ms_p95']} ms "
                         f"(`{ls['command']}`)")
        lines.append("")
    if not lines:
        missing = [k for k in ("eval", "load", "log_summary") if results.get(k, {}).get("status") != "ok"]
        return ("**No measured figures exist yet.** Sections not yet run: " + ", ".join(missing) + ".\n"
                "Nothing below may be quoted until `results/results.json` contains it. Generate with:\n\n"
                "```bash\n" + "\n".join(
                    c for k in ("eval", "load") for c in results.get(k, {}).get("reproduce", [])) + "\n```\n")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-readme", action="store_true")
    a = ap.parse_args()
    block = build(json.loads((ROOT / "results" / "results.json").read_text()))
    if a.write_readme:
        p = ROOT / "README.md"
        txt = p.read_text()
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", txt, flags=re.S)
        p.write_text(new)
    else:
        print(block)
