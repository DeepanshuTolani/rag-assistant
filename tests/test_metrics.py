import json
from argparse import Namespace

import pytest

from app.embeddings import HashingEmbedder
from eval import metrics as m
from eval.common import update_results
from eval.judge import parse_verdict
from eval.run_eval import run

QUOTE = "The warranty period for the router is twenty-four months from the date of purchase."


def test_containment_verbatim_and_noise_tolerant():
    chunk = "Intro text. The warranty period for the router is twenty-four months from the date of " \
            "purchase. More text follows."
    assert m.containment(QUOTE, chunk) == 1.0
    assert m.containment(QUOTE, "warranty period for the rou-\nter is twenty-four MONTHS from the date "
                         "of purchase") >= 0.8
    assert m.containment(QUOTE, "totally unrelated battery charging instructions here") == 0.0


def test_chunk_supports_checks_file():
    gold = [{"file": "a.pdf", "quote": QUOTE}]
    assert m.chunk_supports("a.pdf", QUOTE, gold)
    assert not m.chunk_supports("b.pdf", QUOTE, gold)


def test_rank_hit_mrr():
    gold = [{"file": "a", "quote": QUOTE}]
    hits = [("a", "nothing relevant at all in here today"), ("a", QUOTE), ("a", QUOTE)]
    r = m.first_hit_rank(hits, gold)
    assert r == 2
    assert m.hit_at_k(r, 1) == 0 and m.hit_at_k(r, 2) == 1
    assert m.rr_at_k(r, 3) == 0.5 and m.rr_at_k(r, 1) == 0
    assert m.first_hit_rank(hits[:1], gold) is None and m.rr_at_k(None, 5) == 0


def test_wilson_known_value():
    lo, hi = m.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert m.wilson(0, 0) is None


def test_f1_and_contains():
    assert m.token_f1("24 months", "24 months") == 1.0
    assert m.token_f1("", "x") == 0.0
    assert m.contains_answer("The term is 24 months [1].", "24 months")
    assert not m.contains_answer("about two years", "24 months")


def test_percentile():
    assert m.percentile([1, 2, 3, 4], 50) == 2.5
    assert m.percentile([10], 95) == 10 and m.percentile([], 50) is None
    assert m.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_parse_verdict():
    assert parse_verdict('ok {"verdict": "Correct", "refused": false}')["verdict"] == "correct"
    assert parse_verdict("garbage")["verdict"] == "unparsed"


def test_update_results_preserves_other_sections(tmp_path):
    p = tmp_path / "r.json"
    update_results("a", {"x": 1}, p)
    update_results("b", {"y": 2}, p)
    d = json.loads(p.read_text())
    assert d["a"]["x"] == 1 and d["b"]["y"] == 2 and d["a"]["status"] == "ok"


class EvalFake:
    """Fake generation+judge model, used ONLY to check harness plumbing (numbers are meaningless)."""

    def chat(self, messages, temperature=0.0):
        text = messages[-1]["content"]
        if "You are grading" in text:
            refused = "CANDIDATE: I don't know" in text
            return json.dumps({"verdict": "correct" if refused else "partial", "refused": refused})
        return "I don't know based on the provided documents." if "zebra" in text else "Answer [1]."


def test_harness_end_to_end_with_fakes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(QUOTE + " " + "Filler sentence number one about nothing. " * 30)
    (docs / "b.txt").write_text("Battery charging takes about three hours using the supplied cable. " * 5)
    ds = tmp_path / "ds.jsonl"
    rows = [
        {"id": "q1", "question": "How long is the router warranty?", "answerable": True, "answer": "24 months",
         "gold": [{"file": "a.txt", "quote": QUOTE}], "reviewed": True},
        {"id": "q2", "question": "Who is the zebra CEO?", "answerable": False, "answer": "NOT ANSWERABLE",
         "gold": [], "reviewed": True},
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows))
    args = Namespace(docs=str(docs), dataset=str(ds), chunk_sizes=[20, 40], overlap_frac=0.1, ks=[1, 3],
                     gen_ks=[3], min_scores=[0.0], embedding_model="hashing", retrieval_only=False,
                     allow_unreviewed=False, sleep=0)
    out_path = tmp_path / "results.json"
    payload = run(args, llm=EvalFake(), judge_llm=EvalFake(), embedder=HashingEmbedder(), results_path=out_path)
    cfg = payload["configs"]["chunk_20"]
    assert cfg["retrieval"]["k3"]["n"] == 1 and 0 <= cfg["retrieval"]["k3"]["hit_rate"] <= 1
    gen = cfg["generation"]["k3_min_score0.0"]
    assert gen["n_unanswerable"] == 1 and gen["refusal_rate_on_unanswerable"] == 1.0
    assert payload["no_rag_baseline"]["n_answerable"] == 1
    from scripts.resume_numbers import build

    block = build(json.loads(out_path.read_text()))
    assert "eval.configs.chunk_20.retrieval.k3.hit_rate" in block and "no-RAG baseline" in block
    assert (tmp_path / "eval_details.jsonl").exists() and json.loads(out_path.read_text())["eval"]["status"] == "ok"


def test_unreviewed_dataset_is_rejected(tmp_path):
    from eval.run_eval import load_dataset

    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "q", "answerable": False, "answer": "NOT ANSWERABLE",
                             "gold": [], "reviewed": False}))
    with pytest.raises(SystemExit):
        load_dataset(p, allow_unreviewed=False)
    assert load_dataset(p, allow_unreviewed=True)
