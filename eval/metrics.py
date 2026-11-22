"""Pure metric functions (no model calls). Unit-tested in tests/test_metrics.py."""
from __future__ import annotations

import math
import re
from collections import Counter

_PUNCT = re.compile(r"[^a-z0-9%.\s]")


def normalize(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text.lower())  # undo end-of-line hyphenation from PDFs
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # keep decimal points only
    return re.sub(r"\s+", " ", text).strip()


def _shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def containment(quote: str, chunk_text: str, n: int = 4) -> float:
    """Fraction of the quote's word 4-grams that occur in the chunk (1.0 = verbatim, whitespace/
    punctuation/hyphenation-insensitive)."""
    q = _shingles(normalize(quote).split(), n)
    if not q:
        return 0.0
    c = _shingles(normalize(chunk_text).split(), n)
    return len(q & c) / len(q)


def chunk_supports(chunk_file: str, chunk_text: str, gold: list[dict], thr: float = 0.8) -> bool:
    """A chunk is evidence if it contains >= thr of any gold quote (and is from the gold file)."""
    for g in gold:
        if g.get("file") and g["file"] != chunk_file:
            continue
        if containment(g["quote"], chunk_text) >= thr:
            return True
    return False


def first_hit_rank(hits: list[tuple[str, str]], gold: list[dict], thr: float = 0.8) -> int | None:
    """1-based rank of the first supporting chunk in ``hits`` [(file, text), ...], else None."""
    for i, (f, t) in enumerate(hits, 1):
        if chunk_supports(f, t, gold, thr):
            return i
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def rr_at_k(rank: int | None, k: int) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def wilson(successes: float, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion (honest uncertainty for n ~ 50)."""
    if n == 0:
        return None
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def contains_answer(pred: str, gold: str) -> bool:
    g = normalize(gold)
    return bool(g) and g in normalize(pred)


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0,100]."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
