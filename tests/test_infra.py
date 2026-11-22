from app.rag import REFUSAL, is_refusal, parse_citations
from app.ratelimit import RateLimiter
from app.sessions import SessionStore


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_rate_limiter_window():
    clk = Clock()
    rl = RateLimiter(2, 60, clock=clk)
    assert rl.check("a")[0] and rl.check("a")[0]
    ok, retry = rl.check("a")
    assert not ok and 0 < retry <= 60
    assert rl.check("b")[0]  # other keys unaffected
    clk.t = 61
    assert rl.check("a")[0]


def test_session_ttl_and_cap():
    clk = Clock()
    st = SessionStore(lambda: object(), ttl_s=10, max_sessions=2, clock=clk)
    a = st.get("a")
    assert st.get("a") is a
    clk.t = 11
    assert st.get("a") is not a  # expired -> fresh
    st.get("b"), st.get("c")
    assert len(st) == 2


def test_parse_citations_ignores_out_of_range():
    assert parse_citations("x [2] y [1][2] z [9] [0]", 3) == [1, 2]


def test_refusal_detection():
    assert is_refusal(REFUSAL) and is_refusal('"' + REFUSAL + '"') and not is_refusal("It is 5 [1].")
