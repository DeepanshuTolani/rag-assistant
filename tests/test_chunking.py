import pytest

from app.chunking import chunk_pages


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunk_sizes_and_overlap():
    chunks = chunk_pages("a.txt", [(None, words(100))], chunk_size=30, overlap=10)
    assert [c.n_tokens for c in chunks] == [30, 30, 30, 30, 20]
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]  # overlap is real
    assert [c.chunk_id for c in chunks] == [0, 1, 2, 3, 4]


def test_every_word_is_covered():
    text = words(95)
    chunks = chunk_pages("a.txt", [(None, text)], 25, 5)
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(text.split())


def test_short_doc_single_chunk():
    chunks = chunk_pages("a.txt", [(None, "hello world")], 500, 50)
    assert len(chunks) == 1 and chunks[0].text == "hello world"


def test_empty_and_blank_pages():
    assert chunk_pages("a.pdf", [(1, "  "), (2, "")], 10, 2) == []


def test_page_tracking_across_boundary():
    pages = [(1, words(20, "a")), (2, words(20, "b")), (3, words(20, "c"))]
    chunks = chunk_pages("d.pdf", pages, chunk_size=15, overlap=0)
    assert chunks[0].page_start == chunks[0].page_end == 1
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning and spanning[0].page_label == "pp. 1-2"
    assert chunks[-1].page_end == 3


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages("a", [(None, "x y z")], size, overlap)
