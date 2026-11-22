from app.chunking import chunk_pages
from app.embeddings import HashingEmbedder
from app.retrieval import Index
from app.vectorstore import NumpyStore

DOC = [(1, "The warranty period for the router is twenty four months from purchase."),
       (2, "Battery charging takes about three hours using the supplied USB cable."),
       (3, "To reset the device hold the power button for ten seconds.")]


def build(size=12, overlap=2):
    idx = Index(HashingEmbedder())
    idx.add_chunks(chunk_pages("manual.pdf", DOC, size, overlap))
    return idx


def test_top1_is_relevant_chunk_and_ranks_ordered():
    hits = build().search("how long does battery charging take", k=3)
    assert "charging" in hits[0].chunk.text
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_k_larger_than_corpus():
    idx = build(size=500)
    assert len(idx.search("warranty", k=10)) == len(idx)


def test_empty_index():
    assert Index(HashingEmbedder()).search("anything", 3) == []


def test_remove_file_rebuilds_index():
    idx = build()
    idx.add_chunks(chunk_pages("other.txt", [(None, "completely different zebra content here")], 12, 2))
    idx.remove_file("manual.pdf")
    assert {c.file for c in idx.chunks} == {"other.txt"}
    assert idx.search("zebra", 1)[0].chunk.file == "other.txt"


def test_sessions_do_not_share_documents():
    a, b = build(), Index(HashingEmbedder())
    assert b.search("warranty", 3) == [] and len(a) > 0


def test_numpy_store_exact_cosine():
    import numpy as np

    st = NumpyStore(3)
    st.add(np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]], dtype="float32"))
    res = st.search(np.array([1, 0, 0], dtype="float32"), 2)
    assert [i for i, _ in res] == [0, 2] and abs(res[0][1] - 1.0) < 1e-6
