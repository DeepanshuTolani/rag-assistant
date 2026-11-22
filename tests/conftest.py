import os

import pytest

os.environ.setdefault("EMBEDDING_MODEL", "hashing")

from app import create_app  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.llm import LLMError  # noqa: E402


def make_pdf(pages: list[str]) -> bytes:
    """Minimal valid multi-page PDF with real extractable text (no external deps)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


class FakeLLM:
    """Scripted LLM. Records the messages it was called with."""

    def __init__(self, reply="The warranty is 24 months [1].", error: LLMError | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def chat(self, messages, temperature=0.0):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply

    def stream(self, messages, temperature=0.0):
        self.calls.append(messages)
        if self.error:
            raise self.error
        for w in self.reply.split(" "):
            yield w + " "


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def app(fake_llm, tmp_path):
    s = Settings(embedding_model="hashing", llm_backend="mock", log_dir=str(tmp_path),
                 secret_key="test", rate_limit_per_min=1000, chunk_size=40, chunk_overlap=5,
                 max_upload_bytes=5 * 1024 * 1024)
    return create_app(s, llm=fake_llm, embedder=HashingEmbedder())


@pytest.fixture
def client(app):
    return app.test_client()
