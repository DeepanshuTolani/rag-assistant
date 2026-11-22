"""Upload validation and text extraction (PDF/TXT)."""
from __future__ import annotations

import io
import os
import re

ALLOWED_EXT = {".pdf", ".txt"}


class UploadError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def safe_name(filename: str) -> str:
    base = os.path.basename((filename or "").replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(". ")
    return base[:120]


def validate_and_extract(
    filename: str, data: bytes, max_bytes: int, max_pages: int = 300
) -> tuple[str, list[tuple[int | None, str]]]:
    """Return (clean_filename, [(page_number, text), ...]) or raise UploadError."""
    name = safe_name(filename)
    ext = os.path.splitext(name)[1].lower()
    if not name or ext not in ALLOWED_EXT:
        raise UploadError("Only .pdf and .txt files are accepted", 415)
    if len(data) == 0:
        raise UploadError("File is empty", 400)
    if len(data) > max_bytes:
        raise UploadError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit", 413)

    if ext == ".pdf":
        if not data.startswith(b"%PDF-"):  # content must match the extension
            raise UploadError("File is not a valid PDF", 415)
        pages = _extract_pdf(data, max_pages)
    else:
        pages = [(None, _decode_text(data))]

    if not any(t.strip() for _, t in pages):
        raise UploadError(
            "No extractable text found (scanned PDFs need OCR, which is not supported)", 422
        )
    return name, pages


def _decode_text(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise UploadError("File does not look like plain text", 415)
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UploadError("Text file must be UTF-8 encoded", 415)


def _extract_pdf(data: bytes, max_pages: int) -> list[tuple[int | None, str]]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError("Encrypted PDFs are not supported", 422)
        if len(reader.pages) > max_pages:
            raise UploadError(f"PDF has more than {max_pages} pages", 413)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    except UploadError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError, RecursionError) as e:
        raise UploadError("Could not read PDF (corrupt or unsupported)", 422) from e
