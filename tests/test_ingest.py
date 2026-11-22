
import pytest

from app.ingest import UploadError, safe_name, validate_and_extract
from tests.conftest import make_pdf

MB5 = 5 * 1024 * 1024


def test_txt_ok():
    name, pages = validate_and_extract("notes.txt", b"hello world", MB5)
    assert name == "notes.txt" and pages == [(None, "hello world")]


def test_pdf_pages_extracted():
    name, pages = validate_and_extract("x.pdf", make_pdf(["alpha beta", "gamma delta"]), MB5)
    assert [p for p, _ in pages] == [1, 2]
    assert "alpha" in pages[0][1] and "delta" in pages[1][1]


@pytest.mark.parametrize("fn", ["a.exe", "a.docx", "noext", "a.pdf.exe", ""])
def test_bad_extension(fn):
    with pytest.raises(UploadError) as e:
        validate_and_extract(fn, b"data", MB5)
    assert e.value.status == 415


def test_size_limit_boundary():
    validate_and_extract("a.txt", b"a" * MB5, MB5)  # exactly at limit: ok
    with pytest.raises(UploadError) as e:
        validate_and_extract("a.txt", b"a" * (MB5 + 1), MB5)
    assert e.value.status == 413


def test_empty_file():
    with pytest.raises(UploadError):
        validate_and_extract("a.txt", b"", MB5)


def test_pdf_extension_but_not_pdf():
    with pytest.raises(UploadError) as e:
        validate_and_extract("a.pdf", b"MZ\x90\x00 not a pdf", MB5)
    assert e.value.status == 415


def test_corrupt_pdf():
    with pytest.raises(UploadError) as e:
        validate_and_extract("a.pdf", b"%PDF-1.4\ngarbage", MB5)
    assert e.value.status == 422


def test_binary_txt_rejected():
    with pytest.raises(UploadError):
        validate_and_extract("a.txt", b"abc\x00\x01\x02" * 10, MB5)


def test_pdf_without_text_rejected():
    with pytest.raises(UploadError) as e:
        validate_and_extract("a.pdf", make_pdf([""]), MB5)
    assert e.value.status == 422


def test_safe_name_strips_paths():
    assert safe_name("../../etc/passwd.txt") == "passwd.txt"
    assert safe_name("C:\\x\\y z.pdf") == "y z.pdf"
    assert "/" not in safe_name("a/b/c.txt")
