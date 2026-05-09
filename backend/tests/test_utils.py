from __future__ import annotations

from app.utils import detect_file_type, ensure_supported_file_suffix, safe_filename


def test_safe_filename_preserves_unicode_name() -> None:
    assert safe_filename("战报文档.pdf") == "战报文档.pdf"
    assert safe_filename(r"C:\fakepath\战报文档.pdf") == "战报文档.pdf"


def test_ensure_supported_file_suffix_adds_pdf_suffix(tmp_path) -> None:
    path = tmp_path / "pdf"
    path.write_bytes(b"%PDF-1.7\n%demo")

    renamed = ensure_supported_file_suffix(path)

    assert renamed.name == "pdf.pdf"
    assert renamed.exists()
    assert not path.exists()
    assert detect_file_type(renamed) == 0


def test_ensure_supported_file_suffix_rejects_unknown_file(tmp_path) -> None:
    path = tmp_path / "unknown"
    path.write_bytes(b"not a supported file")

    try:
        ensure_supported_file_suffix(path)
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
