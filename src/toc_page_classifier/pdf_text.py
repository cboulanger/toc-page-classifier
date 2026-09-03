"""Thin wrapper over pypdf for per-page text extraction, shared by every
module that needs to read a PDF's pages rather than treat it as an opaque
file."""

from pathlib import Path

from pypdf import PdfReader


def page_count(pdf_path: str | Path) -> int:
    """The document's declared page count, without reading any page's
    content -- `layout_features` needs it up front to work out which pages
    a head/tail scan should ask pdfalto for."""
    return len(PdfReader(str(pdf_path)).pages)


def page_texts(pdf_path: str | Path, max_pages: int | None = None) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    return [(page.extract_text() or "") for page in pages]


def has_text(pdf_path: str | Path) -> bool:
    """False for a pure-image scan with no OCR/text layer at all -- the case
    `locate_toc.py` cannot handle and that needs an OCR or vision-based
    fallback instead (not implemented yet; see README's "Known gaps")."""
    return any(text.strip() for text in page_texts(pdf_path))
