"""Access to the sibling `dnb-toc-ground-truth` repo's pilot corpus manifest
-- the list of books DNB has digitized a CC0 "Kataloganreicherung" table-of-
contents scan for. Same sibling-checkout-plus-env-var-override pattern as
`chapter-segmentation/evaluation/harness.py`'s `corpus_dir()`, so both repos
can point at the same corpus without hardcoding a path."""

import json
import os
from pathlib import Path
from typing import TypedDict

_ENV_VAR = "DNB_TOC_CORPUS_DIR"
_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dnb-toc-ground-truth" / "data" / "corpus" / "pilot"


class DnbBook(TypedDict):
    filename: str
    title: str
    language: str | None
    doi: str | None
    toc_download_url: str
    license: str
    license_source: str
    lobid_url: str


def corpus_dir() -> Path:
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else _DEFAULT_DIR


def load_dnb_books_by_isbn() -> dict[str, DnbBook]:
    """Returns every DNB pilot-corpus book keyed by normalized ISBN (the
    manifest's `filename` field minus `.pdf`, upper-cased, no hyphens)."""
    manifest_path = corpus_dir() / "manifest.json"
    books = json.loads(manifest_path.read_text())["books"]
    result: dict[str, DnbBook] = {}
    for book in books:
        isbn = book["filename"].removesuffix(".pdf").replace("-", "").upper()
        result[isbn] = book
    return result
