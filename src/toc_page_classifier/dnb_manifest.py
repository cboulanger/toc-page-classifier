"""Access to a local DNB TOC-scan manifest -- the list of books DNB has
digitized a CC0 "Kataloganreicherung" table-of-contents scan for, in the
same `{"books": [{"filename", "title", "toc_download_url", ...}]}` shape
as this repo's own `manifest.json`. Only used by `cli/match_dnb_oa.py`
(the offline, non-authoritative cross-check -- see its docstring); the
main discovery path, `cli/discover_oa_dnb_candidates.py`, checks lobid.org
live instead and doesn't need this at all."""

import json
import os
from pathlib import Path
from typing import TypedDict

_ENV_VAR = "DNB_TOC_CORPUS_DIR"
# No bundled default corpus ships with this repo -- point DNB_TOC_CORPUS_DIR
# at any local checkout that has a manifest.json in the expected shape.
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
