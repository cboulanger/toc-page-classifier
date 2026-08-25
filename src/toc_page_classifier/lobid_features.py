"""Extracting diversity-relevant features from a raw lobid-resources record,
plus the basic "is this a usable DNB TOC-scan candidate" predicate.

Deliberately broad: any `"Book"`-typed record with a table of contents
counts -- monographs and theses, not just edited volumes -- because for
OA-matching purposes a monograph's TOC is just as usable a ground-truth
candidate as an edited volume's, and excluding them would also exclude most
of the language/domain diversity a single template concentrates in.
"""

from typing import Optional, TypedDict
from urllib.parse import urlsplit

# Same ISO 639-2 -> 639-1 mapping as fetch_corpus.py, kept in sync by hand
# (small, stable list; not worth a cross-repo dependency for).
_ISO_639_2_TO_1 = {
    "ger": "de", "eng": "en", "fre": "fr", "fra": "fr", "spa": "es",
    "ita": "it", "dut": "nl", "nld": "nl", "lat": "la", "rus": "ru",
}


class RecordFeatures(TypedDict):
    isbn: str
    title: str
    language: Optional[str]
    volume_type: str
    domain_bucket: str
    toc_download_url: str
    lobid_url: Optional[str]


def record_isbn(record: dict) -> Optional[str]:
    isbns = record.get("isbn") or []
    return isbns[0].replace("-", "").upper() if isbns else None


def record_toc_url(record: dict) -> Optional[str]:
    for entry in record.get("tableOfContents") or []:
        url = entry.get("id")
        if url and urlsplit(url).scheme in ("http", "https"):
            return url
    return None


def record_matches(record: dict) -> bool:
    """A usable candidate: typed as a Book (any kind) and carrying at least
    one http(s) tableOfContents URL, with a resolvable ISBN to match against
    OA sources by."""
    types = record.get("type") or []
    if "Book" not in types:
        return False
    if record_isbn(record) is None:
        return False
    return record_toc_url(record) is not None


def record_language(record: dict) -> Optional[str]:
    languages = record.get("language") or []
    if not languages:
        return None
    code = (languages[0].get("id") or "").rsplit("/", 1)[-1]
    if not code:
        return None
    return _ISO_639_2_TO_1.get(code, code)


def record_volume_type(record: dict) -> str:
    types = record.get("type") or []
    if "EditedVolume" in types:
        return "edited_volume"
    if "Thesis" in types:
        return "thesis"
    return "monograph"


def record_domain_bucket(record: dict) -> str:
    """A coarse discipline bucket from the record's RVK (Regensburger
    Verbundklassifikation) notation, when present -- the leading letter(s)
    of an RVK notation (e.g. "CC 3200" -> "CC") identify its top-level
    class. Falls back to "unknown" when no RVK subject is present (common;
    not every record is RVK-classified)."""
    for subject in record.get("subject") or []:
        source_label = (subject.get("source") or {}).get("label") or ""
        notation = subject.get("notation")
        if "RVK" in source_label and notation:
            letters = "".join(ch for ch in notation.split(" ")[0] if ch.isalpha())
            if letters:
                return letters
    return "unknown"


def record_api_url(record: dict) -> Optional[str]:
    record_id = (record.get("id") or "").rstrip("#!")
    return f"{record_id}?format=json" if record_id else None


def extract_features(record: dict) -> RecordFeatures:
    isbn = record_isbn(record)
    toc_url = record_toc_url(record)
    assert isbn is not None and toc_url is not None, "call record_matches() first"
    return {
        "isbn": isbn,
        "title": record.get("title") or "",
        "language": record_language(record),
        "volume_type": record_volume_type(record),
        "domain_bucket": record_domain_bucket(record),
        "toc_download_url": toc_url,
        "lobid_url": record_api_url(record),
    }
