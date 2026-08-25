"""Text/structural features for TOC-page detection: a standalone
reimplementation of chapter_segmentation's TOC-line structural pattern
(src/chapter_segmentation/segmentation.py's _TOC_LINE_RE and friends -- not
imported, to keep this repo self-contained) plus, once
keyword_hit_* is wired up in a later task, multilingual keyword matching against
data/toc_keywords.json. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Text /
structural" section."""

import re
from itertools import pairwise

# Matches "<title> <dots-or-spaces> <page number>". Requires at least 2
# separator characters so ordinary prose sentences ending in a number don't
# false-positive. Page may be arabic or a lowercase/uppercase roman
# numeral (front-matter pagination).
_TOC_LINE_RE = re.compile(
    r"^(?P<title>.{3,120}?)[.\s]{2,}(?P<page>\d{1,4}|[ivxlcdm]{1,6}|[IVXLCDM]{1,6})\s*$"
)
_STRICT_ROMAN_RE = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_PAGE_MAX_VALUE = 50  # real front-matter roman pagination never runs longer
_URL_OR_DOI_RE = re.compile(r"doi\.org|https?://|www\.", re.IGNORECASE)
_IMPRINT_LINE_RE = re.compile(
    r"^\s*[©@]\s*\d{4}\b|^\s*isbn\b|^\s*gedruckt\b|^\s*printed in\b", re.IGNORECASE
)
# Same separator class _TOC_LINE_RE requires between title and page number.
# Used to identify lines that at least *look* like a TOC entry (so a lone
# heading word like "Contents", with no dot-leader/whitespace run at all,
# doesn't get counted against toc_line_ratio's denominator just for being a
# non-blank line).
_SEPARATOR_RUN_RE = re.compile(r"[.\s]{2,}")

TEXT_FEATURE_NAMES = [
    "toc_line_count",
    "toc_line_ratio",
    "digit_density",
    "monotonic_page_numbers",
    "keyword_hit_any_language",
    "keyword_hit_same_language",
]


def _parse_page_number(raw: str) -> int | None:
    """The integer value of a TOC line's captured page field, or None if
    it's not a plausible page number (invalid/implausibly large roman
    numeral)."""
    if raw.isdigit():
        return int(raw)
    lowered = raw.lower()
    if not lowered or not _STRICT_ROMAN_RE.match(lowered):
        return None
    total = 0
    for ch, nxt in zip(lowered, lowered[1:] + " "):
        value = _ROMAN_VALUES[ch]
        total += -value if nxt != " " and _ROMAN_VALUES.get(nxt, 0) > value else value
    return total if total <= _ROMAN_PAGE_MAX_VALUE else None


def _excluded_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or bool(_URL_OR_DOI_RE.search(line)) or bool(_IMPRINT_LINE_RE.search(line))


def _match_toc_line(line: str) -> re.Match | None:
    if _excluded_line(line):
        return None
    return _TOC_LINE_RE.match(line.strip())


def _is_toc_line_candidate(line: str) -> bool:
    """Whether a line even has the shape a TOC line could take (a
    dot-leader/whitespace separator run), regardless of whether the page
    number it ends in turned out to be valid. Used as toc_line_ratio's
    denominator so a plain heading line like "Contents" -- which has no
    such run at all -- isn't counted against the ratio just for being
    non-blank."""
    if _excluded_line(line):
        return False
    return bool(_SEPARATOR_RUN_RE.search(line.strip()))


def extract_text_features(
    pages: list[str],
    language: str | None = None,
    keywords_path: str | None = None,
) -> dict[int, dict[str, float]]:
    """Structural + keyword text features, one dict per page index.
    `language` is the book's own declared language code (e.g. "en"), used
    only for keyword_hit_same_language. `keywords_path` is wired up in
    a later task -- until then, keyword_hit_* are always 0.0."""
    features: dict[int, dict[str, float]] = {}
    for page_index, text in enumerate(pages):
        lines = [line for line in text.splitlines() if line.strip()]
        matches = [m for line in lines if (m := _match_toc_line(line)) is not None]
        candidate_lines = [line for line in lines if _is_toc_line_candidate(line)]
        page_numbers = [
            n for m in matches if (n := _parse_page_number(m["page"])) is not None
        ]
        monotonic = float(
            len(page_numbers) >= 2 and all(b > a for a, b in pairwise(page_numbers))
        )
        digit_density = sum(c.isdigit() for c in text) / len(text) if text else 0.0

        features[page_index] = {
            "toc_line_count": float(len(matches)),
            "toc_line_ratio": len(matches) / len(candidate_lines) if candidate_lines else 0.0,
            "digit_density": digit_density,
            "monotonic_page_numbers": monotonic,
            "keyword_hit_any_language": 0.0,
            "keyword_hit_same_language": 0.0,
        }
    return features
