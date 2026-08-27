"""Per-page geometric feature extraction from a PDF via pdfplumber -- this
repo's own (pdfalto-free) re-derivation of the layout features
chapter-segmentation's TOC/chapter-first-page classifier pilot proved
useful, trimmed to the subset relevant to TOC-page detection. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's
"Geometry" section."""

import statistics
from itertools import pairwise
from pathlib import Path

import pdfplumber

PAGE_FEATURE_NAMES = [
    "line_count",
    "font_size_max_ratio",
    "line_density",
    "left_margin_mean",
    "left_margin_var",
    "first_text_vpos_fraction",
    "last_text_vpos_fraction",
]

# Computed by add_book_context_features (second pass) -- need neighboring
# pages and book-level aggregates a single-page parse can't see.
CONTEXT_FEATURE_NAMES = [
    "line_count_rel",
    "font_size_max_ratio_book",
    "edge_distance",
]

FEATURE_NAMES = PAGE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES

# Points; chars within this vertical distance of the current line's own
# first char are treated as the same text line. pdfplumber exposes raw
# per-char geometry but no built-in line grouping, unlike ALTO XML's
# explicit TextLine elements.
_LINE_TOLERANCE = 2.0


def _group_chars_into_lines(chars: list[dict]) -> list[list[dict]]:
    """Groups a page's pdfplumber `chars` into text lines by clustering
    consecutive chars (sorted by vertical then horizontal position) within
    _LINE_TOLERANCE of the line's own first char."""
    if not chars:
        return []
    ordered = sorted(chars, key=lambda c: (c["top"], c["x0"]))
    lines: list[list[dict]] = [[ordered[0]]]
    for char in ordered[1:]:
        if abs(char["top"] - lines[-1][0]["top"]) <= _LINE_TOLERANCE:
            lines[-1].append(char)
        else:
            lines.append([char])
    return lines


# Gap-to-font-size ratios used by _reconstruct_line_text to classify the
# horizontal space between two consecutive chars on the same line. Tuned
# to distinguish three cases: kerning within a word (no space), an
# ordinary space between words (one space), and the wide title/page-number
# gap in a TOC entry -- normally rendered via absolute glyph positioning on
# native PDFs, with no literal dot-leader characters pypdf's extract_text()
# could preserve, which is why that gap collapses to a single space there
# (see the comment above _TOC_LINE_RE in text_features.py for the full
# diagnosis this fixes).
_WORD_GAP_RATIO = 0.3
_WIDE_GAP_RATIO = 3.0


def _reconstruct_line_text(line: list[dict]) -> str:
    """Joins one line's pdfplumber chars (each needs `text`, `x0`, `x1`,
    `size`) into a string, left to right. A gap below _WORD_GAP_RATIO of
    the local font size is treated as kerning (no space inserted); a gap
    at or above _WIDE_GAP_RATIO is treated as a TOC-style dot-leader/tab-stop
    gap and gets a 2-char separator (satisfying _TOC_LINE_RE's requirement)
    instead of pypdf's single-space normalization; everything in between
    gets one ordinary space."""
    ordered = sorted(line, key=lambda c: c["x0"])
    parts = [ordered[0]["text"]]
    for prev, cur in pairwise(ordered):
        gap = cur["x0"] - prev["x1"]
        font_size = (prev["size"] + cur["size"]) / 2 or 1.0
        ratio = gap / font_size
        if ratio >= _WIDE_GAP_RATIO:
            parts.append("  ")
        elif ratio >= _WORD_GAP_RATIO:
            parts.append(" ")
        parts.append(cur["text"])
    return "".join(parts)


def extract_page_features_and_texts(
    pdf_path: str | Path,
    head_pages: int | None = None,
    tail_pages: int | None = None,
) -> tuple[dict[int, dict[str, float]], dict[int, str], int]:
    """One pdfplumber pass producing both this book's per-page layout
    feature dict (keyed by 0-based page index, matching PAGE_FEATURE_NAMES
    plus add_book_context_features' underscore-prefixed intermediates),
    its gap-aware reconstructed per-page text (via _reconstruct_line_text --
    use this instead of pdf_text.page_texts() (pypdf) as
    extract_text_features's input, since it preserves the wide
    title/page-number gap in native-PDF TOC lines that pypdf's extraction
    collapses to a single space), and the book's total page count.

    Merged into one pass (2026-08-27) because pdfplumber's own per-page
    parsing -- not model fitting -- is the dominant per-book cost (measured
    at ~1 minute/book on the full 184-book corpus): two separate functions
    each calling pdfplumber.open() independently was parsing every PDF
    twice for no reason.

    By default (head_pages and tail_pages both None) every page is parsed --
    training/evaluation needs honest features for every page, since the
    true TOC location isn't known in advance. Pass head_pages/tail_pages to
    parse only the first head_pages and/or last tail_pages pages instead --
    a real TOC is essentially never buried deep in a long book's interior,
    and that interior is exactly what makes a long book slow to parse (each
    page requires a full pdfminer layout pass). If the two windows overlap
    or together cover the whole book, every page is parsed anyway rather
    than skipping the (nonexistent) gap between them. The returned dicts
    only have entries for the pages actually parsed."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        wanted_pages: set[int] | None = None
        if head_pages is not None or tail_pages is not None:
            head = range(min(head_pages or 0, total_pages))
            tail = range(max(total_pages - (tail_pages or 0), 0), total_pages)
            wanted_pages = set(head) | set(tail)

        features: dict[int, dict[str, float]] = {}
        texts: dict[int, str] = {}
        for page_index, page in enumerate(pdf.pages):
            if wanted_pages is not None and page_index not in wanted_pages:
                continue
            page_height = page.height
            page_width = page.width
            lines = _group_chars_into_lines(page.chars)

            texts[page_index] = "\n".join(_reconstruct_line_text(line) for line in lines)

            if not lines:
                features[page_index] = {
                    **{name: 0.0 for name in PAGE_FEATURE_NAMES},
                    "_max_font_size": 0.0,
                    "_modal_font_size": 0.0,
                }
                continue

            line_sizes = [statistics.mean(c["size"] for c in line) for line in lines]
            line_tops = [min(c["top"] for c in line) for line in lines]
            line_bottoms = [max(c["bottom"] for c in line) for line in lines]
            line_lefts = [min(c["x0"] for c in line) / page_width for line in lines]

            modal_size = statistics.mode(round(s, 1) for s in line_sizes)
            max_size = max(line_sizes)

            features[page_index] = {
                "line_count": float(len(lines)),
                "font_size_max_ratio": max_size / modal_size if modal_size else 1.0,
                "line_density": len(lines) / page_height,
                "left_margin_mean": statistics.mean(line_lefts),
                "left_margin_var": statistics.variance(line_lefts) if len(line_lefts) > 1 else 0.0,
                "first_text_vpos_fraction": min(line_tops) / page_height,
                "last_text_vpos_fraction": max(line_bottoms) / page_height,
                "_max_font_size": max_size,
                "_modal_font_size": modal_size,
            }
    return features, texts, total_pages


def add_book_context_features(
    page_features: dict[int, dict[str, float]], total_pages: int
) -> dict[int, dict[str, float]]:
    """Second pass over one book's extract_page_features_and_texts output: adds
    CONTEXT_FEATURE_NAMES and strips the underscore-prefixed intermediates,
    returning vectors keyed exactly by FEATURE_NAMES.

    Book aggregates are computed over non-empty pages only (a blank page
    would otherwise drag them toward zero). font_size_max_ratio_book uses
    the median of per-page MODAL sizes as the book's body-font estimate
    (stable against a handful of pages with unusual layout), matching
    chapter-segmentation's own add_book_context_features."""
    non_empty = [f for f in page_features.values() if f["line_count"] > 0]
    median_line_count = (
        statistics.median(f["line_count"] for f in non_empty) if non_empty else 1.0
    ) or 1.0
    modal_sizes = [f["_modal_font_size"] for f in non_empty if f["_modal_font_size"] > 0]
    body_font_size = statistics.median(modal_sizes) if modal_sizes else 0.0

    result: dict[int, dict[str, float]] = {}
    for page_index, page in page_features.items():
        out = {name: page[name] for name in PAGE_FEATURE_NAMES}
        out["line_count_rel"] = page["line_count"] / median_line_count
        max_font = page["_max_font_size"]
        out["font_size_max_ratio_book"] = (
            max_font / body_font_size if body_font_size > 0 and max_font > 0 else 1.0
        )
        out["edge_distance"] = float(min(page_index, total_pages - 1 - page_index))
        result[page_index] = out
    return result
