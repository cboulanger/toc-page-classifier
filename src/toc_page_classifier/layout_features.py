"""Per-page geometric feature extraction from a PDF via pdfplumber -- this
repo's own (pdfalto-free) re-derivation of the layout features
chapter-segmentation's TOC/chapter-first-page classifier pilot proved
useful, trimmed to the subset relevant to TOC-page detection. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's
"Geometry" section."""

import statistics
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


def extract_page_features(pdf_path: str | Path) -> dict[int, dict[str, float]]:
    """Parses a PDF into a per-page feature dict, keyed by 0-based page
    index. A page with no extractable characters gets an all-zero feature
    vector plus the underscore-prefixed intermediates
    add_book_context_features needs."""
    features: dict[int, dict[str, float]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_height = page.height
            page_width = page.width
            lines = _group_chars_into_lines(page.chars)

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
    return features


def add_book_context_features(
    page_features: dict[int, dict[str, float]], total_pages: int
) -> dict[int, dict[str, float]]:
    """Second pass over one book's extract_page_features output: adds
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
