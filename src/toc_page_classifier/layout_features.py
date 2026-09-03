"""Per-page geometric feature extraction from a PDF via pdfalto -- this
repo's own re-derivation of the layout features chapter-segmentation's
TOC/chapter-first-page classifier pilot proved useful, trimmed to the
subset relevant to TOC-page detection. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's
"Geometry" section.

pdfalto (GROBID's own PDF front end) converts a whole PDF to ALTO XML in
one subprocess call, replacing pdfplumber here on 2026-09-04. It costs a
single PDF parse per book rather than pdfminer's per-page layout pass:
extracting every page of a 488-page OAPEN book, ALTO parsing included,
went from 59.6s to 9.3s, and the 50-page head+tail scan `predict.py` does
by default from 4.7s to 1.0s. It also hands back explicit <TextLine>
elements and per-word <String> boxes with a document-level font table,
where pdfplumber only exposed raw per-char geometry this module had to
cluster into lines itself.
"""

import statistics
import tempfile
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import pdfalto

from .pdf_text import page_count

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

# Points; ALTO <TextLine> elements whose tops are within this distance of
# the group's own first one are treated as the same visual text line.
_LINE_TOLERANCE = 2.0


def split_into_runs(page_indices: list[int]) -> list[list[int]]:
    """Splits an ascending list of page indices into maximal runs of
    consecutive integers. A head+tail scan skips the pages in between, so
    its scanned pages form two runs, not one -- and both callers care:
    pdfalto takes one contiguous page range per invocation, and
    select_topk_ranges must not be asked to treat the last head page and
    the first tail page as adjacent."""
    runs: list[list[int]] = []
    for index in page_indices:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _font_sizes(root: ET.Element) -> dict[str, float]:
    """ALTO keeps font sizes in one document-level <Styles> table; each
    <String> names its entry through STYLEREFS."""
    return {
        style.get("ID"): float(style.get("FONTSIZE") or 0.0)
        for style in root.iterfind(".//{*}TextStyle")
        if style.get("ID")
    }


def _word(string: ET.Element, font_sizes: dict[str, float]) -> dict:
    """One ALTO <String> (a word box) in the same {text, x0, x1, top,
    bottom, size} shape the pdfplumber implementation used for a single
    char, so the geometry below reads the same as it did per-char."""
    x0 = float(string.get("HPOS") or 0.0)
    top = float(string.get("VPOS") or 0.0)
    height = float(string.get("HEIGHT") or 0.0)
    # STYLEREFS is an IDREFS list (a font, sometimes alongside other style
    # references); the word's own glyph height stands in when none of them
    # resolves to a font size.
    size = next(
        (font_sizes[ref] for ref in (string.get("STYLEREFS") or "").split() if font_sizes.get(ref)),
        height,
    )
    return {
        "text": string.get("CONTENT") or "",
        "x0": x0,
        "x1": x0 + float(string.get("WIDTH") or 0.0),
        "top": top,
        "bottom": top + height,
        "size": size,
    }


def _text_lines(page: ET.Element, font_sizes: dict[str, float]) -> list[dict]:
    """Every non-empty <TextLine> on one ALTO <Page>, as {"top": the
    line's own VPOS, "words": its word boxes}. The line's VPOS is used for
    grouping rather than the words' own, since a superscript's box sits
    higher than the line it belongs to and pdfalto has already decided it
    is part of this line."""
    lines = []
    for line in page.iterfind(".//{*}TextLine"):
        words = [
            word
            for word in (_word(string, font_sizes) for string in line.iterfind("{*}String"))
            if word["text"]
        ]
        if words:
            lines.append({"top": float(line.get("VPOS") or words[0]["top"]), "words": words})
    return lines


def _merge_into_visual_lines(text_lines: list[dict]) -> list[list[dict]]:
    """Groups _text_lines' output into visual lines -- one word list per
    line, left to right -- by clustering consecutive <TextLine>s (sorted
    top to bottom) within _LINE_TOLERANCE of the group's own first.

    pdfalto emits one <TextLine> per run of closely-spaced words, so a TOC
    entry's title and its right-flush page number come back as two
    separate <TextLine>s at the same VPOS, in two different <TextBlock>s --
    precisely because of the wide gap between them. Merging them back into
    one visual line is what lets _reconstruct_line_text see that gap at
    all, and keeps line_count counting visual lines the way the
    pdfplumber implementation's own char clustering did."""
    if not text_lines:
        return []
    ordered = sorted(text_lines, key=lambda line: (line["top"], line["words"][0]["x0"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for line in ordered[1:]:
        if abs(line["top"] - groups[-1][0]["top"]) <= _LINE_TOLERANCE:
            groups[-1].append(line)
        else:
            groups.append([line])
    return [
        sorted((word for line in group for word in line["words"]), key=lambda word: word["x0"])
        for group in groups
    ]


# Gap-to-font-size ratios used by _reconstruct_line_text to classify the
# horizontal space between two consecutive words on the same line. Tuned
# to distinguish three cases: a token pdfalto split without a space in
# between (no space -- a hyphenated compound's two halves, or a word whose
# font changes mid-way), an ordinary space between words (one space), and
# the wide title/page-number gap in a TOC entry -- normally rendered via
# absolute glyph positioning on native PDFs, with no literal dot-leader
# characters pypdf's extract_text() could preserve, which is why that gap
# collapses to a single space there (see the comment above _TOC_LINE_RE in
# text_features.py for the full diagnosis this fixes).
#
# _WORD_GAP_RATIO is 0.1 rather than the 0.3 the pdfplumber implementation
# used, because ALTO measures the gap between two tight word boxes while
# pdfplumber measured it between two glyph boxes that each already include
# their advance width. Measured over ~37k word pairs from two
# independently typeset OAPEN books, the two populations separate cleanly:
# a no-space split sits at or below 0.05, real spaces start at 0.14 and
# peak around 0.18-0.30, and under 0.15% of all gaps fall in between.
_WORD_GAP_RATIO = 0.1
_WIDE_GAP_RATIO = 3.0


def _reconstruct_line_text(line: list[dict]) -> str:
    """Joins one visual line's word boxes (each needs `text`, `x0`, `x1`,
    `size`) into a string, left to right. A gap below _WORD_GAP_RATIO of
    the local font size means the two boxes are halves of one token (no
    space inserted); a gap at or above _WIDE_GAP_RATIO is treated as a
    TOC-style dot-leader/tab-stop gap and gets a 2-char separator
    (satisfying _TOC_LINE_RE's requirement) instead of pypdf's
    single-space normalization; everything in between gets one ordinary
    space."""
    ordered = sorted(line, key=lambda word: word["x0"])
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


def _empty_page_features() -> dict[str, float]:
    return {
        **{name: 0.0 for name in PAGE_FEATURE_NAMES},
        "_max_font_size": 0.0,
        "_modal_font_size": 0.0,
    }


def _convert_to_alto(pdf_path: str | Path, first_page: int, last_page: int) -> ET.Element:
    """Runs pdfalto over pages `first_page`..`last_page` (1-based,
    inclusive) and returns the parsed ALTO root. The XML and pdfalto's
    metadata sidecar are written to a temporary directory and read back
    from there -- nothing is left next to the input PDF."""
    with tempfile.TemporaryDirectory(prefix="toc-page-classifier-alto-") as tmp_dir:
        # skip_graphics: this module reads text geometry only, and without
        # it pdfalto also writes every embedded image out to a sidecar
        # directory for nobody to read.
        result = pdfalto.convert(
            pdf_path,
            Path(tmp_dir) / "pages.xml",
            first_page=first_page,
            last_page=last_page,
            skip_graphics=True,
        )
        return ET.parse(result.alto).getroot()


def extract_page_features_and_texts(
    pdf_path: str | Path,
    head_pages: int | None = None,
    tail_pages: int | None = None,
) -> tuple[dict[int, dict[str, float]], dict[int, str], int]:
    """One pdfalto pass producing both this book's per-page layout feature
    dict (keyed by 0-based page index, matching PAGE_FEATURE_NAMES plus
    add_book_context_features' underscore-prefixed intermediates), its
    gap-aware reconstructed per-page text (via _reconstruct_line_text --
    use this instead of pdf_text.page_texts() (pypdf) as
    extract_text_features's input, since it preserves the wide
    title/page-number gap in native-PDF TOC lines that pypdf's extraction
    collapses to a single space), and the book's total page count.

    Features and text come out of the same conversion (as they did from
    the single pdfplumber pass this replaced) because converting the PDF
    is the dominant per-book cost: two functions each converting
    independently would parse every PDF twice for no reason.

    By default (head_pages and tail_pages both None) every page is
    converted -- training/evaluation needs honest features for every page,
    since the true TOC location isn't known in advance. Pass
    head_pages/tail_pages to convert only the first head_pages and/or last
    tail_pages pages instead: a real TOC is essentially never buried deep
    in a long book's interior, and skipping that interior keeps both
    pdfalto's work and the resulting ALTO document (tens of MB for a long
    book, all of which has to be parsed back into memory here) down to the
    pages that can actually matter. Each contiguous run of wanted pages is
    one pdfalto invocation, so a head+tail scan runs it twice. If the two
    windows overlap or together cover the whole book, every page is
    converted anyway rather than skipping the (nonexistent) gap between
    them. The returned dicts only have entries for the pages actually
    requested."""
    total_pages = page_count(pdf_path)
    if total_pages == 0:
        return {}, {}, 0

    if head_pages is None and tail_pages is None:
        wanted_pages = list(range(total_pages))
    else:
        head = range(min(head_pages or 0, total_pages))
        tail = range(max(total_pages - (tail_pages or 0), 0), total_pages)
        wanted_pages = sorted(set(head) | set(tail))

    features: dict[int, dict[str, float]] = {}
    texts: dict[int, str] = {}
    for run in split_into_runs(wanted_pages):
        root = _convert_to_alto(pdf_path, run[0] + 1, run[-1] + 1)
        font_sizes = _font_sizes(root)
        for page in root.iterfind(".//{*}Page"):
            # PHYSICAL_IMG_NR is the page's 1-based number in the whole
            # document, not its position within the converted range.
            page_index = int(page.get("PHYSICAL_IMG_NR") or 0) - 1
            page_height = float(page.get("HEIGHT") or 0.0)
            page_width = float(page.get("WIDTH") or 0.0)
            lines = _merge_into_visual_lines(_text_lines(page, font_sizes))

            texts[page_index] = "\n".join(_reconstruct_line_text(line) for line in lines)

            if not lines or not page_height or not page_width:
                features[page_index] = _empty_page_features()
                continue

            line_sizes = [statistics.mean(word["size"] for word in line) for line in lines]
            line_tops = [min(word["top"] for word in line) for line in lines]
            line_bottoms = [max(word["bottom"] for word in line) for line in lines]
            line_lefts = [min(word["x0"] for word in line) / page_width for line in lines]

            modal_size = statistics.mode(round(size, 1) for size in line_sizes)
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

    # A page pypdf counted that pdfalto did not emit at all (the two
    # disagreeing on the document's structure) is treated exactly like a
    # page with no text on it, so callers still get an entry for every
    # page they asked for.
    for page_index in wanted_pages:
        if page_index not in features:
            features[page_index] = _empty_page_features()
            texts.setdefault(page_index, "")

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
