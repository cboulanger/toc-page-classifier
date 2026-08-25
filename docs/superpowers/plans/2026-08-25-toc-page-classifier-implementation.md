# TOC Page Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the standalone TOC-page classifier described in
`docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md`: given any
book PDF with a usable text layer (no DNB reference scan required), predict
which contiguous page range(s) contain the table of contents.

**Architecture:** Two independent per-page feature extractors (`pdfplumber`
geometry, structural/keyword text) feed one page-level `LogisticRegression`
scorer (compared against gradient boosting), evaluated leave-one-book-out
over a merged ground-truth table (chapter-segmentation's 89 hand-verified
books + this repo's 95 DNB-located pairs). Page scores are aggregated into
ranked, non-overlapping candidate page-range windows (top-K) rather than
consumed as a per-page classification.

**Tech Stack:** Python 3.12, `pdfplumber` (new dependency), `scikit-learn`,
`pypdf` (already a dependency), `pytest`, `uv`.

---

## Task 0: Add `pdfplumber` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

Edit the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "pypdf>=5.1.0",
    "httpx>=0.27.0",
    "cryptography>=50.0.0",
    "pdfplumber>=0.11.0",
    "scikit-learn>=1.5.0",
]
```

- [ ] **Step 2: Sync and verify import**

```bash
uv sync
uv run python3 -c "import pdfplumber, sklearn; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pdfplumber and scikit-learn dependencies"
```

---

## Task 1: `layout_features.py` -- `pdfplumber`-based per-page geometry

**Files:**
- Create: `src/toc_page_classifier/layout_features.py`
- Test: `tests/test_layout_features.py`

This is this repo's own re-derivation of the layout features
chapter-segmentation's TOC-classifier pilot proved useful
(`evaluation/scripts/layout_features.py` in that repo), ported from
`pdfalto`-produced ALTO XML to `pdfplumber`'s `page.chars`, and trimmed to
the subset the design spec calls out as TOC-relevant (font-size contrast,
line density, left-margin mean/variance, first/last text vertical
position, `edge_distance`, book-context normalization) -- dropping the old
pilot's chapter-opening-specific features (`trailing_number_fraction`,
`top_block_is_large_font`, `top_line_heading_match`, `width_mean/var`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layout_features.py
from toc_page_classifier.layout_features import (
    FEATURE_NAMES,
    _group_chars_into_lines,
    add_book_context_features,
)


def _char(top: float, x0: float, size: float) -> dict:
    return {"top": top, "bottom": top + size, "x0": x0, "size": size}


def test_group_chars_into_lines_clusters_by_vertical_position():
    chars = [
        _char(100.0, 10.0, 12.0),
        _char(100.4, 20.0, 12.0),  # same line as above (within tolerance)
        _char(130.0, 10.0, 12.0),  # a new line, far below
    ]
    lines = _group_chars_into_lines(chars)
    assert len(lines) == 2
    assert len(lines[0]) == 2
    assert len(lines[1]) == 1


def test_group_chars_into_lines_returns_empty_for_no_chars():
    assert _group_chars_into_lines([]) == []


def test_add_book_context_features_computes_relative_and_edge_features():
    # Three pages, hand-built (bypassing extract_page_features/pdfplumber):
    # page 0 is a dense page near the front, page 1 a typical body page,
    # page 2 a typical body page near the back of a 5-page book.
    page_features = {
        0: {
            "line_count": 40.0, "font_size_max_ratio": 1.0, "line_density": 0.05,
            "left_margin_mean": 0.1, "left_margin_var": 0.0,
            "first_text_vpos_fraction": 0.1, "last_text_vpos_fraction": 0.9,
            "_max_font_size": 12.0, "_modal_font_size": 10.0,
        },
        1: {
            "line_count": 20.0, "font_size_max_ratio": 1.0, "line_density": 0.03,
            "left_margin_mean": 0.1, "left_margin_var": 0.0,
            "first_text_vpos_fraction": 0.1, "last_text_vpos_fraction": 0.9,
            "_max_font_size": 10.0, "_modal_font_size": 10.0,
        },
        2: {
            "line_count": 20.0, "font_size_max_ratio": 1.0, "line_density": 0.03,
            "left_margin_mean": 0.1, "left_margin_var": 0.0,
            "first_text_vpos_fraction": 0.1, "last_text_vpos_fraction": 0.9,
            "_max_font_size": 10.0, "_modal_font_size": 10.0,
        },
    }
    result = add_book_context_features(page_features, total_pages=5)
    assert set(result[0].keys()) == set(FEATURE_NAMES)
    # median line_count across the 3 pages is 20.0
    assert result[0]["line_count_rel"] == 2.0
    assert result[1]["line_count_rel"] == 1.0
    # median of _modal_font_size (10, 10, 10) is 10.0 -> page 0's max (12) / 10 = 1.2
    assert result[0]["font_size_max_ratio_book"] == 1.2
    assert result[1]["font_size_max_ratio_book"] == 1.0
    # edge_distance = min(page_index, total_pages - 1 - page_index), total_pages=5
    assert result[0]["edge_distance"] == 0.0
    assert result[1]["edge_distance"] == 1.0
    assert result[2]["edge_distance"] == 2.0


def test_add_book_context_features_handles_all_empty_pages():
    page_features = {
        0: {name: 0.0 for name in [
            "line_count", "font_size_max_ratio", "line_density",
            "left_margin_mean", "left_margin_var",
            "first_text_vpos_fraction", "last_text_vpos_fraction",
        ]} | {"_max_font_size": 0.0, "_modal_font_size": 0.0},
    }
    result = add_book_context_features(page_features, total_pages=1)
    assert result[0]["font_size_max_ratio_book"] == 1.0
    assert result[0]["edge_distance"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'toc_page_classifier.layout_features'`.

- [ ] **Step 3: Implement `layout_features.py`**

```python
# src/toc_page_classifier/layout_features.py
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/toc_page_classifier/layout_features.py tests/test_layout_features.py
git commit -m "feat: add pdfplumber-based per-page layout features"
```

---

## Task 2: `text_features.py` -- structural TOC-line pattern matching

**Files:**
- Create: `src/toc_page_classifier/text_features.py`
- Test: `tests/test_text_features.py`

Standalone reimplementation of `chapter_segmentation`'s TOC-line structural
pattern (`src/chapter_segmentation/segmentation.py`'s `_TOC_LINE_RE`,
`_looks_like_url_or_doi`, `_looks_like_imprint_line`) -- not imported, per
the design spec's "keep this repo self-contained" requirement. Keyword
matching is added in Task 3 once `data/toc_keywords.json` exists; this task
stubs `keyword_hit_any_language`/`keyword_hit_same_language` at `0.0` so
the feature vector shape (`TEXT_FEATURE_NAMES`) is stable from the start.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_text_features.py
from toc_page_classifier.text_features import TEXT_FEATURE_NAMES, extract_text_features


def test_extract_text_features_finds_toc_lines_on_a_real_toc_page():
    toc_page = "Contents\n\nIntroduction ..... 1\nChapter One .......... 5\nChapter Two .......... 12\n"
    other_page = "This is an ordinary paragraph of prose that runs on for a while and\nhas nothing structured about it at all.\n"
    features = extract_text_features([toc_page, other_page])
    assert set(features[0].keys()) == set(TEXT_FEATURE_NAMES)
    assert features[0]["toc_line_count"] == 3.0
    assert features[0]["toc_line_ratio"] == 1.0
    assert features[0]["monotonic_page_numbers"] == 1.0
    assert features[1]["toc_line_count"] == 0.0


def test_extract_text_features_rejects_non_monotonic_page_numbers():
    # An index/bibliography page can superficially match "text ... number"
    # (2+ dot/space separator chars, same as a real TOC line) without a
    # real TOC's monotonically increasing page-number sequence.
    index_page = "Aardvark .. 88\nZebra .. 3\nMongoose .. 45\n"
    features = extract_text_features([index_page])
    assert features[0]["toc_line_count"] == 3.0
    assert features[0]["monotonic_page_numbers"] == 0.0


def test_extract_text_features_excludes_url_and_imprint_lines():
    page = "© 2020 Some Publisher\nISBN 978-0-000-00000-0\nSee https://doi.org/10.1000/xyz123\n"
    features = extract_text_features([page])
    assert features[0]["toc_line_count"] == 0.0


def test_extract_text_features_computes_digit_density():
    features = extract_text_features(["11111"])
    assert features[0]["digit_density"] == 1.0
    features = extract_text_features([""])
    assert features[0]["digit_density"] == 0.0


def test_extract_text_features_keyword_hits_default_to_zero_without_keywords_path():
    features = extract_text_features(["Contents\n"], language="en", keywords_path=None)
    assert features[0]["keyword_hit_any_language"] == 0.0
    assert features[0]["keyword_hit_same_language"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_text_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'toc_page_classifier.text_features'`.

- [ ] **Step 3: Implement `text_features.py`**

```python
# src/toc_page_classifier/text_features.py
"""Text/structural features for TOC-page detection: a standalone
reimplementation of chapter_segmentation's TOC-line structural pattern
(src/chapter_segmentation/segmentation.py's _TOC_LINE_RE and friends -- not
imported, to keep this repo self-contained) plus, once
keyword_hit_* is wired up in Task 3, multilingual keyword matching against
data/toc_keywords.json. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Text /
structural" section."""

import re

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


def _match_toc_line(line: str) -> re.Match | None:
    stripped = line.strip()
    if not stripped or _URL_OR_DOI_RE.search(line) or _IMPRINT_LINE_RE.search(line):
        return None
    return _TOC_LINE_RE.match(stripped)


def extract_text_features(
    pages: list[str],
    language: str | None = None,
    keywords_path: str | None = None,
) -> dict[int, dict[str, float]]:
    """Structural + keyword text features, one dict per page index.
    `language` is the book's own declared language code (e.g. "en"), used
    only for keyword_hit_same_language. `keywords_path` is wired up in
    Task 3 -- until then, keyword_hit_* are always 0.0."""
    features: dict[int, dict[str, float]] = {}
    for page_index, text in enumerate(pages):
        lines = [line for line in text.splitlines() if line.strip()]
        matches = [m for line in lines if (m := _match_toc_line(line)) is not None]
        page_numbers = [
            n for m in matches if (n := _parse_page_number(m["page"])) is not None
        ]
        monotonic = float(
            len(page_numbers) >= 2 and all(b > a for a, b in zip(page_numbers, page_numbers[1:]))
        )
        digit_density = sum(c.isdigit() for c in text) / len(text) if text else 0.0

        features[page_index] = {
            "toc_line_count": float(len(matches)),
            "toc_line_ratio": len(matches) / len(lines) if lines else 0.0,
            "digit_density": digit_density,
            "monotonic_page_numbers": monotonic,
            "keyword_hit_any_language": 0.0,
            "keyword_hit_same_language": 0.0,
        }
    return features
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_text_features.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/toc_page_classifier/text_features.py tests/test_text_features.py
git commit -m "feat: add structural TOC-line text features"
```

---

## Task 3: `data/toc_keywords.json` + keyword matching

**Files:**
- Create: `data/toc_keywords.json`
- Modify: `src/toc_page_classifier/text_features.py`
- Modify: `tests/test_text_features.py`

Hand-seeded starting point covering every language present in the current
100-book manifest (`en=42, de=40, it=6, es=3, fr=2, nl=1`). Task 5's mining
script expands this empirically once run; this seed only needs to be
plausible enough for the feature and its tests to work correctly now.

- [ ] **Step 1: Write the seed keyword file**

```json
{
  "en": ["contents", "table of contents"],
  "de": ["inhalt", "inhaltsverzeichnis"],
  "fr": ["sommaire", "table des matières"],
  "it": ["indice", "sommario"],
  "es": ["índice", "tabla de contenidos", "contenido"],
  "nl": ["inhoud", "inhoudsopgave"]
}
```

Save as `data/toc_keywords.json`.

- [ ] **Step 2: Write the failing tests for keyword matching**

Append to `tests/test_text_features.py`:

```python
import json

from toc_page_classifier.text_features import extract_text_features


def test_extract_text_features_keyword_hit_any_language(tmp_path):
    keywords_path = tmp_path / "keywords.json"
    keywords_path.write_text(json.dumps({"en": ["contents"], "de": ["inhalt"]}))
    features = extract_text_features(["INHALT\n\nKapitel 1 .... 5\n"], keywords_path=str(keywords_path))
    assert features[0]["keyword_hit_any_language"] == 1.0


def test_extract_text_features_keyword_hit_same_language_requires_matching_language(tmp_path):
    keywords_path = tmp_path / "keywords.json"
    keywords_path.write_text(json.dumps({"en": ["contents"], "de": ["inhalt"]}))
    features = extract_text_features(
        ["INHALT\n\nKapitel 1 .... 5\n"], language="en", keywords_path=str(keywords_path)
    )
    # matches "de"'s keyword, but book is declared "en" -- any-language hits,
    # same-language does not
    assert features[0]["keyword_hit_any_language"] == 1.0
    assert features[0]["keyword_hit_same_language"] == 0.0


def test_extract_text_features_keyword_hit_same_language_matches(tmp_path):
    keywords_path = tmp_path / "keywords.json"
    keywords_path.write_text(json.dumps({"en": ["contents"], "de": ["inhalt"]}))
    features = extract_text_features(
        ["INHALT\n\nKapitel 1 .... 5\n"], language="de", keywords_path=str(keywords_path)
    )
    assert features[0]["keyword_hit_same_language"] == 1.0
```

- [ ] **Step 3: Run to verify the new tests fail**

```bash
uv run pytest tests/test_text_features.py -v
```

Expected: the three new tests fail (`keyword_hit_any_language`/`keyword_hit_same_language` are hardcoded `0.0`); the earlier 5 tests still pass.

- [ ] **Step 4: Wire up keyword loading in `text_features.py`**

Replace the module's keyword-related section:

```python
import json
from functools import lru_cache
from pathlib import Path

DEFAULT_KEYWORDS_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "toc_keywords.json"
)


@lru_cache(maxsize=8)
def _load_keywords(keywords_path: str) -> dict[str, list[str]]:
    return json.loads(Path(keywords_path).read_text(encoding="utf-8"))
```

Change `extract_text_features`'s signature and body:

```python
def extract_text_features(
    pages: list[str],
    language: str | None = None,
    keywords_path: str | None = DEFAULT_KEYWORDS_PATH,
) -> dict[int, dict[str, float]]:
    """Structural + keyword text features, one dict per page index.
    `language` is the book's own declared language code (e.g. "en"), used
    only for keyword_hit_same_language -- pass None if unknown.
    `keywords_path` points at a JSON file of {language_code: [keyword, ...]}
    (see data/toc_keywords.json); pass None to disable keyword matching
    entirely (keyword_hit_* always 0.0)."""
    keywords = _load_keywords(keywords_path) if keywords_path else {}
    all_keywords = [kw.lower() for kws in keywords.values() for kw in kws]
    same_language_keywords = [kw.lower() for kw in keywords.get(language, [])] if language else []

    features: dict[int, dict[str, float]] = {}
    for page_index, text in enumerate(pages):
        lines = [line for line in text.splitlines() if line.strip()]
        matches = [m for line in lines if (m := _match_toc_line(line)) is not None]
        page_numbers = [
            n for m in matches if (n := _parse_page_number(m["page"])) is not None
        ]
        monotonic = float(
            len(page_numbers) >= 2 and all(b > a for a, b in zip(page_numbers, page_numbers[1:]))
        )
        digit_density = sum(c.isdigit() for c in text) / len(text) if text else 0.0
        opening_text = " ".join(lines[:5]).lower()

        features[page_index] = {
            "toc_line_count": float(len(matches)),
            "toc_line_ratio": len(matches) / len(lines) if lines else 0.0,
            "digit_density": digit_density,
            "monotonic_page_numbers": monotonic,
            "keyword_hit_any_language": float(any(kw in opening_text for kw in all_keywords)),
            "keyword_hit_same_language": float(any(kw in opening_text for kw in same_language_keywords)),
        }
    return features
```

Update the old `test_extract_text_features_keyword_hits_default_to_zero_without_keywords_path`
test (Task 2) -- it already passes `keywords_path=None` explicitly, so it
keeps passing unchanged; no edit needed.

- [ ] **Step 5: Run to verify all tests pass**

```bash
uv run pytest tests/test_text_features.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add data/toc_keywords.json src/toc_page_classifier/text_features.py tests/test_text_features.py
git commit -m "feat: add multilingual TOC keyword matching"
```

---

## Task 4: `ground_truth.py` -- merge the two label sources

**Files:**
- Create: `src/toc_page_classifier/ground_truth.py`
- Test: `tests/test_ground_truth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ground_truth.py
import json
from pathlib import Path

import pytest

from toc_page_classifier.ground_truth import (
    _margin_to_weight,
    load_chapter_segmentation_rows,
    load_dnb_located_rows,
    merge_ground_truth,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def chapter_segmentation_dir(tmp_path):
    root = tmp_path / "chapter-segmentation"
    corpus = root / "evaluation" / "corpus" / "open-access"
    _write_json(corpus / "manifest.json", {"books": [
        {"filename": "1111111111.pdf", "language": "en", "extraction_type": "native"},
        {"filename": "2222222222.pdf", "language": "de", "extraction_type": "scan"},
    ]})
    _write_json(corpus / "1111111111.expected.json", {"toc": {"toc_start_index": 3, "toc_end_index": 4}})
    (corpus / "1111111111.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_json(corpus / "2222222222.expected.json", {"toc": None})
    (corpus / "2222222222.pdf").write_bytes(b"%PDF-1.4 fake")
    # no "toc" key at all -- must be excluded
    _write_json(corpus / "3333333333.expected.json", {})
    (corpus / "3333333333.pdf").write_bytes(b"%PDF-1.4 fake")
    return root


@pytest.fixture
def dnb_dirs(tmp_path, monkeypatch):
    gt_dir = tmp_path / "dnb" / "ground-truth"
    pdf_dir = tmp_path / "dnb" / "pdf"
    gt_dir.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)
    _write_json(gt_dir / "1111111111.json", {
        "isbn": "1111111111", "status": "located",
        "toc_start_index": 7, "toc_end_index": 8, "margin": 0.5,
    })
    (pdf_dir / "1111111111.fulltext.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_json(gt_dir / "4444444444.json", {
        "isbn": "4444444444", "status": "located",
        "toc_start_index": 1, "toc_end_index": 2, "margin": 0.0,
    })
    (pdf_dir / "4444444444.fulltext.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_json(gt_dir / "5555555555.json", {"isbn": "5555555555", "status": "error"})
    manifest_path = tmp_path / "dnb" / "manifest.json"
    _write_json(manifest_path, {"books": [{"isbn": "4444444444", "language": "fr"}]})

    import toc_page_classifier.ground_truth as gt_module
    monkeypatch.setattr(gt_module, "_DNB_GT_DIR", gt_dir)
    monkeypatch.setattr(gt_module, "_DNB_PDF_DIR", pdf_dir)
    monkeypatch.setattr(gt_module, "_DNB_MANIFEST_PATH", manifest_path)
    return gt_dir, pdf_dir


def test_margin_to_weight_clips_into_expected_range():
    assert _margin_to_weight(0.0) == pytest.approx(0.3)
    assert _margin_to_weight(0.5) == pytest.approx(1.0)
    assert _margin_to_weight(10.0) == pytest.approx(1.0)  # clipped, never > 1.0
    assert _margin_to_weight(-1.0) == pytest.approx(0.3)  # clipped, never < 0.3


def test_load_chapter_segmentation_rows_includes_null_toc_and_excludes_missing_key(chapter_segmentation_dir):
    rows = load_chapter_segmentation_rows(chapter_segmentation_dir)
    keys = {r.key for r in rows}
    assert keys == {"1111111111", "2222222222"}  # "3333333333" has no "toc" key
    by_key = {r.key: r for r in rows}
    assert by_key["1111111111"].toc_start_index == 3
    assert by_key["1111111111"].weight == 1.0
    assert by_key["1111111111"].language == "en"
    assert by_key["1111111111"].extraction_type == "native"
    assert by_key["2222222222"].toc_start_index is None  # confirmed no TOC
    assert by_key["2222222222"].extraction_type == "scan"


def test_load_dnb_located_rows_excludes_non_located_and_maps_margin_to_weight(dnb_dirs):
    rows = load_dnb_located_rows()
    keys = {r.key for r in rows}
    assert keys == {"1111111111", "4444444444"}  # "5555555555" is status=error
    by_key = {r.key: r for r in rows}
    assert by_key["1111111111"].weight == pytest.approx(1.0)  # margin 0.5
    assert by_key["4444444444"].weight == pytest.approx(0.3)  # margin 0.0
    assert by_key["4444444444"].language == "fr"


def test_merge_ground_truth_prefers_chapter_segmentation_on_isbn_collision(chapter_segmentation_dir, dnb_dirs):
    rows = merge_ground_truth(chapter_segmentation_dir)
    by_key = {r.key: r for r in rows}
    # "1111111111" exists in both sources -- chapter_segmentation's row wins
    assert by_key["1111111111"].source == "chapter_segmentation"
    assert by_key["1111111111"].toc_start_index == 3
    # "4444444444" only exists in the DNB source
    assert by_key["4444444444"].source == "dnb_located"
    assert {r.key for r in rows} == {"1111111111", "2222222222", "4444444444"}
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_ground_truth.py -v
```

Expected: `ModuleNotFoundError: No module named 'toc_page_classifier.ground_truth'`.

- [ ] **Step 3: Implement `ground_truth.py`**

```python
# src/toc_page_classifier/ground_truth.py
"""Merges the two ground-truth label sources -- chapter-segmentation's
hand-verified 89-book evaluation corpus and this repo's own 95 DNB-located
pairs -- into one weighted training table. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Ground
truth" section for the merge rule and confidence weighting."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHAPTER_SEGMENTATION_DIR = Path(
    os.environ.get("CHAPTER_SEGMENTATION_DIR", str(_REPO_ROOT.parent / "chapter-segmentation"))
)
_DNB_GT_DIR = _REPO_ROOT / "data" / "corpus" / "pilot" / "ground-truth"
_DNB_PDF_DIR = _REPO_ROOT / "data" / "corpus" / "pilot" / "pdf"
_DNB_MANIFEST_PATH = _REPO_ROOT / "data" / "corpus" / "pilot" / "manifest.json"

_CHAPTER_SEGMENTATION_CORPORA = ["open-access", "copyrighted-scans"]

# Empirical range covering the bulk of the DNB-located corpus's margin
# distribution (95 books: min 0.0006, median 0.43, max 0.71) -- maps onto
# the [0.3, 1.0] sample-weight range below. A margin at or above
# _MARGIN_HI gets the full 1.0 weight; even the least-confident matches
# near 0.0 only get discounted to 0.3, never to zero -- an auto-located
# label is still real evidence, just weaker.
_MARGIN_LO, _MARGIN_HI = 0.0, 0.5
_WEIGHT_LO, _WEIGHT_HI = 0.3, 1.0


@dataclass
class GroundTruthRow:
    key: str
    pdf_path: Path
    toc_start_index: int | None
    toc_end_index: int | None
    weight: float
    language: str | None
    corpus: str  # "open-access" | "copyrighted-scans" | "dnb_located"
    source: str  # "chapter_segmentation" | "dnb_located"
    extraction_type: str | None = None  # "native" | "scan", from chapter-segmentation's
    # manifest.json -- unknown (None) for dnb_located rows, whose manifest carries no
    # such field.


def _margin_to_weight(margin: float) -> float:
    clipped = min(max(margin, _MARGIN_LO), _MARGIN_HI)
    fraction = (clipped - _MARGIN_LO) / (_MARGIN_HI - _MARGIN_LO)
    return _WEIGHT_LO + fraction * (_WEIGHT_HI - _WEIGHT_LO)


def load_chapter_segmentation_rows(chapter_segmentation_dir: Path) -> list[GroundTruthRow]:
    """One row per book in chapter-segmentation's evaluation corpus that has
    a retrofitted "toc" field -- present, whether a real range or null. A
    null-toc book still contributes real training signal (an all-negative
    page-label sequence), so it is included, not skipped."""
    rows = []
    corpus_root = chapter_segmentation_dir / "evaluation" / "corpus"
    for corpus in _CHAPTER_SEGMENTATION_CORPORA:
        corpus_dir = corpus_root / corpus
        manifest_path = corpus_dir / "manifest.json"
        languages: dict[str, str | None] = {}
        extraction_types: dict[str, str | None] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            languages = {
                Path(b["filename"]).stem: b.get("language") for b in manifest["books"]
            }
            extraction_types = {
                Path(b["filename"]).stem: b.get("extraction_type") for b in manifest["books"]
            }
        for expected_path in sorted(corpus_dir.glob("*.expected.json")):
            key = expected_path.name.removesuffix(".expected.json")
            pdf_path = corpus_dir / f"{key}.pdf"
            if not pdf_path.exists():
                continue
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            if "toc" not in expected:
                continue
            toc = expected["toc"]
            rows.append(GroundTruthRow(
                key=key,
                pdf_path=pdf_path,
                toc_start_index=toc["toc_start_index"] if toc else None,
                toc_end_index=toc["toc_end_index"] if toc else None,
                weight=1.0,
                language=languages.get(key),
                corpus=corpus,
                source="chapter_segmentation",
                extraction_type=extraction_types.get(key),
            ))
    return rows


def load_dnb_located_rows() -> list[GroundTruthRow]:
    """One row per DNB-located pair with status "located" -- the other
    statuses (reference_has_no_text, error, no_candidate) have no usable
    range."""
    languages: dict[str, str | None] = {}
    if _DNB_MANIFEST_PATH.exists():
        manifest = json.loads(_DNB_MANIFEST_PATH.read_text(encoding="utf-8"))
        languages = {b["isbn"]: b.get("language") for b in manifest["books"]}

    rows = []
    for gt_path in sorted(_DNB_GT_DIR.glob("*.json")):
        entry = json.loads(gt_path.read_text(encoding="utf-8"))
        if entry.get("status") != "located":
            continue
        isbn = entry["isbn"]
        pdf_path = _DNB_PDF_DIR / f"{isbn}.fulltext.pdf"
        if not pdf_path.exists():
            continue
        rows.append(GroundTruthRow(
            key=isbn,
            pdf_path=pdf_path,
            toc_start_index=entry["toc_start_index"],
            toc_end_index=entry["toc_end_index"],
            weight=_margin_to_weight(entry["margin"]),
            language=languages.get(isbn),
            corpus="dnb_located",
            source="dnb_located",
        ))
    return rows


def merge_ground_truth(chapter_segmentation_dir: Path | None = None) -> list[GroundTruthRow]:
    """Dedups by key, preferring chapter-segmentation's hand-verified row
    over a DNB-located duplicate for the same book -- see the design
    spec's merge rule."""
    chapter_segmentation_rows = load_chapter_segmentation_rows(
        chapter_segmentation_dir or _DEFAULT_CHAPTER_SEGMENTATION_DIR
    )
    dnb_rows = load_dnb_located_rows()
    by_key = {row.key: row for row in dnb_rows}
    for row in chapter_segmentation_rows:
        by_key[row.key] = row
    return list(by_key.values())
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_ground_truth.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/toc_page_classifier/ground_truth.py tests/test_ground_truth.py
git commit -m "feat: merge chapter-segmentation and DNB-located ground truth"
```

---

## Task 5: `cli/mine_toc_keywords.py` -- empirical keyword mining

**Files:**
- Create: `cli/__init__.py` (empty -- makes `cli` an importable package so
  `tests/test_mine_toc_keywords.py` can `import cli.mine_toc_keywords`
  directly, rather than relying on implicit namespace-package resolution)
- Create: `cli/mine_toc_keywords.py`
- Test: `tests/test_mine_toc_keywords.py`

Scans every book in the merged ground truth with a known TOC range and a
known language, reads the first non-blank line of the TOC's first page,
and counts frequency per language -- candidates get written to
`data/toc_keywords.candidates.json` for **human review**, never merged
into `data/toc_keywords.json` automatically (per the design spec: "gets a
human review pass ... before being merged into the committed
`data/toc_keywords.json`").

- [ ] **Step 1: Create `cli/__init__.py`**

```bash
touch cli/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_mine_toc_keywords.py
from toc_page_classifier.ground_truth import GroundTruthRow

from cli.mine_toc_keywords import mine_candidates


def _row(key, pdf_path, toc_start, language):
    return GroundTruthRow(
        key=key, pdf_path=pdf_path, toc_start_index=toc_start, toc_end_index=toc_start,
        weight=1.0, language=language, corpus="test", source="test",
    )


def test_mine_candidates_counts_first_line_per_language(tmp_path):
    # mine_candidates takes a page-text loader so it never touches a real
    # PDF in this test -- see its signature.
    pages_by_key = {
        "a": ["Inhalt\n\nKapitel 1 .... 5\n"],
        "b": ["INHALT\n\nKapitel 1 .... 5\n"],
        "c": ["Contents\n\nChapter 1 .... 5\n"],
    }
    rows = [
        _row("a", tmp_path / "a.pdf", 0, "de"),
        _row("b", tmp_path / "b.pdf", 0, "de"),
        _row("c", tmp_path / "c.pdf", 0, "en"),
    ]
    candidates = mine_candidates(rows, load_pages=lambda row: pages_by_key[row.key])
    assert candidates["de"]["inhalt"] == 2
    assert candidates["en"]["contents"] == 1


def test_mine_candidates_skips_rows_with_no_toc_or_no_language(tmp_path):
    rows = [
        _row("a", tmp_path / "a.pdf", None, "de"),  # no TOC
        _row("b", tmp_path / "b.pdf", 0, None),  # no language
    ]
    candidates = mine_candidates(rows, load_pages=lambda row: ["Inhalt\n"])
    assert candidates == {}
```

- [ ] **Step 3: Run to verify it fails**

```bash
uv run pytest tests/test_mine_toc_keywords.py -v
```

Expected: `ModuleNotFoundError: No module named 'cli.mine_toc_keywords'`.

- [ ] **Step 4: Implement `cli/mine_toc_keywords.py`**

```python
#!/usr/bin/env python3
"""Empirically mines candidate TOC-heading keywords from the merged ground
truth corpus, grouped by each book's declared language -- writes
data/toc_keywords.candidates.json for a HUMAN REVIEW PASS. Never writes to
data/toc_keywords.json directly: a frequent short phrase found here still
needs a human judgment call on whether it's really a TOC-heading phrase
(not, e.g., a frequent but unrelated short word). See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Text /
structural" section.

    uv run python cli/mine_toc_keywords.py
    uv run python cli/mine_toc_keywords.py --min-count 2
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from toc_page_classifier.ground_truth import GroundTruthRow, merge_ground_truth
from toc_page_classifier.pdf_text import page_texts

REPO_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES_PATH = REPO_ROOT / "data" / "toc_keywords.candidates.json"
_MAX_PHRASE_WORDS = 4  # a TOC heading is short ("table of contents"), unlike
# a real chapter title -- filters out long lines that happen to be a TOC
# page's first line for some other reason (e.g. a running header).


def mine_candidates(
    rows: list[GroundTruthRow],
    load_pages: Callable[[GroundTruthRow], list[str]] = page_texts,
) -> dict[str, dict[str, int]]:
    """Returns {language: {phrase: count}} -- the first non-blank line of
    each book's TOC first page, lowercased, for books with a known TOC
    range and a known language, filtered to short phrases."""
    counts: dict[str, Counter] = {}
    for row in rows:
        if row.toc_start_index is None or row.language is None:
            continue
        pages = load_pages(row)
        if row.toc_start_index >= len(pages):
            continue
        lines = [line.strip() for line in pages[row.toc_start_index].splitlines() if line.strip()]
        if not lines:
            continue
        phrase = lines[0].lower()
        if len(phrase.split()) > _MAX_PHRASE_WORDS:
            continue
        counts.setdefault(row.language, Counter())[phrase] += 1
    return {language: dict(counter) for language, counter in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-count", type=int, default=2, help="Minimum frequency to write as a candidate (default: 2).")
    args = parser.parse_args()

    rows = merge_ground_truth()
    all_candidates = mine_candidates(rows)
    filtered = {
        language: {phrase: count for phrase, count in phrases.items() if count >= args.min_count}
        for language, phrases in all_candidates.items()
    }
    filtered = {language: phrases for language, phrases in filtered.items() if phrases}

    _CANDIDATES_PATH.write_text(json.dumps(filtered, indent=2, ensure_ascii=False, sort_keys=True))
    total = sum(len(phrases) for phrases in filtered.values())
    print(f"Wrote {total} candidate phrase(s) across {len(filtered)} language(s) to {_CANDIDATES_PATH}")
    print("Review by hand before merging any of these into data/toc_keywords.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run to verify it passes**

```bash
uv run pytest tests/test_mine_toc_keywords.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add cli/__init__.py cli/mine_toc_keywords.py tests/test_mine_toc_keywords.py
git commit -m "feat: add empirical TOC-keyword mining script"
```

- [ ] **Step 7: Run the miner against the real corpus and review its output by hand**

```bash
uv run python cli/mine_toc_keywords.py
cat data/toc_keywords.candidates.json
```

For each candidate phrase: open the JSON, and for every phrase that is
genuinely a TOC-heading phrase not already in `data/toc_keywords.json`
(case-insensitively), add it to that language's array in
`data/toc_keywords.json`. Discard anything that isn't really a heading
phrase (e.g. a stray frequent short word that isn't about contents/TOC).
Delete `data/toc_keywords.candidates.json` once done reviewing it (it is a
scratch artifact, not meant to be committed).

- [ ] **Step 8: Run the text-feature tests again to confirm the expanded keyword file still parses correctly**

```bash
uv run pytest tests/test_text_features.py -v
```

Expected: all previously-passing tests still pass (this only adds entries
to existing language arrays or new language keys, never removes any).

- [ ] **Step 9: Commit the reviewed keyword additions**

```bash
git add data/toc_keywords.json
git commit -m "data: expand toc_keywords.json with empirically-mined phrases"
```

---

## Task 6: `cli/train_toc_classifier.py` -- LOBO evaluation with range selection

**Files:**
- Create: `src/toc_page_classifier/range_selection.py`
- Create: `cli/train_toc_classifier.py`
- Test: `tests/test_range_selection.py`

Range selection (a pure, easily-testable function) is split into its own
module so it can be unit tested without needing real PDFs or a trained
model; `cli/train_toc_classifier.py` wires it together with the feature
extractors and `ground_truth.py` for the actual LOBO run.

- [ ] **Step 1: Write the failing tests for range selection**

```python
# tests/test_range_selection.py
from toc_page_classifier.range_selection import select_topk_ranges


def test_select_topk_ranges_finds_single_best_window():
    # A clear 2-page peak at indices 3-4.
    scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1]
    ranges = select_topk_ranges(scores, k=1)
    assert len(ranges) == 1
    start, end, score = ranges[0]
    assert (start, end) == (3, 4)
    assert score == 0.9


def test_select_topk_ranges_returns_non_overlapping_windows():
    scores = [0.9, 0.9, 0.1, 0.1, 0.1, 0.8, 0.8]
    ranges = select_topk_ranges(scores, k=2)
    assert len(ranges) == 2
    windows = {(r[0], r[1]) for r in ranges}
    assert (0, 1) in windows
    assert any(w[0] >= 5 for w in windows)  # the second peak, not overlapping the first


def test_select_topk_ranges_never_exceeds_available_non_overlapping_windows():
    scores = [0.9, 0.1]  # too short for many non-overlapping windows
    ranges = select_topk_ranges(scores, k=5)
    assert len(ranges) <= 2


def test_select_topk_ranges_respects_max_window_size():
    scores = [0.5] * 10  # uniform -- every window scores identically
    ranges = select_topk_ranges(scores, k=1, max_window=4)
    start, end, _ = ranges[0]
    assert end - start + 1 <= 4
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_range_selection.py -v
```

Expected: `ModuleNotFoundError: No module named 'toc_page_classifier.range_selection'`.

- [ ] **Step 3: Implement `range_selection.py`**

```python
# src/toc_page_classifier/range_selection.py
"""Aggregates per-page TOC-likelihood scores into ranked, non-overlapping
candidate page ranges -- the "range selection" step of
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Model"
section. Window lengths 1-4 match the observed TOC-length distribution in
the merged ground truth (mean 2.8 pages, max 9 -- 1-4 covers the large
majority; a book with a longer real TOC will still be found by an
overlapping shorter window landing inside it, just not with perfect page
coverage)."""

_DEFAULT_MIN_WINDOW = 1
_DEFAULT_MAX_WINDOW = 4


def select_topk_ranges(
    scores: list[float],
    k: int = 3,
    min_window: int = _DEFAULT_MIN_WINDOW,
    max_window: int = _DEFAULT_MAX_WINDOW,
) -> list[tuple[int, int, float]]:
    """Every contiguous window of length min_window..max_window, ranked by
    mean score, greedily picking the top `k` non-overlapping windows
    (highest mean first; a window overlapping an already-picked one is
    skipped, not merged). Returns fewer than `k` windows if the page count
    doesn't support that many non-overlapping ones."""
    n = len(scores)
    if n == 0:
        return []
    candidates = []
    for size in range(min_window, min(max_window, n) + 1):
        for start in range(n - size + 1):
            window = scores[start : start + size]
            candidates.append((start, start + size - 1, sum(window) / size))
    candidates.sort(key=lambda c: c[2], reverse=True)

    selected: list[tuple[int, int, float]] = []
    for start, end, score in candidates:
        if any(not (end < s or start > e) for s, e, _ in selected):
            continue
        selected.append((start, end, score))
        if len(selected) == k:
            break
    return selected
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_range_selection.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/toc_page_classifier/range_selection.py tests/test_range_selection.py
git commit -m "feat: add top-K non-overlapping range selection"
```

- [ ] **Step 6: Write `cli/train_toc_classifier.py`**

No new failing-test step here: `build_feature_table`/`evaluate_leave_one_book_out`
need real PDFs to exercise meaningfully, so this task is verified by
Step 7's real run against the actual merged corpus rather than a synthetic
unit test (the pieces it composes -- `layout_features`, `text_features`,
`ground_truth`, `range_selection` -- are already unit-tested above).

```python
#!/usr/bin/env python3
"""Leave-one-book-out evaluation of the TOC-page classifier: per-page
layout + text features, a page-level scorer (LogisticRegression or
gradient boosting), and top-K non-overlapping candidate page-range output.
See docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md.

    uv run python cli/train_toc_classifier.py
    uv run python cli/train_toc_classifier.py --model gradient_boosting
    uv run python cli/train_toc_classifier.py --chapter-segmentation-dir ../chapter-segmentation
"""

import argparse
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from toc_page_classifier.ground_truth import GroundTruthRow, merge_ground_truth
from toc_page_classifier.layout_features import (
    FEATURE_NAMES as LAYOUT_FEATURE_NAMES,
    add_book_context_features,
    extract_page_features,
)
from toc_page_classifier.pdf_text import page_texts
from toc_page_classifier.range_selection import select_topk_ranges
from toc_page_classifier.text_features import TEXT_FEATURE_NAMES, extract_text_features

ALL_FEATURE_NAMES = LAYOUT_FEATURE_NAMES + TEXT_FEATURE_NAMES
_TOP_K = 3


def build_feature_table(rows: list[GroundTruthRow]) -> list[dict]:
    """One row per page: {"book_key", "corpus", "extraction_type",
    "features": {...}, "label": bool, "weight": float}."""
    table = []
    for row in rows:
        pages = page_texts(row.pdf_path)
        total_pages = len(pages)
        if total_pages == 0:
            continue
        layout = add_book_context_features(extract_page_features(row.pdf_path), total_pages)
        text = extract_text_features(pages, language=row.language)
        toc_indices = (
            set(range(row.toc_start_index, row.toc_end_index + 1))
            if row.toc_start_index is not None
            else set()
        )
        for page_index in range(total_pages):
            if page_index not in layout:
                continue  # pdfplumber found no page geometry (should not normally happen)
            features = {**layout[page_index], **text[page_index]}
            table.append({
                "book_key": row.key,
                "corpus": row.corpus,
                "extraction_type": row.extraction_type,
                "features": features,
                "label": page_index in toc_indices,
                "weight": row.weight,
            })
    return table


def _make_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier()
    raise ValueError(f"unknown model {name!r}")


def evaluate_leave_one_book_out(table: list[dict], model_name: str) -> dict:
    """Runs LOBO over every book with at least one true TOC page (a book
    with none has nothing to rank a hit against). Returns per-book
    top1/top3 hit flags plus each book's corpus and extraction_type, for
    breakdown reporting."""
    book_keys = sorted({r["book_key"] for r in table})
    per_book = []
    for held_out in book_keys:
        test_rows = [r for r in table if r["book_key"] == held_out]
        true_indices = {i for i, r in enumerate(test_rows) if r["label"]}
        if not true_indices:
            continue

        train_rows = [r for r in table if r["book_key"] != held_out]
        X_train = [[r["features"][name] for name in ALL_FEATURE_NAMES] for r in train_rows]
        y_train = [r["label"] for r in train_rows]
        sample_weight = [r["weight"] for r in train_rows]
        X_test = [[r["features"][name] for name in ALL_FEATURE_NAMES] for r in test_rows]

        scaler = StandardScaler().fit(X_train)
        model = _make_model(model_name)
        model.fit(scaler.transform(X_train), y_train, sample_weight=sample_weight)
        scores = [p[1] for p in model.predict_proba(scaler.transform(X_test))]

        ranges = select_topk_ranges(scores, k=_TOP_K)
        top1_hit = bool(ranges) and true_indices <= set(range(ranges[0][0], ranges[0][1] + 1))
        top3_hit = any(true_indices <= set(range(s, e + 1)) for s, e, _ in ranges)
        per_book.append({
            "book_key": held_out,
            "corpus": test_rows[0]["corpus"],
            "extraction_type": test_rows[0]["extraction_type"],
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
        })

    return {"per_book": per_book}


def _hit_rate(per_book: list[dict], field: str, group_by: str | None = None, group_value=None) -> float:
    subset = [r for r in per_book if group_by is None or r[group_by] == group_value]
    return sum(r[field] for r in subset) / len(subset) if subset else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chapter-segmentation-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["logistic_regression", "gradient_boosting"], default="logistic_regression")
    args = parser.parse_args()

    rows = merge_ground_truth(args.chapter_segmentation_dir)
    print(f"Merged ground truth: {len(rows)} books")
    table = build_feature_table(rows)

    summary = evaluate_leave_one_book_out(table, args.model)
    per_book = summary["per_book"]
    print(f"Books evaluated (with >=1 true TOC page): {len(per_book)}")
    print(f"Top-1 hit rate: {_hit_rate(per_book, 'top1_hit'):.1%}")
    print(f"Top-3 hit rate: {_hit_rate(per_book, 'top3_hit'):.1%}")
    print()
    print("By corpus:")
    for corpus in sorted({r["corpus"] for r in per_book}):
        n = sum(1 for r in per_book if r["corpus"] == corpus)
        print(
            f"  {corpus} (n={n}): top1={_hit_rate(per_book, 'top1_hit', 'corpus', corpus):.1%}, "
            f"top3={_hit_rate(per_book, 'top3_hit', 'corpus', corpus):.1%}"
        )
    print("By extraction_type (chapter-segmentation rows only -- dnb_located rows have no")
    print("extraction_type in their manifest, grouped here as None/'unknown'):")
    for extraction_type in sorted({r["extraction_type"] for r in per_book}, key=lambda v: (v is None, v)):
        n = sum(1 for r in per_book if r["extraction_type"] == extraction_type)
        label = extraction_type or "unknown"
        print(
            f"  {label} (n={n}): top1={_hit_rate(per_book, 'top1_hit', 'extraction_type', extraction_type):.1%}, "
            f"top3={_hit_rate(per_book, 'top3_hit', 'extraction_type', extraction_type):.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run it against the real merged corpus**

```bash
uv run python cli/train_toc_classifier.py
uv run python cli/train_toc_classifier.py --model gradient_boosting
```

Record both models' top-1/top-3 hit rates and per-corpus breakdown --
these are the numbers Task 7 writes into the design spec / README as the
first real measurement. Don't tune anything based on a single run yet;
just capture what happened.

- [ ] **Step 8: Commit**

```bash
git add cli/train_toc_classifier.py
git commit -m "feat: add LOBO TOC classifier training/evaluation script"
```

---

## Task 7: Document the first measured result

**Files:**
- Modify: `README.md`
- Modify: `cli/README.md`
- Modify: `docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md`

- [ ] **Step 1: Add `cli/train_toc_classifier.py` and `cli/mine_toc_keywords.py` sections to `cli/README.md`**

Regenerate each script's `--help` dump and paste it in, following this
file's existing per-script section format (see `AGENTS.md`):

```bash
uv run python cli/mine_toc_keywords.py --help
uv run python cli/train_toc_classifier.py --help
```

Add one section per script, each with a short explanation (mirroring the
existing `## harvest_oapen.py`-style sections) plus the fenced `--help`
dump, and update the numbered pipeline-stage list at the top of the file
to include step 5 (mine keywords, done once/occasionally) and step 6
(train/evaluate).

- [ ] **Step 2: Update `README.md`'s "Methodology" and "Current status"**

Add a step 6 to the numbered methodology list: "**Classifier
training/evaluation** (`cli/train_toc_classifier.py`,
`src/toc_page_classifier/{layout_features,text_features,ground_truth,range_selection}.py`):
merges both ground-truth sources, trains a page-level scorer, and reports
leave-one-book-out top-1/top-3 range-hit rates." Replace the "Classifier
training (not yet implemented)" bullet under "Current status" with the
actual measured numbers from Task 6 Step 7 (both models, overall and
per-corpus), and update the "Known gaps" list to drop "No classifier
training code yet."

Also fix the existing methodology step 4 description's stale wording
("word-token containment") to say "word-token Ochiai overlap" -- the
scoring formula changed after that line was written and the docstring in
`locate_toc.py` was updated at the time but this line was missed.

- [ ] **Step 3: Update the design spec's "Deferred items" section**

In `docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md`, add a
short "Implemented" note above "Out of scope" pointing at this plan
(`docs/superpowers/plans/2026-08-25-toc-page-classifier-implementation.md`)
and the measured LOBO numbers, so a future reader doesn't have to infer
implementation status from the spec's present-tense design language alone.

- [ ] **Step 4: Run the full test suite one more time**

```bash
uv run pytest -q
```

Expected: all tests pass (this task only touches documentation, no code).

- [ ] **Step 5: Commit**

```bash
git add README.md cli/README.md docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md
git commit -m "docs: document the TOC classifier's first measured LOBO result"
```
