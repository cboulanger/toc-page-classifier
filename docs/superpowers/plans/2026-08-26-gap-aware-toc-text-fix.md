# Gap-Aware TOC Text Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the classifier's measured 0% open-access LOBO hit rate by replacing pypdf-based page text (which collapses a TOC entry's title/page-number gap to a single space on native PDFs) with a pdfplumber-geometry-based reconstruction that preserves that gap as a detectable multi-character separator, so `_TOC_LINE_RE` can match TOC lines on native-extraction text the same way it already does on OCR'd text.

**Architecture:** Add a new function `extract_gap_aware_page_texts(pdf_path)` to `layout_features.py` (which already opens every page via pdfplumber and groups its chars into lines via `_group_chars_into_lines`). For each line, walk consecutive chars left-to-right and classify the horizontal gap between them, relative to font size, into three bands: negligible (concatenate directly, e.g. kerning within a word), normal (insert one space, e.g. an ordinary word boundary), or wide (insert two spaces, e.g. the visual gap between a TOC entry's title and its right-flush page number that a dot-leader or absolute positioning creates but pypdf's `extract_text()` normalizes away). `cli/train_toc_classifier.py`'s `build_feature_table` then uses this instead of `pdf_text.page_texts()` as the input to `extract_text_features`.

**Tech Stack:** Python, pdfplumber (already a dependency), pytest.

**Non-goals:** This plan does not touch the model choice, hyperparameters, or add new features beyond fixing this one text-extraction gap. Per the design discussion, we fix this root cause, re-run the full LOBO evaluation once, and report the real numbers before considering any further tuning.

---

### Task 1: Gap-aware line/page text reconstruction in `layout_features.py`

**Files:**
- Modify: `src/toc_page_classifier/layout_features.py`
- Test: `tests/test_layout_features.py`

**Context for the implementer:** `layout_features.py` already has `_group_chars_into_lines(chars)`, which groups a page's pdfplumber `chars` (each a dict with at least `top`, `bottom`, `x0`, `size` keys — see the existing `_char()` test helper in `tests/test_layout_features.py`) into text lines by vertical clustering. pdfplumber chars also carry an `x1` (right edge) and `text` (the single character) key, not currently used by this file. `text_features.py`'s `_TOC_LINE_RE` requires 2+ consecutive `.`/whitespace characters between a TOC entry's title and its page number to avoid false-positiving on ordinary prose; see the comment directly above `_TOC_LINE_RE` in `src/toc_page_classifier/text_features.py` for the full diagnosis of why this fails on native PDFs (pypdf's `extract_text()` collapses the visual gap to one space, so the regex never sees 2+ separator chars there). This task does not touch `text_features.py` at all — it gives `train_toc_classifier.py` (Task 2) a different, gap-preserving text source to feed into the *existing*, unmodified `extract_text_features`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_layout_features.py` (the existing `_char()` helper takes `top, x0, size` and does not include `x1`/`text`; extend it or add a new helper — do not break the existing calls to `_char()` in this file):

```python
from toc_page_classifier.layout_features import _reconstruct_line_text


def _chars(*specs):
    """specs: list of (text, x0, x1, size) tuples, in left-to-right order."""
    return [{"text": t, "x0": x0, "x1": x1, "size": size} for t, x0, x1, size in specs]


def test_reconstruct_line_text_joins_adjacent_chars_with_no_space():
    # Gap of 0.2pt on 10pt font (ratio 0.02) -- ordinary kerning within a word.
    line = _chars(("H", 0.0, 6.0, 10.0), ("i", 6.2, 9.0, 10.0))
    assert _reconstruct_line_text(line) == "Hi"


def test_reconstruct_line_text_inserts_single_space_for_normal_word_gap():
    # Gap of 3.5pt on 10pt font (ratio 0.35) -- an ordinary space between words.
    line = _chars(("A", 0.0, 6.0, 10.0), ("B", 9.5, 15.5, 10.0))
    assert _reconstruct_line_text(line) == "A B"


def test_reconstruct_line_text_inserts_double_space_for_wide_toc_gap():
    # Gap of 35pt on 10pt font (ratio 3.5) -- the title/page-number gap a
    # native PDF's absolute glyph positioning creates for a right-flush TOC
    # page number, with no literal dot-leader characters in between.
    line = _chars(("Foreword", 0.0, 40.0, 10.0), ("7", 75.0, 81.0, 10.0))
    assert _reconstruct_line_text(line) == "Foreword  7"


def test_reconstruct_line_text_output_matches_toc_line_regex():
    # Closes the loop on the actual bug: the reconstructed text for a
    # single-space-collapsed native-PDF TOC line must satisfy
    # _TOC_LINE_RE, which requires a 2+-char separator run.
    from toc_page_classifier.text_features import _TOC_LINE_RE

    line = _chars(
        ("A", 0.0, 6.0, 10.0), ("c", 6.0, 11.0, 10.0), ("k", 11.0, 16.0, 10.0),
        ("n", 16.0, 22.0, 10.0), ("o", 22.0, 28.0, 10.0), ("w", 28.0, 35.0, 10.0),
        ("l", 35.0, 38.0, 10.0), ("e", 38.0, 43.0, 10.0), ("d", 43.0, 49.0, 10.0),
        ("g", 49.0, 55.0, 10.0), ("e", 55.0, 60.0, 10.0), ("m", 60.0, 69.0, 10.0),
        ("e", 69.0, 74.0, 10.0), ("n", 74.0, 80.0, 10.0), ("t", 80.0, 84.0, 10.0),
        ("s", 84.0, 89.0, 10.0),
        ("x", 130.0, 133.0, 10.0), ("i", 133.0, 135.0, 10.0),
    )
    reconstructed = _reconstruct_line_text(line)
    assert _TOC_LINE_RE.match(reconstructed) is not None


def test_extract_gap_aware_page_texts_returns_text_per_page(tmp_path):
    import pdfplumber
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))
    c.setFont("Helvetica", 12)
    c.drawString(20, 150, "Contents")
    c.showPage()
    c.save()

    from toc_page_classifier.layout_features import extract_gap_aware_page_texts

    texts = extract_gap_aware_page_texts(str(pdf_path))
    assert list(texts.keys()) == [0]
    assert "Contents" in texts[0]
```

Check whether `reportlab` is already an installed dependency before writing `test_extract_gap_aware_page_texts_returns_text_per_page`:

Run: `uv run python -c "import reportlab"`

- If it succeeds, keep the test as written.
- If it raises `ModuleNotFoundError`, drop that one test (the four `_reconstruct_line_text` tests above give solid coverage of the actual gap-classification logic without needing a real PDF) rather than adding a new dependency just for one integration-style test — note this decision in the implementer's final report.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_layout_features.py -v`
Expected: FAIL with `ImportError` / `AttributeError` (`_reconstruct_line_text` and `extract_gap_aware_page_texts` don't exist yet).

- [ ] **Step 3: Implement**

In `src/toc_page_classifier/layout_features.py`, add `from itertools import pairwise` to the imports at the top of the file (alongside `statistics` and `pathlib.Path`), then add these constants near `_LINE_TOLERANCE` and these two functions after `_group_chars_into_lines`:

```python
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


def extract_gap_aware_page_texts(pdf_path: str | Path) -> dict[int, str]:
    """Per-page text reconstructed from pdfplumber char geometry via
    _reconstruct_line_text, keyed by 0-based page index. Use this instead
    of pdf_text.page_texts() (pypdf) as extract_text_features's input --
    it preserves the wide title/page-number gap in native-PDF TOC lines
    that pypdf's extraction collapses to a single space."""
    texts: dict[int, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            lines = _group_chars_into_lines(page.chars)
            texts[page_index] = "\n".join(_reconstruct_line_text(line) for line in lines)
    return texts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_layout_features.py -v`
Expected: PASS (all tests, including the pre-existing ones in this file).

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest`
Expected: PASS, no fewer than the pre-existing 47 tests, all green.

- [ ] **Step 6: Commit**

```bash
git add src/toc_page_classifier/layout_features.py tests/test_layout_features.py
git commit -m "feat: reconstruct gap-aware page text from pdfplumber geometry"
```

**Self-review (do before reporting DONE):** Re-read `_reconstruct_line_text`. Confirm: (a) a single trailing char (line of length 1) doesn't crash (`pairwise` of a 1-element list yields nothing, so the loop body never runs -- verify this by reasoning, no new test needed); (b) the function doesn't mutate its input list order in a way that would surprise a caller (it sorts into a new list, `ordered`, not in place -- confirm `line` itself is untouched by checking `sorted()` is used, not `line.sort()`).

---

### Task 2: Wire gap-aware text into the training pipeline

**Files:**
- Modify: `cli/train_toc_classifier.py`

**Context for the implementer:** `cli/train_toc_classifier.py`'s `build_feature_table` currently does:
```python
from toc_page_classifier.pdf_text import page_texts
...
pages = page_texts(row.pdf_path)
total_pages = len(pages)
if total_pages == 0:
    continue
layout = add_book_context_features(extract_page_features(row.pdf_path), total_pages)
text = extract_text_features(pages, language=row.language)
```
Task 1 added `extract_gap_aware_page_texts(pdf_path) -> dict[int, str]` to `layout_features.py`. This task replaces the `pdf_text.page_texts()` call with it. Note the return type difference: `page_texts` returns `list[str]` (implicitly page-index-ordered), `extract_gap_aware_page_texts` returns `dict[int, str]` keyed by page index -- `extract_text_features` takes a `list[str]`, so convert via `[gap_aware_texts[i] for i in range(total_pages)]`. Do not remove `pdf_text.py` or its `page_texts`/`has_text` functions -- `has_text` is still used elsewhere in the pipeline (`locate_toc.py`) and is out of scope for this fix.

- [ ] **Step 1: Make the change**

In `cli/train_toc_classifier.py`:

Replace this import:
```python
from toc_page_classifier.pdf_text import page_texts
```
with:
```python
from toc_page_classifier.layout_features import (
    FEATURE_NAMES as LAYOUT_FEATURE_NAMES,
    add_book_context_features,
    extract_gap_aware_page_texts,
    extract_page_features,
)
```
(merging into the existing `from toc_page_classifier.layout_features import (...)` block a few lines below rather than creating a second import from the same module -- there is already one; add `extract_gap_aware_page_texts` into it and delete the `pdf_text` import line entirely.)

Replace the start of `build_feature_table`'s loop body:
```python
    for row in rows:
        pages = page_texts(row.pdf_path)
        total_pages = len(pages)
        if total_pages == 0:
            continue
        layout = add_book_context_features(extract_page_features(row.pdf_path), total_pages)
        text = extract_text_features(pages, language=row.language)
```
with:
```python
    for row in rows:
        gap_aware_texts = extract_gap_aware_page_texts(row.pdf_path)
        total_pages = len(gap_aware_texts)
        if total_pages == 0:
            continue
        layout = add_book_context_features(extract_page_features(row.pdf_path), total_pages)
        pages = [gap_aware_texts[i] for i in range(total_pages)]
        text = extract_text_features(pages, language=row.language)
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, same count as after Task 1 (this task changes no tested contract -- `build_feature_table` itself has no direct unit test; `evaluate_leave_one_book_out`'s tests construct their own synthetic rows and don't call `build_feature_table`).

- [ ] **Step 3: Smoke-test against 2-3 real books (not the full corpus)**

This confirms the wiring actually runs end-to-end against a real PDF before committing to the ~2h45m-per-model full-corpus run. Run:

```bash
uv run python -c "
from toc_page_classifier.ground_truth import merge_ground_truth
from cli.train_toc_classifier import build_feature_table

rows = merge_ground_truth()[:3]
table = build_feature_table(rows)
print(f'{len(table)} page-rows from {len(rows)} books')
print('toc_line_count > 0 on any page:', any(r[\"features\"][\"toc_line_count\"] > 0 for r in table))
"
```
Expected: prints a page-row count with no traceback. `toc_line_count > 0` may be `True` or `False` depending on which 3 books were sampled -- the point of this step is confirming no crash, not confirming the fix's effect (that's measured in Task 3's full run).

- [ ] **Step 4: Commit**

```bash
git add cli/train_toc_classifier.py
git commit -m "feat: use gap-aware page text in the training pipeline"
```

---

### Task 3 (not subagent-delegated -- run directly by the controlling session)

After Tasks 1-2 are reviewed and committed, run the full LOBO evaluation for both models against the real merged corpus (same commands as the original Task 6 run, now against the fixed pipeline):

```bash
uv run python cli/train_toc_classifier.py --model logistic_regression
uv run python cli/train_toc_classifier.py --model gradient_boosting
```

Each takes on the order of hours (the original run took ~2h45m-2h50m per model) -- dispatch as a background task and monitor via the same `ScheduleWakeup`/background-task pattern used for the original Task 6 run, not via a subagent-driven-development implementer/reviewer cycle (there is no code change in this step, only evaluation output to collect).

Compare the new open-access corpus hit rate against the previous 0.0%/0.0% baseline recorded in `README.md`'s "Current status" section. Then update `README.md`'s "Current status" section and "Known gaps" list with the real new numbers, following the same plain, specific, unemotional reporting style already established there (state what changed, cite the measured numbers, and if the open-access rate is still low, root-cause *that* rather than declaring victory on a partial improvement).
