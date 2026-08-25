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
from collections.abc import Callable
from pathlib import Path

from toc_page_classifier.ground_truth import GroundTruthRow, merge_ground_truth
from toc_page_classifier.pdf_text import page_texts

REPO_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES_PATH = REPO_ROOT / "data" / "toc_keywords.candidates.json"
_MAX_PHRASE_WORDS = 4  # a TOC heading is short ("table of contents"), unlike
# a real chapter title -- filters out long lines that happen to be a TOC
# page's first line for some other reason (e.g. a running header).


def _load_pages_for_row(row: GroundTruthRow) -> list[str]:
    """Default page loader: reads the row's own PDF from disk."""
    return page_texts(row.pdf_path)


def mine_candidates(
    rows: list[GroundTruthRow],
    load_pages: Callable[[GroundTruthRow], list[str]] = _load_pages_for_row,
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
