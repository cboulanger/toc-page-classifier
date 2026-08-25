"""Locate the table-of-contents page range inside a full-text PDF, given a
separate reference PDF (a DNB TOC scan) whose pages contain the same TOC
text.

Validated manually on the first cross-referenced OAPEN/DNB pairs found by
`cli/match_dnb_oa.py` (2026-08-25): an Ochiai (cosine-like) word-token
overlap score -- `|intersection| / sqrt(|candidate| * |reference|)` --
cleanly separates the true TOC page(s) from the rest of the book, in every
case checked where both PDFs had a real text layer. This module has no
fallback for a reference scan with no text layer at all (a pure image, no
OCR) -- see `pdf_text.has_text` to detect that case upstream and route it
to an OCR or vision-based path instead, not implemented here.

Went through two other formulas first, both replaced 2026-08-25 after
ground-truth spot verification surfaced real failures:

- **Symmetric Jaccard** (`|intersection| / |union|`, the original
  implementation): the reference's combined vocabulary (the union of every
  reference page) is typically much larger than any single candidate page's
  own vocabulary, so a genuinely-matching but thin candidate page -- e.g.
  the last, sparsest page of a multi-page TOC, trailing off with just a few
  entries -- gets diluted by a union dominated by the reference's size, even
  though every one of its own tokens is actually present in the reference.
  This systematically truncated real multi-page TOC ranges by one page: the
  true last page's Jaccard score fell just below `select_toc_range`'s
  threshold in every checked case, even though the page was genuinely part
  of the TOC.
- **Plain containment** (`|intersection| / |candidate|`, tried next): fixes
  the truncation above, but a *tiny* candidate page (a title page with only
  a handful of words) can trivially score 1.0 if all of its few tokens
  happen to also appear in the reference -- with no way to distinguish that
  from a genuine, much larger TOC page also scoring 1.0. Found in practice:
  an 11-token title page tied the real 94-token TOC page at containment=1.0,
  and `select_toc_range`'s peak search (ties break toward the first,
  lowest-index match) locked onto the title page instead.

Ochiai fixes both: it rewards a candidate page whose own vocabulary is
substantially explained by the reference (like containment does, avoiding
the Jaccard truncation bug) while still scaling down with the *absolute*
size of the overlap via the sqrt(candidate size) term (avoiding a tiny
page's trivial 1.0 score, unlike plain containment).
"""

import math
import re
from dataclasses import dataclass

from .pdf_text import page_texts

_WORD_RE = re.compile(r"[a-zäöüß]{3,}")


def token_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def score_pages(reference_pages: list[str], candidate_pages: list[str]) -> list[float]:
    """Ochiai overlap between the reference pages' combined vocabulary and
    each candidate page's own vocabulary, one score per candidate page. See
    the module docstring for why this isn't a symmetric Jaccard overlap or
    plain containment."""
    reference_tokens = token_set(" ".join(reference_pages))
    if not reference_tokens:
        raise ValueError("reference has no extractable text -- likely an un-OCR'd scan; not handled here")
    scores = []
    for text in candidate_pages:
        candidate_tokens = token_set(text)
        if not candidate_tokens:
            scores.append(0.0)
            continue
        overlap = len(reference_tokens & candidate_tokens) / math.sqrt(len(candidate_tokens) * len(reference_tokens))
        scores.append(overlap)
    return scores


@dataclass
class TocLocation:
    start_index: int
    end_index: int
    top_score: float
    runner_up_score: float

    @property
    def margin(self) -> float:
        """How much the winning cluster's score clears the next-best page
        outside it -- a low margin means the match is ambiguous and should
        not be trusted without review."""
        return self.top_score - self.runner_up_score


def select_toc_range(scores: list[float], window_size: int) -> TocLocation | None:
    """Finds the contiguous window of exactly `window_size` pages with the
    highest mean score, and reports the highest mean score among windows that
    don't overlap it as the "runner-up" -- the number to check against
    `top_score` for confidence.

    Deliberately a fixed-size window rather than expanding outward from the
    highest-scoring page while neighboring pages stay within some threshold
    of it (the original implementation, replaced 2026-08-25): `window_size`
    is normally the reference TOC scan's own page count (see
    `locate_toc_in_fulltext`), which -- verified against every hand-checked
    ground-truth pair during a spot-verification pass -- matches the true TOC
    range's length exactly, since the DNB scan and the full-text PDF are the
    same edition and the TOC occupies the same number of physical pages in
    both. A relative score threshold needs tuning and trades truncating a
    thin trailing page against over-including a dense, unrelated one right
    next to the true range (both found in practice); a known window size
    needs no threshold at all.
    """
    if not scores or max(scores) == 0.0:
        return None
    window_size = max(1, min(window_size, len(scores)))
    window_means = [
        sum(scores[start : start + window_size]) / window_size
        for start in range(len(scores) - window_size + 1)
    ]
    best_start = max(range(len(window_means)), key=lambda i: window_means[i])
    best_end = best_start + window_size - 1
    top_score = window_means[best_start]

    outside_means = [
        mean
        for start, mean in enumerate(window_means)
        if start + window_size - 1 < best_start or start > best_end
    ]
    runner_up = max(outside_means) if outside_means else 0.0
    return TocLocation(start_index=best_start, end_index=best_end, top_score=top_score, runner_up_score=runner_up)


def locate_toc_in_fulltext(reference_pdf_path, fulltext_pdf_path, max_fulltext_pages: int | None = None) -> TocLocation | None:
    reference_pages = page_texts(reference_pdf_path)
    candidate_pages = page_texts(fulltext_pdf_path, max_pages=max_fulltext_pages)
    scores = score_pages(reference_pages, candidate_pages)
    return select_toc_range(scores, window_size=len(reference_pages))
