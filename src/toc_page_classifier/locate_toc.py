"""Locate the table-of-contents page range inside a full-text PDF, given a
separate reference PDF (a DNB TOC scan) whose pages contain the same TOC
text.

Validated manually on the first cross-referenced OAPEN/DNB pairs found by
`cli/match_dnb_oa.py` (2026-08-25): a plain word-token Jaccard-overlap score
between the reference TOC text and each full-text page cleanly separates the
true TOC page(s) from the rest of the book -- a large score gap between the
top-ranked contiguous cluster and everything else, in every case where both
PDFs had a real text layer. This module has no fallback for a reference scan
with no text layer at all (a pure image, no OCR) -- see `pdf_text.has_text`
to detect that case upstream and route it to an OCR or vision-based path
instead, not implemented here.
"""

import re
from dataclasses import dataclass

from .pdf_text import page_texts

_WORD_RE = re.compile(r"[a-zäöüß]{3,}")


def token_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def score_pages(reference_pages: list[str], candidate_pages: list[str]) -> list[float]:
    """Jaccard overlap between the reference pages' combined vocabulary and
    each candidate page's own vocabulary, one score per candidate page."""
    reference_tokens = token_set(" ".join(reference_pages))
    if not reference_tokens:
        raise ValueError("reference has no extractable text -- likely an un-OCR'd scan; not handled here")
    scores = []
    for text in candidate_pages:
        candidate_tokens = token_set(text)
        if not candidate_tokens:
            scores.append(0.0)
            continue
        overlap = len(reference_tokens & candidate_tokens) / len(reference_tokens | candidate_tokens)
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


def select_toc_range(scores: list[float], score_ratio_threshold: float = 0.5) -> TocLocation | None:
    """Expands outward from the highest-scoring page while neighboring pages
    stay within `score_ratio_threshold` of the peak score, then reports the
    highest score found just outside that contiguous range as the
    "runner-up" -- the number to check against `top_score` for confidence."""
    if not scores or max(scores) == 0.0:
        return None
    peak_index = max(range(len(scores)), key=lambda i: scores[i])
    peak_score = scores[peak_index]
    threshold = peak_score * score_ratio_threshold

    start = end = peak_index
    while start > 0 and scores[start - 1] >= threshold:
        start -= 1
    while end < len(scores) - 1 and scores[end + 1] >= threshold:
        end += 1

    outside_scores = [s for i, s in enumerate(scores) if i < start or i > end]
    runner_up = max(outside_scores) if outside_scores else 0.0
    return TocLocation(start_index=start, end_index=end, top_score=peak_score, runner_up_score=runner_up)


def locate_toc_in_fulltext(reference_pdf_path, fulltext_pdf_path, max_fulltext_pages: int | None = None) -> TocLocation | None:
    reference_pages = page_texts(reference_pdf_path)
    candidate_pages = page_texts(fulltext_pdf_path, max_pages=max_fulltext_pages)
    scores = score_pages(reference_pages, candidate_pages)
    return select_toc_range(scores)
