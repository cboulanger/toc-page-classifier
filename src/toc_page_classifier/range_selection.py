"""Aggregates per-page TOC-likelihood scores into ranked, non-overlapping
candidate page ranges -- the "range selection" step of
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Model"
section. Window lengths 1-6 match the observed TOC-span distribution in
the merged ground truth (181 books: spans of 1-6 pages cover 179/181
(98.9%); only two outliers, at 9 and 17 pages, fall outside that range --
measured 2026-08-27 while diagnosing why top1_hit stayed near 10% despite
the page-level scorer ranking a true TOC page first in ~94% of books. A
book with a longer real TOC will still be found by an overlapping shorter
window landing inside it, just not with perfect page coverage).

Candidates are ranked by SUM of page scores, not mean (changed
2026-08-27; see the module-level history above for the diagnostic that
motivated it). A real multi-page TOC's continuation pages (no visible
"Contents" heading, so no keyword-hit signal) reliably score lower than
its first/headed page, but still meaningfully above baseline. Ranking by
mean systematically prefers a narrow window hugging just the peak page
over a wider one that would also cover those true continuation pages,
because folding in a lower-but-real page always pulls the mean down.
Ranking by sum has no such bias -- since scores are non-negative
probabilities, including a genuinely-elevated continuation page can only
raise a window's sum, never lower it. This is safe specifically because
the design's hit metric already tolerates over-inclusion (a window only
has to CONTAIN the true page set, not match it exactly) -- see
evaluate_leave_one_book_out's top1_hit/top3_hit definition."""

_DEFAULT_MIN_WINDOW = 1
_DEFAULT_MAX_WINDOW = 6


def select_topk_ranges(
    scores: list[float],
    k: int = 3,
    min_window: int = _DEFAULT_MIN_WINDOW,
    max_window: int = _DEFAULT_MAX_WINDOW,
) -> list[tuple[int, int, float]]:
    """Every contiguous window of length min_window..max_window, ranked by
    summed score, greedily picking the top `k` non-overlapping windows
    (highest sum first; a window overlapping an already-picked one is
    skipped, not merged). Returns fewer than `k` windows if the page count
    doesn't support that many non-overlapping ones."""
    n = len(scores)
    if n == 0:
        return []
    candidates = []
    for size in range(min_window, min(max_window, n) + 1):
        for start in range(n - size + 1):
            window = scores[start : start + size]
            candidates.append((start, start + size - 1, sum(window)))
    # Tie-break on window size (widest first) for the residual case of a
    # literal sum tie -- e.g. a wider window whose extra pages all score
    # exactly 0 ties its narrower sub-window on sum -- so the wider,
    # more-coverage window wins instead of whichever happened to be
    # generated first.
    candidates.sort(key=lambda c: (c[2], c[1] - c[0]), reverse=True)

    selected: list[tuple[int, int, float]] = []
    for start, end, score in candidates:
        if any(not (end < s or start > e) for s, e, _ in selected):
            continue
        selected.append((start, end, score))
        if len(selected) == k:
            break
    return selected
