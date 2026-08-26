"""Aggregates per-page TOC-likelihood scores into ranked, non-overlapping
candidate page ranges -- the "range selection" step of
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Model"
section. Window lengths 1-6 match the observed TOC-span distribution in
the merged ground truth (181 books: spans of 1-6 pages cover 179/181
(98.9%); only two outliers, at 9 and 17 pages, fall outside that range --
measured 2026-08-27 while diagnosing why top1_hit stayed near 10% despite
the page-level scorer ranking a true TOC page first in ~94% of books. A
book with a longer real TOC will still be found by an overlapping shorter
window landing inside it, just not with perfect page coverage)."""

_DEFAULT_MIN_WINDOW = 1
_DEFAULT_MAX_WINDOW = 6


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
    # Tie-break on window size (widest first) so that when a wider window
    # ties a narrower one on mean score -- e.g. a uniform-score run, or a
    # single-page peak whose enclosing 2-page window averages the same --
    # the wider, more-coverage window wins the tie instead of whichever
    # window happened to be generated first.
    candidates.sort(key=lambda c: (c[2], c[1] - c[0]), reverse=True)

    selected: list[tuple[int, int, float]] = []
    for start, end, score in candidates:
        if any(not (end < s or start > e) for s, e, _ in selected):
            continue
        selected.append((start, end, score))
        if len(selected) == k:
            break
    return selected
