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


def test_select_topk_ranges_tie_break_prefers_wider_window():
    # Uniform scores -- every window of every length ties on mean score, so
    # the tie-break (not the score) decides the winner. Regression test for
    # a real bug: sorting candidates by score alone breaks ties by
    # generation order (narrowest windows first), so a 1-page window would
    # silently win over an equally-scored wider window. The widest window
    # allowed by max_window must win instead.
    scores = [0.5, 0.5, 0.5, 0.5]
    ranges = select_topk_ranges(scores, k=1, max_window=4)
    assert len(ranges) == 1
    start, end, score = ranges[0]
    assert (start, end) == (0, 3)
    assert score == 0.5
