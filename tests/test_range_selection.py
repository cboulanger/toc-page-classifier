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
