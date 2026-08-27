import pytest

from toc_page_classifier.range_selection import select_topk_ranges


def test_select_topk_ranges_finds_single_best_window():
    # A clear 2-page peak at indices 3-4. max_window=2 pins this test to
    # the original narrow-peak scenario -- sum-ranking with the real
    # default max_window (6) would legitimately prefer a wider window here
    # too (see test_select_topk_ranges_prefers_wider_window_when_it_still_
    # scores_higher below), which is not what this test is about.
    scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1]
    ranges = select_topk_ranges(scores, k=1, max_window=2)
    assert len(ranges) == 1
    start, end, score = ranges[0]
    assert (start, end) == (3, 4)
    assert score == pytest.approx(1.8)


def test_select_topk_ranges_returns_non_overlapping_windows():
    # Two distinct 2-page peaks. max_window=2 keeps them distinct -- see
    # note on test_select_topk_ranges_finds_single_best_window above.
    scores = [0.9, 0.9, 0.1, 0.1, 0.1, 0.8, 0.8]
    ranges = select_topk_ranges(scores, k=2, max_window=2)
    assert len(ranges) == 2
    windows = {(r[0], r[1]) for r in ranges}
    assert (0, 1) in windows
    assert any(w[0] >= 5 for w in windows)  # the second peak, not overlapping the first


def test_select_topk_ranges_prefers_wider_window_when_it_still_scores_higher():
    # A 2-page peak (indices 0-1) immediately followed by a real, lower but
    # still clearly non-baseline continuation page (index 2, matching a
    # true multi-page TOC's headed-page-plus-continuation-page shape). No
    # padding on either side of the array, so the comparison is a clean
    # sum(0.9, 0.9, 0.3)=2.1 vs. sum(0.9, 0.9)=1.8 -- not a padding tie.
    # Under sum ranking the wider window wins on real signal, unlike mean
    # ranking, which would have preferred the narrower, higher-mean peak.
    scores = [0.9, 0.9, 0.3]
    ranges = select_topk_ranges(scores, k=1)
    assert len(ranges) == 1
    start, end, score = ranges[0]
    assert (start, end) == (0, 2)
    assert score == pytest.approx(2.1)


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
    # Uniform scores -- under sum-ranking the widest window (0.5*4=2.0)
    # unambiguously beats any narrower one on sum alone, no tie-break
    # needed. This is a regression test for the tie-break rule's residual
    # case: sorting candidates by score alone breaks LITERAL ties (e.g. a
    # window whose extra pages score exactly 0) by generation order
    # (narrowest first), so a 1-page window could silently win over an
    # equally-scored wider one. The widest window allowed by max_window
    # must win instead.
    scores = [0.5, 0.5, 0.5, 0.5]
    ranges = select_topk_ranges(scores, k=1, max_window=4)
    assert len(ranges) == 1
    start, end, score = ranges[0]
    assert (start, end) == (0, 3)
    assert score == pytest.approx(2.0)
