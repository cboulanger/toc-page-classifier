import pytest

from toc_page_classifier.locate_toc import score_pages, select_toc_range, token_set


def test_token_set_lowercases_and_drops_short_tokens():
    assert token_set("Der Titel 1 ist toll") == {"der", "titel", "ist", "toll"}


def test_score_pages_gives_full_score_for_identical_page():
    reference = ["Inhalt Vorbemerkungen 9 Einleitung 11"]
    candidates = ["Inhalt Vorbemerkungen 9 Einleitung 11", "irrelevant page about something else entirely"]
    scores = score_pages(reference, candidates)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < 0.2


def test_score_pages_raises_when_reference_has_no_text():
    with pytest.raises(ValueError):
        score_pages([""], ["some full text page"])


def test_select_toc_range_finds_contiguous_peak_cluster():
    # a two-page TOC cluster (indices 5, 6) surrounded by low-scoring noise
    scores = [0.05, 0.1, 0.05, 0.02, 0.05, 0.6, 0.55, 0.1, 0.08, 0.12]
    location = select_toc_range(scores, window_size=2)
    assert location is not None
    assert location.start_index == 5
    assert location.end_index == 6
    assert location.top_score == pytest.approx(0.575)
    assert location.runner_up_score == pytest.approx(0.1)


def test_select_toc_range_returns_none_for_all_zero_scores():
    assert select_toc_range([0.0, 0.0, 0.0], window_size=1) is None


def test_select_toc_range_does_not_grow_past_the_known_window_size():
    # Reproduces a real bug found the same day the fixed-window approach
    # replaced threshold-based expansion: a dense, unrelated page sitting
    # right next to the true TOC range (e.g. a sponsors/media-partners list
    # right before a festival catalog's real TOC) can score high enough to
    # pull a threshold-based expansion in with it. Knowing the TOC is exactly
    # `window_size` pages long -- the reference TOC scan's own page count --
    # makes that impossible by construction.
    scores = [0.1, 0.7, 0.65, 0.9, 0.85, 0.1]  # real 2-page TOC at (3, 4); an almost-as-strong decoy at (1, 2)
    location = select_toc_range(scores, window_size=2)
    assert (location.start_index, location.end_index) == (3, 4)


def test_score_pages_does_not_penalize_a_thin_trailing_toc_page():
    # Reproduces a real bug found via manual ground-truth spot-checking
    # (2026-08-25): a reference TOC combining multiple pages has a large
    # combined vocabulary. A genuine but thin trailing TOC page (few of its
    # own tokens, all of them present in the reference) must not score much
    # lower than a denser TOC page just because the reference vocabulary is
    # much bigger than this page's own -- that's what a symmetric Jaccard
    # union penalizes, even though every one of the thin page's own tokens is
    # actually found in the reference.
    reference = [
        "Contents Chapter One Introduction Chapter Two Methods Chapter Three Results",
        "Chapter Four Discussion Chapter Five Conclusion Appendix Bibliography Index",
    ]
    candidates = [
        "Contents Chapter One Introduction Chapter Two Methods Chapter Three Results",
        "Appendix Bibliography Index",
        "some unrelated body text about something else entirely",
    ]
    scores = score_pages(reference, candidates)
    assert scores[1] > 0.5 * scores[0]
    assert scores[1] > 3 * scores[2]


def test_score_pages_scores_tiny_incidental_match_lower_than_real_toc_page():
    # Reproduces a second real bug found the same day, in the fix for the one
    # above: plain containment (intersection / candidate size) lets a tiny
    # page whose few words all happen to appear in the reference score a
    # perfect 1.0, tying or beating a genuine, much larger TOC page that also
    # scores close to 1.0 -- and `select_toc_range`'s peak search breaks ties
    # toward the earliest page, so a title page with a handful of
    # incidentally-matching words won a real match in production. The score
    # must scale with the *absolute* amount of matching evidence, not just
    # the fraction.
    reference = [
        "Contents Chapter One Introduction Chapter Two Methods Chapter Three "
        "Results Chapter Four Analysis Chapter Five Discussion Chapter Six Conclusion"
    ]
    candidates = [
        "Introduction Methods",  # tiny page: both words happen to be in the reference, but weak evidence
        "Contents Chapter One Introduction Chapter Two Methods Chapter Three Results Chapter Four Analysis",
    ]
    scores = score_pages(reference, candidates)
    assert scores[1] > scores[0]
