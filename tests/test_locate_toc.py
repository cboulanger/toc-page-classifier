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
    location = select_toc_range(scores)
    assert location is not None
    assert location.start_index == 5
    assert location.end_index == 6
    assert location.top_score == pytest.approx(0.6)
    assert location.runner_up_score == pytest.approx(0.12)


def test_select_toc_range_returns_none_for_all_zero_scores():
    assert select_toc_range([0.0, 0.0, 0.0]) is None
