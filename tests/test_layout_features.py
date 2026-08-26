from toc_page_classifier.layout_features import (
    FEATURE_NAMES,
    _group_chars_into_lines,
    _reconstruct_line_text,
    add_book_context_features,
)


def _char(top: float, x0: float, size: float) -> dict:
    return {"top": top, "bottom": top + size, "x0": x0, "size": size}


def _chars(*specs):
    """specs: list of (text, x0, x1, size) tuples, in left-to-right order."""
    return [{"text": t, "x0": x0, "x1": x1, "size": size} for t, x0, x1, size in specs]


def test_group_chars_into_lines_clusters_by_vertical_position():
    chars = [
        _char(100.0, 10.0, 12.0),
        _char(100.4, 20.0, 12.0),  # same line as above (within tolerance)
        _char(130.0, 10.0, 12.0),  # a new line, far below
    ]
    lines = _group_chars_into_lines(chars)
    assert len(lines) == 2
    assert len(lines[0]) == 2
    assert len(lines[1]) == 1


def test_group_chars_into_lines_returns_empty_for_no_chars():
    assert _group_chars_into_lines([]) == []


def test_add_book_context_features_computes_relative_and_edge_features():
    # Three pages, hand-built (bypassing extract_page_features/pdfplumber):
    # page 0 is a dense page near the front, page 1 a typical body page,
    # page 2 a typical body page near the back of a 5-page book.
    page_features = {
        0: {
            "line_count": 40.0, "font_size_max_ratio": 1.0, "line_density": 0.05,
            "left_margin_mean": 0.1, "left_margin_var": 0.0,
            "first_text_vpos_fraction": 0.1, "last_text_vpos_fraction": 0.9,
            "_max_font_size": 12.0, "_modal_font_size": 10.0,
        },
        1: {
            "line_count": 20.0, "font_size_max_ratio": 1.0, "line_density": 0.03,
            "left_margin_mean": 0.1, "left_margin_var": 0.0,
            "first_text_vpos_fraction": 0.1, "last_text_vpos_fraction": 0.9,
            "_max_font_size": 10.0, "_modal_font_size": 10.0,
        },
        2: {
            "line_count": 20.0, "font_size_max_ratio": 1.0, "line_density": 0.03,
            "left_margin_mean": 0.1, "left_margin_var": 0.0,
            "first_text_vpos_fraction": 0.1, "last_text_vpos_fraction": 0.9,
            "_max_font_size": 10.0, "_modal_font_size": 10.0,
        },
    }
    result = add_book_context_features(page_features, total_pages=5)
    assert set(result[0].keys()) == set(FEATURE_NAMES)
    # median line_count across the 3 pages is 20.0
    assert result[0]["line_count_rel"] == 2.0
    assert result[1]["line_count_rel"] == 1.0
    # median of _modal_font_size (10, 10, 10) is 10.0 -> page 0's max (12) / 10 = 1.2
    assert result[0]["font_size_max_ratio_book"] == 1.2
    assert result[1]["font_size_max_ratio_book"] == 1.0
    # edge_distance = min(page_index, total_pages - 1 - page_index), total_pages=5
    assert result[0]["edge_distance"] == 0.0
    assert result[1]["edge_distance"] == 1.0
    assert result[2]["edge_distance"] == 2.0


def test_add_book_context_features_handles_all_empty_pages():
    page_features = {
        0: {name: 0.0 for name in [
            "line_count", "font_size_max_ratio", "line_density",
            "left_margin_mean", "left_margin_var",
            "first_text_vpos_fraction", "last_text_vpos_fraction",
        ]} | {"_max_font_size": 0.0, "_modal_font_size": 0.0},
    }
    result = add_book_context_features(page_features, total_pages=1)
    assert result[0]["font_size_max_ratio_book"] == 1.0
    assert result[0]["edge_distance"] == 0.0


def test_reconstruct_line_text_joins_adjacent_chars_with_no_space():
    # Gap of 0.2pt on 10pt font (ratio 0.02) -- ordinary kerning within a word.
    line = _chars(("H", 0.0, 6.0, 10.0), ("i", 6.2, 9.0, 10.0))
    assert _reconstruct_line_text(line) == "Hi"


def test_reconstruct_line_text_inserts_single_space_for_normal_word_gap():
    # Gap of 3.5pt on 10pt font (ratio 0.35) -- an ordinary space between words.
    line = _chars(("A", 0.0, 6.0, 10.0), ("B", 9.5, 15.5, 10.0))
    assert _reconstruct_line_text(line) == "A B"


def test_reconstruct_line_text_inserts_double_space_for_wide_toc_gap():
    # Gap of 35pt on 10pt font (ratio 3.5) -- the title/page-number gap a
    # native PDF's absolute glyph positioning creates for a right-flush TOC
    # page number, with no literal dot-leader characters in between.
    line = _chars(("Foreword", 0.0, 40.0, 10.0), ("7", 75.0, 81.0, 10.0))
    assert _reconstruct_line_text(line) == "Foreword  7"


def test_reconstruct_line_text_output_matches_toc_line_regex():
    # Closes the loop on the actual bug: the reconstructed text for a
    # single-space-collapsed native-PDF TOC line must satisfy
    # _TOC_LINE_RE, which requires a 2+-char separator run.
    from toc_page_classifier.text_features import _TOC_LINE_RE

    line = _chars(
        ("A", 0.0, 6.0, 10.0), ("c", 6.0, 11.0, 10.0), ("k", 11.0, 16.0, 10.0),
        ("n", 16.0, 22.0, 10.0), ("o", 22.0, 28.0, 10.0), ("w", 28.0, 35.0, 10.0),
        ("l", 35.0, 38.0, 10.0), ("e", 38.0, 43.0, 10.0), ("d", 43.0, 49.0, 10.0),
        ("g", 49.0, 55.0, 10.0), ("e", 55.0, 60.0, 10.0), ("m", 60.0, 69.0, 10.0),
        ("e", 69.0, 74.0, 10.0), ("n", 74.0, 80.0, 10.0), ("t", 80.0, 84.0, 10.0),
        ("s", 84.0, 89.0, 10.0),
        ("x", 130.0, 133.0, 10.0), ("i", 133.0, 135.0, 10.0),
    )
    reconstructed = _reconstruct_line_text(line)
    assert _TOC_LINE_RE.match(reconstructed) is not None
