from toc_page_classifier.layout_features import (
    FEATURE_NAMES,
    _group_chars_into_lines,
    add_book_context_features,
)


def _char(top: float, x0: float, size: float) -> dict:
    return {"top": top, "bottom": top + size, "x0": x0, "size": size}


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
