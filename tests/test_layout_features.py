import xml.etree.ElementTree as ET

import pytest

from toc_page_classifier.layout_features import (
    FEATURE_NAMES,
    _merge_into_visual_lines,
    _reconstruct_line_text,
    _text_lines,
    add_book_context_features,
    extract_page_features_and_texts,
    split_into_runs,
)

_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v3#"


def _line_xml(vpos: float, words, height: float = 10.0, style: str = "font0") -> str:
    """One ALTO <TextLine>. `words` is a list of (content, hpos, width)."""
    strings = "".join(
        f'<String ID="w{i}" CONTENT="{content}" HPOS="{hpos}" VPOS="{vpos}" '
        f'WIDTH="{width}" HEIGHT="{height}" STYLEREFS="{style}"/>'
        for i, (content, hpos, width) in enumerate(words)
    )
    hpos = min(word[1] for word in words)
    return f'<TextLine ID="t{vpos}_{hpos}" HPOS="{hpos}" VPOS="{vpos}" HEIGHT="{height}">{strings}</TextLine>'


def _page_xml(number: int, lines_xml: str, width: float = 100.0, height: float = 100.0) -> str:
    """One ALTO <Page>. Each <TextLine> gets its own <TextBlock>, which is
    what pdfalto does whenever a wide gap separates two runs of words."""
    blocks = "".join(f"<TextBlock>{line}</TextBlock>" for line in _split_lines(lines_xml))
    return (
        f'<Page ID="Page{number}" PHYSICAL_IMG_NR="{number}" WIDTH="{width}" '
        f'HEIGHT="{height}"><PrintSpace>{blocks}</PrintSpace></Page>'
    )


def _split_lines(lines_xml: str) -> list[str]:
    parts = [part for part in lines_xml.split("<TextLine") if part]
    return [f"<TextLine{part}" for part in parts]


def _alto(pages_xml: str, font_size: float = 10.0) -> ET.Element:
    return ET.fromstring(
        f'<alto xmlns="{_ALTO_NS}">'
        f'<Styles><TextStyle ID="font0" FONTSIZE="{font_size}"/></Styles>'
        f"<Layout>{pages_xml}</Layout></alto>"
    )


def _words(*specs):
    """specs: (text, x0, x1, size) tuples, in left-to-right order."""
    return [
        {"text": text, "x0": x0, "x1": x1, "top": 0.0, "bottom": size, "size": size}
        for text, x0, x1, size in specs
    ]


def test_split_into_runs_groups_consecutive_indices():
    assert split_into_runs([0, 1, 2, 8, 9, 15]) == [[0, 1, 2], [8, 9], [15]]


def test_split_into_runs_handles_empty_input():
    assert split_into_runs([]) == []


def test_merge_into_visual_lines_joins_text_lines_at_the_same_vertical_position():
    # The case that forces the merge to exist: pdfalto puts a TOC entry's
    # title and its right-flush page number in two <TextBlock>s, so they
    # arrive as two <TextLine>s sharing a VPOS.
    root = _alto(
        _page_xml(
            1,
            _line_xml(100.0, [("Foreword", 10.0, 40.0)])
            + _line_xml(100.4, [("7", 80.0, 6.0)])  # same line, within tolerance
            + _line_xml(130.0, [("Preface", 10.0, 35.0)]),  # a new line, far below
        )
    )
    page = root.find(".//{*}Page")
    lines = _merge_into_visual_lines(_text_lines(page, {"font0": 10.0}))
    assert len(lines) == 2
    assert [word["text"] for word in lines[0]] == ["Foreword", "7"]
    assert [word["text"] for word in lines[1]] == ["Preface"]


def test_merge_into_visual_lines_returns_empty_for_no_lines():
    assert _merge_into_visual_lines([]) == []


def test_text_lines_skips_lines_with_no_content():
    root = _alto(
        _page_xml(1, _line_xml(100.0, [("", 10.0, 0.0)]) + _line_xml(130.0, [("Real", 10.0, 20.0)]))
    )
    lines = _text_lines(root.find(".//{*}Page"), {"font0": 10.0})
    assert [line["words"][0]["text"] for line in lines] == ["Real"]


def test_add_book_context_features_computes_relative_and_edge_features():
    # Three pages, hand-built (bypassing extract_page_features_and_texts/pdfalto):
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


def test_reconstruct_line_text_joins_split_token_halves_with_no_space():
    # Gap of 0.2pt on 10pt font (ratio 0.02) -- the two halves of a
    # hyphenated compound, which pdfalto emits as two <String>s.
    line = _words(("High-", 0.0, 26.0, 10.0), ("Tech", 26.2, 48.0, 10.0))
    assert _reconstruct_line_text(line) == "High-Tech"


def test_reconstruct_line_text_inserts_single_space_for_normal_word_gap():
    # Gap of 1.8pt on 10pt font (ratio 0.18) -- an ordinary space between
    # two ALTO word boxes.
    line = _words(("A", 0.0, 6.0, 10.0), ("B", 7.8, 13.8, 10.0))
    assert _reconstruct_line_text(line) == "A B"


def test_reconstruct_line_text_inserts_double_space_for_wide_toc_gap():
    # Gap of 35pt on 10pt font (ratio 3.5) -- the title/page-number gap a
    # native PDF's absolute glyph positioning creates for a right-flush TOC
    # page number, with no literal dot-leader characters in between.
    line = _words(("Foreword", 0.0, 40.0, 10.0), ("7", 75.0, 81.0, 10.0))
    assert _reconstruct_line_text(line) == "Foreword  7"


def test_reconstruct_line_text_output_matches_toc_line_regex():
    # Closes the loop on the actual bug: the reconstructed text for a
    # single-space-collapsed native-PDF TOC line must satisfy
    # _TOC_LINE_RE, which requires a 2+-char separator run.
    from toc_page_classifier.text_features import _TOC_LINE_RE

    line = _words(("Acknowledgements", 0.0, 89.0, 10.0), ("xi", 130.0, 135.0, 10.0))
    assert _TOC_LINE_RE.match(_reconstruct_line_text(line)) is not None


class _FakeConverter:
    """Stands in for _convert_to_alto, recording every page range pdfalto
    was actually asked for."""

    def __init__(self, total_pages: int):
        self.total_pages = total_pages
        self.ranges: list[tuple[int, int]] = []

    def __call__(self, pdf_path, first_page, last_page):
        self.ranges.append((first_page, last_page))
        return _alto(
            "".join(
                _page_xml(number, _line_xml(20.0, [("word", 10.0, 20.0)]))
                for number in range(first_page, last_page + 1)
            )
        )


@pytest.fixture
def fake_pdfalto(monkeypatch):
    def install(total_pages: int) -> _FakeConverter:
        from toc_page_classifier import layout_features

        converter = _FakeConverter(total_pages)
        monkeypatch.setattr(layout_features, "page_count", lambda path: total_pages)
        monkeypatch.setattr(layout_features, "_convert_to_alto", converter)
        return converter

    return install


def test_extract_page_features_and_texts_converts_only_head_and_tail_pages(fake_pdfalto):
    converter = fake_pdfalto(10)

    features, texts, total_pages = extract_page_features_and_texts(
        "irrelevant.pdf", head_pages=2, tail_pages=2
    )

    assert total_pages == 10
    assert set(features.keys()) == {0, 1, 8, 9}
    assert set(texts.keys()) == {0, 1, 8, 9}
    # The interior pages must never have been sent to pdfalto at all --
    # converting them, and reading the ALTO back, is the cost being
    # skipped. The two windows are two separate invocations, in 1-based
    # inclusive page numbers.
    assert converter.ranges == [(1, 2), (9, 10)]


def test_extract_page_features_and_texts_converts_every_page_when_head_and_tail_cover_book(fake_pdfalto):
    converter = fake_pdfalto(4)

    features, _texts, total_pages = extract_page_features_and_texts(
        "irrelevant.pdf", head_pages=3, tail_pages=3
    )

    assert total_pages == 4
    assert set(features.keys()) == {0, 1, 2, 3}
    assert converter.ranges == [(1, 4)]


def test_extract_page_features_and_texts_converts_every_page_by_default(fake_pdfalto):
    converter = fake_pdfalto(5)

    features, _texts, total_pages = extract_page_features_and_texts("irrelevant.pdf")

    assert total_pages == 5
    assert set(features.keys()) == {0, 1, 2, 3, 4}
    assert converter.ranges == [(1, 5)]


def test_extract_page_features_and_texts_returns_nothing_for_an_empty_pdf(fake_pdfalto):
    converter = fake_pdfalto(0)

    assert extract_page_features_and_texts("irrelevant.pdf") == ({}, {}, 0)
    assert converter.ranges == []


def test_extract_page_features_and_texts_treats_a_page_pdfalto_skipped_as_blank(monkeypatch):
    """pypdf counts the pages; if pdfalto disagrees and emits fewer, every
    page the caller asked for still gets an entry."""
    from toc_page_classifier import layout_features

    monkeypatch.setattr(layout_features, "page_count", lambda path: 3)
    monkeypatch.setattr(
        layout_features,
        "_convert_to_alto",
        # pdfalto returns pages 1 and 3 only -- page 2 is missing.
        lambda pdf_path, first_page, last_page: _alto(
            _page_xml(1, _line_xml(20.0, [("word", 10.0, 20.0)]))
            + _page_xml(3, _line_xml(20.0, [("word", 10.0, 20.0)]))
        ),
    )

    features, texts, total_pages = extract_page_features_and_texts("irrelevant.pdf")

    assert total_pages == 3
    assert set(features.keys()) == {0, 1, 2}
    assert features[1]["line_count"] == 0.0
    assert texts[1] == ""


def test_extract_page_features_and_texts_indexes_pages_by_their_document_page_number(fake_pdfalto):
    """ALTO's PHYSICAL_IMG_NR is the page's 1-based number in the whole
    document, not its position inside the converted range -- the tail
    window's pages must not land back at index 0."""
    fake_pdfalto(10)

    _features, texts, _total = extract_page_features_and_texts(
        "irrelevant.pdf", head_pages=None, tail_pages=3
    )

    assert sorted(texts) == [7, 8, 9]
