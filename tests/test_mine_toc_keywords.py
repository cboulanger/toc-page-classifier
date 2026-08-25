from cli.mine_toc_keywords import mine_candidates
from toc_page_classifier.ground_truth import GroundTruthRow


def _row(key, pdf_path, toc_start, language):
    return GroundTruthRow(
        key=key, pdf_path=pdf_path, toc_start_index=toc_start, toc_end_index=toc_start,
        weight=1.0, language=language, corpus="test", source="test",
    )


def test_mine_candidates_counts_first_line_per_language(tmp_path):
    # mine_candidates takes a page-text loader so it never touches a real
    # PDF in this test -- see its signature.
    pages_by_key = {
        "a": ["Inhalt\n\nKapitel 1 .... 5\n"],
        "b": ["INHALT\n\nKapitel 1 .... 5\n"],
        "c": ["Contents\n\nChapter 1 .... 5\n"],
    }
    rows = [
        _row("a", tmp_path / "a.pdf", 0, "de"),
        _row("b", tmp_path / "b.pdf", 0, "de"),
        _row("c", tmp_path / "c.pdf", 0, "en"),
    ]
    candidates = mine_candidates(rows, load_pages=lambda row: pages_by_key[row.key])
    assert candidates["de"]["inhalt"] == 2
    assert candidates["en"]["contents"] == 1


def test_mine_candidates_skips_rows_with_no_toc_or_no_language(tmp_path):
    rows = [
        _row("a", tmp_path / "a.pdf", None, "de"),  # no TOC
        _row("b", tmp_path / "b.pdf", 0, None),  # no language
    ]
    candidates = mine_candidates(rows, load_pages=lambda row: ["Inhalt\n"])
    assert candidates == {}
