import json
from pathlib import Path

import pytest

from toc_page_classifier.ground_truth import (
    _margin_to_weight,
    load_chapter_segmentation_rows,
    load_dnb_located_rows,
    merge_ground_truth,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def chapter_segmentation_dir(tmp_path):
    root = tmp_path / "chapter-segmentation"
    corpus = root / "evaluation" / "corpus" / "open-access"
    _write_json(corpus / "manifest.json", {"books": [
        {"filename": "1111111111.pdf", "language": "en", "extraction_type": "native"},
        {"filename": "2222222222.pdf", "language": "de", "extraction_type": "scan"},
    ]})
    _write_json(corpus / "1111111111.expected.json", {"toc": {"toc_start_index": 3, "toc_end_index": 4}})
    (corpus / "1111111111.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_json(corpus / "2222222222.expected.json", {"toc": None})
    (corpus / "2222222222.pdf").write_bytes(b"%PDF-1.4 fake")
    # no "toc" key at all -- must be excluded
    _write_json(corpus / "3333333333.expected.json", {})
    (corpus / "3333333333.pdf").write_bytes(b"%PDF-1.4 fake")
    return root


@pytest.fixture
def dnb_dirs(tmp_path, monkeypatch):
    gt_dir = tmp_path / "dnb" / "ground-truth"
    pdf_dir = tmp_path / "dnb" / "pdf"
    gt_dir.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)
    _write_json(gt_dir / "1111111111.json", {
        "isbn": "1111111111", "status": "located",
        "toc_start_index": 7, "toc_end_index": 8, "margin": 0.5,
    })
    (pdf_dir / "1111111111.fulltext.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_json(gt_dir / "4444444444.json", {
        "isbn": "4444444444", "status": "located",
        "toc_start_index": 1, "toc_end_index": 2, "margin": 0.0,
    })
    (pdf_dir / "4444444444.fulltext.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_json(gt_dir / "5555555555.json", {"isbn": "5555555555", "status": "error"})
    manifest_path = tmp_path / "dnb" / "manifest.json"
    _write_json(manifest_path, {"books": [{"isbn": "4444444444", "language": "fr"}]})

    import toc_page_classifier.ground_truth as gt_module
    monkeypatch.setattr(gt_module, "_DNB_GT_DIR", gt_dir)
    monkeypatch.setattr(gt_module, "_DNB_PDF_DIR", pdf_dir)
    monkeypatch.setattr(gt_module, "_DNB_MANIFEST_PATH", manifest_path)
    return gt_dir, pdf_dir


def test_margin_to_weight_clips_into_expected_range():
    assert _margin_to_weight(0.0) == pytest.approx(0.3)
    assert _margin_to_weight(0.5) == pytest.approx(1.0)
    assert _margin_to_weight(10.0) == pytest.approx(1.0)  # clipped, never > 1.0
    assert _margin_to_weight(-1.0) == pytest.approx(0.3)  # clipped, never < 0.3


def test_load_chapter_segmentation_rows_includes_null_toc_and_excludes_missing_key(chapter_segmentation_dir):
    rows = load_chapter_segmentation_rows(chapter_segmentation_dir)
    keys = {r.key for r in rows}
    assert keys == {"1111111111", "2222222222"}  # "3333333333" has no "toc" key
    by_key = {r.key: r for r in rows}
    assert by_key["1111111111"].toc_start_index == 3
    assert by_key["1111111111"].weight == 1.0
    assert by_key["1111111111"].language == "en"
    assert by_key["1111111111"].extraction_type == "native"
    assert by_key["2222222222"].toc_start_index is None  # confirmed no TOC
    assert by_key["2222222222"].extraction_type == "scan"


def test_load_chapter_segmentation_rows_raises_when_corpus_dir_missing():
    with pytest.raises(FileNotFoundError):
        load_chapter_segmentation_rows(Path("/some/nonexistent/path"))


def test_load_dnb_located_rows_excludes_non_located_and_maps_margin_to_weight(dnb_dirs):
    rows = load_dnb_located_rows()
    keys = {r.key for r in rows}
    assert keys == {"1111111111", "4444444444"}  # "5555555555" is status=error
    by_key = {r.key: r for r in rows}
    assert by_key["1111111111"].weight == pytest.approx(1.0)  # margin 0.5
    assert by_key["4444444444"].weight == pytest.approx(0.3)  # margin 0.0
    assert by_key["4444444444"].language == "fr"


def test_merge_ground_truth_prefers_chapter_segmentation_on_isbn_collision(chapter_segmentation_dir, dnb_dirs):
    rows = merge_ground_truth(chapter_segmentation_dir)
    by_key = {r.key: r for r in rows}
    # "1111111111" exists in both sources -- chapter_segmentation's row wins
    assert by_key["1111111111"].source == "chapter_segmentation"
    assert by_key["1111111111"].toc_start_index == 3
    # "4444444444" only exists in the DNB source
    assert by_key["4444444444"].source == "dnb_located"
    assert {r.key for r in rows} == {"1111111111", "2222222222", "4444444444"}
