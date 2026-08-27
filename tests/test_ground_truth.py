import json
from pathlib import Path

import pytest

from toc_page_classifier.ground_truth import (
    _margin_to_weight,
    discover_corpus_dirs,
    load_dnb_located_rows,
    load_expected_json_corpus,
    merge_ground_truth,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def expected_json_corpus_dir(tmp_path):
    """A single expected-json corpus directory (files directly inside,
    no nesting) -- e.g. what a caller would pass via --corpus-dir when
    pointing at one specific corpus."""
    corpus = tmp_path / "open-access"
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
    return corpus


@pytest.fixture
def multi_corpus_root(tmp_path):
    """A root containing two separately-named corpus subdirectories --
    e.g. chapter-segmentation's own evaluation/corpus/ layout, or any
    other project's equivalent."""
    root = tmp_path / "corpus-root"
    for name, key in [("open-access", "5555555555"), ("copyrighted-scans", "6666666666")]:
        corpus = root / name
        _write_json(corpus / f"{key}.expected.json", {"toc": {"toc_start_index": 1, "toc_end_index": 1}})
        (corpus / f"{key}.pdf").write_bytes(b"%PDF-1.4 fake")
    # a subdirectory with no *.expected.json files -- not a corpus, must be skipped
    (root / "not-a-corpus").mkdir()
    (root / "not-a-corpus" / "notes.txt").write_text("irrelevant")
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


@pytest.fixture
def no_auto_discovered_corpora(tmp_path, monkeypatch):
    """Points _CORPUS_ROOT at an empty directory so merge_ground_truth's
    auto-discovery of this repo's own data/corpus/ doesn't pull in the
    real corpus while a test is only exercising extra_corpus_dirs."""
    import toc_page_classifier.ground_truth as gt_module
    monkeypatch.setattr(gt_module, "_CORPUS_ROOT", tmp_path / "empty-corpus-root")


def test_margin_to_weight_clips_into_expected_range():
    assert _margin_to_weight(0.0) == pytest.approx(0.3)
    assert _margin_to_weight(0.5) == pytest.approx(1.0)
    assert _margin_to_weight(10.0) == pytest.approx(1.0)  # clipped, never > 1.0
    assert _margin_to_weight(-1.0) == pytest.approx(0.3)  # clipped, never < 0.3


def test_discover_corpus_dirs_returns_root_itself_when_it_is_one_corpus(expected_json_corpus_dir):
    assert discover_corpus_dirs(expected_json_corpus_dir) == [expected_json_corpus_dir]


def test_discover_corpus_dirs_returns_named_subdirs_skipping_non_corpus_ones(multi_corpus_root):
    found = discover_corpus_dirs(multi_corpus_root)
    assert found == [multi_corpus_root / "copyrighted-scans", multi_corpus_root / "open-access"]


def test_discover_corpus_dirs_returns_empty_for_missing_root(tmp_path):
    assert discover_corpus_dirs(tmp_path / "nonexistent") == []


def test_load_expected_json_corpus_includes_null_toc_and_excludes_missing_key(expected_json_corpus_dir):
    rows = load_expected_json_corpus(expected_json_corpus_dir)
    keys = {r.key for r in rows}
    assert keys == {"1111111111", "2222222222"}  # "3333333333" has no "toc" key
    by_key = {r.key: r for r in rows}
    assert by_key["1111111111"].toc_start_index == 3
    assert by_key["1111111111"].weight == 1.0
    assert by_key["1111111111"].language == "en"
    assert by_key["1111111111"].extraction_type == "native"
    assert by_key["1111111111"].corpus == expected_json_corpus_dir.name
    assert by_key["2222222222"].toc_start_index is None  # confirmed no TOC
    assert by_key["2222222222"].extraction_type == "scan"


def test_load_dnb_located_rows_excludes_non_located_and_maps_margin_to_weight(dnb_dirs):
    rows = load_dnb_located_rows()
    keys = {r.key for r in rows}
    assert keys == {"1111111111", "4444444444"}  # "5555555555" is status=error
    by_key = {r.key: r for r in rows}
    assert by_key["1111111111"].weight == pytest.approx(1.0)  # margin 0.5
    assert by_key["4444444444"].weight == pytest.approx(0.3)  # margin 0.0
    assert by_key["4444444444"].language == "fr"


def test_merge_ground_truth_prefers_expected_json_on_key_collision(
    expected_json_corpus_dir, dnb_dirs, no_auto_discovered_corpora
):
    rows = merge_ground_truth([expected_json_corpus_dir])
    by_key = {r.key: r for r in rows}
    # "1111111111" exists in both sources -- the expected-json row wins
    assert by_key["1111111111"].source == "expected_json"
    assert by_key["1111111111"].toc_start_index == 3
    # "4444444444" only exists in the DNB source
    assert by_key["4444444444"].source == "dnb_located"
    assert {r.key for r in rows} == {"1111111111", "2222222222", "4444444444"}


def test_merge_ground_truth_accepts_a_multi_corpus_root(multi_corpus_root, dnb_dirs, no_auto_discovered_corpora):
    rows = merge_ground_truth([multi_corpus_root])
    keys = {r.key for r in rows}
    assert {"5555555555", "6666666666"} <= keys


def test_merge_ground_truth_auto_discovers_data_corpus(tmp_path, dnb_dirs, monkeypatch):
    import toc_page_classifier.ground_truth as gt_module

    auto_root = tmp_path / "data-corpus"
    corpus = auto_root / "auto-discovered"
    _write_json(corpus / "7777777777.expected.json", {"toc": {"toc_start_index": 2, "toc_end_index": 2}})
    (corpus / "7777777777.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(gt_module, "_CORPUS_ROOT", auto_root)

    rows = merge_ground_truth()
    by_key = {r.key: r for r in rows}
    assert by_key["7777777777"].corpus == "auto-discovered"
