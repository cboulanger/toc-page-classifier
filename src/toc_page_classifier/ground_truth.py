"""Merges the two ground-truth label sources -- chapter-segmentation's
hand-verified 89-book evaluation corpus and this repo's own 95 DNB-located
pairs -- into one weighted training table. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Ground
truth" section for the merge rule and confidence weighting."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHAPTER_SEGMENTATION_DIR = Path(
    os.environ.get("CHAPTER_SEGMENTATION_DIR", str(_REPO_ROOT.parent / "chapter-segmentation"))
)
_DNB_GT_DIR = _REPO_ROOT / "data" / "corpus" / "pilot" / "ground-truth"
_DNB_PDF_DIR = _REPO_ROOT / "data" / "corpus" / "pilot" / "pdf"
_DNB_MANIFEST_PATH = _REPO_ROOT / "data" / "corpus" / "pilot" / "manifest.json"

_CHAPTER_SEGMENTATION_CORPORA = ["open-access", "copyrighted-scans"]

# Empirical range covering the bulk of the DNB-located corpus's margin
# distribution (95 books: min 0.0006, median 0.43, max 0.71) -- maps onto
# the [0.3, 1.0] sample-weight range below. A margin at or above
# _MARGIN_HI gets the full 1.0 weight; even the least-confident matches
# near 0.0 only get discounted to 0.3, never to zero -- an auto-located
# label is still real evidence, just weaker.
_MARGIN_LO, _MARGIN_HI = 0.0, 0.5
_WEIGHT_LO, _WEIGHT_HI = 0.3, 1.0


@dataclass
class GroundTruthRow:
    key: str
    pdf_path: Path
    toc_start_index: int | None
    toc_end_index: int | None
    weight: float
    language: str | None
    corpus: str  # "open-access" | "copyrighted-scans" | "dnb_located"
    source: str  # "chapter_segmentation" | "dnb_located"
    extraction_type: str | None = None  # "native" | "scan", from chapter-segmentation's
    # manifest.json -- unknown (None) for dnb_located rows, whose manifest carries no
    # such field.


def _margin_to_weight(margin: float) -> float:
    clipped = min(max(margin, _MARGIN_LO), _MARGIN_HI)
    fraction = (clipped - _MARGIN_LO) / (_MARGIN_HI - _MARGIN_LO)
    return _WEIGHT_LO + fraction * (_WEIGHT_HI - _WEIGHT_LO)


def load_chapter_segmentation_rows(chapter_segmentation_dir: Path) -> list[GroundTruthRow]:
    """One row per book in chapter-segmentation's evaluation corpus that has
    a retrofitted "toc" field -- present, whether a real range or null. A
    null-toc book still contributes real training signal (an all-negative
    page-label sequence), so it is included, not skipped."""
    rows = []
    corpus_root = chapter_segmentation_dir / "evaluation" / "corpus"
    for corpus in _CHAPTER_SEGMENTATION_CORPORA:
        corpus_dir = corpus_root / corpus
        manifest_path = corpus_dir / "manifest.json"
        languages: dict[str, str | None] = {}
        extraction_types: dict[str, str | None] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            languages = {
                Path(b["filename"]).stem: b.get("language") for b in manifest["books"]
            }
            extraction_types = {
                Path(b["filename"]).stem: b.get("extraction_type") for b in manifest["books"]
            }
        for expected_path in sorted(corpus_dir.glob("*.expected.json")):
            key = expected_path.name.removesuffix(".expected.json")
            pdf_path = corpus_dir / f"{key}.pdf"
            if not pdf_path.exists():
                continue
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            if "toc" not in expected:
                continue
            toc = expected["toc"]
            rows.append(GroundTruthRow(
                key=key,
                pdf_path=pdf_path,
                toc_start_index=toc["toc_start_index"] if toc else None,
                toc_end_index=toc["toc_end_index"] if toc else None,
                weight=1.0,
                language=languages.get(key),
                corpus=corpus,
                source="chapter_segmentation",
                extraction_type=extraction_types.get(key),
            ))
    return rows


def load_dnb_located_rows() -> list[GroundTruthRow]:
    """One row per DNB-located pair with status "located" -- the other
    statuses (reference_has_no_text, error, no_candidate) have no usable
    range."""
    languages: dict[str, str | None] = {}
    if _DNB_MANIFEST_PATH.exists():
        manifest = json.loads(_DNB_MANIFEST_PATH.read_text(encoding="utf-8"))
        languages = {b["isbn"]: b.get("language") for b in manifest["books"]}

    rows = []
    for gt_path in sorted(_DNB_GT_DIR.glob("*.json")):
        entry = json.loads(gt_path.read_text(encoding="utf-8"))
        if entry.get("status") != "located":
            continue
        isbn = entry["isbn"]
        pdf_path = _DNB_PDF_DIR / f"{isbn}.fulltext.pdf"
        if not pdf_path.exists():
            continue
        rows.append(GroundTruthRow(
            key=isbn,
            pdf_path=pdf_path,
            toc_start_index=entry["toc_start_index"],
            toc_end_index=entry["toc_end_index"],
            weight=_margin_to_weight(entry["margin"]),
            language=languages.get(isbn),
            corpus="dnb_located",
            source="dnb_located",
        ))
    return rows


def merge_ground_truth(chapter_segmentation_dir: Path | None = None) -> list[GroundTruthRow]:
    """Dedups by key, preferring chapter-segmentation's hand-verified row
    over a DNB-located duplicate for the same book -- see the design
    spec's merge rule."""
    chapter_segmentation_rows = load_chapter_segmentation_rows(
        chapter_segmentation_dir or _DEFAULT_CHAPTER_SEGMENTATION_DIR
    )
    dnb_rows = load_dnb_located_rows()
    by_key = {row.key: row for row in dnb_rows}
    for row in chapter_segmentation_rows:
        by_key[row.key] = row
    return list(by_key.values())
