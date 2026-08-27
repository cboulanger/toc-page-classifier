"""Merges ground-truth label sources -- any number of external
"expected-json" evaluation corpora (see load_expected_json_corpus) plus
this repo's own DNB-located pairs -- into one weighted training table. See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Ground
truth" section for the merge rule and confidence weighting."""

import json
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CORPUS_ROOT = _REPO_ROOT / "data" / "corpus"
_DNB_GT_DIR = _CORPUS_ROOT / "pilot" / "ground-truth"
_DNB_PDF_DIR = _CORPUS_ROOT / "pilot" / "pdf"
_DNB_MANIFEST_PATH = _CORPUS_ROOT / "pilot" / "manifest.json"

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
    corpus: str  # the expected-json corpus's own directory name, or "dnb_located"
    source: str  # "expected_json" | "dnb_located"
    extraction_type: str | None = None  # "native" | "scan", from an expected-json corpus's
    # own manifest.json -- unknown (None) for dnb_located rows, whose manifest carries no
    # such field.


def _margin_to_weight(margin: float) -> float:
    clipped = min(max(margin, _MARGIN_LO), _MARGIN_HI)
    fraction = (clipped - _MARGIN_LO) / (_MARGIN_HI - _MARGIN_LO)
    return _WEIGHT_LO + fraction * (_WEIGHT_HI - _WEIGHT_LO)


def discover_corpus_dirs(root: Path) -> list[Path]:
    """A `root` directory can itself be one expected-json corpus (has
    *.expected.json files directly inside it), or a container of one or
    more separately-named corpus subdirectories (each with their own
    *.expected.json files, e.g. chapter-segmentation's own
    evaluation/corpus/{open-access,copyrighted-scans}/ layout) -- returns
    the actual corpus directories to load either way. Non-corpus
    subdirectories (no *.expected.json files of their own, e.g. this
    repo's own data/corpus/pilot/, which is the differently-shaped
    DNB-located format) are silently skipped."""
    if not root.is_dir():
        return []
    if any(root.glob("*.expected.json")):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("*.expected.json")))


def load_expected_json_corpus(corpus_dir: Path) -> list[GroundTruthRow]:
    """One row per book in `corpus_dir` that has a "<key>.expected.json"
    file with a "toc" field -- present, whether a real range or null. A
    null-toc book still contributes real training signal (an
    all-negative page-label sequence), so it is included, not skipped.
    The row's `corpus` label is `corpus_dir`'s own directory name.

    An optional manifest.json in `corpus_dir`, with a top-level "books"
    list of {"filename", "language", "extraction_type"} objects (matching
    chapter-segmentation's evaluation-corpus convention), supplies
    per-book language/extraction_type; both are left None without one."""
    rows = []
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
            corpus=corpus_dir.name,
            source="expected_json",
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


def merge_ground_truth(extra_corpus_dirs: list[Path] | None = None) -> list[GroundTruthRow]:
    """Dedups by key, preferring an expected-json corpus's hand-verified
    row over a DNB-located duplicate for the same book -- see the design
    spec's merge rule.

    Expected-json corpora come from two places: any subdirectory of this
    repo's own data/corpus/ that has the expected-json shape (auto-
    discovered via discover_corpus_dirs -- data/corpus/pilot/, the
    DNB-located format, doesn't and is skipped), plus every directory
    passed in `extra_corpus_dirs` (each resolved via discover_corpus_dirs
    too, so passing either a single corpus or a root of several named
    ones both work) -- e.g. a sibling chapter-segmentation checkout's
    evaluation/corpus/ directory, passed explicitly by the caller. This
    repo makes no assumption about any particular sibling project being
    present."""
    expected_json_dirs = discover_corpus_dirs(_CORPUS_ROOT)
    for extra_dir in extra_corpus_dirs or []:
        expected_json_dirs.extend(discover_corpus_dirs(extra_dir))

    expected_json_rows = [row for corpus_dir in expected_json_dirs for row in load_expected_json_corpus(corpus_dir)]
    dnb_rows = load_dnb_located_rows()
    by_key = {row.key: row for row in dnb_rows}
    for row in expected_json_rows:
        by_key[row.key] = row
    return list(by_key.values())
