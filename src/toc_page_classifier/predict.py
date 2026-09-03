"""Deployable inference: given a PDF path, predicts which pages are its
table of contents. Uses the model bundle `cli/train_final_model.py`
fits on the full ground truth and serializes to data/model.pkl (bundled
as package data, shipped alongside this module)."""

import pickle
from pathlib import Path

from .layout_features import (
    add_book_context_features,
    extract_page_features_and_texts,
    split_into_runs,
)
from .range_selection import select_topk_ranges
from .text_features import extract_text_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "model.pkl"

# A real TOC is essentially never buried in a long book's interior, so
# locate_toc_pages only converts the first _DEFAULT_HEAD_PAGES and last
# _DEFAULT_TAIL_PAGES pages by default -- skipping the interior keeps both
# pdfalto's work and the ALTO document read back from it proportional to
# the pages that can matter (see extract_page_features_and_texts).
_DEFAULT_HEAD_PAGES = 30
_DEFAULT_TAIL_PAGES = 20


def _score_pages(
    features_per_page: dict[int, dict[str, float]],
    bundle: dict,
    page_indices: list[int] | None = None,
) -> list[float]:
    """Pure scoring step -- split out from locate_toc_pages so it's
    testable without a real PDF/pdfalto. `features_per_page` must have
    every name in bundle["feature_names"] for each index in `page_indices`
    (every page 0..n-1 if not given)."""
    feature_names = bundle["feature_names"]
    indices = page_indices if page_indices is not None else range(len(features_per_page))
    X = [[features_per_page[i][name] for name in feature_names] for i in indices]
    return [p[1] for p in bundle["model"].predict_proba(bundle["scaler"].transform(X))]


def locate_toc_pages(
    pdf_path: str | Path,
    language: str | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    top_k: int = 3,
    head_pages: int | None = _DEFAULT_HEAD_PAGES,
    tail_pages: int | None = _DEFAULT_TAIL_PAGES,
) -> list[int]:
    """Returns the predicted 0-based TOC page indices for `pdf_path`: the
    highest-ranked candidate range among select_topk_ranges' top `top_k`,
    for the model bundled at `model_path`. Returns [] if the PDF has no
    extractable pages. `language` is the book's declared language code
    (e.g. "en"), used only for one text feature's same-language keyword
    match -- pass None if unknown.

    Only the first `head_pages` and last `tail_pages` pages are actually
    converted (pass None/None to scan every page instead) -- see
    extract_page_features_and_texts for why this is where a long book's
    processing time goes, and why skipping its interior is safe."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    page_features, gap_aware_texts, total_pages = extract_page_features_and_texts(
        pdf_path, head_pages=head_pages, tail_pages=tail_pages
    )
    if total_pages == 0:
        return []
    layout = add_book_context_features(page_features, total_pages)
    scanned_pages = sorted(gap_aware_texts)
    pages = [gap_aware_texts[i] for i in scanned_pages]
    text = extract_text_features(pages, language=language)
    features_per_page = {
        page_index: {**layout[page_index], **text[local_index]}
        for local_index, page_index in enumerate(scanned_pages)
    }

    candidates: list[tuple[int, int, float]] = []
    for run in split_into_runs(scanned_pages):
        scores = _score_pages(features_per_page, bundle, page_indices=run)
        for start, end, score in select_topk_ranges(scores, k=top_k):
            candidates.append((run[start], run[end], score))
    if not candidates:
        return []
    # Same tie-break as select_topk_ranges (widest window first on a
    # literal score tie) -- each run's own candidates are already
    # non-overlapping and sorted this way; merging is safe since pages in
    # different runs never overlap.
    candidates.sort(key=lambda c: (c[2], c[1] - c[0]), reverse=True)
    start, end, _ = candidates[0]
    return list(range(start, end + 1))
