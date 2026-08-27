"""Deployable inference: given a PDF path, predicts which pages are its
table of contents. Uses the model bundle `cli/train_final_model.py`
fits on the full ground truth and serializes to data/model.pkl (bundled
as package data, shipped alongside this module)."""

import pickle
from pathlib import Path

from .layout_features import add_book_context_features, extract_page_features_and_texts
from .range_selection import select_topk_ranges
from .text_features import extract_text_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "model.pkl"


def _score_pages(features_per_page: dict[int, dict[str, float]], bundle: dict) -> list[float]:
    """Pure scoring step -- split out from locate_toc_pages so it's
    testable without a real PDF/pdfplumber. `features_per_page` must have
    every name in bundle["feature_names"] for each page index 0..n-1."""
    feature_names = bundle["feature_names"]
    total_pages = len(features_per_page)
    X = [[features_per_page[i][name] for name in feature_names] for i in range(total_pages)]
    return [p[1] for p in bundle["model"].predict_proba(bundle["scaler"].transform(X))]


def locate_toc_pages(
    pdf_path: str | Path,
    language: str | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    top_k: int = 3,
) -> list[int]:
    """Returns the predicted 0-based TOC page indices for `pdf_path`: the
    highest-ranked candidate range among select_topk_ranges' top `top_k`,
    for the model bundled at `model_path`. Returns [] if the PDF has no
    extractable pages. `language` is the book's declared language code
    (e.g. "en"), used only for one text feature's same-language keyword
    match -- pass None if unknown."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    page_features, gap_aware_texts = extract_page_features_and_texts(pdf_path)
    total_pages = len(gap_aware_texts)
    if total_pages == 0:
        return []
    layout = add_book_context_features(page_features, total_pages)
    pages = [gap_aware_texts[i] for i in range(total_pages)]
    text = extract_text_features(pages, language=language)
    features_per_page = {i: {**layout[i], **text[i]} for i in range(total_pages)}

    scores = _score_pages(features_per_page, bundle)
    ranges = select_topk_ranges(scores, k=top_k)
    if not ranges:
        return []
    start, end, _ = ranges[0]
    return list(range(start, end + 1))
