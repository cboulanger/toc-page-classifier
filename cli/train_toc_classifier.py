#!/usr/bin/env python3
"""Leave-one-book-out evaluation of the TOC-page classifier: per-page
layout + text features, a page-level scorer (LogisticRegression or
gradient boosting), and top-K non-overlapping candidate page-range output.
See docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md.

    uv run python cli/train_toc_classifier.py
    uv run python cli/train_toc_classifier.py --model gradient_boosting
    uv run python cli/train_toc_classifier.py --chapter-segmentation-dir ../chapter-segmentation
"""

import argparse
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from toc_page_classifier.ground_truth import GroundTruthRow, merge_ground_truth
from toc_page_classifier.layout_features import (
    FEATURE_NAMES as LAYOUT_FEATURE_NAMES,
    add_book_context_features,
    extract_gap_aware_page_texts,
    extract_page_features,
)
from toc_page_classifier.range_selection import select_topk_ranges
from toc_page_classifier.text_features import TEXT_FEATURE_NAMES, extract_text_features

ALL_FEATURE_NAMES = LAYOUT_FEATURE_NAMES + TEXT_FEATURE_NAMES
_TOP_K = 3


def build_feature_table(rows: list[GroundTruthRow]) -> list[dict]:
    """One row per page: {"book_key", "corpus", "extraction_type",
    "page_index", "features": {...}, "label": bool, "weight": float}."""
    table = []
    for row in rows:
        gap_aware_texts = extract_gap_aware_page_texts(row.pdf_path)
        total_pages = len(gap_aware_texts)
        if total_pages == 0:
            continue
        layout = add_book_context_features(extract_page_features(row.pdf_path), total_pages)
        pages = [gap_aware_texts[i] for i in range(total_pages)]
        text = extract_text_features(pages, language=row.language)
        toc_indices = (
            set(range(row.toc_start_index, row.toc_end_index + 1))
            if row.toc_start_index is not None
            else set()
        )
        dropped_pages = []
        for page_index in range(total_pages):
            if page_index not in layout:
                dropped_pages.append(page_index)
                continue  # pdfplumber found no page geometry (should not normally happen)
            features = {**layout[page_index], **text[page_index]}
            table.append({
                "book_key": row.key,
                "corpus": row.corpus,
                "extraction_type": row.extraction_type,
                "page_index": page_index,
                "features": features,
                "label": page_index in toc_indices,
                "weight": row.weight,
            })
        if dropped_pages:
            print(
                f"WARNING: {row.key}: pdfplumber found no page geometry for "
                f"{len(dropped_pages)}/{total_pages} page(s) {dropped_pages} -- "
                f"dropped from the feature table (pypdf and pdfplumber disagreed "
                f"on page count/content, or the page has no extractable geometry)."
            )
    return table


def _make_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier()
    raise ValueError(f"unknown model {name!r}")


def evaluate_leave_one_book_out(table: list[dict], model_name: str) -> dict:
    """Runs LOBO over every book with at least one true TOC page (a book
    with none has nothing to rank a hit against). Returns per-book
    top1/top3 hit flags plus each book's corpus and extraction_type, for
    breakdown reporting."""
    book_keys = sorted({r["book_key"] for r in table})
    per_book = []
    for held_out in book_keys:
        test_rows = [r for r in table if r["book_key"] == held_out]
        true_indices = {i for i, r in enumerate(test_rows) if r["label"]}
        if not true_indices:
            continue

        train_rows = [r for r in table if r["book_key"] != held_out]
        X_train = [[r["features"][name] for name in ALL_FEATURE_NAMES] for r in train_rows]
        y_train = [r["label"] for r in train_rows]
        sample_weight = [r["weight"] for r in train_rows]
        X_test = [[r["features"][name] for name in ALL_FEATURE_NAMES] for r in test_rows]

        scaler = StandardScaler().fit(X_train)
        model = _make_model(model_name)
        model.fit(scaler.transform(X_train), y_train, sample_weight=sample_weight)
        scores = [p[1] for p in model.predict_proba(scaler.transform(X_test))]

        ranges = select_topk_ranges(scores, k=_TOP_K)
        top1_hit = bool(ranges) and true_indices <= set(range(ranges[0][0], ranges[0][1] + 1))
        top3_hit = any(true_indices <= set(range(s, e + 1)) for s, e, _ in ranges)
        per_book.append({
            "book_key": held_out,
            "corpus": test_rows[0]["corpus"],
            "extraction_type": test_rows[0]["extraction_type"],
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
        })

    return {"per_book": per_book}


def _hit_rate(per_book: list[dict], field: str, group_by: str | None = None, group_value=None) -> float:
    subset = [r for r in per_book if group_by is None or r[group_by] == group_value]
    return sum(r[field] for r in subset) / len(subset) if subset else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chapter-segmentation-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["logistic_regression", "gradient_boosting"], default="logistic_regression")
    args = parser.parse_args()

    rows = merge_ground_truth(args.chapter_segmentation_dir)
    print(f"Merged ground truth: {len(rows)} books")
    table = build_feature_table(rows)

    summary = evaluate_leave_one_book_out(table, args.model)
    per_book = summary["per_book"]
    print(f"Books evaluated (with >=1 true TOC page): {len(per_book)}")
    print(f"Top-1 hit rate: {_hit_rate(per_book, 'top1_hit'):.1%}")
    print(f"Top-3 hit rate: {_hit_rate(per_book, 'top3_hit'):.1%}")
    print()
    print("By corpus:")
    for corpus in sorted({r["corpus"] for r in per_book}):
        n = sum(1 for r in per_book if r["corpus"] == corpus)
        print(
            f"  {corpus} (n={n}): top1={_hit_rate(per_book, 'top1_hit', 'corpus', corpus):.1%}, "
            f"top3={_hit_rate(per_book, 'top3_hit', 'corpus', corpus):.1%}"
        )
    print("By extraction_type (chapter-segmentation rows only -- dnb_located rows have no")
    print("extraction_type in their manifest, grouped here as None/'unknown'):")
    for extraction_type in sorted({r["extraction_type"] for r in per_book}, key=lambda v: (v is None, v)):
        n = sum(1 for r in per_book if r["extraction_type"] == extraction_type)
        label = extraction_type or "unknown"
        print(
            f"  {label} (n={n}): top1={_hit_rate(per_book, 'top1_hit', 'extraction_type', extraction_type):.1%}, "
            f"top3={_hit_rate(per_book, 'top3_hit', 'extraction_type', extraction_type):.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
