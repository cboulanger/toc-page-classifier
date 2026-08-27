#!/usr/bin/env python3
"""Leave-one-book-out evaluation of the TOC-page classifier: per-page
layout + text features, a page-level scorer (LogisticRegression or
gradient boosting), and top-K non-overlapping candidate page-range output.
See docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md.

    uv run python cli/train_toc_classifier.py
    uv run python cli/train_toc_classifier.py --model gradient_boosting
    uv run python cli/train_toc_classifier.py --chapter-segmentation-dir ../chapter-segmentation

The feature table (the pdfplumber pass over every book, ~1 minute/book on
the full corpus) is cached to data/feature_table_cache.pkl and reused
across --model choices; pass --rebuild-features after a feature-extraction
code change.
"""

import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from toc_page_classifier.ground_truth import GroundTruthRow, merge_ground_truth
from toc_page_classifier.layout_features import (
    FEATURE_NAMES as LAYOUT_FEATURE_NAMES,
    add_book_context_features,
    extract_page_features_and_texts,
)
from toc_page_classifier.range_selection import select_topk_ranges
from toc_page_classifier.text_features import TEXT_FEATURE_NAMES, extract_text_features

ALL_FEATURE_NAMES = LAYOUT_FEATURE_NAMES + TEXT_FEATURE_NAMES
_TOP_K = 3
_FEATURE_TABLE_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "feature_table_cache.pkl"


def build_feature_table(rows: list[GroundTruthRow]) -> list[dict]:
    """One row per page: {"book_key", "corpus", "extraction_type",
    "page_index", "features": {...}, "label": bool, "weight": float}."""
    table = []
    for book_number, row in enumerate(rows, start=1):
        print(f"[build_feature_table {book_number}/{len(rows)}] {row.key}", flush=True)
        page_features, gap_aware_texts = extract_page_features_and_texts(row.pdf_path)
        total_pages = len(gap_aware_texts)
        if total_pages == 0:
            continue
        layout = add_book_context_features(page_features, total_pages)
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
    breakdown reporting.

    Also returns each book's best_true_page_rank: the 1-indexed rank (by
    page score, descending) of whichever true TOC page the model scored
    highest. This is a page-level diagnostic independent of range
    selection -- it distinguishes "the scorer ranked a true TOC page near
    the top but select_topk_ranges' contiguous-window logic didn't
    capture it" (small rank, top1/top3 miss) from "the scorer never gave
    any true TOC page a competitive score at all" (large rank).

    Also returns top1_overlap/top3_overlap: a looser hit definition that
    only requires the selected window to intersect the true TOC page set
    (not fully contain it, like top1_hit/top3_hit do). The gap between
    top1_overlap and top1_hit isolates "found the right neighborhood but
    didn't cover every true page" from "missed the neighborhood
    entirely"."""
    book_keys = sorted({r["book_key"] for r in table})
    per_book = []
    for fold_number, held_out in enumerate(book_keys, start=1):
        test_rows = [r for r in table if r["book_key"] == held_out]
        true_indices = {i for i, r in enumerate(test_rows) if r["label"]}
        if not true_indices:
            print(
                f"[evaluate_leave_one_book_out {fold_number}/{len(book_keys)}] "
                f"{held_out}: skipped, no true TOC page",
                flush=True,
            )
            continue
        print(
            f"[evaluate_leave_one_book_out {fold_number}/{len(book_keys)}] "
            f"{held_out}: fitting on {len(table) - len(test_rows)} rows, "
            f"testing on {len(test_rows)}",
            flush=True,
        )

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
        top1_overlap = bool(ranges) and bool(true_indices & set(range(ranges[0][0], ranges[0][1] + 1)))
        top3_overlap = any(true_indices & set(range(s, e + 1)) for s, e, _ in ranges)

        ranked_page_indices = [
            i for i, _ in sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        ]
        best_true_page_rank = 1 + min(ranked_page_indices.index(i) for i in true_indices)

        per_book.append({
            "book_key": held_out,
            "corpus": test_rows[0]["corpus"],
            "extraction_type": test_rows[0]["extraction_type"],
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top1_overlap": top1_overlap,
            "top3_overlap": top3_overlap,
            "best_true_page_rank": best_true_page_rank,
        })

    return {"per_book": per_book}


def _hit_rate(per_book: list[dict], field: str, group_by: str | None = None, group_value=None) -> float:
    subset = [r for r in per_book if group_by is None or r[group_by] == group_value]
    return sum(r[field] for r in subset) / len(subset) if subset else 0.0


def _load_or_build_feature_table(rows: list[GroundTruthRow], rebuild: bool) -> list[dict]:
    """Loads the feature table from disk if it was built from the exact
    same set of book keys as `rows`, since build_feature_table's pdfplumber
    pass over every book's PDF -- not model fitting -- is the dominant
    per-run cost (~1 minute/book on the full 184-book corpus, identical
    regardless of --model), and re-running with a different --model or
    after a range-selection/model-only code change shouldn't have to pay
    it again. Pass rebuild=True (--rebuild-features) to force a fresh
    build, e.g. after a feature-extraction code change."""
    book_keys = sorted(row.key for row in rows)
    if not rebuild and _FEATURE_TABLE_CACHE_PATH.exists():
        with open(_FEATURE_TABLE_CACHE_PATH, "rb") as f:
            cached_keys, cached_table = pickle.load(f)
        if cached_keys == book_keys:
            print(f"Loaded cached feature table ({len(cached_table)} rows) from {_FEATURE_TABLE_CACHE_PATH}")
            return cached_table
        print("Ground truth's book keys changed since the cache was built -- rebuilding")

    table = build_feature_table(rows)
    _FEATURE_TABLE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_FEATURE_TABLE_CACHE_PATH, "wb") as f:
        pickle.dump((book_keys, table), f)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chapter-segmentation-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["logistic_regression", "gradient_boosting"], default="logistic_regression")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Force a fresh feature-table build instead of using the cached one on disk "
        "(needed after a feature-extraction code change; a book-key-set change is detected automatically).",
    )
    args = parser.parse_args()

    rows = merge_ground_truth(args.chapter_segmentation_dir)
    print(f"Merged ground truth: {len(rows)} books")
    table = _load_or_build_feature_table(rows, rebuild=args.rebuild_features)

    summary = evaluate_leave_one_book_out(table, args.model)
    per_book = summary["per_book"]
    print(f"Books evaluated (with >=1 true TOC page): {len(per_book)}")
    print(f"Top-1 hit rate: {_hit_rate(per_book, 'top1_hit'):.1%}")
    print(f"Top-3 hit rate: {_hit_rate(per_book, 'top3_hit'):.1%}")
    print(
        f"Top-1 overlap rate: {_hit_rate(per_book, 'top1_overlap'):.1%} "
        f"(loose: any true page in the window, vs. full-coverage hit above)"
    )
    print(f"Top-3 overlap rate: {_hit_rate(per_book, 'top3_overlap'):.1%}")
    print()
    ranks = sorted(r["best_true_page_rank"] for r in per_book)
    n = len(ranks)
    median_rank = ranks[n // 2] if n % 2 else (ranks[n // 2 - 1] + ranks[n // 2]) / 2
    print(
        f"Page-level diagnostic (rank, by score, of the best-scored true TOC "
        f"page -- independent of range selection):"
    )
    print(f"  median best_true_page_rank: {median_rank:.1f}")
    for cutoff in (1, 3, 10):
        pct = sum(1 for rank in ranks if rank <= cutoff) / n
        print(f"  books with best_true_page_rank <= {cutoff}: {pct:.1%}")
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
