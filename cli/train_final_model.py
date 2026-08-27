#!/usr/bin/env python3
"""Fits the deployable TOC-page model on the FULL merged ground truth (no
held-out book, unlike train_toc_classifier.py's leave-one-book-out
evaluation) and serializes it to src/toc_page_classifier/data/model.pkl,
where toc_page_classifier.predict.locate_toc_pages loads it from.

    uv run python -m cli.train_final_model
    uv run python -m cli.train_final_model --model logistic_regression
    uv run python -m cli.train_final_model --corpus-dir ../chapter-segmentation/evaluation/corpus

Run as `-m cli.train_final_model`, not `python cli/train_final_model.py`
directly -- unlike this repo's other cli/ scripts, this one imports from
cli.train_toc_classifier (to reuse its feature-table cache and model
factory), which only resolves when cli/ is imported as a package from the
repo root.

Defaults to gradient_boosting: per README.md's LOBO results, it beats
logistic_regression's top-1 hit rate on every corpus/extraction_type
slice, and top-1 (a single best answer) is what a deployable predictor
needs, unlike top-3.
"""

import argparse
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from cli.train_toc_classifier import ALL_FEATURE_NAMES, _load_or_build_feature_table, _make_model
from toc_page_classifier.ground_truth import merge_ground_truth

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "toc_page_classifier" / "data" / "model.pkl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        action="append",
        default=None,
        help="Additional expected-json evaluation corpus (repeatable) -- a single corpus "
        "directory, or a root containing several named ones. Corpora under this repo's "
        "own data/corpus/ are auto-discovered and don't need this flag.",
    )
    parser.add_argument("--model", choices=["logistic_regression", "gradient_boosting"], default="gradient_boosting")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--out", type=Path, default=_DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    rows = merge_ground_truth(args.corpus_dir)
    print(f"Merged ground truth: {len(rows)} books")
    table = _load_or_build_feature_table(rows, rebuild=args.rebuild_features)

    X = [[r["features"][name] for name in ALL_FEATURE_NAMES] for r in table]
    y = [r["label"] for r in table]
    weight = [r["weight"] for r in table]

    scaler = StandardScaler().fit(X)
    model = _make_model(args.model)
    model.fit(scaler.transform(X), y, sample_weight=weight)

    bundle = {"model_name": args.model, "feature_names": ALL_FEATURE_NAMES, "scaler": scaler, "model": model}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes, trained on {len(X)} rows from {len(rows)} books)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
