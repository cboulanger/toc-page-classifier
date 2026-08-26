from sklearn.preprocessing import StandardScaler

from cli.train_toc_classifier import ALL_FEATURE_NAMES, evaluate_leave_one_book_out


def _row(book_key, page_index, label, corpus="test-corpus", extraction_type=None, weight=1.0):
    """A synthetic feature-table row -- no PDFs, no real feature extraction.
    Every ALL_FEATURE_NAMES key must be present (evaluate_leave_one_book_out
    indexes into "features" by name), so fill them all with 0.0; callers that
    care about a specific feature's value overwrite it afterwards."""
    return {
        "book_key": book_key,
        "corpus": corpus,
        "extraction_type": extraction_type,
        "page_index": page_index,
        "features": {name: 0.0 for name in ALL_FEATURE_NAMES},
        "label": label,
        "weight": weight,
    }


def test_evaluate_leave_one_book_out_fits_scaler_only_on_training_rows(monkeypatch):
    """Locks in the leakage-avoidance property: for each LOBO fold, the
    StandardScaler (and therefore the model) must be fit on every row
    EXCEPT the held-out book's own rows -- never on the full table.
    Both books carry a true TOC page (so LogisticRegression sees both
    classes in whichever one supplies the training data for a given fold)
    and have different row counts, so the two folds' training-set sizes
    are distinguishable rather than accidentally symmetric."""
    book_a_rows = [_row("book_a", i, label=(i == 1)) for i in range(5)]
    book_b_rows = [_row("book_b", i, label=(i == 2)) for i in range(4)]
    table = book_a_rows + book_b_rows

    fit_sizes = []
    original_fit = StandardScaler.fit

    def spy_fit(self, X, *args, **kwargs):
        fit_sizes.append(len(X))
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy_fit)

    evaluate_leave_one_book_out(table, "logistic_regression")

    # Both books have a true page, so both are evaluated as folds. book_a's
    # fold must be fit on book_b's 4 rows only (not the full 9-row table --
    # that would mean book_a's own held-out rows leaked in), and vice versa.
    assert sorted(fit_sizes) == sorted([len(book_b_rows), len(book_a_rows)])


def test_evaluate_leave_one_book_out_top1_requires_full_coverage_top3_allows_it():
    """Locks in the top1_hit/top3_hit boundary logic: a hit requires the
    selected window to fully CONTAIN the true TOC page set (a subset check,
    not mere overlap). Stubs the model's predict_proba with fixed,
    hand-picked scores (via monkeypatching _make_model) so this test
    exercises select_topk_ranges' real greedy/overlap logic deterministically,
    independent of any actual classifier's behavior."""
    # True TOC is pages {1, 2}. Fixed page scores: page 0 has the single
    # highest score of all (a false-positive elsewhere), pages 1 and 2 are
    # tied just below it, page 3 is low. Worked through select_topk_ranges'
    # greedy overlap logic by hand: rank-1 window is the lone page-0 peak
    # (misses the true set entirely), but the 2-page window covering {1, 2}
    # (tied score, wins the width tie-break over the two solo sub-windows)
    # is still available and lands at rank 2 -- inside the top-3.
    fixed_scores = [0.95, 0.8, 0.8, 0.1]
    filler_rows = [_row("filler", i, label=False) for i in range(4)]
    target_rows = [_row("target", i, label=(i in (1, 2))) for i in range(4)]
    table = filler_rows + target_rows

    class _FixedScoreModel:
        def fit(self, X, y, sample_weight=None):
            pass

        def predict_proba(self, X):
            return [[1.0 - s, s] for s in fixed_scores]

    import cli.train_toc_classifier as mod

    def fake_make_model(name):
        return _FixedScoreModel()

    original_make_model = mod._make_model
    mod._make_model = fake_make_model
    try:
        summary = evaluate_leave_one_book_out(table, "logistic_regression")
    finally:
        mod._make_model = original_make_model

    result = next(r for r in summary["per_book"] if r["book_key"] == "target")
    assert result["top1_hit"] is False
    assert result["top3_hit"] is True
    # Page-level diagnostic: true pages are {1, 2}, scored 0.8 each -- tied
    # for 2nd/3rd highest score behind page 0's 0.95 false-positive peak, so
    # the best-scored true page ranks 2nd overall.
    assert result["best_true_page_rank"] == 2
