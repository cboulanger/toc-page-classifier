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
    exercises select_topk_ranges' real greedy/sum-ranked/overlap logic
    deterministically, independent of any actual classifier's behavior."""
    # True TOC is pages {6, 7, 8} (a headed page at 6 plus two lower-scoring
    # continuation pages, matching the real shape a multi-page TOC's
    # keyword-hit-only-on-the-first-page takes). A false-positive region at
    # pages 0-1 scores even higher. Worked through select_topk_ranges' real
    # sum-ranked greedy/overlap logic by hand (max_window defaults to 6):
    # every window spanning some prefix of {0, 1} sums to 1.98 (0.99+0.99,
    # padding with zeros doesn't change the sum) and the widest of those
    # ties, (0, 5), wins the width tie-break -- so rank 1 is a 6-page
    # window that never touches the true set at all. Once (0, 5) is
    # removed, the best remaining window is (6, 9) at sum 1.4 (again a
    # zero-padding tie against the tighter (6, 8), won by width) -- this
    # is rank 2, and it fully contains {6, 7, 8}.
    fixed_scores = [0.99, 0.99, 0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.2, 0.0]
    filler_rows = [_row("filler", i, label=False) for i in range(4)]
    target_rows = [_row("target", i, label=(i in (6, 7, 8))) for i in range(10)]
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
    # Page-level diagnostic: individual page scores rank 0 and 1 (0.99 each)
    # above page 6 (0.9), so the best-scored true page ranks 3rd overall.
    assert result["best_true_page_rank"] == 3
    # The rank-1 window (0, 5) doesn't intersect the true set {6, 7, 8} at
    # all -- a full miss even under the loose overlap definition. But rank
    # 2, (6, 9), does, so top3_overlap is True, same as top3_hit here.
    assert result["top1_overlap"] is False
    assert result["top3_overlap"] is True
