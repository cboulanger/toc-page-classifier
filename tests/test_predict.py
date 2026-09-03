from toc_page_classifier.predict import _score_pages, locate_toc_pages


class _FixedScoreModel:
    """Ignores its input entirely and returns hand-picked scores -- lets
    tests exercise _score_pages'/locate_toc_pages' wiring deterministically,
    independent of any real classifier's behavior."""

    def __init__(self, scores):
        self._scores = scores

    def predict_proba(self, X):
        assert len(X) == len(self._scores)
        return [[1.0 - s, s] for s in self._scores]


class _IdentityScaler:
    def transform(self, X):
        return X


class _KeyedScoreModel:
    """Looks up each row's score by its first feature value -- unlike
    _FixedScoreModel, this tolerates being called once per run (a
    different row count each time) rather than once for the whole book."""

    def __init__(self, scores_by_first_feature):
        self._scores_by_first_feature = scores_by_first_feature

    def predict_proba(self, X):
        return [[1.0 - (s := self._scores_by_first_feature[row[0]]), s] for row in X]


def test_score_pages_orders_features_by_bundle_feature_names():
    # Feature dicts have extra, irrelevantly-ordered keys; _score_pages must
    # select and order strictly by bundle["feature_names"].
    features_per_page = {
        0: {"b": 2.0, "a": 1.0, "extra": 99.0},
        1: {"b": 20.0, "a": 10.0, "extra": 99.0},
    }
    bundle = {
        "feature_names": ["a", "b"],
        "scaler": _IdentityScaler(),
        "model": _FixedScoreModel([0.1, 0.9]),
    }
    scores = _score_pages(features_per_page, bundle)
    assert scores == [0.1, 0.9]


def test_locate_toc_pages_returns_empty_list_for_pdf_with_no_pages(tmp_path, monkeypatch):
    import toc_page_classifier.predict as predict_module

    monkeypatch.setattr(
        predict_module,
        "extract_page_features_and_texts",
        lambda pdf_path, head_pages=None, tail_pages=None: ({}, {}, 0),
    )
    model_path = tmp_path / "model.pkl"
    import pickle

    with open(model_path, "wb") as f:
        pickle.dump({"feature_names": [], "scaler": _IdentityScaler(), "model": _FixedScoreModel([])}, f)

    assert locate_toc_pages("irrelevant.pdf", model_path=model_path) == []


def test_locate_toc_pages_returns_top_ranked_range(tmp_path, monkeypatch):
    import pickle

    import toc_page_classifier.predict as predict_module

    # A 2-page book: page 0 a headed/high-scoring TOC page, page 1 a
    # lower-scoring but still real continuation page (matching the real
    # shape a multi-page TOC takes) -- no padding pages, so the sum-ranked
    # winner is unambiguous: the 2-page window (0, 1) beats either 1-page
    # window on sum, covering both real pages, not just the peak.
    layout = {0: {"line_count": 0.0}, 1: {"line_count": 0.0}}
    texts = {0: "", 1: ""}
    monkeypatch.setattr(
        predict_module,
        "extract_page_features_and_texts",
        lambda pdf_path, head_pages=None, tail_pages=None: (layout, texts, 2),
    )
    monkeypatch.setattr(predict_module, "add_book_context_features", lambda layout, total_pages: layout)
    monkeypatch.setattr(
        predict_module,
        "extract_text_features",
        lambda pages, language=None: {i: {"toc_line_ratio": 0.0} for i in range(len(pages))},
    )

    bundle = {
        "feature_names": ["line_count", "toc_line_ratio"],
        "scaler": _IdentityScaler(),
        "model": _FixedScoreModel([0.9, 0.3]),
    }
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    assert locate_toc_pages("irrelevant.pdf", model_path=model_path) == [0, 1]


def test_locate_toc_pages_treats_head_and_tail_scan_as_separate_runs(tmp_path, monkeypatch):
    """A long book scanned as head pages {0, 1} + tail pages {10, 11} (the
    middle skipped) must not let select_topk_ranges treat page 1 and page
    10 as adjacent -- the winning window must stay within one real run,
    never span pages that were never scanned."""
    import pickle

    import toc_page_classifier.predict as predict_module

    scanned = [0, 1, 10, 11]
    # Unique line_count per page so _KeyedScoreModel can identify each row
    # regardless of which run's call it arrives in.
    layout = {i: {"line_count": float(i)} for i in scanned}
    texts = {i: "" for i in scanned}
    monkeypatch.setattr(
        predict_module,
        "extract_page_features_and_texts",
        lambda pdf_path, head_pages=None, tail_pages=None: (layout, texts, 20),
    )
    monkeypatch.setattr(predict_module, "add_book_context_features", lambda layout, total_pages: layout)
    monkeypatch.setattr(
        predict_module,
        "extract_text_features",
        lambda pages, language=None: {i: {"toc_line_ratio": 0.0} for i in range(len(pages))},
    )

    # Scores in scanned order: [page0=0.1, page1=0.9, page10=0.9, page11=0.1].
    # Treating all 4 scanned pages as one contiguous run (the bug this
    # guards against) would let the 2-page window (page1, page10) win on
    # sum (0.9+0.9=1.8, higher than any real window) and get returned as
    # range(1, 11) -- silently claiming every skipped interior page (2-9)
    # is part of the TOC too. Splitting into the real runs [0, 1] and
    # [10, 11] confines each candidate window to pages actually scanned;
    # both runs' own full-window sum ties at 1.0, and the tie resolves to
    # whichever run's candidate was generated first -- run [0, 1], scanned
    # first since it sorts before [10, 11].
    bundle = {
        "feature_names": ["line_count"],
        "scaler": _IdentityScaler(),
        "model": _KeyedScoreModel({0.0: 0.1, 1.0: 0.9, 10.0: 0.9, 11.0: 0.1}),
    }
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    assert locate_toc_pages("irrelevant.pdf", model_path=model_path, top_k=1) == [0, 1]
