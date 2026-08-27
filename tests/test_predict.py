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

    monkeypatch.setattr(predict_module, "extract_page_features_and_texts", lambda pdf_path: ({}, {}))
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
    monkeypatch.setattr(predict_module, "extract_page_features_and_texts", lambda pdf_path: (layout, texts))
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
