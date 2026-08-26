import json

from toc_page_classifier.text_features import TEXT_FEATURE_NAMES, extract_text_features


def test_extract_text_features_finds_toc_lines_on_a_real_toc_page():
    toc_page = "Contents\n\nIntroduction ..... 1\nChapter One .......... 5\nChapter Two .......... 12\n"
    other_page = "This is an ordinary paragraph of prose that runs on for a while and\nhas nothing structured about it at all.\n"
    features = extract_text_features([toc_page, other_page])
    assert set(features[0].keys()) == set(TEXT_FEATURE_NAMES)
    assert features[0]["toc_line_ratio"] == 1.0
    assert features[0]["monotonic_page_numbers"] == 1.0
    assert features[1]["toc_line_ratio"] == 0.0


def test_extract_text_features_rejects_non_monotonic_page_numbers():
    # An index/bibliography page can superficially match "text ... number"
    # (2+ dot/space separator chars, same as a real TOC line) without a
    # real TOC's monotonically increasing page-number sequence.
    index_page = "Aardvark .. 88\nZebra .. 3\nMongoose .. 45\n"
    features = extract_text_features([index_page])
    assert features[0]["toc_line_ratio"] == 1.0
    assert features[0]["monotonic_page_numbers"] == 0.0


def test_extract_text_features_excludes_url_and_imprint_lines():
    page = "© 2020 Some Publisher\nISBN 978-0-000-00000-0\nSee https://doi.org/10.1000/xyz123\n"
    features = extract_text_features([page])
    assert features[0]["toc_line_ratio"] == 0.0


def test_extract_text_features_computes_digit_density():
    features = extract_text_features(["11111"])
    assert features[0]["digit_density"] == 1.0
    features = extract_text_features([""])
    assert features[0]["digit_density"] == 0.0


def test_extract_text_features_keyword_hits_default_to_zero_without_keywords_path():
    features = extract_text_features(["Contents\n"], language="en", keywords_path=None)
    assert features[0]["keyword_hit_any_language"] == 0.0
    assert features[0]["keyword_hit_same_language"] == 0.0


def test_extract_text_features_keyword_hit_any_language(tmp_path):
    keywords_path = tmp_path / "keywords.json"
    keywords_path.write_text(json.dumps({"en": ["contents"], "de": ["inhalt"]}))
    features = extract_text_features(["INHALT\n\nKapitel 1 .... 5\n"], keywords_path=str(keywords_path))
    assert features[0]["keyword_hit_any_language"] == 1.0


def test_extract_text_features_keyword_hit_same_language_requires_matching_language(tmp_path):
    keywords_path = tmp_path / "keywords.json"
    keywords_path.write_text(json.dumps({"en": ["contents"], "de": ["inhalt"]}))
    features = extract_text_features(
        ["INHALT\n\nKapitel 1 .... 5\n"], language="en", keywords_path=str(keywords_path)
    )
    # matches "de"'s keyword, but book is declared "en" -- any-language hits,
    # same-language does not
    assert features[0]["keyword_hit_any_language"] == 1.0
    assert features[0]["keyword_hit_same_language"] == 0.0


def test_extract_text_features_keyword_hit_same_language_matches(tmp_path):
    keywords_path = tmp_path / "keywords.json"
    keywords_path.write_text(json.dumps({"en": ["contents"], "de": ["inhalt"]}))
    features = extract_text_features(
        ["INHALT\n\nKapitel 1 .... 5\n"], language="de", keywords_path=str(keywords_path)
    )
    assert features[0]["keyword_hit_same_language"] == 1.0
