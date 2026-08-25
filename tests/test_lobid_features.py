from toc_page_classifier.lobid_features import (
    extract_features,
    record_domain_bucket,
    record_language,
    record_matches,
    record_volume_type,
)

MONOGRAPH = {
    "type": ["BibliographicResource", "Book"],
    "isbn": ["978-3-16-148410-0"],
    "title": "A Monograph",
    "language": [{"id": "http://id.loc.gov/vocabulary/iso639-2/ger"}],
    "tableOfContents": [{"id": "https://digitale-objekte.hbz-nrw.de/storage/x.pdf"}],
    "subject": [{"notation": "CC 3200", "source": {"label": "RVK (Regensburger Verbundklassifikation)"}}],
    "id": "http://lobid.org/resources/12345#!",
}

EDITED_VOLUME = {**MONOGRAPH, "type": ["BibliographicResource", "EditedVolume", "Book"]}
THESIS = {**MONOGRAPH, "type": ["BibliographicResource", "Thesis", "Book"]}
NOT_A_BOOK = {**MONOGRAPH, "type": ["BibliographicResource", "Journal"]}
NO_TOC = {**MONOGRAPH, "tableOfContents": []}
NO_ISBN = {**MONOGRAPH, "isbn": []}


def test_record_matches_requires_book_isbn_and_toc():
    assert record_matches(MONOGRAPH) is True
    assert record_matches(NOT_A_BOOK) is False
    assert record_matches(NO_TOC) is False
    assert record_matches(NO_ISBN) is False


def test_record_language_maps_iso639_2_to_1():
    assert record_language(MONOGRAPH) == "de"
    assert record_language({"language": []}) is None


def test_record_volume_type():
    assert record_volume_type(MONOGRAPH) == "monograph"
    assert record_volume_type(EDITED_VOLUME) == "edited_volume"
    assert record_volume_type(THESIS) == "thesis"


def test_record_domain_bucket_from_rvk_notation():
    assert record_domain_bucket(MONOGRAPH) == "CC"
    assert record_domain_bucket({"subject": []}) == "unknown"


def test_extract_features_shape():
    features = extract_features(MONOGRAPH)
    assert features["isbn"] == "9783161484100"
    assert features["language"] == "de"
    assert features["volume_type"] == "monograph"
    assert features["domain_bucket"] == "CC"
    assert features["toc_download_url"].endswith(".pdf")
