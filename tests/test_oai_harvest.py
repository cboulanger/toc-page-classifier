from toc_page_classifier.oai_harvest import _handle_re_for_host, normalize_isbn, parse_record

SAMPLE_RECORD = """
<record><header><identifier>oai:library.oapen.org:20.500.12657/43833</identifier></header>
<metadata><oai_dc:dc xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:oapen="http://purl.org/dc/elements/1.1/">
<dc:title>Aggression in Pornography</dc:title>
<dc:identifier type="URL">https://library.oapen.org/handle/20.500.12657/43833</dc:identifier>
<oapen:relationisbn>9780429276105</oapen:relationisbn>
</oai_dc:dc></metadata></record>
"""


def test_normalize_isbn_strips_hyphens_and_upcases():
    assert normalize_isbn("3-16-148410-x") == "316148410X"
    assert normalize_isbn("978-3-16-148410-0") == "9783161484100"


def test_parse_record_extracts_isbn_handle_and_title():
    handle_re = _handle_re_for_host("library.oapen.org")
    rec = parse_record(SAMPLE_RECORD, handle_re)
    assert rec is not None
    assert rec.isbns == ["9780429276105"]
    assert rec.handle == "20.500.12657/43833"
    assert rec.title == "Aggression in Pornography"


def test_parse_record_returns_none_without_isbn():
    handle_re = _handle_re_for_host("library.oapen.org")
    no_isbn = SAMPLE_RECORD.replace("<oapen:relationisbn>9780429276105</oapen:relationisbn>", "")
    assert parse_record(no_isbn, handle_re) is None


def test_parse_record_returns_none_for_wrong_host():
    handle_re = _handle_re_for_host("directory.doabooks.org")
    assert parse_record(SAMPLE_RECORD, handle_re) is None
