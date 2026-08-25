"""OAI-PMH harvesting for DSpace-based OA book repositories (OAPEN, DOAB).

Both repositories expose the same DSpace OAI-PMH interface and the same
oai_dc record shape, including a shared, non-standard `oapen:relationisbn`
element carrying each book's ISBN(s) -- this module knows how to walk one
repository's "Books" set to completion via `resumptionToken` paging and
extract that shared shape. It does not know anything about the DNB side;
see `dnb_manifest.py` for that.
"""

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ISBN_RE = re.compile(r"<oapen:relationisbn>([0-9Xx-]+)</oapen:relationisbn>")
TITLE_RE = re.compile(r"<dc:title>(.*?)</dc:title>", re.S)
RECORD_RE = re.compile(r"<record>.*?</record>", re.S)
TOKEN_RE = re.compile(r"<resumptionToken[^>]*>([^<]*)</resumptionToken>")
PDF_IDENTIFIER_RE = re.compile(r"<dc:identifier>(https?://[^<]+\.pdf)</dc:identifier>", re.I)

USER_AGENT = "oa-toc-ground-truth/0.1 (research tool, see https://github.com/cboulanger)"


@dataclass
class BookRecord:
    isbns: list[str]
    handle: str
    title: str | None
    pdf_urls: list[str] = field(default_factory=list)


def normalize_isbn(raw: str) -> str:
    return raw.replace("-", "").upper()


def fetch_url(url: str, retries: int = 3, timeout: int = 30) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _handle_re_for_host(handle_host: str) -> re.Pattern:
    escaped = re.escape(handle_host)
    return re.compile(rf'<dc:identifier type="URL">https://{escaped}/handle/([\d.]+/\d+)</dc:identifier>')


def parse_record(record_xml: str, handle_re: re.Pattern) -> BookRecord | None:
    isbns = sorted({normalize_isbn(m) for m in ISBN_RE.findall(record_xml)})
    handle_m = handle_re.search(record_xml)
    if not isbns or not handle_m:
        return None
    title_m = TITLE_RE.search(record_xml)
    pdf_urls = PDF_IDENTIFIER_RE.findall(record_xml)
    return BookRecord(
        isbns=isbns,
        handle=handle_m.group(1),
        title=(title_m.group(1)[:200] if title_m else None),
        pdf_urls=pdf_urls,
    )


def harvest_books_set(oai_base: str, set_spec: str, handle_host: str, sleep: float = 0.5):
    """Yields BookRecord for every OAI record in `set_spec` that carries at
    least one ISBN and a resolvable handle. Pages through the full set via
    `resumptionToken` -- for OAPEN's ~49.8k-record and DOAB's ~72k-record
    "Books" sets, expect several hundred requests and several minutes."""
    handle_re = _handle_re_for_host(handle_host)
    url = f"{oai_base}?verb=ListRecords&set={set_spec}&metadataPrefix=oai_dc"
    page = 0
    while True:
        page += 1
        xml = fetch_url(url)
        for record_xml in RECORD_RE.findall(xml):
            rec = parse_record(record_xml, handle_re)
            if rec is not None:
                yield rec
        token_m = TOKEN_RE.search(xml)
        token = token_m.group(1) if token_m else ""
        if not token:
            return
        url = f"{oai_base}?verb=ListRecords&resumptionToken={token}"
        time.sleep(sleep)
