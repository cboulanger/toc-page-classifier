"""Live per-ISBN lookup against lobid.org/resources -- used instead of
streaming the full ~21.5GB lobid-resources dump (`dnb-toc-ground-truth`'s
`fetch_corpus.py --from-dump` approach) because we already know the exact,
much smaller set of ISBNs worth checking: every OA book found by
`harvest_oapen.py`/`harvest_doab.py`. Checking that ISBN list live is far
cheaper than scanning the entire dump hoping to hit it by chance.
"""

import json

from .oai_harvest import fetch_url

_SEARCH_URL = "https://lobid.org/resources/search"


def search_by_isbn(isbn: str) -> dict | None:
    """Returns the first matching lobid-resources record for this ISBN, or
    None if lobid has no record at all (common -- lobid/hbz skews toward
    German-published or German-catalogued books, so most OAPEN/DOAB ISBNs,
    which are majority English-language, won't have any record)."""
    url = f"{_SEARCH_URL}?q=isbn:{isbn}&format=json"
    data = json.loads(fetch_url(url))
    members = data.get("member") or []
    return members[0] if members else None
