#!/usr/bin/env python3
"""Harvests OAPEN's "Books" OAI-PMH set into a local cache of ISBN -> handle
records, for `discover_oa_dnb_candidates.py` (or `match_dnb_oa.py`) to
intersect against DNB TOC-scan records.

    uv run python cli/harvest_oapen.py

Re-running overwrites the cache from scratch -- OAPEN's Books set has ~49.8k
records as of 2026-08-25, so a full run takes several minutes and issues
~500 HTTP requests against library.oapen.org. Not something to run on every
commit; re-run it periodically (e.g. before a `discover_oa_dnb_candidates.py`
pass) to pick
up newly-added OAPEN books.
"""

import argparse
import json
from pathlib import Path

from toc_page_classifier.oai_harvest import harvest_books_set

OAI_BASE = "https://library.oapen.org/oai/request"
SET_SPEC = "com_20.500.12657_5"  # "Books"
HANDLE_HOST = "library.oapen.org"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "corpus" / "pilot" / ".oapen-cache" / "books.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w") as f:
        for record in harvest_books_set(OAI_BASE, SET_SPEC, HANDLE_HOST):
            f.write(json.dumps({
                "isbns": record.isbns,
                "handle": record.handle,
                "title": record.title,
                "pdf_urls": record.pdf_urls,
            }) + "\n")
            count += 1
            if count % 500 == 0:
                print(f"  {count} records with ISBN+handle so far...")
    print(f"Done: {count} OAPEN records written to {args.out}")


if __name__ == "__main__":
    main()
