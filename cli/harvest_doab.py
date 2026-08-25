#!/usr/bin/env python3
"""Harvests DOAB's "Books" OAI-PMH set the same way `harvest_oapen.py`
harvests OAPEN's -- see that script's docstring for the shared mechanism.

    uv run python cli/harvest_doab.py

DOAB's Books set has ~72.5k records as of 2026-08-25 (~725 requests, several
minutes). DOAB largely mirrors OAPEN plus other publishers/repositories, so
expect real but partial overlap with `harvest_oapen.py`'s output -- and
unlike OAPEN, many DOAB records have no directly resolvable full-text PDF at
all (DOAB is a directory that often links out to a publisher's own site
rather than hosting a copy); `match_dnb_oa.py` records this rather than
silently dropping such matches.
"""

import argparse
import json
from pathlib import Path

from toc_page_classifier.oai_harvest import harvest_books_set

OAI_BASE = "https://directory.doabooks.org/oai/request"
SET_SPEC = "com_20.500.12854_5"  # "Books"
HANDLE_HOST = "directory.doabooks.org"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "corpus" / "pilot" / ".doab-cache" / "books.jsonl"


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
    print(f"Done: {count} DOAB records written to {args.out}")


if __name__ == "__main__":
    main()
