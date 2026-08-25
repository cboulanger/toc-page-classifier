#!/usr/bin/env python3
"""Intersects the harvested OAPEN/DOAB ISBN caches (`harvest_oapen.py`,
`harvest_doab.py`) against a local DNB TOC-scan manifest's own ISBNs
(`toc_page_classifier.dnb_manifest`, configured via `DNB_TOC_CORPUS_DIR`),
resolves each match's actual PDF download link, and writes the result to
`--out` (default: `data/corpus/pilot/match_dnb_oa_check.json`, gitignored).

    uv run python cli/match_dnb_oa.py

This is a narrower, offline-only cross-check against whatever local DNB
manifest `DNB_TOC_CORPUS_DIR` points at -- no live lobid search, no
diversity sampling. It deliberately does NOT write to
`data/corpus/pilot/manifest.json`: that file is owned by
`discover_oa_dnb_candidates.py`, and overwriting it here would silently
replace the actual diverse sample with whatever this narrower check finds.

Requires both harvest caches to already exist (run the two harvest scripts
first). Safe to re-run: it rebuilds its output from scratch each time
rather than appending, so it can't accumulate stale duplicate entries from
an earlier partial run.
"""

import argparse
import json
from pathlib import Path

from toc_page_classifier.dnb_manifest import load_dnb_books_by_isbn
from toc_page_classifier.oa_repository import resolve_pdf_bitstream

REPO_ROOT = Path(__file__).resolve().parent.parent
OAPEN_CACHE = REPO_ROOT / "data" / "corpus" / "pilot" / ".oapen-cache" / "books.jsonl"
DOAB_CACHE = REPO_ROOT / "data" / "corpus" / "pilot" / ".doab-cache" / "books.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "corpus" / "pilot" / "match_dnb_oa_check.json"

OAPEN_REST_BASE = "https://library.oapen.org"
DOAB_REST_BASE = "https://directory.doabooks.org"


def load_cache(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skipping {path.name} -- not harvested yet)")
        return []
    return [json.loads(line) for line in path.open()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    dnb_books = load_dnb_books_by_isbn()
    print(f"DNB corpus: {len(dnb_books)} books")

    sources = [
        ("oapen", OAPEN_CACHE, OAPEN_REST_BASE),
        ("doab", DOAB_CACHE, DOAB_REST_BASE),
    ]

    seen_isbns: set[str] = set()
    entries = []
    for source_name, cache_path, rest_base in sources:
        records = load_cache(cache_path)
        print(f"{source_name}: {len(records)} harvested records")
        for rec in records:
            for isbn in rec["isbns"]:
                if isbn not in dnb_books or isbn in seen_isbns:
                    continue
                seen_isbns.add(isbn)
                dnb_book = dnb_books[isbn]

                resolved = None
                try:
                    resolved = resolve_pdf_bitstream(rest_base, rec["handle"])
                except Exception as e:
                    print(f"  {isbn}: bitstream lookup failed ({e})")

                fallback_urls = rec.get("pdf_urls") or []
                pdf_url = resolved.url if resolved else (fallback_urls[0] if fallback_urls else None)
                entries.append({
                    "isbn": isbn,
                    "dnb_title": dnb_book["title"],
                    "dnb_toc_download_url": dnb_book["toc_download_url"],
                    "oa_source": source_name,
                    "oa_handle": rec["handle"],
                    "oa_title": rec["title"],
                    "oa_pdf_url": pdf_url,
                    "oa_pdf_resolved_via": ("bitstream" if resolved else ("oai_identifier" if pdf_url else None)),
                    "oa_license_code": resolved.license_code if resolved else None,
                })
                status = "OK" if pdf_url else "NO DIRECT PDF LINK"
                print(f"  + {isbn} ({source_name}) -- {status}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"books": entries}, indent=2, ensure_ascii=False))
    n_with_pdf = sum(1 for e in entries if e["oa_pdf_url"])
    print(f"\nWrote {len(entries)} matches ({n_with_pdf} with a resolvable PDF link) to {args.out}")


if __name__ == "__main__":
    main()
