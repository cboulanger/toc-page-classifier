#!/usr/bin/env python3
"""Downloads both PDFs (DNB TOC scan + OA full text) for every matched pair
in `data/corpus/pilot/manifest.json` (written by `discover_oa_dnb_candidates.py`), into
`data/corpus/pilot/pdf/<isbn>.dnb_toc.pdf` and `<isbn>.fulltext.pdf`.

    uv run python cli/fetch_pairs.py

PDFs are gitignored (see `.gitignore`). Skips a pair entirely if
`oa_pdf_url` is null (no resolvable OA PDF link yet -- see
`match_dnb_oa.py`'s "NO DIRECT PDF LINK" entries) or if both files already
exist locally.
"""

import argparse
import json
import ssl
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "corpus" / "pilot" / "manifest.json"
PDF_DIR = REPO_ROOT / "data" / "corpus" / "pilot" / "pdf"


def fetch_binary(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "toc-page-classifier/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except Exception:
        # A couple of DNB `d-nb.info` links serve a self-signed cert -- not
        # a security-sensitive fetch (public, unauthenticated PDF scans of
        # public-domain TOC pages), so retry once without verification.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            return r.read()


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    books = json.loads(MANIFEST.read_text())["books"]
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    fetched, skipped, failed = 0, 0, 0
    for book in books:
        isbn = book["isbn"]
        if not book["oa_pdf_url"]:
            skipped += 1
            continue

        toc_path = PDF_DIR / f"{isbn}.dnb_toc.pdf"
        full_path = PDF_DIR / f"{isbn}.fulltext.pdf"
        if toc_path.exists() and full_path.exists():
            continue

        print(f"{isbn}: fetching...")
        try:
            if not toc_path.exists():
                toc_path.write_bytes(fetch_binary(book["dnb_toc_download_url"]))
            if not full_path.exists():
                full_path.write_bytes(fetch_binary(book["oa_pdf_url"]))
            fetched += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\nFetched {fetched}, skipped (no PDF link) {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
