#!/usr/bin/env python3
"""Runs the text-overlap TOC locator (`src/toc_page_classifier/locate_toc.py`)
against every downloaded pair in `data/corpus/pilot/pdf/` and writes one
ground-truth JSON file per book to `data/corpus/pilot/ground-truth/`.

    uv run python cli/locate_toc.py

A book whose DNB TOC scan has no extractable text (a pure image, no OCR --
see `pdf_text.has_text`) is written with `"status": "reference_has_no_text"`
and no located range, rather than silently skipped -- these are exactly the
cases needing an OCR-first or vision-based fallback (not implemented).
`"margin"` (top score minus the best score just outside the located range)
is a rough confidence signal: a low margin means the match is ambiguous and
should be reviewed by hand before being trusted as ground truth.

A book that raises an unexpected error while its PDF is being read (a
malformed file, a pypdf limitation) is written with `"status": "error"` and
its error message, rather than aborting the whole batch -- one bad PDF
among a hundred shouldn't cost the rest their ground truth.
"""

import argparse
import json
from pathlib import Path

from toc_page_classifier.locate_toc import locate_toc_in_fulltext
from toc_page_classifier.pdf_text import has_text

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = REPO_ROOT / "data" / "corpus" / "pilot" / "pdf"
GT_DIR = REPO_ROOT / "data" / "corpus" / "pilot" / "ground-truth"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    GT_DIR.mkdir(parents=True, exist_ok=True)
    toc_paths = sorted(PDF_DIR.glob("*.dnb_toc.pdf"))
    print(f"Found {len(toc_paths)} downloaded pairs")

    for toc_path in toc_paths:
        isbn = toc_path.name.removesuffix(".dnb_toc.pdf")
        full_path = PDF_DIR / f"{isbn}.fulltext.pdf"
        if not full_path.exists():
            continue

        out_path = GT_DIR / f"{isbn}.json"
        try:
            if not has_text(toc_path):
                print(f"{isbn}: DNB TOC scan has no text layer -- needs OCR/vision fallback")
                out_path.write_text(json.dumps({"isbn": isbn, "status": "reference_has_no_text"}, indent=2))
                continue

            location = locate_toc_in_fulltext(toc_path, full_path)
        except Exception as e:
            print(f"{isbn}: ERROR reading PDF ({e})")
            out_path.write_text(json.dumps({"isbn": isbn, "status": "error", "error": str(e)}, indent=2))
            continue

        if location is None:
            print(f"{isbn}: no candidate found")
            out_path.write_text(json.dumps({"isbn": isbn, "status": "no_candidate"}, indent=2))
            continue

        print(f"{isbn}: pages {location.start_index}-{location.end_index} "
              f"(score {location.top_score:.3f}, margin {location.margin:.3f})")
        out_path.write_text(json.dumps({
            "isbn": isbn,
            "status": "located",
            "toc_start_index": location.start_index,
            "toc_end_index": location.end_index,
            "top_score": round(location.top_score, 4),
            "margin": round(location.margin, 4),
            "method": "text_token_ochiai_fixed_window",
            "verified": False,
        }, indent=2))


if __name__ == "__main__":
    main()
