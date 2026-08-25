#!/usr/bin/env python3
"""Builds a diverse sample of at least `--target` books that are BOTH
open access (found via `harvest_oapen.py`/`harvest_doab.py`) AND have a DNB
"Kataloganreicherung" TOC scan (checked live against lobid.org, per-ISBN --
see `lobid_search.py`'s docstring for why this beats streaming the full
lobid dump).

    uv run python cli/discover_oa_dnb_candidates.py --target 100

Diversity is enforced with simple per-value caps (language, RVK domain
bucket, edited-volume/monograph/thesis), not real stratified balancing --
see `diversity_sampler.py`. A candidate only counts toward the target once
its OA full-text PDF link is actually resolved (not just "catalogued as
OA") -- an unresolvable DOAB entry (common; DOAB often only links out to a
publisher's own site) is skipped, not counted.

**Resumable by construction, not just by re-running:** every accepted
record's raw lobid JSON is cached to `data/corpus/pilot/.lobid-cache/`
(gitignored) immediately on acceptance, and on startup this script loads
whatever's already cached there, re-derives its features and OA link
(cheap -- no live lobid search needed, just one PDF-bitstream lookup per
cached record), and seeds the diversity sampler with it before resuming the
live search for the rest. So killing a long run mid-flight never wastes the
live-lobid-search cost already spent -- e.g. to relax `--*-cap-fraction`
once it's clear the default caps make the last few slots too hard to fill,
kill and re-run with looser caps; the already-found candidates carry over.
`data/corpus/pilot/manifest.json` is rewritten after every new acceptance
(not just at the end), so an interrupted run still leaves a usable manifest.
"""

import argparse
import json
import random
import time
from pathlib import Path

from toc_page_classifier.diversity_sampler import DiversitySampler
from toc_page_classifier.lobid_features import extract_features, record_matches
from toc_page_classifier.lobid_search import search_by_isbn
from toc_page_classifier.oa_repository import resolve_pdf_bitstream

REPO_ROOT = Path(__file__).resolve().parent.parent
OAPEN_CACHE = REPO_ROOT / "data" / "corpus" / "pilot" / ".oapen-cache" / "books.jsonl"
DOAB_CACHE = REPO_ROOT / "data" / "corpus" / "pilot" / ".doab-cache" / "books.jsonl"
LOBID_CACHE_DIR = REPO_ROOT / "data" / "corpus" / "pilot" / ".lobid-cache"
OUT_MANIFEST = REPO_ROOT / "data" / "corpus" / "pilot" / "manifest.json"

OAPEN_REST_BASE = "https://library.oapen.org"
DOAB_REST_BASE = "https://directory.doabooks.org"

DEFAULT_CAP_FRACTIONS = {
    "language": 0.4,
    "domain_bucket": 0.3,
    "volume_type": 0.6,
}


def load_oa_index() -> dict[str, dict]:
    """isbn -> {source, handle, title, rest_base}, OAPEN preferred over DOAB
    on an ISBN collision (OAPEN more reliably hosts a direct PDF copy)."""
    index: dict[str, dict] = {}
    for cache_path, source, rest_base in [
        (DOAB_CACHE, "doab", DOAB_REST_BASE),
        (OAPEN_CACHE, "oapen", OAPEN_REST_BASE),  # loaded second so it wins ties
    ]:
        if not cache_path.exists():
            continue
        for line in cache_path.open():
            rec = json.loads(line)
            for isbn in rec["isbns"]:
                index[isbn] = {"source": source, "handle": rec["handle"], "title": rec["title"], "rest_base": rest_base}
    return index


def load_cached_records() -> list[tuple[str, dict]]:
    """Returns (isbn, record) pairs, using the filename's ISBN (the one
    actually matched against the OA index at acceptance time) rather than
    re-deriving it from the record's own ISBN list -- a lobid record often
    carries several ISBN variants (print/ebook, 10/13-digit), and its own
    list's first entry isn't necessarily the one that matched."""
    if not LOBID_CACHE_DIR.exists():
        return []
    return [
        (p.name.removesuffix(".lobid.json"), json.loads(p.read_text()))
        for p in sorted(LOBID_CACHE_DIR.glob("*.lobid.json"))
    ]


def build_manifest_entry(isbn: str, features: dict, oa_entry: dict, resolved) -> dict:
    return {
        "isbn": isbn,
        "dnb_title": features["title"],
        "dnb_toc_download_url": features["toc_download_url"],
        "language": features["language"],
        "volume_type": features["volume_type"],
        "domain_bucket": features["domain_bucket"],
        "oa_source": oa_entry["source"],
        "oa_handle": oa_entry["handle"],
        "oa_title": oa_entry["title"],
        "oa_pdf_url": resolved.url,
        "oa_license_code": resolved.license_code,
    }


def recover_from_cache(oa_index: dict[str, dict], sampler: DiversitySampler) -> tuple[list[dict], set[str]]:
    """Seeds the sampler from whatever's already in `.lobid-cache/` (a
    previous, possibly-killed run) without re-doing any live lobid search --
    only a one-time PDF-bitstream re-resolution per cached record."""
    recovered = []
    seen_isbns: set[str] = set()
    for isbn, record in load_cached_records():
        if isbn in seen_isbns:
            continue
        seen_isbns.add(isbn)
        if isbn not in oa_index:
            print(f"  (skip cached {isbn}: no longer in OA index)")
            continue
        features = extract_features(record)
        if not sampler.would_accept(features):
            print(f"  (skip cached {isbn}: no longer fits diversity caps)")
            continue
        oa_entry = oa_index[isbn]
        try:
            resolved = resolve_pdf_bitstream(oa_entry["rest_base"], oa_entry["handle"])
        except Exception as e:
            print(f"  (skip cached {isbn}: bitstream re-resolution failed: {e})")
            continue
        if resolved is None:
            print(f"  (skip cached {isbn}: PDF no longer resolvable)")
            continue
        # Only now, with a confirmed real PDF link in hand, does this candidate
        # actually consume a diversity slot -- offering any earlier (as a prior
        # version of this function did) let a resolution failure silently burn
        # a slot with nothing to show for it, letting the sampler believe it was
        # full while the manifest held fewer real entries than the target.
        sampler.offer(features)
        recovered.append(build_manifest_entry(isbn, features, oa_entry, resolved))
    return recovered, seen_isbns


def ordered_isbns(oa_index: dict[str, dict], already_seen: set[str], seed: int) -> list[str]:
    rest = [isbn for isbn in oa_index if isbn not in already_seen]
    random.Random(seed).shuffle(rest)
    return rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--max-checked", type=int, default=None, help="Stop after checking this many ISBNs even if target isn't reached")
    parser.add_argument("--language-cap-fraction", type=float, default=DEFAULT_CAP_FRACTIONS["language"])
    parser.add_argument("--domain-cap-fraction", type=float, default=DEFAULT_CAP_FRACTIONS["domain_bucket"])
    parser.add_argument("--volume-type-cap-fraction", type=float, default=DEFAULT_CAP_FRACTIONS["volume_type"])
    args = parser.parse_args()
    cap_fractions = {
        "language": args.language_cap_fraction,
        "domain_bucket": args.domain_cap_fraction,
        "volume_type": args.volume_type_cap_fraction,
    }

    oa_index = load_oa_index()
    print(f"OA index: {len(oa_index)} unique ISBNs (OAPEN + DOAB)")

    LOBID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sampler = DiversitySampler(target_size=args.target, cap_fractions=cap_fractions)
    accepted, already_seen = recover_from_cache(oa_index, sampler)
    if accepted:
        OUT_MANIFEST.write_text(json.dumps({"books": accepted}, indent=2, ensure_ascii=False))
        print(f"Recovered {len(accepted)} candidates from .lobid-cache/ (no live lobid search needed for these).")
    isbns = ordered_isbns(oa_index, already_seen, args.seed)
    checked = 0
    lobid_hits = 0

    for isbn in isbns:
        if sampler.is_full():
            break
        if args.max_checked is not None and checked >= args.max_checked:
            print(f"Stopping: reached --max-checked {args.max_checked}")
            break
        checked += 1

        try:
            record = search_by_isbn(isbn)
        except Exception as e:
            print(f"  {isbn}: lobid lookup failed ({e})")
            continue
        finally:
            time.sleep(args.sleep_seconds)

        if record is None or not record_matches(record):
            continue
        lobid_hits += 1
        features = extract_features(record)
        if not sampler.would_accept(features):
            continue

        oa_entry = oa_index[isbn]
        try:
            resolved = resolve_pdf_bitstream(oa_entry["rest_base"], oa_entry["handle"])
        except Exception as e:
            print(f"  {isbn}: bitstream lookup failed ({e})")
            resolved = None
        if resolved is None:
            continue  # "of course it must be open access" -- no resolvable PDF, doesn't count

        sampler.offer(features)
        (LOBID_CACHE_DIR / f"{isbn}.lobid.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        accepted.append(build_manifest_entry(isbn, features, oa_entry, resolved))
        OUT_MANIFEST.write_text(json.dumps({"books": accepted}, indent=2, ensure_ascii=False))
        print(f"  + {isbn} [{sampler.accepted_count}/{args.target}] "
              f"lang={features['language']} domain={features['domain_bucket']} type={features['volume_type']}")

    print(f"\nChecked {checked} ISBNs live against lobid ({lobid_hits} had a matching Book+TOC record).")
    print(f"Accepted {sampler.accepted_count}/{args.target}. Diversity counts: {sampler.counts_summary()}")
    print(f"Wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
