# TOC Page Classifier

A classifier that locates the table-of-contents page(s) inside a full-text
book PDF, plus the ground-truth generation pipeline it's trained on. Unlike
the [dnb-toc-ground-truth](https://github.com/cboulanger/dnb-toc-ground-truth)
repo (which produces TOC-entry ground truth and deliberately leaves every
consumer -- fine-tuning, classifiers -- to a separate repo), **this repo
keeps ground-truth generation and the classifier itself together**: the
discovery/matching/localization pipeline below exists specifically to feed
this classifier, not as a general-purpose, repo-agnostic dataset.

## Motivation

A layout-based TOC-page classifier pilot already exists in the
`chapter-segmentation` project
(`evaluation/experiments/toc-classifier-pilot.md`) and has never cleared its
own decision bar, in part for lack of ground truth beyond ~90 hand-verified
books. This repo explores a different, more scalable way to generate that
ground truth: the Deutsche Nationalbibliothek's CC0-licensed
"Kataloganreicherung" program has independently scanned a TOC excerpt for
over a million German-catalog books (a small pilot subset lives in the
sibling `dnb-toc-ground-truth` repo). For any such book that also has an
**open-access full-text PDF** available elsewhere, the DNB scan's TOC text
can be automatically located inside that full text -- turning a one-time
manual-transcription bottleneck into a scalable, automatable pairing.

## Methodology

1. **OA harvest** (`cli/harvest_oapen.py`, `cli/harvest_doab.py`): harvest
   OAPEN's and DOAB's public OAI-PMH "Books" sets (DSpace-based OA book
   repositories, ~50k and ~72k records respectively) into a local
   ISBN → handle cache -- the pool of candidate ISBNs worth checking against
   DNB.
2. **Diverse discovery** (`cli/discover_oa_dnb_candidates.py`): for each OA
   ISBN, check live against `lobid.org` (per-ISBN search, not the full
   ~21.5GB lobid-resources dump -- see the script's docstring for why) for a
   matching DNB "Kataloganreicherung" TOC scan, applying simple per-value
   caps on language / RVK domain bucket / edited-volume-vs-monograph-vs-thesis
   so no single value dominates the sample (`src/toc_page_classifier/diversity_sampler.py`
   -- caps, not real stratified balancing). Resumable by construction: every
   accepted record's raw lobid JSON is cached immediately
   (`data/corpus/pilot/.lobid-cache/`, gitignored) and the manifest is
   rewritten after every acceptance, so killing a long run to relax the caps
   never loses progress already found. Writes `data/corpus/pilot/manifest.json`.

   (`cli/match_dnb_oa.py` is an older, narrower variant: no live lobid
   search, no diversity sampling, just an offline intersection against the
   sibling `dnb-toc-ground-truth` checkout's 1,251-book pilot subset --
   kept as a cheap sanity check, superseded by `discover_oa_dnb_candidates.py`
   as the actual discovery path.)
3. **Fetching** (`cli/fetch_pairs.py`): download both PDFs (the DNB TOC scan
   and the OA full text) for every matched pair.
4. **Localization** (`cli/locate_toc.py`,
   `src/toc_page_classifier/locate_toc.py`): score every full-text page
   against the DNB TOC scan's own text via word-token Jaccard overlap, and
   report the highest-scoring contiguous page range as the located TOC.
5. **Classifier training** (not yet implemented): once enough located pairs
   accumulate, train the actual TOC-page classifier on this ground truth
   (a natural next evolution of `chapter-segmentation`'s existing
   layout-based pilot, which this repo's ground truth is meant to feed).

## Current status (2026-08-25)

A 100-book diverse sample, found by checking every OAPEN/DOAB OA ISBN
(158,909 unique) live against lobid for a matching DNB TOC scan, with
per-value caps of 50% (language), 40% (domain bucket), 70% (volume type):

| Dimension | Breakdown |
| --- | --- |
| language | en=42, de=40, it=6, es=3, fr=2, nl=1 |
| volume_type | monograph=59, edited_volume=25, thesis=16 |
| domain_bucket (RVK top-level, coarse) | unknown=32 (not RVK-classified), MS=7, LB=5, AP=3, EC=3, ~40 other buckets at 1-2 each |

Every one of the 100 matched pairs was downloaded and run through the
text-overlap locator:

- **95/100 located**: a clean, high-margin text-overlap match (see
  `data/corpus/pilot/ground-truth/*.json`, `"status": "located"`).
- **3/100 `reference_has_no_text`**: the DNB TOC scan itself is a pure
  image with no OCR/text layer -- needs an OCR-first or
  vision-embedding-based fallback, not implemented yet.
- **2/100 `error`**: the DNB `toc_download_url` didn't actually serve a
  PDF -- one returned an empty body, the other an HTML "externer Link"
  catalog redirect page (a known DNB corpus artifact, not a bug here; see
  `dnb-toc-ground-truth`'s own acquisition history for the same failure
  mode). Not retried automatically -- a dead reference, not a transient
  fetch error.

None of the 95 `located` entries have been manually verified beyond a
couple of spot-checks -- `"verified": false` on every one.

**Known gaps / not yet done:**

- Discovery only checks ISBNs already known to OAPEN/DOAB; it never
  streams the full lobid-resources dump, so it can't find DNB TOC scans
  for OA books lobid doesn't already know the ISBN link for by other
  means. In practice this hasn't been a limiting factor (100/100 found
  entirely via live per-ISBN lookups).
- No OCR/vision fallback for the 3 `reference_has_no_text` books.
- No classifier training code yet -- this repo is currently at the
  ground-truth-generation stage.

## Development

```bash
uv sync
uv run python cli/harvest_oapen.py            # ~5 min, ~500 requests
uv run python cli/harvest_doab.py             # ~10 min, ~725 requests
uv run python cli/discover_oa_dnb_candidates.py --target 100  # live per-ISBN lobid checks; resumable, see its docstring
uv run python cli/fetch_pairs.py
uv run python cli/locate_toc.py
uv run pytest
```

`DNB_TOC_CORPUS_DIR` overrides the default sibling-checkout path for the
DNB manifest, same convention as `chapter-segmentation/evaluation/harness.py`.
