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

1. **Discovery** (`cli/harvest_oapen.py`, `cli/harvest_doab.py`): harvest
   OAPEN's and DOAB's public OAI-PMH "Books" sets (DSpace-based OA book
   repositories, ~50k and ~72k records respectively) into a local
   ISBN → handle cache.
2. **Matching** (`cli/match_dnb_oa.py`): intersect those caches' ISBNs
   against the DNB TOC-scan corpus's own ISBNs (via the sibling
   `dnb-toc-ground-truth` checkout), resolve each match's actual PDF
   download link via the repository's DSpace REST API, and write
   `data/corpus/pilot/manifest.json`.
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

## Current status (2026-08-25, initial exploration)

Discovery so far, run against the `dnb-toc-ground-truth` pilot corpus's
1,251 books (**not** the full DNB "Kataloganreicherung" universe, which is
far larger but requires an hours-long full lobid-dump pull -- not done yet):

| Source | Records harvested | DNB-corpus matches |
| --- | --- | --- |
| OAPEN | 49,814 | 15 (14 unique ISBNs, 1 book listed twice under two OAPEN records) |
| DOAB | 72,511 | 7 (0 overlap with OAPEN's matches) |
| **Combined** | | **~21 unique books** |

Of the resolvable pairs downloaded and tested so far (OAPEN subset, 14
books):

- **10/14**: both the DNB TOC scan and the OA full text have a real,
  OCR'd/native text layer. Plain word-token Jaccard overlap between the
  DNB scan's text and every full-text page cleanly isolates the true TOC
  page range at the top of the ranking, with a large score margin over the
  next-best page (e.g. top score 0.67 vs. runner-up 0.31). Spot-checked one
  match by hand (`9781447324775`, page 5) -- genuinely the "Contents" page.
- **3/14**: the DNB TOC scan itself is a pure image with no OCR/text layer
  at all (confirmed via embedded-image inspection). Text-overlap matching
  cannot work here at all -- this is the scanned-TOC case that needs an
  OCR-first or vision-embedding-based fallback, not implemented yet.
- **1/14**: PDF download failed (self-signed TLS cert on `d-nb.info`) --
  a fixable technical issue, not investigated further yet.

**Known gaps / not yet done:**

- The full lobid-resources dump (~21.5GB) hasn't been pulled -- the DNB side
  of the match is currently limited to the 1,251-book pilot subset, not the
  full DNB TOC-scan universe. Doing this would very likely multiply the
  match count.
- No OCR/vision fallback for the 3 known un-OCR'd DNB TOC scans.
- `cli/fetch_pairs.py` and `cli/locate_toc.py` (see Methodology above)
  process pairs from the manifest one at a time -- not yet run as a batch
  and written to `data/corpus/pilot/ground-truth/`.
- No classifier training code yet -- this repo is currently at the
  ground-truth-generation stage.

## Development

```bash
uv sync
uv run python cli/harvest_oapen.py   # ~5 min, ~500 requests
uv run python cli/harvest_doab.py    # ~10 min, ~725 requests
uv run python cli/match_dnb_oa.py    # requires a sibling ../dnb-toc-ground-truth checkout
uv run python cli/fetch_pairs.py
uv run python cli/locate_toc.py
uv run pytest
```

`DNB_TOC_CORPUS_DIR` overrides the default sibling-checkout path for the
DNB manifest, same convention as `chapter-segmentation/evaluation/harness.py`.
