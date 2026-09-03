# TOC Page Classifier

A classifier that locates the table-of-contents page(s) inside a full-text
book PDF, plus the ground-truth generation pipeline it's trained on. Both
live in this one repo: the discovery/matching/localization pipeline below
exists specifically to feed this classifier's training data.

## Usage

```python
from toc_page_classifier.predict import locate_toc_pages

page_indices = locate_toc_pages("path/to/book.pdf")  # e.g. [6, 7, 8]
```

`locate_toc_pages` returns the predicted 0-based TOC page indices (an
empty list if the PDF has no extractable pages), using the model bundle
committed at `src/toc_page_classifier/data/model.pkl` (~370 KB, shipped
as package data -- see `cli/train_final_model.py` to retrain it). Pass
`language="de"` (an ISO language code) if you know the book's language,
for a small keyword-matching feature; omit it if you don't.

By default only the first 30 and last 20 pages are actually parsed
(`head_pages=30, tail_pages=20`) -- a real TOC is essentially never buried
in a long book's interior, and that interior is what makes a long PDF slow
to process (each page needs a full layout pass). Pass `head_pages=None,
tail_pages=None` to scan every page instead, or your own page counts.

## Demo

Try it hosted: [cmboulanger/toc-page-classifier-demo](https://huggingface.co/spaces/cmboulanger/toc-page-classifier-demo)
-- upload a full book PDF, or try one of the example open-access books
(one per language).

The demo (`space/`) is a thin Gradio wrapper around `locate_toc_pages` --
see `space/README.md`. Run it locally:

```bash
uv run --with gradio python space/app.py
```

Deploy (or redeploy) it with `cli/upload_space.py` (see `cli/README.md`).

## Motivation

The Deutsche Nationalbibliothek's CC0-licensed "Kataloganreicherung" program
has independently scanned a TOC excerpt for over a million German-catalog
books. For any such book that also has an **open-access full-text PDF**
available elsewhere, the DNB scan's TOC text can be automatically located
inside that full text -- turning what would otherwise be a one-time
manual-transcription bottleneck into a scalable, automatable way to generate
ground truth for a TOC-page classifier.

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
   search, no diversity sampling, just an offline intersection against a
   local DNB TOC-scan manifest -- see `cli/README.md` for its exact
   requirements. Kept as a cheap sanity check, superseded by
   `discover_oa_dnb_candidates.py` as the actual discovery path.)
3. **Fetching** (`cli/fetch_pairs.py`): download both PDFs (the DNB TOC scan
   and the OA full text) for every matched pair.
4. **Localization** (`cli/locate_toc.py`,
   `src/toc_page_classifier/locate_toc.py`): score every full-text page
   against the DNB TOC scan's own text via word-token Ochiai overlap, and
   report the highest-scoring contiguous page range as the located TOC.
5. **Keyword mining** (`cli/mine_toc_keywords.py`, run once/occasionally,
   not on every commit): empirically scans the merged ground truth for
   frequent TOC-heading phrases per language, writing candidates to `data/toc_keywords.candidates.json` for a human review pass before merging into
   `src/toc_page_classifier/data/toc_keywords.json`. Already run once against the full merged
   corpus (2026-08-25): every frequent candidate found was already in the
   hand-seeded list, and everything else was noise (page numbers, OCR
   garbage, one unrelated word).
6. **Classifier training/evaluation** (`cli/train_toc_classifier.py`,
   `src/toc_page_classifier/{layout_features,text_features,ground_truth,range_selection}.py`):
   merges both ground-truth sources, trains a page-level scorer, and
   reports leave-one-book-out top-1/top-3 range-hit rates -- a
   predicted range counts as a "hit" only if it fully contains every true
   TOC page (over-inclusion is tolerated -- the classifier's job is
   narrowing the page list for a downstream parser, not exact boundaries
   -- but missing even one true page is not).

## Current status

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
  catalog redirect page (a DNB-side data quality issue, not a bug in this
  pipeline). Not retried automatically -- a dead reference, not a
  transient fetch error.

None of the 95 `located` entries have been manually verified beyond a
couple of spot-checks -- `"verified": false` on every one.

**Classifier performance** (`cli/train_toc_classifier.py`, full LOBO over
the merged ground truth -- 89 `chapter_segmentation` books + 95
`dnb_located` pairs; 181 of them have at least one true TOC page and were
scored):

| Model | Top-1 | Top-3 |
| --- | --- | --- |
| `logistic_regression` (default) | 85.6% | 96.1% |
| `gradient_boosting` | 90.1% | 92.3% |

By corpus:

| Corpus | `logistic_regression` top1 / top3 | `gradient_boosting` top1 / top3 |
| --- | --- | --- |
| copyrighted-scans (n=29) | 89.7% / 100.0% | 93.1% / 96.6% |
| dnb_located (n=95) | 85.3% / 94.7% | 88.4% / 91.6% |
| open-access (n=57) | 84.2% / 96.5% | 91.2% / 91.2% |

By extraction_type (chapter_segmentation rows only):

| extraction_type | `logistic_regression` top1 / top3 | `gradient_boosting` top1 / top3 |
| --- | --- | --- |
| native (n=74) | 83.8% / 97.3% | 91.9% / 93.2% |
| scan (n=12) | 100.0% / 100.0% | 91.7% / 91.7% |

See `docs/history/classifier-results.md` for how this was reached --
the 2026-08-25 baseline (single digits, one root cause), the
2026-08-26 gap-aware-text fix, and the 2026-08-27 range-selection fix
that was the actual turning point.

**Known gaps / not yet done:**

- The bundled `src/toc_page_classifier/data/model.pkl` -- and the LOBO
  numbers above -- were fitted on features extracted by `pdfplumber`,
  which `layout_features.py` replaced with `pdfalto` on 2026-09-04. The
  feature *names* are unchanged, but their values shift (ALTO word boxes
  and its own line segmentation, against pdfplumber's raw per-char
  geometry), so the shipped model is being fed slightly
  out-of-distribution inputs until it is refitted. Re-run
  `cli/fetch_pairs.py`, then `cli/train_toc_classifier.py
  --rebuild-features` and `cli/train_final_model.py`, to close this.
- Discovery only checks ISBNs already known to OAPEN/DOAB; it never
  streams the full lobid-resources dump, so it can't find DNB TOC scans
  for OA books lobid doesn't already know the ISBN link for by other
  means. In practice this hasn't been a limiting factor (100/100 found
  entirely via live per-ISBN lookups).
- No OCR/vision fallback for the 3 `reference_has_no_text` books.
- None of the 95 `dnb_located` entries have been manually verified beyond
  a couple of spot-checks (`"verified": false` on every one) -- the LOBO
  numbers above trust that ground truth as-is.
- The remaining top1_hit misses (and the top1_hit-vs-top3_hit gap) haven't
  been individually diagnosed -- `best_true_page_rank` and the overlap
  metrics narrow down *where* to look (scorer vs. range selection) but no
  per-book failure analysis has been done yet.

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

See `cli/README.md` for a description of every script under `cli/`, including
full `--help` output and what `DNB_TOC_CORPUS_DIR` (used by `cli/match_dnb_oa.py`)
expects.

---

Agents working in this repository should read `AGENTS.md` before making changes.
