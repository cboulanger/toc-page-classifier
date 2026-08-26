# TOC Page Classifier

A classifier that locates the table-of-contents page(s) inside a full-text
book PDF, plus the ground-truth generation pipeline it's trained on. Both
live in this one repo: the discovery/matching/localization pipeline below
exists specifically to feed this classifier's training data.

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
   `data/toc_keywords.json`. Already run once against the full merged
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
  catalog redirect page (a DNB-side data quality issue, not a bug in this
  pipeline). Not retried automatically -- a dead reference, not a
  transient fetch error.

None of the 95 `located` entries have been manually verified beyond a
couple of spot-checks -- `"verified": false` on every one.

**Classifier: first measured LOBO result (`cli/train_toc_classifier.py`).**
Trained and leave-one-book-out evaluated over the merged ground truth
(89 `chapter_segmentation` books + 95 `dnb_located` pairs; 181 of them
have at least one true TOC page and were scored):

| Model | Top-1 | Top-3 |
| --- | --- | --- |
| `logistic_regression` (default) | 3.9% | 5.0% |
| `gradient_boosting` | 9.4% | 9.4% |

By corpus:

| Corpus | `logistic_regression` top1 / top3 | `gradient_boosting` top1 / top3 |
| --- | --- | --- |
| copyrighted-scans (n=29) | 6.9% / 6.9% | 13.8% / 13.8% |
| dnb_located (n=95) | 5.3% / 7.4% | 13.7% / 13.7% |
| open-access (n=57) | 0.0% / 0.0% | 0.0% / 0.0% |

Both models score a flat 0% on the entire open-access corpus. This is
root-caused, not a mystery left dangling: `text_features.py`'s
`_TOC_LINE_RE` requires 2+ consecutive dot/whitespace separator
characters between a TOC entry's title and its page number, but
`pypdf`'s text extraction on born-digital ("native") open-access PDFs
collapses that visual gap to a single space (e.g. `"Foreword vii"`), so
the regex essentially never matches on native-extraction text --
measured directly at 0 of 80 TOC-line-shaped lines sampled across 6 real
open-access books, versus 18.6%/9.1% match rates on the OCR'd
`dnb_located`/`copyrighted-scans` corpora, where literal repeated
separator glyphs do survive extraction (see the comment above
`_TOC_LINE_RE` in `src/toc_page_classifier/text_features.py`).
`gradient_boosting`'s higher headline numbers (9.4% vs. 3.9%) come
entirely from doing better on the corpora where that text feature
already fires (13.7-13.8% there, vs. `logistic_regression`'s 5.3-6.9%) --
it is not solving the open-access blind spot either, it is still 0.0%
there too.

Neither model is anywhere close to production-useful at these hit
rates -- this is a first-measurement checkpoint, not a finished
deliverable. The single most actionable next step is fixing
`_TOC_LINE_RE` to also match a single-space-collapsed title/page-number
gap (e.g. a minimum-gap-width heuristic derived from some other signal,
or a looser separator requirement paired with a compensating precision
check), since that -- not model choice -- is what suppresses the
open-access half of the corpus to zero. The design spec's already-listed
deferred directions (a small LM/VLM finetune, per-`extraction_type`
calibration) remain candidates too, but should now be weighed against
this more specific, measured gap.

**Known gaps / not yet done:**

- Discovery only checks ISBNs already known to OAPEN/DOAB; it never
  streams the full lobid-resources dump, so it can't find DNB TOC scans
  for OA books lobid doesn't already know the ISBN link for by other
  means. In practice this hasn't been a limiting factor (100/100 found
  entirely via live per-ISBN lookups).
- No OCR/vision fallback for the 3 `reference_has_no_text` books.
- The classifier's text features are effectively blind on native-text
  (non-OCR) PDFs -- see the open-access result above and
  `_TOC_LINE_RE`'s known limitation in `text_features.py`.
- No page-level precision/recall diagnostic yet, though the design spec
  calls for one (as a secondary metric to distinguish "wrong boundary"
  from "completely missed") -- `evaluate_leave_one_book_out` currently
  only reports the top-1/top-3 hit booleans described above. Would help
  explain *why* a book's range prediction failed, not just that it did.

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
