# CLI scripts

Every script accepts `-h`/`--help`. **This file must be kept in sync with
each script's own `--help` output whenever a script changes -- see
`AGENTS.md`.**

## Using the classifier

Just want predicted TOC pages for a PDF? You don't need any script on this
page -- call `toc_page_classifier.predict.locate_toc_pages(pdf_path)`
directly (see the main `README.md`'s "Usage" section). The one script here
is for producing/refreshing the model that call loads:

### `train_final_model.py`

Fits the deployable model on the FULL merged ground truth (no held-out
book, unlike `train_toc_classifier.py`'s LOBO evaluation below) and
serializes it to `src/toc_page_classifier/data/model.pkl`, the bundled
package data `toc_page_classifier.predict.locate_toc_pages` loads at
import time. Defaults to `gradient_boosting` (the stronger model on
top-1 hit rate, per `README.md`'s current-status table). Must be run as
`python -m cli.train_final_model`, not `python cli/train_final_model.py`
directly -- see the script's own docstring for why.

```
usage: python3 -m cli.train_final_model [-h] [--corpus-dir CORPUS_DIR]
                                        [--model {logistic_regression,gradient_boosting}]
                                        [--rebuild-features] [--out OUT]

Fits the deployable TOC-page model on the FULL merged ground truth (no
held-out book, unlike train_toc_classifier.py's leave-one-book-out
evaluation) and serializes it to src/toc_page_classifier/data/model.pkl,
where toc_page_classifier.predict.locate_toc_pages loads it from.

options:
  -h, --help            show this help message and exit
  --corpus-dir CORPUS_DIR
                        Additional expected-json evaluation corpus
                        (repeatable) -- a single corpus directory, or a root
                        containing several named ones. Corpora under this
                        repo's own data/corpus/ are auto-discovered and don't
                        need this flag.
  --model {logistic_regression,gradient_boosting}
  --rebuild-features
  --out OUT
```

## Training/evaluating the classifier

The pipeline that produces and evaluates the classifier's training data,
run in the order below.

### `harvest_oapen.py`

Harvests OAPEN's public OAI-PMH "Books" set (~49.8k records) into a local
`isbn -> handle` cache. Takes several minutes; not run on every commit, only
periodically to refresh the OA ISBN pool.

```
usage: harvest_oapen.py [-h] [--out OUT]

Harvests OAPEN's "Books" OAI-PMH set into a local cache of ISBN -> handle
records, for `discover_oa_dnb_candidates.py` (or `match_dnb_oa.py`) to
intersect against DNB TOC-scan records. uv run python cli/harvest_oapen.py Re-
running overwrites the cache from scratch -- OAPEN's Books set has ~49.8k
records as of 2026-08-25, so a full run takes several minutes and issues ~500
HTTP requests against library.oapen.org. Not something to run on every commit;
re-run it periodically (e.g. before a `discover_oa_dnb_candidates.py` pass) to
pick up newly-added OAPEN books.

options:
  -h, --help  show this help message and exit
  --out OUT
```

### `harvest_doab.py`

Same mechanism as `harvest_oapen.py`, against DOAB's "Books" set (~72.5k
records). DOAB partially overlaps OAPEN and often has no directly
downloadable PDF of its own (it frequently just links out to a publisher's
site).

```
usage: harvest_doab.py [-h] [--out OUT]

Harvests DOAB's "Books" OAI-PMH set the same way `harvest_oapen.py` harvests
OAPEN's -- see that script's docstring for the shared mechanism. uv run python
cli/harvest_doab.py DOAB's Books set has ~72.5k records as of 2026-08-25 (~725
requests, several minutes). DOAB largely mirrors OAPEN plus other
publishers/repositories, so expect real but partial overlap with
`harvest_oapen.py`'s output -- and unlike OAPEN, many DOAB records have no
directly resolvable full-text PDF at all (DOAB is a directory that often links
out to a publisher's own site rather than hosting a copy);
`discover_oa_dnb_candidates.py` records this rather than silently dropping
such matches.

options:
  -h, --help  show this help message and exit
  --out OUT
```

### `discover_oa_dnb_candidates.py`

The actual discovery path: for every OA ISBN from the two harvest caches,
checks live against lobid.org for a matching DNB TOC scan, applies
diversity caps (language / RVK domain bucket / volume type), and only
counts a candidate once its OA PDF link actually resolves. Resumable by
construction -- safe to kill and re-run with different `--*-cap-fraction`
values; already-found candidates carry over via `data/corpus/pilot/.lobid-cache/`.
Owns `data/corpus/pilot/manifest.json`.

```
usage: discover_oa_dnb_candidates.py [-h] [--target TARGET] [--seed SEED]
                                     [--sleep-seconds SLEEP_SECONDS]
                                     [--max-checked MAX_CHECKED]
                                     [--language-cap-fraction LANGUAGE_CAP_FRACTION]
                                     [--domain-cap-fraction DOMAIN_CAP_FRACTION]
                                     [--volume-type-cap-fraction VOLUME_TYPE_CAP_FRACTION]

Builds a diverse sample of at least `--target` books that are BOTH
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

options:
  -h, --help            show this help message and exit
  --target TARGET
  --seed SEED
  --sleep-seconds SLEEP_SECONDS
  --max-checked MAX_CHECKED
                        Stop after checking this many ISBNs even if target
                        isn't reached
  --language-cap-fraction LANGUAGE_CAP_FRACTION
  --domain-cap-fraction DOMAIN_CAP_FRACTION
  --volume-type-cap-fraction VOLUME_TYPE_CAP_FRACTION
```

### `match_dnb_oa.py`

An older, narrower alternative: an offline-only intersection against
whatever local DNB manifest `DNB_TOC_CORPUS_DIR` points at, with no live
lobid search and no diversity sampling. Writes to its own `--out` file
(default `data/corpus/pilot/match_dnb_oa_check.json`, gitignored) --
**never** to `manifest.json`, which `discover_oa_dnb_candidates.py` owns.
Useful as a cheap sanity check when a local DNB manifest happens to be
available; not required for the main pipeline.

```
usage: match_dnb_oa.py [-h] [--out OUT]

Intersects the harvested OAPEN/DOAB ISBN caches (`harvest_oapen.py`,
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

options:
  -h, --help  show this help message and exit
  --out OUT
```

### `fetch_pairs.py`

Downloads both PDFs (DNB TOC scan + OA full text) for every matched pair in
`manifest.json`. Idempotent -- skips a pair whose files already exist, and
skips a pair with no resolved `oa_pdf_url` outright.

```
usage: fetch_pairs.py [-h]

Downloads both PDFs (DNB TOC scan + OA full text) for every matched pair
in `data/corpus/pilot/manifest.json` (written by `discover_oa_dnb_candidates.py`), into
`data/corpus/pilot/pdf/<isbn>.dnb_toc.pdf` and `<isbn>.fulltext.pdf`.

    uv run python cli/fetch_pairs.py

PDFs are gitignored (see `.gitignore`). Skips a pair entirely if
`oa_pdf_url` is null (no resolvable OA PDF link yet -- see
`match_dnb_oa.py`'s "NO DIRECT PDF LINK" entries) or if both files already
exist locally.

options:
  -h, --help  show this help message and exit
```

### `locate_toc.py`

Runs the text-overlap locator against every downloaded pair, writing one
ground-truth JSON per book. Never aborts the whole batch on a single bad
PDF -- a per-book error is recorded as `"status": "error"` and processing
continues.

```
usage: locate_toc.py [-h]

Runs the text-overlap TOC locator (`src/toc_page_classifier/locate_toc.py`)
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

options:
  -h, --help  show this help message and exit
```

### `mine_toc_keywords.py`

Empirically mines candidate multilingual TOC-heading keywords (e.g. "contents",
"inhaltsverzeichnis") from the merged ground truth corpus, grouped by each
book's declared language, and writes `data/toc_keywords.candidates.json` for a
human review pass -- never writes to the committed
`src/toc_page_classifier/data/toc_keywords.json` directly. Run once/occasionally,
not part of the per-book pipeline above; safe to re-run whenever the ground
truth corpus grows to pick up new languages or phrasings.

```
usage: mine_toc_keywords.py [-h] [--min-count MIN_COUNT]

Empirically mines candidate TOC-heading keywords from the merged ground truth
corpus, grouped by each book's declared language -- writes
data/toc_keywords.candidates.json for a HUMAN REVIEW PASS. Never writes to
src/toc_page_classifier/data/toc_keywords.json directly: a frequent short
phrase found here still needs a human judgment call on whether it's really a
TOC-heading phrase (not, e.g., a frequent but unrelated short word). See
docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md's "Text /
structural" section.

options:
  -h, --help            show this help message and exit
  --min-count MIN_COUNT
                        Minimum frequency to write as a candidate (default:
                        2).
```

### `train_toc_classifier.py`

Leave-one-book-out (LOBO) training/evaluation of the TOC-page classifier
itself: merges both ground-truth sources
(`src/toc_page_classifier/ground_truth.py`), extracts per-page layout
(`layout_features.py`) and text (`text_features.py`) features, trains a
page-level scorer (`LogisticRegression` by default, or gradient boosting),
selects top-K non-overlapping candidate page ranges
(`range_selection.py`), and reports top-1/top-3 range-hit rates overall and
broken down by corpus and `extraction_type`. See `README.md`'s "Current
status" section for the first measured result.

```
usage: train_toc_classifier.py [-h] [--corpus-dir CORPUS_DIR]
                               [--model {logistic_regression,gradient_boosting}]
                               [--rebuild-features]

Leave-one-book-out evaluation of the TOC-page classifier: per-page
layout + text features, a page-level scorer (LogisticRegression or
gradient boosting), and top-K non-overlapping candidate page-range output.
See docs/superpowers/specs/2026-08-25-toc-page-classifier-design.md.

options:
  -h, --help            show this help message and exit
  --corpus-dir CORPUS_DIR
                        Additional expected-json evaluation corpus
                        (repeatable) -- a single corpus directory, or a root
                        containing several named ones. Corpora under this
                        repo's own data/corpus/ are auto-discovered and don't
                        need this flag.
  --model {logistic_regression,gradient_boosting}
  --rebuild-features    Force a fresh feature-table build instead of using the
                        cached one on disk (needed after a feature-extraction
                        code change; a book-key-set change is detected
                        automatically).
```
