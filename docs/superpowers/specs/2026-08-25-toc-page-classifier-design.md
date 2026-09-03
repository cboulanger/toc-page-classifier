# TOC-page classifier design

## Motivation

`toc-page-classifier`'s methodology (see `README.md`) ends at step 4,
localization: given a full-text PDF *and* a separate DNB reference TOC
scan of the same book, find which full-text pages contain the same TOC
text via word-token Jaccard overlap. That only works for books that
already have a matching DNB scan. Step 5 -- "classifier training" -- has
never been implemented: a model that, given *any* book PDF with a usable
text layer, predicts which of its own pages contain the table of
contents, with no reference scan required at all. This is what lets the
pipeline (or any other consumer) locate a TOC for an arbitrary book, not
just the subset lucky enough to have a DNB cross-reference.

Constraints, from the requesting conversation:

- Small, local, fast, CPU-capable -- not a hosted-API dependency, and must
  not require a GPU to run.
- OCR is a prior pipeline step (making sure a PDF has a usable text
  layer), never something this classifier does itself.
- A separate, related pilot already exists in the `chapter-segmentation`
  repo (`evaluation/experiments/toc-classifier-pilot.md`): a
  leave-one-book-out (LOBO) classifier trained purely on page-layout
  geometry (from `pdfalto`-derived ALTO XML) that predicts both TOC pages
  and chapter-opening pages. Across eleven follow-up investigations it
  never cleared its own 90%-recall/15%-candidate-budget bar on the mixed
  89-book corpus it was tested against (best: 65.2% full-recall / 10.3%
  candidate fraction), though it *did* clear that bar on open-access books
  in isolation (94.7%/13.8%) with per-corpus threshold tuning. That
  pilot's model deliberately used geometry only, no text content -- this
  design's explicit brief is to combine text signal with geometry, since
  text was the one lever that pilot never had access to.
- Text and geometry should be tried before reaching for a small LLM or
  VLM finetune.

## Scope

This design covers a **TOC-only** classifier (no chapter-opening
prediction -- that stays a `chapter-segmentation` concern) that is a
**standalone deliverable of this repo**: it must work for a book with no
DNB reference scan available, which is the normal case this repo needs to
handle to be useful beyond its own ground-truth-generation pipeline.

**Implemented.** See
`docs/superpowers/plans/2026-08-25-toc-page-classifier-implementation.md`
for the task-by-task build of this design, and `README.md`'s "Current
status" section for the first measured leave-one-book-out result:
`logistic_regression` (the default model) gets 3.9%/5.0% top-1/top-3
range-hit rate, `gradient_boosting` gets 9.4%/9.4% -- and both are flat at
0.0% on the entire open-access corpus, root-caused to a text-extraction
gap in `_TOC_LINE_RE` (see that section and the comment above
`_TOC_LINE_RE` in `text_features.py`). Neither model is production-useful
yet; this is a first-measurement checkpoint that should inform which of
the follow-up directions below (plus the newly-identified `_TOC_LINE_RE`
fix) actually matter most, rather than an assumption made in the abstract.

Out of scope for this document (named as follow-up directions, not
designed here):

- A small multilingual encoder LM (e.g. a DistilBERT/MiniLM-class model)
  finetuned on page text, if text+geometry plateaus below a useful bar.
- A small VLM / document-image model reading rendered pages directly, to
  address the old pilot's residual scan/degenerate-OCR-font-metadata
  failure mode -- named in that pilot's own history as the untouched
  candidate direction for exactly that ceiling.
- Per-`extraction_type` (native vs. scan) calibration split, if the same
  corpus-composition gap the old pilot found reappears here. Cheap to add
  later; not designed now since it's premature before a first model
  exists to calibrate.
- Full manual verification of all 95 DNB-located ground-truth pairs.

## Ground truth

Two existing label sources get merged into one training table:

1. **chapter-segmentation's 89-book evaluation corpus** -- the `toc` field
   in each book's `.expected.json` (`{"toc_start_index": ..., "toc_end_index":
   ...}` or `null`), hand-verified, spanning both open-access and
   copyrighted-scans books.
2. **toc-page-classifier's own 95 DNB-located pairs**
   (`data/corpus/pilot/ground-truth/*.json`) -- `toc_start_index`/
   `toc_end_index` located automatically via text-token-overlap against a
   DNB reference scan, every entry currently `"verified": false`.

**Merge rule:** dedup by ISBN. If a book appears in both sources, keep
chapter-segmentation's hand-verified label (higher trust) and drop the
DNB-located duplicate.

**Confidence weighting, not blind trust:** hand-verified rows get sample
weight `1.0`. DNB-located rows get a weight derived from their own
`margin` field (`top_score - runner_up_score`, already computed by
`locate_toc.py`) -- a low-margin auto-located page is weaker evidence
than a high-margin one, and the model should be told so rather than
treating every auto-located label as equally certain. Exact weighting
function (e.g. a clipped linear map of `margin` into `[0.3, 1.0]`) is an
implementation detail to tune empirically, not fixed here.

**Spot-check done, and it found a real bug -- since fixed.** A 20%
hand-inspected sample (19 of 95 DNB-located books) found a **47% error
rate**: 8 of 9 wrong cases shared the same root cause, a systematic
one-page truncation of the true TOC range. Root-caused to `locate_toc.py`'s
scoring formula -- symmetric Jaccard overlap penalized a genuine but thin
trailing TOC page because the reference's *combined* vocabulary (unioned
across all reference pages) is much larger than that one page's own
vocabulary, pushing its score below the range-expansion threshold even
though every one of its own tokens matched the reference. Critically,
`margin` did not predict this failure -- some truncated books had high
margins -- so the weighting scheme above would not have caught it.

Fixed in two steps, both now shipped in `locate_toc.py`: the per-page score
became an Ochiai coefficient (`|intersection| / sqrt(|candidate| *
|reference|)`, which rewards a thin-but-genuine page without letting a
tiny incidental match win by raw fraction alone -- an intermediate plain-
containment attempt fixed the truncation but introduced exactly that new
failure); and `select_toc_range` moved from expanding outward from a peak
while neighbors clear a tunable score-ratio threshold, to picking the
best-scoring contiguous window of a *known* fixed length -- the DNB
reference scan's own page count, which was found to match the true TOC
range's length exactly in all 18 hand-verified cases (same edition, same
physical page count). This removes the threshold entirely rather than
re-tuning it. Re-validated against all 18 hand-verified cases: 18/18 exact
matches. The full 95-book corpus was regenerated with the fix; the
lowest-margin entries dropped from a wide, bug-correlated spread to just 4
books under `margin=0.05`, worth a light manual look but no longer a sign
of a systematic defect.

## Features

Two families, computed per page, both new relative to the old
geometry-only pilot's excluded-text-content constraint.

### Geometry (new to this repo)

Extracted via `pdfalto`, the same ALTO XML converter the old pilot used --
no longer a separately built sibling checkout, but a pip dependency whose
wheels bundle the compiled binary. (Written against `pdfplumber` first, to
avoid that build step entirely; swapped back on 2026-09-04, once the
bindings existed, for a ~6x faster extraction and explicit line/word
boxes.) A lighter re-derivation of the old pilot's proven feature set,
keeping what's TOC-relevant and dropping what was chapter-opening-specific:

- Font-size contrast: page's max line font size vs. its own modal
  (body-text) font size.
- Line density (lines per page-height fraction).
- Left-margin mean and variance across the page's lines.
- First/last text line's vertical-position fraction on the page.
- `edge_distance`: `min(page_index, total_pages - 1 - page_index)` -- the
  old pilot's single most effective late addition, added specifically to
  catch back-of-book TOCs that a front-loaded `page_position_fraction`
  can't represent (fixed a real failure case, `9782821895607`, a
  French-language book with its TOC at pages 189-190 of 193).
- Book-context features: this page's line count relative to the book's
  own median line count, and similar per-book-normalized ratios -- the old
  pilot's context-feature follow-up found these meaningfully separate
  scan noise from real signal.

### Text / structural (the new signal this design is actually about)

The old pilot never used page text content at all. Two complementary
signals:

**Structural pattern matching** -- a standalone, lightweight port of the
"3+ lines matching `title ... dot-leader/whitespace ... page-number`"
shape that `chapter_segmentation`'s own TOC-page heuristic already uses
(reimplemented here, not imported as a dependency, to keep this repo
self-contained): count and ratio of lines matching that shape, digit
density on the page, and a monotonic-page-number-sequence check across
matched lines (a real TOC's page numbers increase down the page; an index
or bibliography page can superficially match the same "words ... number"
shape without that monotonic property, per the old pilot's own
"mid-book/back-matter false positive" diagnosis).

**Multilingual keyword matching -- data-driven, not hardcoded.** Keywords
live in a standalone, human-editable resource file,
`data/toc_keywords.json`, structured as:

```json
{
  "en": ["contents", "table of contents"],
  "de": ["inhalt", "inhaltsverzeichnis"],
  "fr": ["sommaire", "table des matières"]
}
```

The feature extractor loads this file at runtime and does a
case-insensitive match against each page's first few lines. No language
is hardcoded into the matching *logic* -- adding a new language, or a new
phrasing for an existing one, is a data-file edit, not a code change.
Two features result: an any-language keyword-hit flag/count, and (when
the book's `language` field is known, from either source corpus's
manifest) a same-declared-language-specific hit flag, since a match in an
unrelated language is weaker evidence than a match in the book's own
declared language.

`data/toc_keywords.json` is **seeded empirically, not hand-guessed**: a
one-off script scans every known true TOC page across the merged corpus,
extracts their opening lines, and reports frequent short heading-like
phrases grouped by each book's declared language. That candidate list
gets a human review pass (filtering noise -- a frequent word that isn't
actually a TOC-heading phrase) before being merged into the committed
`data/toc_keywords.json`. The same mining script can be re-run whenever
the ground-truth corpus grows, to pick up languages or phrasings not yet
represented, without redesigning the matching code.

## Model

Page-level scorer, chosen empirically rather than assumed: train and
compare `LogisticRegression` (with per-fold `StandardScaler`, matching
the old pilot's own finding that this generalizes better than tree
ensembles on small/imbalanced per-book data) against a small
gradient-boosted-trees model, under leave-one-book-out (LOBO)
cross-validation over the merged corpus. Don't assume the old pilot's
conclusion carries over unchanged -- the feature set and label
distribution are both different here (TOC-only, plus new text features)
-- but do start from that prior instead of a blind architecture search.

**Range selection.** A page-level score alone doesn't answer "where is
the TOC" -- score every page in the book, then evaluate contiguous
windows of length 1-4 pages (matches the observed TOC-length distribution
in the existing ground truth: mean 2.8 pages, max 9) by aggregate window
score, and return the top-K non-overlapping windows, ranked, as candidate
TOC page ranges.

## Evaluation

Leave-one-book-out over the full merged corpus.

**Primary metric: top-1 and top-3 range-hit rate** -- does the model's
top-ranked candidate range exactly match (or meaningfully overlap) the
true TOC range, and how often is the true range found somewhere in the
top 3? This matches the actual consumption pattern (a best guess, with a
couple of runner-ups available) better than the old pilot's book-level
"≥90% page recall within a candidate budget" framing, which was tuned for
a downstream LLM-confirmation consumer that doesn't apply here.

**Secondary: page-level precision/recall**, kept as a diagnostic to
understand *why* a book's range prediction failed (e.g. right pages,
wrong window boundary vs. completely missed), not as the pass/fail bar
itself.

**Breakdowns**, since these splits dominated the old pilot's results and
should be checked here too before assuming they don't apply: open-access
vs. copyrighted-scans, and native vs. scanned `extraction_type`.

## Deliverables

New modules under `src/toc_page_classifier/`:

- `layout_features.py` -- `pdfalto`-based per-page geometry extraction.
- `text_features.py` -- structural pattern matching + keyword matching
  (reads `data/toc_keywords.json`).
- `ground_truth.py` -- merges the two label sources into one weighted
  training table.

New scripts under `cli/`:

- `cli/mine_toc_keywords.py` -- the empirical keyword-seeding script
  described above; writes/updates `data/toc_keywords.json` (human-reviewed
  before commit).
- `cli/train_toc_classifier.py` / an evaluation script mirroring the old
  pilot's LOBO harness structure (`evaluate_layout_toc_classifier.py`),
  scoped to TOC-only prediction and range-based output instead of
  per-page candidate-fraction thresholding.

`data/toc_keywords.json` -- committed, human-editable keyword resource
file (see Features above).

Following this repo's own `AGENTS.md` convention, any new `cli/` script
gets a real `argparse` parser and a `cli/README.md` section with its
`--help` dump.
