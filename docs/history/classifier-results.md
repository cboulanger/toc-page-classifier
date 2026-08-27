# Classifier results: history

Chronological record of `cli/train_toc_classifier.py`'s leave-one-book-out
(LOBO) results and the root causes found and fixed along the way. See the
main `README.md`'s "Current status" section for where things stand now --
this file is the "how we got there," kept for anyone who needs the
reasoning behind a since-changed default, feature, or metric.

## 2026-08-25: first measured LOBO result

Trained and leave-one-book-out evaluated over the merged ground truth (89
`chapter_segmentation` books + 95 `dnb_located` pairs; 181 of them have at
least one true TOC page and were scored):

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

Both models scored a flat 0% on the entire open-access corpus. This was
root-caused, not a mystery left dangling: `text_features.py`'s
`_TOC_LINE_RE` requires 2+ consecutive dot/whitespace separator
characters between a TOC entry's title and its page number, but
`pypdf`'s text extraction on born-digital ("native") open-access PDFs
collapses that visual gap to a single space (e.g. `"Foreword vii"`), so
the regex essentially never matched on native-extraction text --
measured directly at 0 of 80 TOC-line-shaped lines sampled across 6 real
open-access books, versus 18.6%/9.1% match rates on the OCR'd
`dnb_located`/`copyrighted-scans` corpora, where literal repeated
separator glyphs do survive extraction.

## 2026-08-26: gap-aware text reconstruction fix, re-measured

`layout_features.extract_gap_aware_page_texts` was added: it reconstructs
each page's text from pdfplumber's own char geometry instead of pypdf's
`extract_text()`, inserting a real 2-character separator wherever the
horizontal gap between consecutive chars is wide relative to font size
(preserving the title/page-number gap that pypdf collapses to one
space), and `cli/train_toc_classifier.py` now uses it. Verified directly
before re-running the full corpus: `toc_line_count` now fires with real,
non-zero values (4-27) on every true TOC page sampled across 6
open-access books that previously scored a hard 0 -- the root cause is
confirmed fixed at the feature level. Fresh full LOBO run:

| Model | Top-1 | Top-3 |
| --- | --- | --- |
| `logistic_regression` (default) | 6.1% | 6.1% |
| `gradient_boosting` | 10.5% | 11.6% |

By corpus:

| Corpus | `logistic_regression` top1 / top3 | `gradient_boosting` top1 / top3 |
| --- | --- | --- |
| copyrighted-scans (n=29) | 10.3% / 10.3% | 20.7% / 24.1% |
| dnb_located (n=95) | 8.4% / 8.4% | 10.5% / 11.6% |
| open-access (n=57) | 0.0% / 0.0% | 5.3% / 5.3% |

By extraction_type (chapter_segmentation rows only):

| extraction_type | `logistic_regression` top1 / top3 | `gradient_boosting` top1 / top3 |
| --- | --- | --- |
| native (n=74) | 2.7% / 2.7% | 9.5% / 10.8% |
| scan (n=12) | 8.3% / 8.3% | 16.7% / 16.7% |

Both models improved overall, and `gradient_boosting` moved off zero on
open-access entirely (0.0%->5.3%, native 2.7%->9.5%/10.8%) -- real
movement, not noise, given the feature-level fix is directly confirmed
above. **`logistic_regression`'s open-access/native numbers did not move
at all (0.0% and 2.7%, identical to before the fix)**, despite the same
underlying feature now carrying real signal for every book in this
corpus. This wasn't a sign the fix failed -- it turned out (see the
2026-08-27 entry below) to be a symptom of the range-selection bug, not
a `logistic_regression`-specific model-behavior gap as first suspected.

Neither model was anywhere close to production-useful at these hit
rates -- this remained a measurement checkpoint, not a finished
deliverable.

## 2026-08-27: the actual bottleneck was range selection, not the model or features

Prompted by the still-poor ~10% hit rates above, two diagnostics were
added to `evaluate_leave_one_book_out`: `best_true_page_rank` (the rank,
by page score, of the best-scored true TOC page -- independent of range
selection) and `top1_overlap`/`top3_overlap` (a loose hit definition
requiring only overlap with the true set, not full coverage). These
revealed that the page-level scorer was already excellent --
`best_true_page_rank` <= 1 for ~94-98% of books, and `top1_overlap` at
94-98% -- while `top1_hit` stayed near 10%. The scorer was finding the
right pages; `range_selection.select_topk_ranges` was failing to turn
that into a correct range.

Root cause: candidate windows were ranked by **mean** score. A real
multi-page TOC's continuation pages (no visible "Contents" heading, so no
`keyword_hit` signal) reliably score lower than the headed first page, but
still meaningfully above baseline -- and folding a lower-but-real page
into a window always pulls its mean down relative to a narrower window
hugging just the peak. Since the design's hit metric already tolerates
over-inclusion (a window only needs to *contain* the true page set, not
match it exactly), switching the ranking criterion to **sum** removes this
bias entirely: including a genuinely-elevated continuation page can only
raise a window's sum, never lower it. `select_topk_ranges`' `max_window`
was also raised from 4 to 6 (covering 179/181 ground-truth books' TOC
spans instead of 169/181), though this alone moved the needle only
slightly -- the sum-vs-mean change was the real fix, confirmed by
re-measuring after each change in isolation.

A second, unrelated bug was found and fixed along the way:
`extract_gap_aware_page_texts` and `extract_page_features` each
independently called `pdfplumber.open()` and re-ran the same per-page
char-grouping -- every book's PDF was being parsed twice for no reason,
at a measured cost of ~1 minute/book (the full corpus's feature-table
build alone was taking 3-4 hours). Merged into one `pdfplumber` pass
(`extract_page_features_and_texts`), and the built feature table is now
cached to `data/feature_table_cache.pkl` (keyed by the ground truth's
book-key set) so re-running with a different `--model`, or after a
model/range-selection-only code change, skips PDF-parsing entirely.

Fresh full LOBO run after all of the above:

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

Both models moved from "barely better than nothing" (5.5-11.6%) to
solidly production-plausible territory (85.6-96.1%), with the
open-access/native slice -- the one that was hard-zero just one day
earlier -- now at 84.2%/96.5% (`logistic_regression`) and 91.2%/91.2%
(`gradient_boosting`). This was the first LOBO result in this project's
history that looked like a usable classifier rather than a measurement
checkpoint.
