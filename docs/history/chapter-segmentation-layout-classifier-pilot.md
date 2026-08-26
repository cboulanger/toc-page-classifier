# Prior art: the layout-only classifier pilot in `chapter-segmentation`

**Historical document, imported verbatim-in-spirit (condensed) from
`chapter-segmentation`'s `evaluation/experiments/toc-classifier-pilot.md`
on 2026-08-26, when that pilot was retired from the `chapter-segmentation`
repo.** This repo (`toc-page-classifier`) is not a continuation of that
pilot -- it uses a different ground-truth source (DNB TOC scans matched to
open-access full text via lobid, not hand-transcribed `.expected.json`
files) and a different feature set (pdfplumber geometry + text/keyword
features, not ALTO-XML-only geometry). It's preserved here because the
failure modes and methodology lessons below are directly relevant to this
repo's own classifier work, and the original document no longer exists in
`chapter-segmentation` once this file was written.

## What it was

`evaluate_layout_toc_classifier.py` in `chapter-segmentation` trained a
leave-one-book-out (LOBO) `LogisticRegression` classifier on purely
geometric per-page layout features derived from `pdfalto`-generated ALTO
XML (font-size ratios, line density, left margin, vertical position,
book-context features like `edge_distance` -- no text content at all) to
identify table-of-contents pages and chapter-opening pages, as a candidate
pre-filter for a downstream stage (e.g. an LLM confirmation pass). Decision
bar: ≥90% `full_recall_fraction` (share of books whose predicted candidates
caught ~all true TOC + chapter-opening pages) at ≤15% `avg_candidate_fraction`
(how much the candidate set narrowed the page list).

## Bottom line

**Never cleared its own bar on the full mixed corpus, across eleven
follow-up investigations over about two weeks of work.** Best full-corpus
result: 65.2% full_recall_fraction / 10.3% avg_candidate_fraction (16
features including `edge_distance`, `LogisticRegression`,
`recall_target=0.90`). The **open-access corpus alone** did clear the bar
once, in isolation, at a hand-tuned `recall_target=0.988` (94.7%/13.8%) --
explicitly not proposed as a global default, since `copyrighted-scans`'
candidate-fraction cost rises much faster with `recall_target` than
open-access's does.

Persistent structural failure modes, never resolved: `copyrighted-scans`
(OCR'd) systematically weaker font-size/layout signal than born-digital
`open-access`; some open-access TOC layouts don't match the training
majority; a residual scan `chapter_first_recall` ceiling. Named-but-unscoped
future directions: TOC-anchored chapter matching (parse a detected TOC page,
locate chapter openings by title/page-number matching against it), and
document-image deep learning bypassing OCR font metadata entirely.

## Findings that generalize beyond this specific pilot

- **A linear model (`LogisticRegression` + `StandardScaler`) generalized
  across held-out books far better than a tree-based one
  (`HistGradientBoostingClassifier`)** for this kind of small, imbalanced,
  per-page geometric feature set. Root cause: tree models calibrate a
  threshold against one fold's own leaf-region probability distribution,
  and a held-out book whose feature distribution sits slightly off from
  that fold's training pool can fall into a threshold gap even with a real,
  obvious signal on the page. A smooth linear score has no such
  discontinuity. `RandomForestClassifier` was even worse --
  badly-miscalibrated `predict_proba` on this data shape made its accept
  threshold reject almost everything on held-out books.
- **Document-relative top-K candidate selection beat a single globally
  calibrated probability threshold, on every axis, with no per-corpus
  retuning.** Instead of picking one absolute probability cutoff from
  training data and applying it uniformly (which made per-book candidate
  volume swing from under 2% to over 58% of a book's pages, depending on
  how separated that book's own score distribution happened to be),
  ranking each held-out book's own pages by score and taking the top
  `candidate_fraction_cap` share (e.g. 15%) needs no ground truth for the
  document being scored -- only its own model output -- and made per-book
  candidate volume land within ~1 point of the target cap for every book,
  while also reaching equal-or-better recall than any single global
  threshold found in the whole investigation. This repo's own
  `range_selection.select_topk_ranges` (top-K non-overlapping page ranges)
  is an independent design that arrived at a related idea -- window-level
  rather than page-level top-K -- for the same underlying reason: a fixed
  global cutoff doesn't transfer well across documents with different score
  separation.
- **Corpus growth saturates once the model is already fitting the
  well-represented case.** A learning-curve check (fixed held-out test set,
  retrained on random subsets at increasing pool sizes) found
  `full_recall_fraction` flat within noise across pool sizes 10-35 books,
  despite the model having very few parameters (unlikely to be
  data-starved). More books *like the ones already well-represented* add
  little; the underrepresented templates the model still fails on are where
  new ground truth actually moves the numbers. If this repo's own LOBO
  numbers plateau despite a growing corpus, check whether new books are
  actually diversifying the hard cases before assuming more data helps.
- **Absolute edge-distance beat fractional page-position as a "is this
  near the front or back of the book" feature.** A TOC near the very back
  of a book (common in some French-language traditions) was invisible to a
  monotonic `page_index / total_pages` feature, which can only reward one
  direction. `edge_distance = min(page_index, total_pages - 1 - page_index)`
  (0 at either edge, rising toward the middle) fixed it outright, and its
  correlation analysis first (front-matter length before a TOC barely
  scales with book length; TOC length scales moderately with book length)
  is worth re-checking before adding an analogous feature here.
- **Per-feature regularization (L2 strength) was tested directly and ruled
  out as a fix for one feature dominating the model.** Swept `C` over three
  orders of magnitude with zero effect on outcomes, because uniform L2
  shrinkage rescales every coefficient together without changing pages'
  *relative* ranking within a book -- and ranking, not absolute probability,
  is what a per-fold recalibrated threshold actually uses. Only far more
  extreme regularization (well past any normal hyperparameter search range)
  eventually changed anything, and even then only weakly. If a feature
  looks like it's dominating a linear model's decision here, this pilot's
  experience says the fix is more likely in feature engineering or
  threshold/selection strategy than in regularization.
- **A feature that looks separative in isolation on a handful of
  hand-picked target books can still be net-negative once wired into the
  full model.** Happened twice (`max_font_vpos_fraction`, `early_gap_ratio`)
  -- both showed real, measured separation on the specific books they were
  designed to fix, and both made the full LOBO numbers worse or neutral
  once actually added, at the operating point that mattered. Only a full
  LOBO re-run, not an isolated single-feature check, is a reliable signal
  for whether a new feature helps.
- **A ground-truth labeling bug can look exactly like a classifier failure
  until traced by hand.** One "0% toc_recall" book turned out to have a
  TOC page mislabeled as a `chapter_first` page in its own ground truth
  (an "Inhalt" section that should have been marked `{"skip": true}` during
  transcription, per that project's own documented convention, but wasn't)
  -- overwriting the true label and injecting a false positive at the same
  time. Worth ruling out before spending more effort on a stubborn
  zero-recall book here.
- **`extraction_type` (native vs. scan) is not the same split as
  which corpus/source a book came from**, and calibrating a threshold
  separately per `extraction_type` (rather than per corpus, or globally)
  bought a real +9-point full-recall gain at matched candidate budget in
  one measured comparison -- worth remembering if this repo ever
  reintroduces a global threshold/calibration knob instead of (or alongside)
  its current per-document top-K range selection.

## What did not carry over / does not apply here

This repo's classifier already differs from the pilot in ways that make
several of its specific numeric conclusions (recall_target values, exact
feature list, `pdfalto`/ALTO dependency) not directly reusable: different
ground-truth source and scale, different feature extraction library
(pdfplumber vs. pdfalto/ALTO XML), and this repo's own text/keyword
features (which the pilot deliberately excluded by design, to isolate pure
layout signal). Read this document for the *shape* of what was tried and
what generalized, not as a source of constants to copy in.
