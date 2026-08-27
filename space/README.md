---
title: TOC Page Classifier
emoji: 📑
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

Demo for the classifier trained in the
[toc-page-classifier](https://github.com/cboulanger/toc-page-classifier)
repo: upload a full book PDF and see which pages it predicts are the
table of contents. See that repo's README.md for training/evaluation
details (leave-one-book-out hit rates, corpus composition) and its
"Usage" section for calling the same function directly in your own code.

Unlike a fine-tuned-LLM demo, there is no separate model checkpoint to
push here: the classifier is a small (~370KB) scikit-learn model
committed directly inside the `toc_page_classifier` package
(`src/toc_page_classifier/data/model.pkl`). This Space runs entirely on
the free CPU tier -- no GPU, no runtime model download.

`toc_page_classifier/` in this directory is bundled in at deploy time by
`cli/upload_space.py` from the main repo's `src/toc_page_classifier/` --
not maintained as a separate copy here.
