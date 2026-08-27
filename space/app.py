#!/usr/bin/env python3
"""Gradio demo: given a full book PDF, predicts which pages are its table
of contents, using toc_page_classifier.predict.locate_toc_pages.

Runs entirely on CPU -- no GPU, and no model download at request time:
the model is a small (~370KB) scikit-learn artifact checked into the
toc_page_classifier package itself (src/toc_page_classifier/data/model.pkl;
see the main repo's README.md "Usage" section). This Space just needs
that package installed, which cli/upload_space.py handles by bundling
the whole src/toc_page_classifier/ directory into this one at deploy
time (not maintained as a separate copy here -- see this directory's
README.md).
"""

from pathlib import Path

import gradio as gr

from toc_page_classifier.predict import locate_toc_pages

_MAX_PREVIEW_PAGES = 6

# Real open-access OAPEN books, one per language, drawn from this
# project's own ground-truth corpus (data/corpus/pilot/manifest.json) --
# these were part of the data the shipped model was trained on, so treat
# them as a quick illustration of the tool, not a held-out accuracy test;
# upload your own PDF for that.
#
# Fetched once by cli/upload_space.py into this local examples/ directory
# (gitignored -- these are real full-text books, several MB each) rather
# than referenced by their live OAPEN URL: Gradio derives a File
# component's displayed/downloaded name from the example value's own
# path, not from gr.Examples' example_labels (which only labels the
# button) -- so a URL example would always show as OAPEN's extension-less
# "retrieve", no matter what label it's given. Each file's own name here
# -- "LANG — Title.pdf" -- is what gets shown and downloaded once
# selected, and each example_labels entry below (its filename stem)
# matches it for the button too, deliberately keeping one source of
# truth (this directory's actual filenames) rather than two.
_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
_EXAMPLE_FILES = sorted(_EXAMPLES_DIR.glob("*.pdf")) if _EXAMPLES_DIR.is_dir() else []


def predict(pdf_path: str):
    if not pdf_path:
        return "Upload a PDF first.", []

    try:
        page_indices = locate_toc_pages(pdf_path)
    except Exception as exc:  # noqa: BLE001 -- surface any extraction failure to the user, not a 500
        return f"Could not process this PDF: {exc}", []

    if not page_indices:
        return "No table of contents found (or the PDF has no extractable text).", []

    summary = f"Predicted table-of-contents pages (0-indexed): {page_indices}"

    previews = []
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_index in page_indices[:_MAX_PREVIEW_PAGES]:
                image = pdf.pages[page_index].to_image(resolution=120).original
                previews.append((image, f"page {page_index}"))
    except Exception:
        pass  # the preview is a bonus -- the page-index summary above is the real answer

    return summary, previews


with gr.Blocks(title="TOC Page Classifier") as demo:
    gr.Markdown(
        "# TOC page classifier\n"
        "Upload a full book PDF and this predicts which pages are its "
        "table of contents -- for use as a preprocessing step before a "
        "downstream TOC parser, not as an end-user reader. The examples "
        "below are real open-access books, one per language.\n\n"
        "See [toc-page-classifier](https://github.com/cboulanger/toc-page-classifier) "
        "for training/evaluation details and the library API."
    )
    pdf_input = gr.File(label="Book PDF", type="filepath")
    run_btn = gr.Button("Locate TOC", variant="primary")
    gr.Examples(
        examples=[[str(path)] for path in _EXAMPLE_FILES],
        example_labels=[path.stem for path in _EXAMPLE_FILES],
        inputs=pdf_input,
    )
    summary_output = gr.Textbox(label="Result")
    gallery_output = gr.Gallery(label="Predicted pages", columns=3)
    run_btn.click(predict, inputs=pdf_input, outputs=[summary_output, gallery_output])
    # Selecting a new file (an example, a fresh upload, or clearing the
    # input) should drop any prior result immediately, not leave the
    # previous book's prediction on screen looking like it belongs to
    # the newly-selected one until "Locate TOC" is pressed again.
    pdf_input.change(lambda: ("", []), outputs=[summary_output, gallery_output])

if __name__ == "__main__":
    demo.launch()
