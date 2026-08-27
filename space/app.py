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

import gradio as gr

from toc_page_classifier.predict import locate_toc_pages

_MAX_PREVIEW_PAGES = 6

# Real open-access OAPEN books, one per language, drawn from this
# project's own ground-truth corpus (data/corpus/pilot/manifest.json) --
# these were part of the data the shipped model was trained on, so treat
# them as a quick illustration of the tool, not a held-out accuracy test;
# upload your own PDF for that. OAPEN's bitstream URLs have no filename
# or extension (they all end in plain "/retrieve"), so each entry also
# carries the book's own title -- from the same manifest -- as its
# example_labels caption; without one, Gradio falls back to showing the
# meaningless literal "retrieve" for every example.
_EXAMPLES = [
    ("https://library.oapen.org/rest/bitstreams/fb942a48-c1a1-4ba9-b859-0e2a1aecdfad/retrieve", "EN — Covid-19 in Asia"),
    ("https://library.oapen.org/rest/bitstreams/5e7031bd-743f-4474-99fb-5f729792b7a6/retrieve", "DE — Wider die Verunsicherung"),
    ("https://library.oapen.org/rest/bitstreams/a8ca8e7c-855b-4708-90f2-13892191075f/retrieve", "ES — Resignificar la vida"),
    ("https://library.oapen.org/rest/bitstreams/5b3bcd76-0b00-49d4-906a-a137b614c602/retrieve", "FR — Discours sur l'éducation au XVIIIe siècle"),
    ("https://library.oapen.org/rest/bitstreams/563219e8-1d1b-4b22-954e-66947fe1727a/retrieve", "IT — Le lingue della Chiesa"),
    ("https://library.oapen.org/rest/bitstreams/baade1b2-4ab3-4401-a62b-a7447dfa5dd4/retrieve", "NL — Over de grens"),
]


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
    # No file_types restriction: OAPEN's example URLs resolve to real PDFs
    # but have no ".pdf" (or any) extension, which would otherwise fail
    # Gradio's client-side extension check before predict() ever runs --
    # actual content validation happens inside predict() via pdfplumber.
    pdf_input = gr.File(label="Book PDF", type="filepath")
    run_btn = gr.Button("Locate TOC", variant="primary")
    gr.Examples(
        examples=[[url] for url, _label in _EXAMPLES],
        example_labels=[label for _url, label in _EXAMPLES],
        inputs=pdf_input,
    )
    summary_output = gr.Textbox(label="Result")
    gallery_output = gr.Gallery(label="Predicted pages", columns=3)
    run_btn.click(predict, inputs=pdf_input, outputs=[summary_output, gallery_output])

if __name__ == "__main__":
    demo.launch()
