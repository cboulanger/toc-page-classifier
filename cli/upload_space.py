#!/usr/bin/env python3
"""Packages and uploads the Gradio demo (space/) to a Hugging Face Space
repo.

This classifier's whole model is a ~370KB scikit-learn artifact already
committed inside this repo (see src/toc_page_classifier/data/model.pkl),
so there's no separate checkpoint to push anywhere first, and no GPU
hardware to request -- the Space just needs this package installed.
Bundles the whole src/toc_page_classifier/ package into
space/toc_page_classifier/ before uploading (removed again afterwards --
not a separate copy kept in this directory, see space/README.md) so the
demo always ships whatever this repo's own package currently contains.

Also fetches space/examples/*.pdf (the demo's example books) if not
already present on disk -- these are gitignored, not committed to this
repo (they're real full-text books, several MB each), so a fresh
checkout needs this step before its first deploy. Downloaded once, real
static files from then on: app.py's examples no longer depend on OAPEN's
server being reachable (or a bitstream not 500'ing) at demo request time,
and each file's own name -- not just its Gradio example-button label --
is the book's real title, since Gradio derives a File component's
displayed/downloaded name from the example value's own path, not from
example_labels (which only labels the button).

Run (needs a .env file with an HF_TOKEN= line, not committed to this repo):

    uv run python cli/upload_space.py --space-id <your-hf-namespace>/toc-page-classifier-demo \\
        --env-file ~/.env
"""

import argparse
import shutil
import sys
from pathlib import Path

import httpx
from huggingface_hub import HfApi

_ROOT = Path(__file__).resolve().parent.parent
_SPACE_DIR = _ROOT / "space"
_PACKAGE_SRC = _ROOT / "src" / "toc_page_classifier"
_PACKAGE_DEST = _SPACE_DIR / "toc_page_classifier"
_EXAMPLES_DIR = _SPACE_DIR / "examples"

# Real open-access OAPEN books, one per language, drawn from this
# project's own ground-truth corpus (data/corpus/pilot/manifest.json) --
# these were part of the data the shipped model was trained on, so treat
# the demo's examples as a quick illustration of the tool, not a
# held-out accuracy test; uploading your own PDF is that test.
_EXAMPLE_SOURCES = [
    ("https://library.oapen.org/rest/bitstreams/fb942a48-c1a1-4ba9-b859-0e2a1aecdfad/retrieve", "EN — Covid-19 in Asia.pdf"),
    ("https://library.oapen.org/rest/bitstreams/5e7031bd-743f-4474-99fb-5f729792b7a6/retrieve", "DE — Wider die Verunsicherung.pdf"),
    ("https://library.oapen.org/rest/bitstreams/a8ca8e7c-855b-4708-90f2-13892191075f/retrieve", "ES — Resignificar la vida.pdf"),
    ("https://library.oapen.org/rest/bitstreams/5b3bcd76-0b00-49d4-906a-a137b614c602/retrieve", "FR — Discours sur l'éducation au XVIIIe siècle.pdf"),
    ("https://library.oapen.org/rest/bitstreams/563219e8-1d1b-4b22-954e-66947fe1727a/retrieve", "IT — Le lingue della Chiesa.pdf"),
    ("https://library.oapen.org/rest/bitstreams/baade1b2-4ab3-4401-a62b-a7447dfa5dd4/retrieve", "NL — Over de grens.pdf"),
]


def _ensure_examples() -> None:
    _EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for url, filename in _EXAMPLE_SOURCES:
        dest = _EXAMPLES_DIR / filename
        if dest.exists():
            continue
        print(f"Fetching example {filename!r}")
        # OAPEN's bitstream endpoint has returned a transient HTTP 500 to a
        # plain urllib request with no User-Agent during testing -- a
        # browser-like one avoided it every time since.
        response = httpx.get(url, follow_redirects=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        dest.write_bytes(response.content)


def _read_token(env_file: Path) -> str:
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            token = line[len("HF_TOKEN=") :].strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError(f"No non-empty HF_TOKEN= line found in {env_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space-id", required=True, help="e.g. <your-hf-namespace>/toc-page-classifier-demo")
    parser.add_argument("--env-file", required=True, type=Path, help=".env file with an HF_TOKEN= line")
    parser.set_defaults(private=True)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true", help="default")
    visibility.add_argument("--public", dest="private", action="store_false")
    args = parser.parse_args()

    if not _SPACE_DIR.is_dir():
        print(f"No such directory: {_SPACE_DIR}", file=sys.stderr)
        return 1

    token = _read_token(args.env_file)
    api = HfApi(token=token)

    _ensure_examples()
    shutil.copytree(_PACKAGE_SRC, _PACKAGE_DEST, dirs_exist_ok=True)
    try:
        print(f"Creating/reusing Space {args.space_id} (private={args.private})")
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        print(f"Uploading {_SPACE_DIR} -> {args.space_id}")
        api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=str(_SPACE_DIR))
    finally:
        shutil.rmtree(_PACKAGE_DEST, ignore_errors=True)

    print(f"Done: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
