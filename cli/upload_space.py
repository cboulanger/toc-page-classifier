#!/usr/bin/env python3
"""Packages and uploads the Gradio demo (space/) to a Hugging Face Space
repo.

Unlike a fine-tuned-LLM demo, this classifier's whole model is a ~370KB
scikit-learn artifact already committed inside this repo (see
src/toc_page_classifier/data/model.pkl) -- there's no separate
HPC-trained checkpoint to push to its own model repo first, and no GPU
hardware to request; the Space just needs this package installed.
Bundles the whole src/toc_page_classifier/ package into
space/toc_page_classifier/ before uploading (removed again afterwards --
not a separate copy kept in this directory, see space/README.md) so the
demo always ships whatever this repo's own package currently contains.

Run (needs a .env file with an HF_TOKEN= line, not committed to this repo):

    uv run python cli/upload_space.py --space-id <your-hf-namespace>/toc-page-classifier-demo \\
        --env-file ~/.env
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

_ROOT = Path(__file__).resolve().parent.parent
_SPACE_DIR = _ROOT / "space"
_PACKAGE_SRC = _ROOT / "src" / "toc_page_classifier"
_PACKAGE_DEST = _SPACE_DIR / "toc_page_classifier"


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
