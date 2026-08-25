# Agent conventions for this repository

## Keep `cli/README.md` in sync with the scripts it documents

`cli/README.md` has one section per script under `cli/`: a short
explanation plus a verbatim `--help` dump. Whenever you change a CLI
script's arguments, docstring, or behavior in a way that changes what
`--help` prints (or what the short explanation above it should say), update
that script's section in `cli/README.md` in the same change -- regenerate
the dump with:

```bash
uv run python cli/<script>.py --help
```

and paste the output into the fenced code block, updating the prose above
it if the explanation itself is now stale. Do this before committing, not
as a follow-up -- a `cli/README.md` section that describes an old set of
flags is worse than no documentation at all.

If you add a new script under `cli/`, give it a real `argparse` parser
(even with zero flags) so `--help` is always safe to run -- several
scripts originally had none, which meant `--help` silently ran the script
for real instead of printing usage.

## Never let `cli/match_dnb_oa.py` write to `data/corpus/pilot/manifest.json`

That file is owned by `cli/discover_oa_dnb_candidates.py`. `match_dnb_oa.py`
is a narrower, offline-only sanity check and defaults its `--out` elsewhere
specifically so it can't silently clobber the real manifest -- don't
"simplify" this by pointing it back at `manifest.json`.
