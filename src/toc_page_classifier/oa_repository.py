"""Resolving an OAPEN/DOAB handle to an actual downloadable full-text PDF.

Both repositories run the same DSpace 6 REST API
(`/rest/handle/{handle}?expand=bitstreams`), so one function covers both --
callers pass the right `rest_base` for the repository the handle came from.
"""

import json
from dataclasses import dataclass

from .oai_harvest import fetch_url


@dataclass
class ResolvedPdf:
    url: str
    size_bytes: int | None
    license_code: str | None


def resolve_pdf_bitstream(rest_base: str, handle: str) -> ResolvedPdf | None:
    """Looks up the item's bitstreams and returns the first PDF found, or
    None if the repository hosts no direct copy (common for DOAB, which
    often only links out to a publisher's own site -- see
    `harvest_doab.py`'s docstring)."""
    url = f"{rest_base}/rest/handle/{handle}?expand=bitstreams"
    data = json.loads(fetch_url(url))
    for bitstream in data.get("bitstreams") or []:
        if bitstream.get("mimeType") == "application/pdf":
            return ResolvedPdf(
                url=rest_base + bitstream["retrieveLink"],
                size_bytes=bitstream.get("sizeBytes"),
                license_code=bitstream.get("code"),
            )
    return None
