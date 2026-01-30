from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _stringify_meta(meta: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _pdf_metadata(provenance: dict[str, Any], *, title: str) -> dict[str, str]:
    # Matplotlib's PDF backend accepts a constrained set of keys (Title, Author, Subject, Keywords, ...).
    # Store provenance as JSON in Keywords (reviewer-auditable without custom tooling).
    prov_str = json.dumps(_stringify_meta(provenance), sort_keys=True, ensure_ascii=False)
    return {
        "Title": title,
        "Keywords": prov_str,
    }


def save_figure(
    *,
    fig,
    out_pdf: Path,
    out_png: Path,
    provenance: dict[str, Any] | None,
    dpi: int = 200,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    prov = provenance or {}
    title = out_pdf.stem

    fig.savefig(out_pdf, metadata=_pdf_metadata(prov, title=title))
    # For PNG, matplotlib forwards metadata to Pillow as tEXt chunks.
    fig.savefig(out_png, dpi=int(dpi), metadata=_stringify_meta(prov))

