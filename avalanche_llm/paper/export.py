from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..canon import get_canon


class PaperExportError(RuntimeError):
    pass


def validate_no_forbidden_markers(paper_dir: Path) -> None:
    patterns = [
        "to do",
        "to be determined",
        "...",
        "<",
        ">",
    ]
    for md in paper_dir.glob("*.md"):
        text = md.read_text(encoding="utf-8").lower()
        for pat in patterns:
            if pat in text:
                raise PaperExportError(f"Forbidden marker {pat!r} in {md}")


def _load_bib_keys(bib_path: Path) -> set[str]:
    if not bib_path.is_file():
        raise PaperExportError(f"Missing bib file: {bib_path}")
    text = bib_path.read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", text))
    return {k.strip() for k in keys if k.strip()}


def validate_citations(paper_dir: Path, bib_path: Path) -> None:
    keys = _load_bib_keys(bib_path)
    cite_pat = re.compile(r"\[@([A-Za-z0-9_:-]+)\]")
    for md in paper_dir.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        for k in cite_pat.findall(text):
            if k not in keys:
                raise PaperExportError(f"Missing bib key {k} referenced in {md}")


def update_paper_provenance_footnotes(
    paper_path: Path,
    *,
    evidence: dict[str, dict[str, Any]],
) -> None:
    """
    Insert or replace a provenance footer block with per-artifact footnotes.
    """
    text = paper_path.read_text(encoding="utf-8")
    begin = "PROVENANCE_BEGIN"
    end = "PROVENANCE_END"

    keys = sorted(evidence.keys())
    artifacts_line = ", ".join([f"{k}[^{k}]" for k in keys])

    lines = [begin, "", "## Artifact provenance", ""]
    if keys:
        lines.append(f"Artifacts: {artifacts_line}")
        lines.append("")
        for k in keys:
            e = evidence.get(k, {})
            run_id = e.get("run_id")
            config_hash = e.get("config_hash")
            if not run_id or not config_hash:
                raise PaperExportError(f"Missing provenance for {k} when updating {paper_path}")
            lines.append(f"[^{k}]: run_id={run_id} config_hash={config_hash}")
    else:
        lines.append("Artifacts: none")
    lines += ["", end, ""]
    block = "\n".join(lines)

    if begin in text and end in text:
        pre, rest = text.split(begin, 1)
        _, post = rest.split(end, 1)
        out = pre.rstrip("\n") + "\n" + block + post
    else:
        out = text.rstrip("\n") + "\n\n" + block

    paper_path.write_text(out, encoding="utf-8", newline="\n")


def validate_artifact_references(
    paper_dir: Path,
    *,
    run_dirs: list[Path],
) -> None:
    canon = get_canon()
    fig_pdf = canon["OUTPUT"]["FIG_FILE_PDF"]
    fig_png = canon["OUTPUT"]["FIG_FILE_PNG"]
    tab_csv = canon["OUTPUT"]["TABLE_FILE_CSV"]
    tab_parq = canon["OUTPUT"]["TABLE_FILE_PARQUET"]

    id_pat = re.compile(r"\b([FT]\d{2}_[A-Z0-9_]+)\b")
    referenced: set[str] = set()
    for md in paper_dir.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        referenced.update(id_pat.findall(text))

    for k in sorted(referenced):
        if k in fig_pdf and k in fig_png:
            rel_pdf = Path(str(fig_pdf[k]))
            rel_png = Path(str(fig_png[k]))
            ok = any((run_dir / rel_pdf).is_file() and (run_dir / rel_png).is_file() for run_dir in run_dirs)
            if not ok:
                raise PaperExportError(f"Missing figure artifacts for {k} in producing runs")
        elif k in tab_csv and k in tab_parq:
            rel_csv = Path(str(tab_csv[k]))
            rel_parq = Path(str(tab_parq[k]))
            ok = any((run_dir / rel_csv).is_file() and (run_dir / rel_parq).is_file() for run_dir in run_dirs)
            if not ok:
                raise PaperExportError(f"Missing table artifacts for {k} in producing runs")
        else:
            raise PaperExportError(f"Unrecognized artifact id referenced in paper: {k}")


def update_paper_snapshot(
    paper_snapshot_path: Path,
    *,
    evidence: dict[str, dict[str, Any]],
) -> None:
    """
    Insert or replace a provenance block in paper/00_PAPER_SNAPSHOT.md.
    """
    text = paper_snapshot_path.read_text(encoding="utf-8")
    begin = "PROVENANCE_BEGIN"
    end = "PROVENANCE_END"

    lines = [begin, "", "## Provenance", ""]
    for k in sorted(evidence.keys()):
        e = evidence[k]
        lines.append(f"- {k}: run_id={e.get('run_id')} config_hash={e.get('config_hash')}")
    lines += ["", end, ""]
    block = "\n".join(lines)

    if begin in text and end in text:
        pre, rest = text.split(begin, 1)
        _, post = rest.split(end, 1)
        out = pre.rstrip("\n") + "\n" + block + post
    else:
        out = text.rstrip("\n") + "\n\n" + block

    paper_snapshot_path.write_text(out, encoding="utf-8", newline="\n")


def validate_snapshot_provenance(
    paper_snapshot_path: Path,
    *,
    evidence: dict[str, dict[str, Any]],
    required_keys: set[str],
) -> None:
    text = paper_snapshot_path.read_text(encoding="utf-8")
    missing: list[str] = []
    for k in sorted(required_keys):
        e = evidence.get(k, {})
        run_id = e.get("run_id")
        config_hash = e.get("config_hash")
        if not run_id or not config_hash:
            missing.append(k)
            continue
        needle = f"{k}: run_id={run_id} config_hash={config_hash}"
        if needle not in text:
            missing.append(k)
    if missing:
        raise PaperExportError(f"Paper snapshot missing required provenance entries: {', '.join(missing)}")
