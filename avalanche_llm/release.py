from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _normalize_exclude_dirs(root: Path, exclude_dirs: list[Path] | None) -> list[Path] | None:
    if not exclude_dirs:
        return None
    out: list[Path] = []
    for ex in exclude_dirs:
        ex_path = ex
        if not ex_path.is_absolute():
            ex_path = root / ex_path
        out.append(ex_path)
    return out


def _iter_files(
    *,
    root: Path,
    manifest_path: Path,
    zip_path: Path,
    include_manifest: bool,
    exclude_dirs: list[Path] | None,
) -> list[Path]:
    exclude_dirs = _normalize_exclude_dirs(root, exclude_dirs)
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if ".git" in p.parts:
            continue
        if exclude_dirs and any(_is_under(p, ex) for ex in exclude_dirs):
            continue
        if p == zip_path:
            continue
        if not include_manifest and p == manifest_path:
            continue
        out.append(p)
    return sorted(out, key=lambda x: x.as_posix())


def _add_dir_entry(zf: zipfile.ZipFile, arc_dir: str) -> None:
    name = arc_dir.replace("\\", "/")
    if not name.endswith("/"):
        name += "/"
    # Avoid duplicates: ZipFile.NameToInfo is populated as we add entries.
    if name in zf.NameToInfo:
        return
    info = zipfile.ZipInfo(filename=name)
    # Mark as a directory on Unix.
    info.external_attr = 0o40775 << 16
    zf.writestr(info, b"")


def _ensure_run_layout_dirs_in_zip(
    zf: zipfile.ZipFile,
    *,
    root: Path,
    canon: dict[str, Any],
    exclude_dirs: list[Path] | None,
) -> None:
    exclude_dirs = _normalize_exclude_dirs(root, exclude_dirs)
    runs_dir = root / str(canon["PATH"]["RUNS_DIR"])
    if not runs_dir.is_dir():
        return

    required_subdirs = [str(v) for v in canon.get("OUTPUT", {}).get("RUN_SUBDIR", {}).values()]
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not run_dir.name.startswith("RUN_"):
            continue
        if exclude_dirs and any(_is_under(run_dir, ex) for ex in exclude_dirs):
            continue
        for sub in required_subdirs:
            abs_sub = run_dir / sub
            if not abs_sub.is_dir():
                continue
            rel = abs_sub.relative_to(root).as_posix()
            _add_dir_entry(zf, rel)


def compute_expected_manifest_text(
    *,
    root: Path,
    canon: dict[str, Any],
    exclude_dirs: list[Path] | None = None,
) -> str:
    exclude_dirs = _normalize_exclude_dirs(root, exclude_dirs)
    manifest_path = root / str(canon["OUTPUT"]["MANIFEST_SHA256_BASENAME"])
    zip_name = f"{canon['PROJECT']['PACK_NAME']}.zip"
    zip_path = root / zip_name

    lines: list[str] = []
    for p in _iter_files(
        root=root,
        manifest_path=manifest_path,
        zip_path=zip_path,
        include_manifest=False,
        exclude_dirs=exclude_dirs,
    ):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = p.relative_to(root).as_posix()
        lines.append(f"{digest}  {rel}")
    return "\n".join(lines) + "\n"


def verify_manifest_matches_recomputation(
    *,
    root: Path,
    canon: dict[str, Any],
    exclude_dirs: list[Path] | None = None,
) -> None:
    exclude_dirs = _normalize_exclude_dirs(root, exclude_dirs)
    manifest_path = root / str(canon["OUTPUT"]["MANIFEST_SHA256_BASENAME"])
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest missing for verification: {manifest_path}")
    expected = compute_expected_manifest_text(root=root, canon=canon, exclude_dirs=exclude_dirs)
    actual = manifest_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if actual != expected:
        raise RuntimeError("MANIFEST.sha256 does not match recomputation")


def write_release_manifest_and_zip(
    *,
    root: Path,
    canon: dict[str, Any],
    exclude_dirs: list[Path] | None = None,
) -> tuple[Path, Path]:
    """
    Create MANIFEST.sha256 and the release zip at repo root.

    The zip includes directory entries for empty required run subdirectories (spec/23),
    so unzipping preserves run layouts even when subdirectories contain no files.
    """
    manifest_path = root / str(canon["OUTPUT"]["MANIFEST_SHA256_BASENAME"])
    zip_name = f"{canon['PROJECT']['PACK_NAME']}.zip"
    zip_path = root / zip_name

    # Regenerate MANIFEST.sha256 (exclude manifest and zip to avoid cyclic dependencies).
    expected = compute_expected_manifest_text(root=root, canon=canon, exclude_dirs=exclude_dirs)
    manifest_path.write_text(expected, encoding="utf-8", newline="\n")

    # Create release zip (exclude itself, include manifest).
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _ensure_run_layout_dirs_in_zip(zf, root=root, canon=canon, exclude_dirs=exclude_dirs)
        for p in _iter_files(
            root=root,
            manifest_path=manifest_path,
            zip_path=zip_path,
            include_manifest=True,
            exclude_dirs=exclude_dirs,
        ):
            rel = p.relative_to(root).as_posix()
            zf.write(p, rel)

    # Windows can leave file handles open briefly; hard-fail if the zip is not present.
    if not zip_path.is_file():
        raise RuntimeError(f"Release zip missing after write: {zip_path}")
    return (manifest_path, zip_path)
