from __future__ import annotations

import ast
from pathlib import Path

from avalanche_llm.canon import get_canon


def _iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def test_forbidden_marker_scan() -> None:
    canon = get_canon()
    spec_dir = str(canon["PATH"]["SPEC_DIR"])
    patterns = [
        "to do",
        "to be determined",
        "...",
        "<",
        ">",
    ]
    root = Path.cwd()
    for md in root.rglob("*.md"):
        if spec_dir in md.parts:
            continue
        text = md.read_text(encoding="utf-8").lower()
        for pat in patterns:
            assert pat not in text, f"Forbidden marker {pat!r} in {md}"


def test_sdr_scan_no_canon_string_literals_in_python() -> None:
    canon = get_canon()
    # SDR scan is concerned with drift-prone literals (paths, outputs, command strings),
    # not symbolic IDs used as internal keys (e.g. "T01_SUMMARY").
    sdr_roots = []
    for k in ("PROJECT", "PATH", "CLI", "OUTPUT", "MODEL", "DATASET", "CONST"):
        v = canon.get(k)
        if v is not None:
            sdr_roots.append(v)
    canon_strings = {s for root in sdr_roots for s in _iter_strings(root) if len(s) >= 4}

    root = Path.cwd() / "avalanche_llm"
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lit = node.value
                assert lit not in canon_strings, f"Found CANON literal {lit!r} in {py}"
