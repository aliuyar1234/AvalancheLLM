from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class CanonError(RuntimeError):
    pass


def _find_canon_file(start: Path) -> Path:
    env_path = os.environ.get("AVALANCHE_LLM_CANON_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        raise CanonError(f"AVALANCHE_LLM_CANON_PATH does not exist: {p}")

    candidates = list(start.rglob("00_CANONICAL.md"))
    if candidates:
        if len(candidates) > 1:
            raise CanonError(
                "Multiple 00_CANONICAL.md files found; set AVALANCHE_LLM_CANON_PATH. "
                f"Found: {candidates}"
            )
        return candidates[0]

    raise CanonError(
        "Could not locate 00_CANONICAL.md from current working directory; "
        "set AVALANCHE_LLM_CANON_PATH."
    )


def _extract_yaml_block(text: str) -> str:
    match = re.search(r"```yaml\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not match:
        raise CanonError("Could not find ```yaml ... ``` block in 00_CANONICAL.md")
    return match.group(1)


@lru_cache(maxsize=1)
def get_canon() -> dict[str, Any]:
    canon_path = _find_canon_file(Path.cwd())
    text = canon_path.read_text(encoding="utf-8")
    yaml_block = _extract_yaml_block(text)
    parsed = yaml.safe_load(yaml_block)
    if not isinstance(parsed, dict) or "CANON" not in parsed:
        raise CanonError("Parsed CANON registry missing top-level CANON key")
    canon = parsed["CANON"]
    if not isinstance(canon, dict):
        raise CanonError("CANON registry is not a mapping")
    return canon


def canon_get(path: str) -> Any:
    """
    Resolve a string like "CANON.CONST.SEQ_LEN_TOKENS" against the registry.
    """
    if not path.startswith("CANON."):
        raise CanonError(f"Not a CANON path: {path}")
    canon = get_canon()
    parts = path.split(".")[1:]
    cur: Any = canon
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        raise CanonError(f"Missing CANON key path: {path}")
    return cur
