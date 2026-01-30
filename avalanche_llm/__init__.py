"""Avalanche LLM ICLR pack implementation."""

from __future__ import annotations

__all__ = [
    "__version__",
]


def __version__() -> str:
    from .canon import get_canon

    canon = get_canon()
    return str(canon["PROJECT"]["PACK_VERSION"])

