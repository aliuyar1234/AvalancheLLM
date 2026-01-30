from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..canon import get_canon


@dataclass(frozen=True)
class PermNullResult:
    a_perm: np.ndarray
    x_perm: np.ndarray


def _seed_from_fields(fields: list[str]) -> int:
    h = hashlib.sha256()
    for f in fields:
        h.update(f.encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest()[:8], 16)


def within_layer_time_permutation(
    a: np.ndarray, *, run_id: str, cond_id: int, seq_id: int
) -> PermNullResult:
    """
    Apply within-layer token permutation independently per layer.
    a: uint16 array [L, T]
    """
    if a.ndim != 2:
        raise ValueError("a must be [L, T]")
    canon = get_canon()
    seed0 = int(canon["CONST"]["BOOTSTRAP_SEED"])

    L, T = a.shape
    a_perm = np.empty_like(a)
    for l in range(L):
        seed = _seed_from_fields([str(seed0), str(run_id), str(cond_id), str(seq_id), str(l)])
        rng = np.random.default_rng(seed)
        pi = rng.permutation(T)
        a_perm[l, :] = a[l, pi]
        # Invariant: sorted marginals preserved
        if not np.array_equal(np.sort(a[l, :]), np.sort(a_perm[l, :])):
            raise ValueError("Marginal preservation failed")
    x_perm = (a_perm > 0).astype(np.uint8)
    return PermNullResult(a_perm=a_perm, x_perm=x_perm)


def within_layer_time_circular_shift(
    a: np.ndarray, *, run_id: str, cond_id: int, seq_id: int
) -> PermNullResult:
    """
    Apply a circular time shift independently per layer.
    This preserves per-layer marginals exactly while preserving within-layer autocorrelation structure.

    a: uint16 array [L, T]
    """
    if a.ndim != 2:
        raise ValueError("a must be [L, T]")
    canon = get_canon()
    seed0 = int(canon["CONST"]["BOOTSTRAP_SEED"])

    L, T = a.shape
    a_perm = np.empty_like(a)
    for l in range(L):
        seed = _seed_from_fields([str(seed0), str(run_id), str(cond_id), str(seq_id), "circ_shift", str(l)])
        rng = np.random.default_rng(seed)
        if T <= 1:
            shift = 0
        else:
            shift = int(rng.integers(1, T))
        a_perm[l, :] = np.roll(a[l, :], shift=shift)
        if not np.array_equal(np.sort(a[l, :]), np.sort(a_perm[l, :])):
            raise ValueError("Marginal preservation failed")
    x_perm = (a_perm > 0).astype(np.uint8)
    return PermNullResult(a_perm=a_perm, x_perm=x_perm)
