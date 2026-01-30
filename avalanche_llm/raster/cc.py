from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..canon import get_canon


@dataclass(frozen=True)
class ComponentStats:
    size: int
    span_tokens: int
    span_layers: int
    t_min: int | None = None
    t_max: int | None = None
    l_min: int | None = None
    l_max: int | None = None


def _neighbors_4(t: int, l: int) -> Iterable[tuple[int, int]]:
    yield (t - 1, l)
    yield (t + 1, l)
    yield (t, l - 1)
    yield (t, l + 1)


def _neighbors_8(t: int, l: int) -> Iterable[tuple[int, int]]:
    for dt in (-1, 0, 1):
        for dl in (-1, 0, 1):
            if dt == 0 and dl == 0:
                continue
            yield (t + dt, l + dl)


def connected_components(x: np.ndarray, *, adjacency_id: str) -> list[ComponentStats]:
    """
    x: 2D occupancy array [L, T] or [T, L]; this function expects [L, T].
    adjacency_id: ADJ_4N or ADJ_8N
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D")

    canon = get_canon()
    hardfail = int(canon["CONST"]["CC_MAX_COMPONENTS_PER_SEQ_HARDFAIL"])
    adj_4n = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_4N"])
    adj_8n = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_8N"])
    neigh = _neighbors_4 if adjacency_id == adj_4n else _neighbors_8
    if adjacency_id not in {adj_4n, adj_8n}:
        raise ValueError(f"Unknown adjacency_id: {adjacency_id}")

    L, T = x.shape
    x_bool = x.astype(bool)
    visited = np.zeros_like(x_bool, dtype=bool)
    comps: list[ComponentStats] = []

    comp_count = 0
    for l0 in range(L):
        for t0 in range(T):
            if not x_bool[l0, t0] or visited[l0, t0]:
                continue
            stack = [(t0, l0)]
            visited[l0, t0] = True
            comp_count += 1
            if comp_count > hardfail:
                raise ValueError(f"Component count exceeded hard fail threshold: {hardfail}")

            size = 0
            t_min = t0
            t_max = t0
            l_min = l0
            l_max = l0
            while stack:
                t, l = stack.pop()
                size += 1
                if t < t_min:
                    t_min = t
                if t > t_max:
                    t_max = t
                if l < l_min:
                    l_min = l
                if l > l_max:
                    l_max = l
                for tn, ln in neigh(t, l):
                    if tn < 0 or tn >= T or ln < 0 or ln >= L:
                        continue
                    if visited[ln, tn] or not x_bool[ln, tn]:
                        continue
                    visited[ln, tn] = True
                    stack.append((tn, ln))

            span_tokens = t_max - t_min + 1
            span_layers = l_max - l_min + 1
            comps.append(
                ComponentStats(
                    size=int(size),
                    span_tokens=int(span_tokens),
                    span_layers=int(span_layers),
                    t_min=int(t_min),
                    t_max=int(t_max),
                    l_min=int(l_min),
                    l_max=int(l_max),
                )
            )
    return comps


def connected_components_weighted(a: np.ndarray, *, adjacency_id: str) -> list[ComponentStats]:
    """
    a: 2D count array [L, T]; occupancy is a > 0 and component size is sum of a over the component.
    """
    if a.ndim != 2:
        raise ValueError("a must be 2D")

    canon = get_canon()
    hardfail = int(canon["CONST"]["CC_MAX_COMPONENTS_PER_SEQ_HARDFAIL"])
    adj_4n = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_4N"])
    adj_8n = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_8N"])
    neigh = _neighbors_4 if adjacency_id == adj_4n else _neighbors_8
    if adjacency_id not in {adj_4n, adj_8n}:
        raise ValueError(f"Unknown adjacency_id: {adjacency_id}")

    L, T = a.shape
    x_bool = (a > 0).astype(bool)
    visited = np.zeros_like(x_bool, dtype=bool)
    comps: list[ComponentStats] = []

    comp_count = 0
    for l0 in range(L):
        for t0 in range(T):
            if not x_bool[l0, t0] or visited[l0, t0]:
                continue
            stack = [(t0, l0)]
            visited[l0, t0] = True
            comp_count += 1
            if comp_count > hardfail:
                raise ValueError(f"Component count exceeded hard fail threshold: {hardfail}")

            size = 0
            t_min = t0
            t_max = t0
            l_min = l0
            l_max = l0
            while stack:
                t, l = stack.pop()
                size += int(a[l, t])
                if t < t_min:
                    t_min = t
                if t > t_max:
                    t_max = t
                if l < l_min:
                    l_min = l
                if l > l_max:
                    l_max = l
                for tn, ln in neigh(t, l):
                    if tn < 0 or tn >= T or ln < 0 or ln >= L:
                        continue
                    if visited[ln, tn] or not x_bool[ln, tn]:
                        continue
                    visited[ln, tn] = True
                    stack.append((tn, ln))

            span_tokens = t_max - t_min + 1
            span_layers = l_max - l_min + 1
            comps.append(
                ComponentStats(
                    size=int(size),
                    span_tokens=int(span_tokens),
                    span_layers=int(span_layers),
                    t_min=int(t_min),
                    t_max=int(t_max),
                    l_min=int(l_min),
                    l_max=int(l_max),
                )
            )
    return comps
